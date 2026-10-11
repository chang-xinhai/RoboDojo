"""DepthUMI-compatible active stereo exposure and sensor response.

The ambient images come from each independent IR camera. Projector response
comes from its linear HDR on/off buffers. This is a visible-renderer sensor
proxy, without a claim of spectral NIR material transport.
"""

from contextlib import contextmanager

import numpy as np


def grayscale_linear(image):
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError("Sensor radiance must be HWC with at least three channels")
    return image[..., :3] @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def projector_response(on, off):
    if np.asarray(on).shape != np.asarray(off).shape:
        raise ValueError("HDR on/off images must have matching shapes")
    return np.maximum(grayscale_linear(on) - grayscale_linear(off), 0)


def validate_ir_config(config):
    for key in ("saturation_radiance", "full_well_electrons"):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if not isinstance(config["adc_bits"], int) or not 1 <= config["adc_bits"] <= 16:
        raise ValueError("adc_bits must be an integer between 1 and 16")
    for key in ("ambient_scale", "shot_noise_scale", "read_noise_electrons"):
        if not np.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")


def shared_ir_gain(ambient_images, on_images, off_images, config):
    validate_ir_config(config)
    ae = config.get("auto_exposure", {})
    if not ae.get("enabled", False):
        return 1.0
    if not ae.get("stereo_pair_shared", False):
        raise ValueError("IR exposure must be shared between stereo eyes")
    if not (len(ambient_images) == len(on_images) == len(off_images) == 2):
        raise ValueError("IR exposure requires two corresponding eye images")
    percentile = float(ae["measurement_percentile"])
    target = float(ae["target_dn"])
    lower, upper = float(ae["minimum_gain"]), float(ae["maximum_gain"])
    if not 0 < percentile < 100 or not 0 < target <= 255 or not 0 < lower <= upper:
        raise ValueError("Invalid bounded IR exposure configuration")
    response = []
    for ambient, on, off in zip(ambient_images, on_images, off_images, strict=True):
        passive = np.clip(grayscale_linear(ambient), 0, 255).astype(np.uint8).astype(np.float32)
        active = np.nan_to_num(projector_response(on, off), nan=0, posinf=config["saturation_radiance"], neginf=0)
        response.append(passive / 255 * config["ambient_scale"] + active / config["saturation_radiance"])
    measured = float(np.percentile(np.concatenate([a.reshape(-1) for a in response]), percentile))
    return upper if measured <= np.finfo(np.float32).eps else float(np.clip(target / 255 / measured, lower, upper))


def simulate_ir(ambient, on, off, config, rng, *, exposure_gain=1.0):
    """Apply the declared electron/ADC model, preserving real passive texture."""
    validate_ir_config(config)
    if not np.isfinite(exposure_gain) or exposure_gain <= 0:
        raise ValueError("IR exposure gain must be positive and finite")
    response = np.nan_to_num(projector_response(on, off), nan=0, posinf=config["saturation_radiance"], neginf=0)
    full_well = float(config["full_well_electrons"])
    adc_max = (1 << config["adc_bits"]) - 1
    electrons = exposure_gain * response * (full_well / config["saturation_radiance"])
    electrons += rng.normal(0, np.sqrt(electrons) * config["shot_noise_scale"])
    electrons += rng.normal(0, config["read_noise_electrons"], electrons.shape)
    passive = np.clip(grayscale_linear(ambient), 0, 255).astype(np.uint8).astype(np.float32)
    dn = exposure_gain * passive / 255 * adc_max * config["ambient_scale"] + electrons / full_well * adc_max
    saturated = dn >= adc_max
    image = np.rint(np.clip(np.rint(dn), 0, adc_max) / adc_max * 255).astype(np.uint8)
    return image, saturated, response


@contextmanager
def scoped_render_settings(settings, overrides):
    """Restore all renderer settings, including after failed capture phases."""
    original = {key: settings.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            settings.set(key, value)
        yield
    finally:
        for key, value in original.items():
            if value is None:
                settings.destroy_item(key)
            else:
                settings.set(key, value)
