"""Physical proxy response, phase scope, and independent-render averaging."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


response = load_module("sensor_response", ROOT / "env/camera_manager/sensor_response.py")
sl = load_module("structured_light_response", ROOT / "env/camera_manager/structured_light.py")
CFG = {
    "saturation_radiance": 5500.0,
    "full_well_electrons": 12000.0,
    "adc_bits": 10,
    "ambient_scale": 0.85,
    "shot_noise_scale": 0.0,
    "read_noise_electrons": 0.0,
}


class SensorResponseTests(unittest.TestCase):
    def test_passive_scene_texture_survives_without_projector_signal(self):
        ambient = np.repeat(np.array([[40, 100, 180]], np.uint8)[..., None], 3, axis=2)
        hdr = np.zeros((1, 3, 3), np.float32)
        image, saturated, active = response.simulate_ir(ambient, hdr, hdr, CFG, np.random.default_rng(1))
        np.testing.assert_array_equal(image, [[34, 85, 153]])
        self.assertFalse(saturated.any())
        self.assertFalse(active.any())

    def test_projector_signal_is_local_and_retains_ambient_texture(self):
        ambient = np.full((4, 5, 3), 100, np.uint8)
        off = np.full((4, 5, 3), 20, np.float32)
        on = off.copy()
        on[1, 3] += 1000
        image, _, active = response.simulate_ir(ambient, on, off, CFG, np.random.default_rng(1))
        self.assertEqual(image[0, 0], 85)
        self.assertGreater(image[1, 3], image[0, 0])
        self.assertEqual(np.count_nonzero(active), 1)
        self.assertEqual(active[1, 3], 1000)

    def test_black_radiance_does_not_become_fake_scene_detail(self):
        zero = np.zeros((3, 4, 3), np.float32)
        image, _, _ = response.simulate_ir(zero, zero, zero, CFG, np.random.default_rng(0))
        self.assertFalse(image.any())

    def test_one_pair_gain_is_clamped_and_requires_stereo_shared_mode(self):
        cfg = dict(
            CFG,
            auto_exposure={
                "enabled": True,
                "stereo_pair_shared": True,
                "measurement_percentile": 95,
                "target_dn": 180,
                "minimum_gain": 1,
                "maximum_gain": 1.25,
            },
        )
        zero = np.zeros((3, 4, 3), np.float32)
        dark = np.full_like(zero, 30)
        self.assertEqual(response.shared_ir_gain((dark, dark), (zero, zero), (zero, zero), cfg), 1.25)
        cfg["auto_exposure"]["stereo_pair_shared"] = False
        with self.assertRaisesRegex(ValueError, "shared"):
            response.shared_ir_gain((dark, dark), (zero, zero), (zero, zero), cfg)

    def test_noise_is_seeded_and_model_parameters_are_validated(self):
        cfg = dict(CFG, shot_noise_scale=0.15, read_noise_electrons=3)
        zero = np.zeros((30, 40, 3), np.float32)
        ambient = np.full_like(zero, 100)
        on = np.full_like(zero, 500)
        first = response.simulate_ir(ambient, on, zero, cfg, np.random.default_rng(42))[0]
        repeated = response.simulate_ir(ambient, on, zero, cfg, np.random.default_rng(42))[0]
        second = response.simulate_ir(ambient, on, zero, cfg, np.random.default_rng(43))[0]
        np.testing.assert_array_equal(first, repeated)
        self.assertTrue(np.any(first != second))
        self.assertLess(float(first.std()), 2)
        with self.assertRaises(ValueError):
            response.simulate_ir(ambient, on, zero, dict(cfg, adc_bits=0), np.random.default_rng(0))

    def test_phase_settings_restore_after_renderer_failure(self):
        values = {"aa": 0, "denoiser": True}
        settings = SimpleNamespace(get=values.get, set=values.__setitem__, destroy_item=values.pop)
        with self.assertRaisesRegex(RuntimeError, "render failure"):
            with response.scoped_render_settings(settings, {"aa": 2, "denoiser": False, "new": 10}):
                self.assertEqual(values, {"aa": 2, "denoiser": False, "new": 10})
                raise RuntimeError("render failure")
        self.assertEqual(values, {"aa": 0, "denoiser": True})

    def test_render_average_reads_distinct_deliveries_at_the_same_physics_state(self):
        capture = sl.StructuredLightCapture.__new__(sl.StructuredLightCapture)
        delivered = [0]

        def render(min_passes):
            delivered[0] += 1
            return 1

        capture._render = render
        capture._array = lambda data, *args: data
        sensor = SimpleNamespace(get_data=lambda name: np.full((2, 3, 3), delivered[0] * 20, np.uint8))
        hdr = SimpleNamespace(get_data=lambda: np.full((2, 3, 3), delivered[0] * 100, np.float32))
        ldr, linear, passes = capture._average_views({"rgb": (sensor, (2, 3))}, {"ir": (hdr, (2, 3))}, 3)
        self.assertEqual(passes, 3)
        self.assertTrue((ldr["rgb"] == 40).all())
        self.assertTrue((linear["ir"] == 200).all())
        with self.assertRaises(ValueError):
            capture._average_views({}, {}, 0)


if __name__ == "__main__":
    unittest.main()
