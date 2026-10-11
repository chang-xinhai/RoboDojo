"""RTX projected-pattern stereo capture at a frozen physics timestamp.

The IR streams combine each independent eye's passive exposure with linear
HDR projector-on/off response and a declared electron/ADC model. RTX is a visible-spectrum renderer:
this is an explicitly tagged IR illumination proxy, not spectral NIR transport.
``depth_gt`` is renderer ground truth; ``depth_sl`` is triangulated stereo depth.
"""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
import os
from pathlib import Path
import time

import numpy as np


def physics_timestamp_ns(sim):
    """Isaac Lab 3 owns a physics step counter; legacy Sim exposes seconds."""
    if hasattr(sim, "get_physics_step_count"):
        return round(sim.get_physics_step_count() * sim.get_physics_dt() * 1e9)
    return round(float(sim.current_time) * 1e9)


def stereo_intrinsics(resolution, focal_length, horizontal_aperture, vertical_aperture):
    width, height = resolution
    return np.array(
        [
            [width * focal_length / horizontal_aperture, 0, width / 2],
            [0, height * focal_length / vertical_aperture, height / 2],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )


def render_camera_model(params, shape):
    """Read optical calibration from the exact delivered RTX render product."""

    def vector(key, size):
        value = np.asarray(params[key], dtype=np.float64).reshape(-1)
        if value.size != size or not np.isfinite(value).all():
            raise ValueError(f"CameraParams {key} must contain {size} finite values")
        return value

    view = vector("cameraViewTransform", 16).reshape(4, 4)
    projection = vector("cameraProjection", 16).reshape(4, 4)
    focal = vector("cameraFocalLength", 1)[0]
    aperture = vector("cameraAperture", 2)
    offsets = vector("cameraApertureOffset", 2)
    resolution = vector("renderProductResolution", 2)
    if not np.allclose(resolution, shape[::-1], atol=0, rtol=0):
        raise ValueError("CameraParams resolution differs from delivered image")
    if focal <= 0 or (aperture <= 0).any():
        raise ValueError("CameraParams focal length and aperture must be positive")
    if not np.isclose(vector("metersPerSceneUnit", 1)[0], 1.0, atol=1e-9, rtol=0):
        raise ValueError("Structured stereo requires metric stage units")
    k = stereo_intrinsics(resolution, focal, *aperture)
    k[:2, 2] += resolution * offsets / aperture
    world = np.linalg.inv(view).T
    if not np.allclose(world[3], [0, 0, 0, 1], atol=1e-6, rtol=0):
        raise ValueError("CameraParams view transform is not affine")
    optical = world @ np.diag([1.0, -1.0, -1.0, 1.0])
    return dict(K=k, T_world_from_optical=optical, view_row_major=view, projection_row_major=projection)


def active_ir(on, off, saturation_radiance):
    """Expose projector-only radiance using a shared fixed sensor gain."""
    if not np.isfinite(saturation_radiance) or saturation_radiance <= 0:
        raise ValueError("saturation_radiance must be positive and finite")
    on, off = np.asarray(on), np.asarray(off)
    if on.shape != off.shape or on.ndim != 3 or on.shape[-1] < 3:
        raise ValueError("HDR on/off must have the same HWC shape with >=3 channels")
    radiance = np.maximum(on[..., :3] - off[..., :3], 0).mean(axis=-1)
    radiance = np.nan_to_num(radiance, nan=0, posinf=saturation_radiance, neginf=0)
    return np.rint(np.clip(radiance / saturation_radiance, 0, 1) * 255).astype(np.uint8)


def reconstruct_stereo(
    left,
    right,
    fx,
    baseline_m,
    *,
    max_disparity=128,
    min_depth_m=0.05,
    max_depth_m=10,
    lr_max_diff_px=0.5,
    min_texture_std_dn=2.0,
    min_patch_correlation=0.5,
):
    """Triangulate only independently consistent, textured IR matches.

    The right-to-left SGBM solve is independent of the left-to-right solve.
    A bilinear correspondence must agree within the declared pixel tolerance
    and both local windows must contain texture. Occlusions and ambiguous
    matches remain NaN; neither confidence nor depth accesses renderer GT.
    """
    import cv2

    if left.shape != right.shape or left.ndim != 2 or left.dtype != np.uint8 or right.dtype != np.uint8:
        raise ValueError("Stereo images must be same-shaped uint8 HW arrays")
    if not np.isfinite(fx) or not np.isfinite(baseline_m) or fx <= 0 or baseline_m <= 0:
        raise ValueError("fx and baseline_m must be positive")
    if not np.isfinite(lr_max_diff_px) or lr_max_diff_px <= 0:
        raise ValueError("Left-right disparity tolerance must be positive and finite")
    if not np.isfinite(min_texture_std_dn) or min_texture_std_dn < 0:
        raise ValueError("Minimum IR texture must be finite and nonnegative")
    if not np.isfinite(min_patch_correlation) or not -1 <= min_patch_correlation <= 1:
        raise ValueError("Minimum IR patch correlation must be finite and within [-1, 1]")
    if not np.isfinite(min_depth_m) or not np.isfinite(max_depth_m) or not 0 < min_depth_m < max_depth_m:
        raise ValueError("Stereo depth range must be finite, positive and increasing")
    if not np.isfinite(max_disparity) or max_disparity <= 0:
        raise ValueError("Maximum disparity must be positive and finite")
    num_disparities = min(int(np.ceil(max_disparity / 16) * 16), (left.shape[1] // 16 - 1) * 16)
    if num_disparities < 16:
        raise ValueError("Stereo width must be at least 32 pixels")
    settings = dict(
        numDisparities=num_disparities,
        blockSize=5,
        P1=8 * 25,
        P2=32 * 25,
        disp12MaxDiff=1,
        uniquenessRatio=8,
        speckleWindowSize=50,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disparity = cv2.StereoSGBM_create(minDisparity=0, **settings).compute(left, right).astype(np.float32) / 16
    reverse = (
        cv2.StereoSGBM_create(minDisparity=-num_disparities, **settings).compute(right, left).astype(np.float32) / 16
    )
    rows, cols = np.indices(left.shape, dtype=np.float32)
    right_cols = cols - disparity
    reverse_at_match = cv2.remap(
        reverse,
        right_cols,
        rows,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=-num_disparities - 1,
    )
    reverse_support = cv2.remap(
        ((reverse >= -num_disparities) & (reverse < 0)).astype(np.float32),
        right_cols,
        rows,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    valid = (
        (disparity > 0)
        & (right_cols >= 0)
        & (right_cols < left.shape[1])
        & (reverse_at_match < 0)
        & (reverse_at_match >= -num_disparities)
        & (reverse_support >= 1 - 1e-6)
        & (np.abs(disparity + reverse_at_match) <= lr_max_diff_px)
    )
    if min_texture_std_dn > 0:

        def local_std(image):
            image = image.astype(np.float32)
            mean = cv2.boxFilter(image, -1, (5, 5))
            return np.sqrt(np.maximum(cv2.boxFilter(image * image, -1, (5, 5)) - mean * mean, 0))

        right_std = cv2.remap(local_std(right), right_cols, rows, cv2.INTER_LINEAR)
        valid &= (local_std(left) >= min_texture_std_dn) & (right_std >= min_texture_std_dn)
    # Normalize local photometric evidence for each eye's brightness/contrast.
    # A fixed DN difference rejects bright aliased dots even when they match.
    # Independent wrong textures can occasionally satisfy reciprocal disparity;
    # they must also agree in their measured local intensity arrangement.
    warped_right = cv2.remap(right.astype(np.float32), right_cols, rows, cv2.INTER_LINEAR)
    left_float = left.astype(np.float32)
    left_mean = cv2.boxFilter(left_float, -1, (5, 5))
    right_mean = cv2.boxFilter(warped_right, -1, (5, 5))
    left_variance = np.maximum(cv2.boxFilter(left_float * left_float, -1, (5, 5)) - left_mean * left_mean, 0)
    right_variance = np.maximum(cv2.boxFilter(warped_right * warped_right, -1, (5, 5)) - right_mean * right_mean, 0)
    covariance = cv2.boxFilter(left_float * warped_right, -1, (5, 5)) - left_mean * right_mean
    correlation = covariance / np.sqrt(np.maximum(left_variance * right_variance, 1e-8))
    valid &= correlation >= min_patch_correlation
    depth = np.full(left.shape, np.nan, dtype=np.float32)
    depth[valid] = fx * baseline_m / disparity[valid]
    depth[(depth < min_depth_m) | (depth > max_depth_m)] = np.nan
    return depth


def generate_projector_assets(
    directory, *, seed=7, resolution=(640, 480), fov_deg=(112, 80), pattern_path=None, lut_resolution=512
):
    """Deterministic dots and an octahedral direction-to-pattern UV lookup."""
    # OpenCV's EXR codec reads this environment variable when first invoked.
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    import cv2

    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    width, height = map(int, resolution)
    if width < 32 or height < 32 or any(not 0 < value < 180 for value in fov_deg):
        raise ValueError("Invalid projector resolution or field of view")
    if not isinstance(lut_resolution, int) or lut_resolution < 32:
        raise ValueError("Projector LUT resolution must be an integer >=32")
    pattern = Path(pattern_path).resolve() if pattern_path else directory / f"dots_{width}x{height}_seed{seed}.png"
    if pattern_path and not pattern.is_file():
        raise FileNotFoundError(f"Supplied projector pattern not found: {pattern}")
    lut = directory / f"direction_{fov_deg[0]}x{fov_deg[1]}_{lut_resolution}.exr"
    if not pattern_path and not pattern.exists():
        rng = np.random.default_rng(seed)
        dots = np.zeros((height, width), dtype=np.uint8)
        for x, y in rng.integers([0, 0], [width, height], size=(width * height // 40, 2)):
            cv2.circle(dots, (int(x), int(y)), 1, 255, -1)
        if not cv2.imwrite(str(pattern), dots):
            raise RuntimeError(f"Could not write {pattern}")
    if not lut.exists():
        axis = (np.arange(lut_resolution, dtype=np.float32) + 0.5) / lut_resolution * 2 - 1
        x, y = np.meshgrid(axis, axis)
        z = 1 - np.abs(x) - np.abs(y)
        old_x = x.copy()
        x = np.where(z <= 0, (1 - np.abs(y)) * np.where(x >= 0, 1, -1), x)
        y = np.where(z <= 0, (1 - np.abs(old_x)) * np.where(y >= 0, 1, -1), y)
        safe_z = np.where(np.abs(z) < 1e-12, 1, z)
        u = x / safe_z / (2 * np.tan(np.deg2rad(fov_deg[0]) / 2)) + 0.5
        v = y / safe_z / (2 * np.tan(np.deg2rad(fov_deg[1]) / 2)) + 0.5
        valid = (z > 0) & (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
        bgr = np.zeros((lut_resolution, lut_resolution, 3), dtype=np.float32)
        bgr[..., 2], bgr[..., 1] = np.where(valid, u, -1), np.where(valid, v, -1)
        if not cv2.imwrite(str(lut), bgr):
            raise RuntimeError(f"Could not write EXR projector LUT {lut}; install OpenCV EXR support")
    return pattern, lut


def _load_native_camera_types():
    """Enable native camera dependencies for every Kit experience at capture entry."""
    from isaacsim.core.utils.extensions import enable_extension

    # IsaacLab's canonical experience does not enable this optional extension.
    # Enable it before importing its modules or the experimental core dependency.
    enable_extension("isaacsim.sensors.experimental.rtx")
    from isaacsim.core.experimental.objects import Camera as CameraAuthoring
    from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera

    return CameraAuthoring, CameraSensor, RtxCamera


class StructuredLightCapture:
    """One mounted stereo/projector rig per existing RoboDojo RGB camera."""

    def __init__(self, manager, config):
        CameraAuthoring, CameraSensor, RtxCamera = _load_native_camera_types()
        from isaacsim.core.utils.stage import get_current_stage
        import omni.replicator.core as rep
        from pxr import Gf, Sdf, UsdGeom, UsdLux

        self.manager, self.config = manager, dict(config)
        self.stage = get_current_stage()
        self.baseline = float(config.get("baseline_m", 0.05))
        if not np.isfinite(self.baseline) or self.baseline <= 0:
            raise ValueError("structured_light.baseline_m must be positive and finite")
        asset_dir = config.get(
            "asset_cache", str(Path(__file__).resolve().parents[2] / ".cache/isaac61/structured_light/assets")
        )
        pattern_path = config.get("pattern_path")
        if pattern_path and not Path(pattern_path).is_absolute():
            pattern_path = Path(__file__).resolve().parents[2] / pattern_path
        pattern, lut = generate_projector_assets(
            asset_dir,
            seed=int(config.get("pattern_seed", 7)),
            pattern_path=pattern_path,
            lut_resolution=int(config.get("direction_lut_resolution", 512)),
        )
        self.pattern, self.direction_texture = str(pattern), str(lut)
        self.pattern_sha256 = hashlib.sha256(pattern.read_bytes()).hexdigest()
        if config.get("pattern_sha256") and self.pattern_sha256 != config["pattern_sha256"]:
            raise ValueError("Supplied projector pattern differs from configured SHA-256")
        self.rigs = {}
        for env_id in range(manager.num_envs):
            self.rigs[env_id] = {}
            for cam_id, name in enumerate(manager.camera_names[env_id]):
                rgb_path = manager.cameras[env_id][cam_id].prim_path
                # Wrap the authored camera without changing mounted transforms.
                self.stage.GetPrimAtPath(rgb_path).ApplyAPI("OmniSensorAPI")
                rgb = RtxCamera(rgb_path, reset_xform_op_properties=False)
                # RtxCamera.camera lazily creates another wrapper whose default
                # rewrites xform ops. Preserve the existing mounted transform.
                rgb._camera = CameraAuthoring(rgb_path, reset_xform_op_properties=False)
                optical = UsdGeom.Camera(self.stage.GetPrimAtPath(rgb_path))
                width, height = manager.cameras[env_id][cam_id].get_resolution()
                rgb_sensor = CameraSensor(
                    rgb, resolution=(height, width), annotators=["rgb", "distance_to_image_plane"]
                )
                params = {"rgb": rep.AnnotatorRegistry.get_annotator("CameraParams")}
                params["rgb"].attach([str(rgb_sensor.render_product.GetPath())])
                # CameraSensor enforces square pixels; read the resulting
                # authored aperture after its constructor, not before.
                k = stereo_intrinsics(
                    (width, height),
                    optical.GetFocalLengthAttr().Get(),
                    optical.GetHorizontalApertureAttr().Get(),
                    optical.GetVerticalApertureAttr().Get(),
                )
                eyes, hdr = {}, {}
                for eye, offset in (("ir_left", -self.baseline / 2), ("ir_right", self.baseline / 2)):
                    path = f"{rgb_path}/{eye}"
                    camera = RtxCamera(path, translations=[[offset, 0, 0]])
                    eye_optical = UsdGeom.Camera(self.stage.GetPrimAtPath(path))
                    for attr in ("FocalLength", "HorizontalAperture", "VerticalAperture", "ClippingRange"):
                        getattr(eye_optical, f"Get{attr}Attr")().Set(getattr(optical, f"Get{attr}Attr")().Get())
                    sensor = CameraSensor(camera, resolution=(height, width), annotators=["rgb"])
                    eyes[eye] = sensor
                    hdr[eye] = rep.AnnotatorRegistry.get_annotator("HdrColor", device="cpu", do_array_copy=True)
                    hdr[eye].attach([str(sensor.render_product.GetPath())])
                    params[eye] = rep.AnnotatorRegistry.get_annotator("CameraParams")
                    params[eye].attach([str(sensor.render_product.GetPath())])
                projector = UsdLux.RectLight.Define(self.stage, f"{rgb_path}/projector")
                prim = projector.GetPrim()
                UsdLux.ShapingAPI.Apply(prim)
                intensity = float(
                    config.get("projector_intensity_by_station", {}).get(
                        name.removeprefix("cam_"), config.get("projector_intensity", 1500)
                    )
                )
                projector.CreateIntensityAttr(intensity)
                projector.CreateWidthAttr(1.0)
                projector.CreateHeightAttr(1.0)
                UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
                prim.CreateAttribute("inputs:texture:file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(pattern)))
                prim.CreateAttribute("projector:directionTexture:file", Sdf.ValueTypeNames.Asset).Set(
                    Sdf.AssetPath(str(lut))
                )
                prim.CreateAttribute("isProjector", Sdf.ValueTypeNames.Bool).Set(True)
                prim.CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool).Set(False)
                UsdGeom.Imageable(prim).MakeInvisible()
                self.rigs[env_id][name] = dict(
                    cam_id=cam_id,
                    rgb=rgb_sensor,
                    eyes=eyes,
                    hdr=hdr,
                    projector=prim,
                    K=k,
                    shape=(height, width),
                    rgb_path=rgb_path,
                    params=params,
                    projector_intensity=intensity,
                )

        from env.camera_manager.capture.render_sync import RenderProductDeliveryFence

        sensors = [
            sensor
            for rigs in self.rigs.values()
            for rig in rigs.values()
            for sensor in (rig["rgb"], *rig["eyes"].values())
        ]
        self._delivery_fence = RenderProductDeliveryFence(sensors)

    def set_projectors(self, enabled, *, env_id=None, camera_name=None):
        from pxr import UsdGeom

        for idx, rigs in self.rigs.items():
            for name, rig in rigs.items():
                if not rig["projector"].IsValid():
                    continue  # a hard scene reset already removed this prim
                image = UsdGeom.Imageable(rig["projector"])
                if enabled and (env_id is None or idx == env_id) and (camera_name is None or name == camera_name):
                    image.MakeVisible()
                else:
                    image.MakeInvisible()

    def _render(self, min_passes=None):
        from env.camera_manager.capture.render_sync import wait_for_camera_data, zero_delay_active

        # render_sync expects env.render(); the Isaac simulation context has the
        # same render-only API and does not advance PhysX or the timeline.
        if not zero_delay_active():
            raise RuntimeError(
                "Structured-light phase fencing requires zero-delay Kit settings; "
                "call add_zero_delay_kit_args before AppLauncher"
            )

        def read_rgbd():
            data = []
            for rigs in self.rigs.values():
                for rig in rigs.values():
                    rgb = self._array(rig["rgb"].get_data("rgb"), rig["shape"], 3)
                    depth = self._array(rig["rgb"].get_data("distance_to_image_plane"), rig["shape"])
                    data.append({"rgb": [{"data": rgb}], "depth_gt": [{"data": depth}]})
            return data

        _, passes = wait_for_camera_data(
            self.manager.sim.sim,
            read_rgbd,
            physics_state=lambda: physics_timestamp_ns(self.manager.sim.sim),
            min_passes=int(self.config.get("settle_render_passes", 3)) if min_passes is None else min_passes,
            max_passes=int(self.config.get("max_render_passes", 32)),
            delivery_clock=self._delivery_fence.snapshot,
        )
        return passes

    @staticmethod
    def _array(data, shape, channels=None):
        from env.camera_manager.capture.render_sync import CameraFrameNotReady

        if isinstance(data, tuple):
            data = data[0]
        if isinstance(data, dict):
            data = data.get("data")
        if data is None:
            raise CameraFrameNotReady("RTX camera annotator has no delivered frame")
        if hasattr(data, "numpy"):
            data = data.numpy()
        value = np.asarray(data)
        if value.ndim == 1:
            expected = int(np.prod(shape))
            if value.size < expected or value.size % expected:
                raise CameraFrameNotReady(f"RTX camera annotator delivered {value.size} values for image {shape}")
            value = value.reshape((*shape, -1) if channels else shape)
        if value.shape[:2] != shape or (channels and (value.ndim != 3 or value.shape[-1] < channels)):
            raise CameraFrameNotReady(f"Unexpected RTX image shape {value.shape}, expected {shape}")
        return value.copy()

    def _phase_scope(self, phase):
        rendering = getattr(self, "config", {}).get("rendering", {})
        if not rendering:
            return nullcontext()
        import carb.settings

        from env.camera_manager.sensor_response import scoped_render_settings

        aa = int(rendering[f"{phase}_antialiasing_op"])
        overrides = {
            "/rtx/post/aa/op": aa,
            "/rtx-defaults/post/aa/op": aa,
            "/rtx-transient/post/aa/limitedOps": False,
        }
        if phase == "ir":
            overrides["/rtx-transient/dldenoiser/enabled"] = False
        return scoped_render_settings(carb.settings.get_settings(), overrides)

    def _average_views(self, views, hdr_views, sample_count):
        """Average independent RTX deliveries while retaining the frozen pose."""
        if not isinstance(sample_count, int) or sample_count <= 0:
            raise ValueError("Sensor samples_per_pass must be a positive integer")
        ldr, hdr = {}, {}
        passes = 0
        for _ in range(sample_count):
            passes += self._render(min_passes=1)
            for name, (sensor, shape) in views.items():
                sample = self._array(sensor.get_data("rgb"), shape, 3)[..., :3].astype(np.float32)
                ldr[name] = ldr.get(name, np.zeros_like(sample)) + sample
            for name, (annotator, shape) in hdr_views.items():
                sample = self._array(annotator.get_data(), shape, 3)[..., :3].astype(np.float32)
                hdr[name] = hdr.get(name, np.zeros_like(sample)) + sample
        return (
            {name: np.rint(total / sample_count).clip(0, 255).astype(np.uint8) for name, total in ldr.items()},
            {name: total / sample_count for name, total in hdr.items()},
            passes,
        )

    def _record_render_state(self, phase):
        """Read effective renderer state after live deliveries, rather than only config."""
        import carb.settings

        settings = carb.settings.get_settings()
        paths = (
            "/rtx/rendermode",
            "/rtx/post/aa/op",
            "/rtx-defaults/post/aa/op",
            "/rtx-transient/dldenoiser/enabled",
            "/rtx/sdg/force/disableColorRender",
            "/rtx/post/tonemap/filmIso",
            "/rtx/post/tonemap/exposureTime",
            "/rtx/post/tonemap/fNumber",
            "/rtx/post/tonemap/responsivity",
            "/rtx/post/histogram/enabled",
        )
        self._phase_states[phase] = {path: settings.get(path) for path in paths}
        self._phase_states[phase]["physics_timestamp_ns"] = physics_timestamp_ns(self.manager.sim.sim)
        self._phase_states[phase]["native_render_product_completions"] = dict(
            zip(self._delivery_fence.paths, self._delivery_fence.snapshot(), strict=True)
        )
        from isaacsim.core.utils.stage import get_current_stage

        stage = get_current_stage()
        self._phase_states[phase]["render_product_aa_overrides"] = {
            path: stage.GetPrimAtPath(path).GetAttribute("omni:rtx:post:aa:op").Get()
            for path in self._delivery_fence.paths
        }

    def capture(self, env_id=0):
        rendering = getattr(self, "config", {}).get("rendering", {})
        if not rendering:
            return self._capture(env_id)
        import carb.settings

        from env.camera_manager.sensor_response import scoped_render_settings

        exposure = rendering.get("exposure", {})
        settings = {
            f"/rtx/post/tonemap/{key}": float(value)
            for key, value in exposure.items()
            if key in {"filmIso", "exposureTime", "fNumber", "responsivity"}
        }
        settings["/rtx/post/histogram/enabled"] = bool(rendering.get("auto_exposure_enabled", False))
        with scoped_render_settings(carb.settings.get_settings(), settings):
            return self._capture(env_id)

    def _capture(self, env_id=0):
        timestamp_ns = physics_timestamp_ns(self.manager.sim.sim)
        result = {}
        self.set_projectors(False)
        try:
            self._phase_states = {}
            rendering = getattr(self, "config", {}).get("rendering", {})
            with self._phase_scope("rgb"):
                rgb_settle = (
                    self._render(min_passes=int(rendering.get("rgb_settle_renders", 1)))
                    if rendering
                    else self._render()
                )
                rgb_samples = int(rendering.get("rgb_samples_per_pass", 1))
                rgb_views = {name: (rig["rgb"], rig["shape"]) for name, rig in self.rigs[env_id].items()}
                rgb_images, _, rgb_sample_passes = self._average_views(rgb_views, {}, rgb_samples)
                self._record_render_state("rgb")
                gt_images = {
                    name: self._array(rig["rgb"].get_data("distance_to_image_plane"), rig["shape"])
                    .reshape(rig["shape"])
                    .astype(np.float32)
                    for name, rig in self.rigs[env_id].items()
                }
            for name, rig in self.rigs[env_id].items():
                shape = rig["shape"]
                rgb, gt = rgb_images[name], gt_images[name]
                ir_samples = int(rendering.get("ir_samples_per_pass", 1))
                with self._phase_scope("ir"):
                    self.set_projectors(False)
                    off_passes = self._render(
                        min_passes=int(rendering.get("ir_settle_renders", self.config.get("settle_render_passes", 3)))
                    )
                    ambient, off, off_sample_passes = self._average_views(
                        {eye: (sensor, shape) for eye, sensor in rig["eyes"].items()},
                        {eye: (annotator, shape) for eye, annotator in rig["hdr"].items()},
                        ir_samples,
                    )
                    phase_off_ns = time.time_ns()
                    self._record_render_state(name + "/ir_ambient")
                    # Isolate each projector so other emitters cannot
                    # contaminate this independently illuminated stereo pair.
                    self.set_projectors(True, env_id=env_id, camera_name=name)
                    on_passes = self._render(
                        min_passes=int(rendering.get("ir_settle_renders", self.config.get("settle_render_passes", 3)))
                    )
                    _, on, on_sample_passes = self._average_views(
                        {}, {eye: (annotator, shape) for eye, annotator in rig["hdr"].items()}, ir_samples
                    )
                    phase_on_ns = time.time_ns()
                    self._record_render_state(name + "/ir_active")
                active = {
                    eye + "_active": active_ir(on[eye], off[eye], float(self.config.get("saturation_radiance", 5500.0)))
                    for eye in on
                }
                from env.camera_manager.sensor_response import projector_response

                radiance = {eye + "_active_radiance": projector_response(on[eye], off[eye]) for eye in on}
                sensor_cfg = self.config.get("ir_sensor_model")
                response_stats = {}
                if sensor_cfg:
                    from env.camera_manager.sensor_response import shared_ir_gain, simulate_ir

                    eyes = ("ir_left", "ir_right")
                    gain = shared_ir_gain(
                        tuple(ambient[e] for e in eyes),
                        tuple(on[e] for e in eyes),
                        tuple(off[e] for e in eyes),
                        sensor_cfg,
                    )
                    ir = {}
                    for eye_id, eye in enumerate(eyes):
                        seed = (
                            int(self.config.get("sensor_seed", 42))
                            + timestamp_ns
                            + env_id * 1000
                            + rig["cam_id"] * 2
                            + eye_id
                        )
                        ir[eye], saturated, response = simulate_ir(
                            ambient[eye], on[eye], off[eye], sensor_cfg, np.random.default_rng(seed), exposure_gain=gain
                        )
                        response_stats[eye] = {
                            "exposure_gain": gain,
                            "noise_seed": seed,
                            "saturation_fraction": float(saturated.mean()),
                            "projector_response_p995": float(np.percentile(response, 99.5)),
                            "ambient_mean_dn": float(ambient[eye].mean()),
                        }
                else:
                    ir = {eye: active[eye + "_active"] for eye in on}
                models = {view: render_camera_model(a.get_data(), shape) for view, a in rig["params"].items()}
                k = models["ir_left"]["K"]
                relative = (
                    np.linalg.inv(models["ir_left"]["T_world_from_optical"])
                    @ models["ir_right"]["T_world_from_optical"]
                )
                expected = np.eye(4)
                expected[0, 3] = self.baseline
                if not np.allclose(relative, expected, atol=1e-5, rtol=0) or not np.allclose(
                    k, models["ir_right"]["K"], atol=1e-3, rtol=0
                ):
                    raise RuntimeError("Rendered stereo pair is not parallel with the configured horizontal baseline")
                confidence = {
                    key: float(self.config.get("stereo_confidence", {}).get(key, default))
                    for key, default in (
                        ("lr_max_diff_px", 0.5),
                        ("min_texture_std_dn", 2.0),
                        ("min_patch_correlation", 0.5),
                    )
                }
                depth_sl = reconstruct_stereo(
                    ir["ir_left"],
                    ir["ir_right"],
                    k[0, 0],
                    self.baseline,
                    max_disparity=int(self.config.get("max_disparity", 128)),
                    **confidence,
                )
                transforms = {view: model["T_world_from_optical"] for view, model in models.items()}
                result[name] = dict(
                    rgb=rgb,
                    depth_gt=gt,
                    depth_sl=depth_sl,
                    **ir,
                    **active,
                    **radiance,
                    timestamp_ns=timestamp_ns,
                    calibration=dict(
                        K=k.tolist(),
                        K_by_view={view: model["K"].tolist() for view, model in models.items()},
                        projection_authority="Replicator CameraParams from each exact RGB/IR render product",
                        camera_view_row_major={
                            view: model["view_row_major"].tolist() for view, model in models.items()
                        },
                        camera_projection_row_major={
                            view: model["projection_row_major"].tolist() for view, model in models.items()
                        },
                        baseline_m=self.baseline,
                        T_world_from_optical={k: v.tolist() for k, v in transforms.items()},
                        depth_gt_frame="rgb",
                        depth_sl_frame="ir_left",
                        depth_unit="m",
                        stereo_confidence={**confidence, "texture_window_px": 5, "valid_depth_range_m": [0.05, 10.0]},
                    ),
                    provenance=dict(
                        ir="independent_eye_ambient_plus_active_hdr_electron_adc_proxy"
                        if sensor_cfg
                        else "rtx_active_projector_hdr_on_minus_off_proxy",
                        ir_active="projector_only_hdr_on_minus_off_fixed_gain_proxy",
                        ir_active_radiance="linear_hdr_weighted_projector_response_preserved_float32",
                        ir_active_radiance_sha256={
                            key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in radiance.items()
                        },
                        ir_spectral_model="visible_radiance_proxy_no_NIR_material_response",
                        depth_gt="distance_to_image_plane_ground_truth",
                        depth_sl="rectified_active_stereo_sgbm_triangulation",
                        stereo_confidence={
                            **confidence,
                            "method": "independent_reverse_sgbm_bilinear_consistency_texture_and_normalized_patch_correlation",
                            "texture_window_px": 5,
                            "rejected_matches": "NaN; no GT filtering or filling",
                        },
                        pattern=self.pattern,
                        pattern_sha256=self.pattern_sha256,
                        direction_texture=self.direction_texture,
                        projector_intensity=rig["projector_intensity"],
                        rgb_prim_path=rig["rgb_path"],
                        phase_off_host_timestamp_ns=phase_off_ns,
                        phase_on_host_timestamp_ns=phase_on_ns,
                        off_render_passes=off_passes,
                        on_render_passes=on_passes,
                        rgb_render_passes=rgb_settle + rgb_sample_passes,
                        ambient_sample_render_passes=off_sample_passes,
                        active_sample_render_passes=on_sample_passes,
                        rendering=rendering,
                        actual_render_states=dict(self._phase_states),
                        ir_sensor_model=sensor_cfg,
                        response_stats=response_stats,
                    ),
                )
                self.set_projectors(False)
                # Restore baseline before processing another independently
                # illuminated rig, preventing radiance cache contamination.
                self._render()
            if physics_timestamp_ns(self.manager.sim.sim) != timestamp_ns:
                raise RuntimeError("Physics advanced during synchronized structured-light capture")
            return result
        finally:
            self.set_projectors(False)

    def destroy(self):
        self.set_projectors(False)
        self._delivery_fence.close()
        for rigs in self.rigs.values():
            for rig in rigs.values():
                for annotator in rig["hdr"].values():
                    annotator.detach()
                for annotator in rig["params"].values():
                    annotator.detach()
                for sensor in (rig["rgb"], *rig["eyes"].values()):
                    sensor.detach_annotators(sensor.annotators)
                    sensor._hydra_texture.destroy()
        self.rigs.clear()
