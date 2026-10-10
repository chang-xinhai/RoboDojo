"""RTX projected-pattern stereo capture at a frozen physics timestamp.

The IR streams model active illumination using independent camera views and
linear HDR projector-on/off subtraction. RTX is a visible-spectrum renderer:
this is an explicitly tagged IR illumination proxy, not spectral NIR transport.
``depth_gt`` is renderer ground truth; ``depth_sl`` is triangulated stereo depth.
"""

from __future__ import annotations

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


def reconstruct_stereo(left, right, fx, baseline_m, *, max_disparity=128, min_depth_m=0.05, max_depth_m=10):
    """Triangulate rectified active stereo; invalid pixels are NaN, never GT."""
    import cv2

    if left.shape != right.shape or left.ndim != 2 or left.dtype != np.uint8 or right.dtype != np.uint8:
        raise ValueError("Stereo images must be same-shaped uint8 HW arrays")
    if fx <= 0 or baseline_m <= 0:
        raise ValueError("fx and baseline_m must be positive")
    num_disparities = min(int(np.ceil(max_disparity / 16) * 16), (left.shape[1] // 16 - 1) * 16)
    if num_disparities < 16:
        raise ValueError("Stereo width must be at least 32 pixels")
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
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
    disparity = matcher.compute(left, right).astype(np.float32) / 16
    depth = np.full(left.shape, np.nan, dtype=np.float32)
    valid = disparity > 0
    depth[valid] = fx * baseline_m / disparity[valid]
    depth[(depth < min_depth_m) | (depth > max_depth_m)] = np.nan
    return depth


def generate_projector_assets(directory, *, seed=7, resolution=(640, 480), fov_deg=(112, 80)):
    """Deterministic dots and an octahedral direction-to-pattern UV lookup."""
    # OpenCV's EXR codec reads this environment variable when first invoked.
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    import cv2

    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    width, height = map(int, resolution)
    if width < 32 or height < 32 or any(not 0 < value < 180 for value in fov_deg):
        raise ValueError("Invalid projector resolution or field of view")
    pattern = directory / f"dots_{width}x{height}_seed{seed}.png"
    lut = directory / f"direction_{fov_deg[0]}x{fov_deg[1]}_512.exr"
    if not pattern.exists():
        rng = np.random.default_rng(seed)
        dots = np.zeros((height, width), dtype=np.uint8)
        for x, y in rng.integers([0, 0], [width, height], size=(width * height // 40, 2)):
            cv2.circle(dots, (int(x), int(y)), 1, 255, -1)
        if not cv2.imwrite(str(pattern), dots):
            raise RuntimeError(f"Could not write {pattern}")
    if not lut.exists():
        axis = (np.arange(512, dtype=np.float32) + 0.5) / 512 * 2 - 1
        x, y = np.meshgrid(axis, axis)
        z = 1 - np.abs(x) - np.abs(y)
        old_x = x.copy()
        x = np.where(z <= 0, (1 - np.abs(y)) * np.where(x >= 0, 1, -1), x)
        y = np.where(z <= 0, (1 - np.abs(old_x)) * np.where(y >= 0, 1, -1), y)
        safe_z = np.where(np.abs(z) < 1e-12, 1, z)
        u = x / safe_z / (2 * np.tan(np.deg2rad(fov_deg[0]) / 2)) + 0.5
        v = y / safe_z / (2 * np.tan(np.deg2rad(fov_deg[1]) / 2)) + 0.5
        valid = (z > 0) & (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
        bgr = np.zeros((512, 512, 3), dtype=np.float32)
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
        pattern, lut = generate_projector_assets(asset_dir, seed=int(config.get("pattern_seed", 7)))
        self.pattern, self.direction_texture = str(pattern), str(lut)
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
                projector.CreateIntensityAttr(float(config.get("projector_intensity", 1500)))
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
                )

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

    def _render(self):
        from env.camera_manager.capture.render_sync import wait_for_latest_cameras, zero_delay_active

        # render_sync expects env.render(); the Isaac simulation context has the
        # same render-only API and does not advance PhysX or the timeline.
        if not zero_delay_active():
            raise RuntimeError(
                "Structured-light phase fencing requires zero-delay Kit settings; "
                "call add_zero_delay_kit_args before AppLauncher"
            )
        return wait_for_latest_cameras(
            self.manager.sim.sim,
            min_passes=int(self.config.get("settle_render_passes", 3)),
            max_passes=int(self.config.get("max_render_passes", 32)),
        )

    @staticmethod
    def _array(data, shape, channels=None):
        if isinstance(data, tuple):
            data = data[0]
        if isinstance(data, dict):
            data = data.get("data")
        if data is None:
            raise RuntimeError("RTX camera annotator has no delivered frame")
        if hasattr(data, "numpy"):
            data = data.numpy()
        value = np.asarray(data)
        if value.ndim == 1:
            value = value.reshape((*shape, -1) if channels else shape)
        if value.shape[:2] != shape or (channels and (value.ndim != 3 or value.shape[-1] < channels)):
            raise RuntimeError(f"Unexpected RTX image shape {value.shape}, expected {shape}")
        return value.copy()

    def capture(self, env_id=0):
        timestamp_ns = physics_timestamp_ns(self.manager.sim.sim)
        result = {}
        self.set_projectors(False)
        try:
            off_passes = self._render()
            for name, rig in self.rigs[env_id].items():
                shape = rig["shape"]
                rgb = self._array(rig["rgb"].get_data("rgb"), shape, 3)[..., :3].astype(np.uint8)
                gt = (
                    self._array(rig["rgb"].get_data("distance_to_image_plane"), shape).reshape(shape).astype(np.float32)
                )
                off = {eye: self._array(a.get_data(), shape, 3) for eye, a in rig["hdr"].items()}
                phase_off_ns = time.time_ns()
                # Isolate each projector so wrist and head emitters cannot
                # contaminate one another's off/on subtraction.
                self.set_projectors(True, env_id=env_id, camera_name=name)
                on_passes = self._render()
                on = {eye: self._array(a.get_data(), shape, 3) for eye, a in rig["hdr"].items()}
                ir = {
                    eye: active_ir(on[eye], off[eye], float(self.config.get("saturation_radiance", 5500.0)))
                    for eye in on
                }
                phase_on_ns = time.time_ns()
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
                depth_sl = reconstruct_stereo(
                    ir["ir_left"],
                    ir["ir_right"],
                    k[0, 0],
                    self.baseline,
                    max_disparity=int(self.config.get("max_disparity", 128)),
                )
                transforms = {view: model["T_world_from_optical"] for view, model in models.items()}
                result[name] = dict(
                    rgb=rgb,
                    depth_gt=gt,
                    depth_sl=depth_sl,
                    **ir,
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
                    ),
                    provenance=dict(
                        ir="rtx_active_projector_hdr_on_minus_off_proxy",
                        ir_spectral_model="visible_radiance_proxy_no_NIR_material_response",
                        depth_gt="distance_to_image_plane_ground_truth",
                        depth_sl="rectified_active_stereo_sgbm_triangulation",
                        pattern=self.pattern,
                        direction_texture=self.direction_texture,
                        rgb_prim_path=rig["rgb_path"],
                        phase_off_host_timestamp_ns=phase_off_ns,
                        phase_on_host_timestamp_ns=phase_on_ns,
                        off_render_passes=off_passes,
                        on_render_passes=on_passes,
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
