"""Optional native Lab stereo cameras and official RTX pattern projection.

This captures grayscale visible-spectrum illumination, not spectral infrared or
reconstructed depth. The scene owns all camera sensors and render products.
"""
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from isaaclab.sensors import CameraCfg
from pxr import Gf, Sdf, UsdGeom, UsdRender


class StructuredLightManager:
    ROLES = ("cam_head", "cam_left_wrist", "cam_right_wrist")
    # Virtual stereo geometry and projector field of view from the supplied rig.
    BASELINE_M = 0.05
    FOV_DEG = (112.0, 80.0)
    # Current RTX baseline: reference power / 100 and a 2ms physical exposure.
    # Native projected-pattern measurements keep saturation below 1% on insert_tubes.
    INTENSITY = {"cam_head": 35.0, "cam_left_wrist": 6.0, "cam_right_wrist": 6.0}
    EXPOSURE = {"exposure:iso": 400.0, "exposure:time": 0.002,
                "exposure:fStop": 10.0, "exposure:responsivity": 1.10267}

    def __init__(self, config, camera_manager):
        self.camera_manager = camera_manager
        self.pattern = Path(config.get("pattern", "gemini335L"))
        if str(self.pattern) == "gemini335L":
            self.pattern = Path(__file__).resolve().parents[2] / "env_cfg/camera/pattern/gemini335L.png"
        self.pattern = self.pattern.expanduser().resolve()
        if not self.pattern.is_file():
            raise ValueError(f"Structured-light pattern does not exist: {self.pattern}")
        with Image.open(self.pattern) as image:
            self.pattern_pixels = np.asarray(image.convert("L")).copy()
        if min(self.pattern_pixels.shape) < 2:
            raise ValueError("Structured-light pattern must be at least 2 by 2 pixels.")
        self.pattern_sha256 = hashlib.sha256(self.pattern.read_bytes()).hexdigest()
        self.eyes, self.projectors, self.authors, self.products = {}, {}, [], {}
        self._temporary = None
        self._observations = {}

    def scene_configs(self, ordinary_configs):
        """Declare child eyes before Lab prepares partitions and the native scene."""
        result = {}
        self._keys = {}
        for role in self.ROLES:
            ordinary = ordinary_configs.get(f"camera_{role}")
            if ordinary is None:
                raise ValueError(f"Structured light requires camera role {role!r}.")
            self._keys[role] = []
            for eye, x in (("left", -self.BASELINE_M / 2), ("right", self.BASELINE_M / 2)):
                cfg = deepcopy(ordinary)
                cfg.prim_path += f"/structured_light_{eye}"
                cfg.offset = CameraCfg.OffsetCfg(pos=(x, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0), convention="opengl")
                cfg.data_types = ["rgb"]
                cfg.spawn.semantic_tags = [("class", f"{role}_ir_{eye}")]
                key = f"structured_light_{role}_{eye}"
                result[key] = cfg
                self._keys[role].append(key)
        return result

    def _projection_assets(self):
        """Generate octahedral pinhole LUT; black borders reject invalid rays.

        Mapping follows NVIDIA's public Projector Direction Texture Generation
        specification. Padding prevents an invalid UV=(0,0) ray from sampling a
        white corner in the supplied pattern. No image intensities are scaled.
        """
        self._temporary = TemporaryDirectory(prefix="robodojo-structured-light-")
        directory = Path(self._temporary.name)
        projected_pattern = directory / "pattern.png"
        Image.fromarray(np.pad(self.pattern_pixels, 1)).save(projected_pattern)
        size = 2048
        axis = (np.arange(size, dtype=np.float32) + 0.5) / size * 2 - 1
        x, y = np.meshgrid(axis, axis)
        z = 1 - np.abs(x) - np.abs(y)
        safe_z = np.where(z > 0, z, 1)
        u = 0.5 + x / safe_z / (2 * np.tan(np.deg2rad(self.FOV_DEG[0]) / 2))
        v = 0.5 + y / safe_z / (2 * np.tan(np.deg2rad(self.FOV_DEG[1]) / 2))
        valid = (z > 0) & (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
        height, width = self.pattern_pixels.shape
        # OpenCV BGR storage yields EXR R=u, G=v. Invalid directions sample black.
        pixels = np.stack((np.zeros_like(u), np.where(valid, (v * height + 1) / (height + 2), 0),
                           np.where(valid, (u * width + 1) / (width + 2), 0)), axis=-1).astype(np.float32)
        array_path, lut_path = directory / "direction.npy", directory / "direction.exr"
        np.save(array_path, pixels)
        code = ("import cv2,numpy as np,sys; a=np.load(sys.argv[1]); "
                "ok=cv2.imwrite(sys.argv[2],a,[cv2.IMWRITE_EXR_TYPE,cv2.IMWRITE_EXR_TYPE_FLOAT]); "
                "assert ok, 'EXR encoding failed'")
        subprocess.run([sys.executable, "-c", code, str(array_path), str(lut_path)],
                       env=dict(os.environ, OPENCV_IO_ENABLE_OPENEXR="1"),
                       check=True, timeout=30, capture_output=True, text=True)
        array_path.unlink()
        return projected_pattern, lut_path

    def bind(self, scene):
        import omni.replicator.core as rep
        from isaaclab.sim.utils import enable_extension
        enable_extension("isaacsim.sensors.experimental.rtx")
        from isaacsim.sensors.experimental.rtx import StructuredLightCamera

        self.eyes = {role: [scene[key] for key in keys] for role, keys in self._keys.items()}
        stage = scene.stage
        all_cameras = list(self.camera_manager.cameras) + [c for pair in self.eyes.values() for c in pair]
        paths_by_camera = {
            id(camera): {str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdGeom.Camera)
                         and re.fullmatch(camera.cfg.prim_path, str(p.GetPath()))}
            for camera in all_cameras
        }
        if any(len(paths) != self.camera_manager.num_envs for paths in paths_by_camera.values()):
            raise RuntimeError("Native structured-light camera paths do not cover every environment.")
        wanted = set().union(*paths_by_camera.values())
        paths = [str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdRender.Product)]
        owners = {path: [] for path in wanted}
        for path in paths:
            product = UsdRender.Product(stage.GetPrimAtPath(path))
            cameras = list(map(str, product.GetCameraRel().GetTargets()))
            if not set(cameras) & wanted:
                continue
            if not set(cameras) <= wanted:
                raise RuntimeError(f"Ambiguous structured-light render product ownership: {path}")
            resolution = tuple(product.GetResolutionAttr().Get())
            borrowed = rep.create.render_product(cameras[0], resolution, force_new=False)
            if (borrowed.path != path or tuple(product.GetResolutionAttr().Get()) != resolution
                    or list(map(str, product.GetCameraRel().GetTargets())) != cameras):
                raise RuntimeError(f"Native render product reuse failed: {path}")
            self.products[path] = borrowed  # Borrowed; never destroy scene-owned products.
            for camera_path in cameras:
                owners[camera_path].append(path)
        if set(paths) != {str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdRender.Product)}:
            raise RuntimeError("Borrowing native camera products unexpectedly created a render product.")
        if any(len(products) != 1 for products in owners.values()):
            raise RuntimeError(f"Could not resolve native camera product ownership: {owners}")
        self._camera_products = {id(c): {owners[p][0] for p in paths_by_camera[id(c)]} for c in all_cameras}
        self._ordinary_products = set().union(*(self._camera_products[id(c)] for c in self.camera_manager.cameras))
        self._ir_products = set(self.products) - self._ordinary_products
        for path in self._ir_products:
            product = stage.GetPrimAtPath(path)
            product.ApplyAPI("OmniRtxPostDebugSettingsAPI_1")
            product.GetAttribute("omni:rtx:post:aa:limitedOps").Set(False)
            # Dense dots are smoothed away by DLAA at these resolutions. Keep native
            # pixel sampling for IR only; ordinary RGB retains its default AA.
            product.GetAttribute("omni:rtx:post:aa:op").Set("none")
            self.products[path].hydra_texture.updates_enabled = False
        pattern, lut = self._projection_assets()
        for role, pair in self.eyes.items():
            for camera in pair:
                for path in paths_by_camera[id(camera)]:
                    prim = stage.GetPrimAtPath(path)
                    for name, value in self.EXPOSURE.items():
                        prim.GetAttribute(name).Set(value)
                    prim.ApplyAPI("OmniRtxCameraAutoExposureAPI_1")
                    prim.GetAttribute("omni:rtx:autoExposure:enabled").Set(False)
            self.projectors[role] = []
            for env_id in range(self.camera_manager.num_envs):
                matches = [p for p in paths_by_camera[id(pair[0])] if p.startswith(f"/World/envs/env_{env_id}/")]
                if len(matches) != 1:
                    raise RuntimeError(f"Native structured-light eye ownership is ambiguous: {matches}")
                eye_path = matches[0]
                gate = UsdGeom.Xform.Define(stage, f"/World/StructuredLight/env_{env_id}/{role}")
                token = stage.GetPrimAtPath(eye_path).GetAttribute("omni:scenePartition").Get()
                if token is not None:
                    gate.GetPrim().CreateAttribute("primvars:omni:scenePartition", Sdf.ValueTypeNames.Token).Set(token)
                author = StructuredLightCamera(eye_path, projector_light_patterns=[str(pattern)],
                    projector_direction_texture=str(lut), projector_prim_path=str(gate.GetPath()) + "/projectors",
                    projector_intensity=self.INTENSITY[role], reset_xform_op_properties=False)
                self.authors.append(author)
                gate.GetVisibilityAttr().Set("invisible")
                xform = UsdGeom.Xformable(stage.GetPrimAtPath(str(gate.GetPath()) + "/projectors"))
                xform.ClearXformOpOrder()
                self.projectors[role].append((gate, xform.AddTransformOp()))
        self._observations.clear()

    @staticmethod
    def _optical_pose(camera, env_id):
        data = camera.data
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_quat(data.quat_w_ros.torch[env_id].cpu().numpy()).as_matrix()
        transform[:3, 3] = data.pos_w.torch[env_id].cpu().numpy()
        return transform

    def capture(self, render):
        """Isolate projectors from ordinary products at one physics timestamp."""
        self._observations.clear()
        saved = {p: w.hydra_texture.updates_enabled for p, w in self.products.items()}
        pending_pair = None
        try:
            for path in self._ordinary_products:
                self.products[path].hydra_texture.updates_enabled = False
            for role, pair in self.eyes.items():
                pending_pair = pair
                role_index = self.camera_manager.camera_names[0].index(role)
                projector_poses = [self.camera_manager.get_camera_extrinsics(role_index, i)
                                   for i in range(self.camera_manager.num_envs)]
                for (gate, op), pose in zip(self.projectors[role], projector_poses):
                    op.Set(Gf.Matrix4d((pose @ np.diag([1., -1., -1., 1.])).T.tolist()))
                    gate.GetVisibilityAttr().Set("inherited")
                active = set().union(*(self._camera_products[id(c)] for c in pair))
                for path in active:
                    self.products[path].hydra_texture.updates_enabled = True
                # Product enable changes take one render update to reach the renderer.
                # The second native capture reads the newly delivered image.
                for _ in range(2):
                    render()
                    for camera in pair:
                        camera.update(0.0, force_recompute=True)
                for env_id in range(self.camera_manager.num_envs):
                    images = []
                    for camera in pair:
                        rgb = camera.data.output["rgb"].torch[env_id, ..., :3].cpu().numpy()
                        images.append(np.rint(rgb.astype(np.float32) @ np.array([.299, .587, .114], dtype=np.float32)).astype(np.uint8))
                    self._observations[(env_id, role)] = {
                        "left": images[0], "right": images[1],
                        "calibration": {
                            "left_intrinsic_matrix": pair[0].data.intrinsic_matrices.torch[env_id].cpu().numpy().copy(),
                            "right_intrinsic_matrix": pair[1].data.intrinsic_matrices.torch[env_id].cpu().numpy().copy(),
                            "left_extrinsic_matrix": self._optical_pose(pair[0], env_id),
                            "right_extrinsic_matrix": self._optical_pose(pair[1], env_id),
                            "projector_extrinsic_matrix": projector_poses[env_id],
                            "resolution_hw": tuple(images[0].shape), "baseline_m": self.BASELINE_M,
                            "projector_fov_deg": self.FOV_DEG,
                        },
                        "metadata": {"representation": "grayscale_visible_spectrum_projection",
                            "pattern_name": self.pattern.name, "pattern_sha256": self.pattern_sha256,
                            "pattern_wh": tuple(reversed(self.pattern_pixels.shape)),
                            "quaternion_order": "xyzw", "extrinsic_convention": "optical_ros_to_world",
                            "projector_intensity": self.INTENSITY[role], "exposure": dict(self.EXPOSURE),
                            "antialiasing": "none", "projector_size_m": (1.0, 1.0),
                            "projector_normalize": False},
                    }
                for gate, _ in self.projectors[role]:
                    gate.GetVisibilityAttr().Set("invisible")
                # Consume OFF before any ordinary product may resume.
                for _ in range(2):
                    render()
                    for camera in pair:
                        camera.update(0.0, force_recompute=True)
                for path in active:
                    self.products[path].hydra_texture.updates_enabled = False
                pending_pair = None
        finally:
            for projectors in self.projectors.values():
                for gate, _ in projectors:
                    gate.GetVisibilityAttr().Set("invisible")
            try:
                if pending_pair is not None:
                    for _ in range(2):
                        render()
                        for camera in pending_pair:
                            camera.update(0.0, force_recompute=True)
            finally:
                for path, enabled in saved.items():
                    self.products[path].hydra_texture.updates_enabled = enabled

    def observation(self, env_id, role):
        if (env_id, role) not in self._observations:
            raise RuntimeError("Capture structured light before requesting its observation.")
        return deepcopy(self._observations[(env_id, role)])

    def reset(self):
        self._observations.clear()
        for pair in self.eyes.values():
            for camera in pair:
                camera.reset()

    def destroy(self):
        self._observations.clear()
        for author in self.authors:
            author.destroy()
        self.authors.clear()
        self.projectors.clear()
        self.products.clear()
        self.eyes.clear()
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None
