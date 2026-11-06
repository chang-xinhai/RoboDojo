"""Role-batched Isaac Lab cameras owned by the native interactive scene."""

from collections.abc import Sequence

from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import PinholeCameraCfg
from isaaclab.sim.views import FrameView
from isaaclab_physx.renderers import IsaacRtxRendererCfg
import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.transform import Rotation
import torch
import warp as wp

from env_cfg.camera.template import D435, GEMINI_345LG, LARGE_D435, THIRD_VIEW

CAMERA_MODELS = {
    "Gemini_345Lg": GEMINI_345LG,
    "third_view": THIRD_VIEW,
    "d435": D435,
    "large_d435": LARGE_D435,
}

# Configuration aliases only: buffers and observations use the current SDK names.
OUTPUT_ALIASES = {
    "depth": "distance_to_image_plane",
    "instance_segmentation_fast": "instance_segmentation",
    "instance_id_segmentation": "instance_id_segmentation_fast",
}


class CameraManager:
    """Declare cameras before scene construction, then bind after physics is ready.

    Each role is one native Camera across all environments. Mount poses are local
    OpenGL camera poses; all quaternions are XYZW. The scene owns sensor lifetime,
    render products and normal buffer updates. The transform views below only
    implement mount-local reset/randomization through the public writer API.
    """

    def __init__(
        self, num_envs: int, camera_config: DictConfig, device: torch.device, seeds_per_env: Sequence[int] | None = None
    ):
        self.num_envs = num_envs
        self.device = str(device)
        self.config = camera_config
        self.cameras: list[Camera] = []
        self._mount_views = []
        self._scene_keys = []
        self.camera_names = [[] for _ in range(num_envs)]
        self.output_types: list[list[str]] = []
        self.num_cams = 0
        self.update_env_seeds(seeds_per_env)

    def update_env_seeds(self, seeds: Sequence[int] | None):
        self._seeds_per_env = None if seeds is None else [int(seed) for seed in seeds]
        if self._seeds_per_env is not None and len(self._seeds_per_env) != self.num_envs:
            raise ValueError("Camera seed count must equal num_envs.")

    def scene_configs(self) -> dict[str, CameraCfg]:
        """Return fresh declarations after RobotManager has resolved mount links."""
        config = OmegaConf.to_container(self.config, resolve=True)
        self.camera_config = {
            name: value for name, value in config.items() if isinstance(value, dict) and "camera" in value
        }
        self.random_config = config.get("random", {}) or {}
        names = list(self.camera_config)
        self.camera_names = [names.copy() for _ in range(self.num_envs)]
        self.num_cams = len(names)
        self.output_types = []
        self._scene_keys = []
        result = {}
        annotators = config.get("annotator", {}) or {}
        for cam_id, (name, role) in enumerate(self.camera_config.items()):
            camera = role["camera"]
            if camera.get("mesh", "pinhole") not in ("pinhole", "d435"):
                raise ValueError(f"Camera {name}: no native housing definition for mesh {camera['mesh']!r}.")
            try:
                model = CAMERA_MODELS[camera["type"]]
            except KeyError as exc:
                raise ValueError(f"Unknown camera model for {name}: {camera.get('type')!r}") from exc
            outputs = annotators.get(name, annotators.get("common", {})) or {}
            types = []
            if outputs.get("enabled", False):
                for key, setting in outputs.items():
                    if key != "enabled":
                        kind = OUTPUT_ALIASES.get(setting["type"], setting["type"])
                        if kind not in types:
                            types.append(kind)
            self.output_types.append(types)
            mount = str(camera.get("mount_link", "")).strip("/")
            parent = "{ENV_REGEX_NS}" + (f"/{mount}" if mount else "")
            position, orientation = self._local_pose(camera)
            width, height = model["resolution"]
            renderer = IsaacRtxRendererCfg(
                colorize_semantic_segmentation=False,
                colorize_instance_segmentation=False,
                colorize_instance_id_segmentation=False,
                semantic_filter=role.get("semantic_filter", "*:*"),
            )
            key = f"camera_{name}"
            cfg = CameraCfg(
                prim_path=f"{parent}/Camera_{cam_id}",
                width=width,
                height=height,
                # ObsManager schedules captures. A zero period also makes repeated
                # captures at the same physics time refresh native sensor buffers.
                update_period=0.0,
                update_latest_camera_pose=True,
                offset=CameraCfg.OffsetCfg(pos=tuple(position), rot=tuple(orientation), convention="opengl"),
                spawn=PinholeCameraCfg(
                    focal_length=model["focal_length"],
                    horizontal_aperture=model["horizontal_aperture"],
                    vertical_aperture=model["vertical_aperture"],
                    clipping_range=model["clipping_range"],
                    semantic_tags=[("class", str(role.get("semantic_label", name)))],
                ),
                data_types=types,
                renderer_cfg=renderer,
            )
            cfg.validate_config()
            result[key] = cfg
            self._scene_keys.append(key)
        return result

    @staticmethod
    def _local_pose(camera, random_pos=None, random_ori=None):
        position = np.asarray(camera.get("pos", (0, 0, 1)), dtype=np.float32)
        orientation = np.asarray(camera.get("ori", (0, 0, 0)), dtype=np.float64)
        if random_pos is not None:
            position = position + random_pos
        if orientation.shape == (3,):
            if random_ori is not None:
                orientation = orientation + random_ori
            # Preserve the benchmark's intrinsic XYZ Euler-angle convention.
            orientation = Rotation.from_euler("XYZ", orientation, degrees=True).as_quat()
        elif orientation.shape != (4,):
            raise ValueError("Camera orientation must be XYZ Euler degrees or an XYZW quaternion.")
        orientation = Rotation.from_quat(orientation).as_quat()
        return position.astype(np.float32), orientation.astype(np.float32)

    def bind(self, scene):
        """Bind scene-owned sensors after the DirectRLEnv constructor returns."""
        self.destroy()
        self.cameras = [scene[key] for key in self._scene_keys]
        for camera in self.cameras:
            if not camera.is_initialized or camera.num_instances != self.num_envs:
                raise RuntimeError(f"Camera {camera.cfg.prim_path} is not ready for all {self.num_envs} environments.")
            self._mount_views.append(FrameView(camera.cfg.prim_path, device=self.device, stage=scene.stage))

    def reset(self, env_ids: Sequence[int] | None = None, use_randomization: bool = True):
        """Restore selected mount-local poses and invalidate buffers without stepping."""
        ids = list(range(self.num_envs)) if env_ids is None else [int(i) for i in env_ids]
        if not ids:
            return
        if len(set(ids)) != len(ids) or any(i < 0 or i >= self.num_envs for i in ids):
            raise ValueError("Camera reset requires unique, in-range environment IDs.")
        if len(self.cameras) != self.num_cams:
            raise RuntimeError("Bind cameras before resetting them.")
        positions = [[] for _ in self.cameras]
        orientations = [[] for _ in self.cameras]
        for env_id in ids:
            # Local RNGs leave task/robot random streams untouched.
            rng = np.random.RandomState(None if self._seeds_per_env is None else self._seeds_per_env[env_id])
            for cam_id, (name, role) in enumerate(self.camera_config.items()):
                random = self.random_config.get(name, self.random_config.get("common", {})) or {}
                pos_noise = ori_noise = None
                if use_randomization and random.get("enabled", False):
                    if random.get("pos") is not None:
                        pos_noise = rng.uniform(random["pos"]["min"], random["pos"]["max"])
                    if random.get("ori") is not None:
                        ori_noise = rng.uniform(random["ori"]["min"], random["ori"]["max"])
                position, orientation = self._local_pose(role["camera"], pos_noise, ori_noise)
                positions[cam_id].append(position)
                orientations[cam_id].append(orientation)
        indices = wp.array(ids, dtype=wp.int32, device=self.device)
        for cam_id, (camera, view) in enumerate(zip(self.cameras, self._mount_views)):
            with view.xform_local_space_writer() as writer:
                writer.set_poses(
                    positions=wp.array(np.asarray(positions[cam_id]), dtype=wp.vec3f, device=self.device),
                    orientations=wp.array(np.asarray(orientations[cam_id]), dtype=wp.vec4f, device=self.device),
                    indices=indices,
                )
            camera.reset(env_ids=ids)

    def get_camera_intrinsics(self, cam_id: int, env_id: int = 0) -> np.ndarray:
        """Return the final native projection matrix in pixels."""
        return self.cameras[cam_id].data.intrinsic_matrices.torch[env_id].cpu().numpy().copy()

    def get_camera_extrinsics(self, cam_id: int, env_id: int) -> np.ndarray:
        """Return camera optical (ROS: +Z forward, +Y down) to world transform."""
        data = self.cameras[cam_id].data
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = Rotation.from_quat(data.quat_w_ros.torch[env_id].cpu().numpy()).as_matrix()
        transform[:3, 3] = data.pos_w.torch[env_id].cpu().numpy()
        return transform

    def destroy(self):
        """Release our transform views and references; the scene owns sensors."""
        for view in self._mount_views:
            view.close()
        self._mount_views.clear()
        self.cameras.clear()
