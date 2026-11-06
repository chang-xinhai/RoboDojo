"""Read ordinary images from scene-owned Isaac Lab Camera buffers."""

from collections.abc import Callable, Sequence
from copy import deepcopy

import torch

from env.camera_manager.camera_manager import CameraManager


class CaptureManager:
    """Expose camera outputs without owning render products or a physics loop.

    Output layout is ``data[role_index][output_name][env_index]`` with ``data``
    and ``info`` entries. Arrays are CPU snapshots, not mutable SDK buffer views.
    RGB is uint8 HWC/3, GT optical-Z depth is float32 HWC/1 in metres, normals
    HWC/3, motion vectors HWC/2, and segmentation IDs int32 HWC/1. Segmentation
    dictionaries are retained in ``info``. No depth reconstruction is performed.
    """

    def __init__(self, num_envs: int, config, camera_manager: CameraManager, device: torch.device):
        self.num_envs = num_envs
        self.camera_manager = camera_manager
        self.sim = None

    def initialize(self, sim):
        self.sim = sim

    def capture(self, render: Callable[[], None], passes: int = 1):
        """Render and refresh cameras through public APIs without advancing physics.

        The SDK owns rendering completion. Its camera frame counter records a
        capture request, not an independent delivery timestamp; this method does
        not infer image freshness from changing counters or pixel content.
        """
        if passes < 1:
            raise ValueError("Capture requires at least one render pass.")
        for _ in range(passes):
            render()
            for camera in self.camera_manager.cameras:
                if not camera.is_initialized:
                    raise RuntimeError(f"Camera {camera.cfg.prim_path} is not initialized.")
                camera.update(0.0, force_recompute=True)

    def step(self, env_ids: Sequence[int] | None = None, cam_ids: Sequence[int] | None = None):
        """Snapshot native buffers in caller-selected environment/role order."""
        env_ids = list(range(self.num_envs)) if env_ids is None else list(env_ids)
        cam_ids = list(range(self.camera_manager.num_cams)) if cam_ids is None else list(cam_ids)
        if any(i < 0 or i >= self.num_envs for i in env_ids):
            raise IndexError("Capture environment index out of range.")
        result = []
        for cam_id in cam_ids:
            camera = self.camera_manager.cameras[cam_id]
            data = camera.data
            role_data = {}
            for kind in self.camera_manager.output_types[cam_id]:
                output = data.output[kind].torch
                if tuple(output.shape[:3]) != (self.num_envs, *camera.image_shape):
                    raise RuntimeError(f"Unexpected {kind} output shape: {tuple(output.shape)}.")
                snapshots = output[env_ids].cpu().numpy().copy()
                info = deepcopy(data.info.get(kind))
                role_data[kind] = [{"data": snapshots[i], "info": info} for i in range(len(env_ids))]
            result.append(role_data)
        return result

    def reset(self):
        """Sensors are invalidated by scene/CameraManager.reset, with no reattach."""

    def destroy(self):
        self.sim = None
