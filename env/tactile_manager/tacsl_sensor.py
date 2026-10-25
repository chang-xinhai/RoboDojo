# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Official TacSL with narrow Isaac Lab 3 sensor boundary adapters.

Each sensor observes one concrete environment, permitting heterogeneous objects.
Environment counting, array access, exact view paths and native SDF coordinate
units are adapted. Contact equations, gel sampling and Taxim rendering retain
the official model.
"""

from isaaclab.renderers import CameraRenderSpec
from isaaclab.sensors import Camera, SensorBase
import isaaclab.sim as sim_utils
from isaaclab.sim.views import FrameView
from isaaclab.utils.warp import ProxyArray
from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensor
from isaaclab_contrib.sensors.tacsl_sensor.visuotactile_render import GelsightRender
import numpy as np
from pxr import UsdGeom
import torch
import warp as wp

from .camera_boundary import TacSLCameraBoundary, cleanup_camera_once
from .contact_query import create_contact_query_view, find_warp_contact_components, validate_query_backend
from .sdf_boundary import BodyFrameSdfView, shape_pose_in_body


class SingleEnvironmentSensorBase(SensorBase):
    """Select one actual USD parent instead of the global clone-plan count.

    EA's SensorBase initializes every sensor against the complete clone plan,
    including absolute paths. This mirrors its USD branch and buffer setup,
    without changing any global simulation/renderer state or clone-plan getter.
    """

    def _initialize_impl(self):
        sim = sim_utils.SimulationContext.instance()
        if sim is None:
            raise RuntimeError("Simulation Context is not initialized")
        self._device = sim.device
        self._backend = sim.backend
        self._sim_physics_dt = sim.get_physics_dt()
        self._clone_plan = None
        parent_path = str(self.cfg.prim_path).rsplit("/", 1)[0]
        self._parent_prims = sim_utils.find_matching_prims(parent_path)
        self._num_envs = len(self._parent_prims)
        if self._num_envs != 1:
            raise RuntimeError(
                f"An absolute TacSL sensor must resolve one parent: {parent_path}, found {self._num_envs}"
            )
        self._ALL_ENV_MASK = wp.ones(1, dtype=wp.bool, device=self._device)
        self._reset_mask = wp.zeros(1, dtype=wp.bool, device=self._device)
        self._reset_mask_torch = wp.to_torch(self._reset_mask)
        self._is_outdated = wp.ones(1, dtype=wp.bool, device=self._device)
        self._timestamp = wp.zeros(1, dtype=wp.float32, device=self._device)
        self._timestamp_last_update = wp.zeros_like(self._timestamp)
        self._data_generation = 0
        self._data_generation_last_update = -1
        if self._debug_vis_handle is None:
            self.set_debug_vis(self.cfg.debug_vis)


class SingleEnvironmentCamera(Camera, SingleEnvironmentSensorBase):
    """Official Camera/RTX renderer with one-instance parent/buffer allocation."""

    def __del__(self):
        cleanup_camera_once(self, super().__del__)

    def _initialize_impl(self):
        # Mirrors the pinned official Camera initialize. Sensor/view/buffers
        # represent one camera, but stage preprocessing must retain the global
        # environment count shared with head/wrist camera renderers.
        SingleEnvironmentSensorBase._initialize_impl(self)
        sim = sim_utils.SimulationContext.instance()
        if self._renderer is None:
            self._renderer = sim.render_context.get_renderer(self.cfg.renderer_cfg)
        paths = tuple(str(p.GetPath()) for p in sim_utils.find_matching_prims(self.cfg.prim_path, self.stage))
        prefix = "/World/envs/env_0/"
        relative = paths[0].removeprefix(prefix) if paths and paths[0].startswith(prefix) else ""
        spec = CameraRenderSpec(
            cfg=self.cfg,
            device=str(self._device),
            num_instances=1,
            camera_prim_paths=paths,
            view_count=1,
            camera_path_relative_to_env_0=relative,
        )
        self._renderer.prepare_cameras(self.stage, spec)
        clone_plan = sim.get_clone_plan()
        scene_count = int(clone_plan.env_ids.numel()) if clone_plan is not None else 1
        sim.render_context.ensure_prepare_stage(self.stage, scene_count)
        self._view = FrameView(self.cfg.prim_path, device=self._device, stage=self.stage)
        if self._view.count != 1:
            raise RuntimeError(f"Expected one actual TacSL camera, found {self._view.count}: {self.cfg.prim_path}")
        self._ALL_INDICES = wp.array(np.arange(1, dtype=np.int32), device=self._device)
        self._frame = ProxyArray(wp.zeros(1, device=self._device, dtype=wp.int64))
        self._sensor_prims.clear()
        for prim in self._view.prims:
            if not prim.IsA(UsdGeom.Camera):
                raise RuntimeError(f"TacSL prim is not a Camera: {prim.GetPath()}")
            self._sensor_prims.append(UsdGeom.Camera(prim))
        self._render_data = self._renderer.create_render_data(spec)
        self._create_buffers()


class RoboDojoTacSLSensor(TacSLCameraBoundary, VisuoTactileSensor, SingleEnvironmentSensorBase):
    """Official TacSL equations with exact per-environment PhysX view paths."""

    def __init__(self, cfg, *, contact_query_backend="auto"):
        self._requested_contact_query_backend = validate_query_backend(contact_query_backend)
        super().__init__(cfg)

    def _initialize_camera_tactile(self):
        cfg = self.cfg.camera_cfg
        if cfg is None:
            raise ValueError("TacSL camera configuration is required")
        if (cfg.height, cfg.width) != (self.cfg.render_cfg.image_height, self.cfg.render_cfg.image_width):
            raise ValueError("TacSL camera dimensions must match official calibration")
        if not all(
            key in ("distance_to_image_plane", "depth", "depth/distance_to_image_plane") for key in cfg.data_types
        ):
            raise ValueError("TacSL requires image-plane depth only")
        cfg.update_period = self.cfg.update_period
        self._tactile_rgb_render = GelsightRender(self.cfg.render_cfg, device=self.device)
        self._camera_sensor = SingleEnvironmentCamera(cfg)
        if not self._camera_sensor.is_initialized:
            self._camera_sensor._initialize_impl()
            self._camera_sensor._is_initialized = True
        self._data.tactile_rgb_image = torch.zeros((1, cfg.height, cfg.width, 3), device=self._device)
        self._data.tactile_depth_image = torch.zeros((1, cfg.height, cfg.width, 1), device=self._device)

    def _create_physx_views(self):
        # Upstream unconditionally replaces env_0 with env_* for SDF/body paths.
        # Exact views isolate real contacts in the intended environment.
        elastomer_path = str(self._parent_prims[0].GetPath())
        self._elastomer_body_view = self._physics_sim_view.create_rigid_body_view([elastomer_path])
        if self._elastomer_body_view.count != 1:
            raise RuntimeError(f"TacSL requires one initialized elastomer rigid body: {elastomer_path}")
        self._elastomer_com_b = wp.to_torch(self._elastomer_body_view.get_coms()).to(self._device)[..., :3]
        if self.cfg.contact_object_prim_path_expr is None:
            return
        if self._requested_contact_query_backend in ("auto", "warp_triangle_mesh"):
            mesh, body = find_warp_contact_components(self.stage, self.cfg.contact_object_prim_path_expr)
        else:
            mesh, body = self._find_contact_object_components()
        self._contact_object_sdf_view, self.contact_query_backend, self.contact_query_metadata = (
            create_contact_query_view(
                self._physics_sim_view,
                mesh,
                self.cfg.tactile_array_size[0] * self.cfg.tactile_array_size[1],
                self._device,
                self._requested_contact_query_backend,
                body=body,
            )
        )
        self._contact_object_body_view = self._physics_sim_view.create_rigid_body_view([str(body.GetPath())])
        if self._contact_object_sdf_view.count != 1 or self._contact_object_body_view.count != 1:
            raise RuntimeError(
                f"TacSL requires one initialized SDF collider/body at {mesh.GetPath()}, {body.GetPath()}; "
                f"matched shapes={self._contact_object_sdf_view.count}, bodies={self._contact_object_body_view.count}."
            )
        if self.contact_query_metadata.get("query_coordinate_frame") != "rigid_body_metres":
            rotation, translation, scale = shape_pose_in_body(mesh, body)
            self._contact_object_sdf_view = BodyFrameSdfView(
                self._contact_object_sdf_view, rotation, translation, self._device, geometry_scale=scale
            )
        self._contact_object_com_b = wp.to_torch(self._contact_object_body_view.get_coms()).to(self._device)[..., :3]
