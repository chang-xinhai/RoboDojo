"""Interactive task mechanisms composed with a native Lab Articulation."""
import numpy as np
import torch
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim.schemas import apply_articulation_root_properties
from pxr import Usd, UsdGeom, UsdPhysics

from .scene_object import SceneObject, matrix_pose, pose_matrix, world_matrix


class ArticulationObject(SceneObject):
    def __init__(self, prim_path, usd_path, inst_config, env_origin, default_pos, default_ori, scale):
        super().__init__(prim_path, usd_path, inst_config, env_origin, default_pos, default_ori, scale)
        self.physics_cfg = inst_config.get('physics', {})
        if self.physics_cfg.get('fixed_base', False):
            roots = [p for p in Usd.PrimRange(self.prim) if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
            if len(roots) != 1:
                raise ValueError(f'Expected one articulation root in {prim_path}, got {len(roots)}')
            if not apply_articulation_root_properties(str(roots[0].GetPath()), [], stage=self.stage, fix_root_link=True):
                raise RuntimeError(f'Could not fix articulation root at {prim_path}')
        self._authored_targets = {}
        for prim in Usd.PrimRange(self.prim):
            token = 'angular' if prim.IsA(UsdPhysics.RevoluteJoint) else 'linear'
            if prim.HasAPI(UsdPhysics.DriveAPI, token):
                value = UsdPhysics.DriveAPI.Get(prim, token).GetTargetPositionAttr().Get()
                if value is not None:
                    self._authored_targets[prim.GetName()] = np.deg2rad(value) if token == 'angular' else value
        self.asset = Articulation(ArticulationCfg(
            prim_path=prim_path, spawn=None,
            actuators={'mechanism': ImplicitActuatorCfg(joint_names_expr=['.*'], stiffness=None, damping=None)},
        ))
        self.native_assets = [self.asset]
        # Resolve public native body order only after physics initializes.
        self._body_prims = [p for p in Usd.PrimRange(self.prim) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        self._authored_frames = {str(p.GetPath()): pose_matrix(*matrix_pose(world_matrix(p)))
                                 for p in self._body_prims}
        self._authored_root = pose_matrix(*matrix_pose(world_matrix(self.prim)))
        self._authored_link_frames = {
            str(p.GetPath()): world_matrix(p) for p in Usd.PrimRange(self.prim) if p.IsA(UsdGeom.Xformable)
        }
        self._body_sources = {str(p.GetPath()): (self.asset, None) for p in self._body_prims}
        self._cache_geometry()

    def post_init(self):
        super().post_init()
        names = self.asset.body_names
        for prim in self._body_prims:
            if prim.GetName() not in names:
                raise ValueError(f'Physics link missing from native asset: {prim.GetPath()}')
            self._body_sources[str(prim.GetPath())] = (self.asset, names.index(prim.GetName()))
        # PhysX root link is index zero in the default native ordering.
        roots = [p for p in self._body_prims if p.GetName() == names[0]]
        if len(roots) != 1:
            raise ValueError(f'Ambiguous articulation root body in {self.prim_path}')
        self._root_body_key = str(roots[0].GetPath())
        # Saved articulation layouts specify the physics root-link pose, as did
        # the original set_local_pose API, not the enclosing USD Xform pose.
        # Preserve child geometry offsets without adding the authored root-link
        # offset a second time when restoring a layout.
        self._root_from_body = np.eye(4)
        position, quaternion = matrix_pose(self._authored_root)
        self._initial_poses = [torch.as_tensor(
            np.r_[position, quaternion], dtype=torch.float32, device=self.asset.device).reshape(1, 7)]
        self.initial_joint_positions = self.asset.data.joint_pos.torch.clone()
        self._drive_targets = self.initial_joint_positions.clone()
        for index, name in enumerate(self.asset.joint_names):
            if name in self._authored_targets:
                self._drive_targets[0, index] = float(self._authored_targets[name])
        if self.physics_cfg.get('reset_drive', False):
            limits = self.asset.data.joint_pos_limits.torch
            self.initial_joint_positions = self._drive_targets.clamp(limits[..., 0], limits[..., 1]).clone()
        self.reset_drive_targets()

    def get_current_joint_positions(self):
        return self.asset.data.joint_pos.torch[0].detach().clone()

    def set_current_joint_positions(self, positions):
        positions = torch.as_tensor(positions, dtype=torch.float32, device=self.asset.device).reshape(1, -1)
        self.asset.write_joint_state_to_sim_index(
            position=positions, velocity=torch.zeros_like(positions), env_ids=[0])

    def reset_drive_targets(self):
        self.asset.actuators.target_command.set_position_index(value=self._drive_targets, env_ids=[0])
        self.asset.actuators.target_command.set_velocity_index(value=torch.zeros_like(self._drive_targets), env_ids=[0])

    def _place(self, offset):
        super()._place(offset)
        self.set_current_joint_positions(self.initial_joint_positions)
        self.reset_drive_targets()

    def get_link_pose(self, link_name):
        matches = [p for p in Usd.PrimRange(self.prim) if p.GetName() == link_name]
        if len(matches) != 1:
            raise ValueError(f'Expected one link named {link_name} in {self.prim_path}, got {len(matches)}')
        prim = matches[0]
        body = prim
        while body and str(body.GetPath()) not in self._body_sources and body != self.prim:
            body = body.GetParent()
        key = str(body.GetPath())
        if key in self._body_sources:
            # Authored mesh/link offsets are constant; moving link state is native.
            relative = np.linalg.inv(self._authored_frames[key]) @ self._authored_link_frames[str(prim.GetPath())]
            frame = self._frame(key) @ relative
        else:
            frame = (self._root_frame() @ np.linalg.inv(self._authored_frames[self._root_body_key])
                     @ self._authored_link_frames[str(prim.GetPath())])
        pos, quat = matrix_pose(frame)
        return np.r_[pos, quat].astype(np.float32)

    def get_joint_info(self, joint_name='button_joint'):
        index = self.asset.joint_names.index(joint_name)
        limits = self.asset.data.joint_pos_limits.torch[0, index]
        return {'index': index, 'position': self.asset.data.joint_pos.torch[0, index].item(),
                'lower': limits[0].item(), 'upper': limits[1].item()}

    def get_all_joints_info(self):
        return {name: self.get_joint_info(name) for name in self.asset.joint_names}
