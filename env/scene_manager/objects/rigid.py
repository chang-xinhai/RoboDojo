"""Benchmark rigid object backed by one native Lab RigidObject."""
from isaaclab.assets import RigidObject as NativeRigidObject, RigidObjectCfg
from pxr import Usd, UsdGeom, UsdPhysics
import numpy as np
import torch

from .physics_material import PhysicsMaterial
from .scene_object import SceneObject, bind_material, matrix_pose


def _author_compound_rigid_body(root):
    """A declared rigid object is one root body with its authored child shapes.

    Validate before editing: independent child dynamics belong to an articulation
    or dynamic object. All edits affect the composed stage, not the referenced USD.
    """
    prims = list(Usd.PrimRange(root))
    for prim in prims:
        if prim.IsA(UsdPhysics.Joint):
            raise ValueError(f'Rigid object contains a joint: {prim.GetPath()}')
        if prim.IsA(UsdGeom.Xformable) and UsdGeom.Xformable(prim).GetResetXformStack():
            raise ValueError(f'Rigid object contains a reset transform stack: {prim.GetPath()}')
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        body = UsdPhysics.RigidBodyAPI(prim)
        if not body.GetRigidBodyEnabledAttr().Get() or body.GetKinematicEnabledAttr().Get():
            raise ValueError(f'Rigid object contains a disabled or kinematic body: {prim.GetPath()}')
        if prim != root:
            allowed = {'physics:rigidBodyEnabled', 'physics:kinematicEnabled'}
            tuning = [a.GetName() for a in prim.GetAuthoredAttributes()
                      if a.GetName().startswith(('physics:', 'physxRigidBody:')) and a.GetName() not in allowed]
            relationships = [r for r in prim.GetAuthoredRelationships() if r.GetName().startswith('physics:')]
            if prim.HasAPI(UsdPhysics.MassAPI) or tuning or relationships:
                raise ValueError(f'Rigid object contains independently configured child dynamics: {prim.GetPath()}')
    for prim in prims:
        schemas = prim.GetAppliedSchemas()
        for schema in ('PhysicsArticulationRootAPI', 'PhysxArticulationAPI'):
            if schema in schemas:
                prim.RemoveAPI(schema)
        if prim != root:
            for schema in ('PhysicsRigidBodyAPI', 'PhysxRigidBodyAPI'):
                if schema in schemas:
                    prim.RemoveAPI(schema)
    UsdPhysics.RigidBodyAPI.Apply(root)
    return root


class RigidObject(SceneObject):
    def __init__(self, prim_path, usd_path, inst_config, env_origin, default_pos, default_ori, scale):
        super().__init__(prim_path, usd_path, inst_config, env_origin, default_pos, default_ori, scale)
        body = _author_compound_rigid_body(self.prim)
        physics = inst_config.get('physics', {})
        mass = min(float(physics.get('mass', 0.5)), 0.5)
        UsdPhysics.MassAPI.Apply(body).CreateMassAttr().Set(mass if mass > 0 else 0.05)
        material = PhysicsMaterial(prim_path + '/physics_material',
                                   static_friction=physics.get('static_friction', 0.6),
                                   dynamic_friction=physics.get('dynamic_friction', 1.5),
                                   restitution=physics.get('restitution', 0.0))
        bind_material(body, str(material.material.GetPath()), physics=True)
        self.asset = NativeRigidObject(RigidObjectCfg(prim_path=prim_path, spawn=None))
        self.native_assets = [self.asset]
        key = str(body.GetPath())
        self._body_sources[key] = (self.asset, None)
        self._root_body_key = key
        self._root_from_body = np.eye(4)
        self._cache_geometry()

    def post_init(self):
        super().post_init()
        # Native startup advances physics before post_init. Restore the saved
        # layout, not the displaced warmup pose, when staging the next episode.
        position, quaternion = matrix_pose(self._authored_root)
        self._initial_poses = [torch.as_tensor(
            np.r_[position, quaternion], dtype=torch.float32, device=self.asset.device).reshape(1, 7)]
