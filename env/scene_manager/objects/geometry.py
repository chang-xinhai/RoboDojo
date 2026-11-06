"""Static task geometry authored with public USD schemas."""
from pxr import Usd, UsdGeom, UsdPhysics

from .scene_object import SceneObject


class GeometryObject(SceneObject):
    def __init__(self, prim_path, usd_path, inst_config, env_origin, default_pos, default_ori, scale,
                 primitive_type=None):
        super().__init__(prim_path, usd_path, inst_config, env_origin, default_pos, default_ori, scale, primitive_type)
        physics = inst_config.get('physics', {})
        for prim in Usd.PrimRange(self.prim):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr().Set(False)
            if prim.IsA(UsdGeom.Gprim) and physics.get('collision', False):
                UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr().Set(True)
        self._cache_geometry()
