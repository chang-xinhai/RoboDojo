"""Table collider and material with optional native rigid dynamics."""
import numpy as np
from isaaclab.assets import RigidObject, RigidObjectCfg
from pxr import UsdPhysics

from .geometry import GeometryObject
from .physics_material import PhysicsMaterial
from .primitives import CubeMesh
from .scene_object import bind_material, mdl_material


class Table(GeometryObject):
    def __init__(self, prim_path, mdl_file_path, instance_config, mdl_name='Ceiling_Tiles', resolution=10):
        CubeMesh(prim_path, resolution=resolution)
        cfg = {'physics': {'collision': True, 'static_friction': 0.8, 'dynamic_friction': 0.8}, 'visual': {}}
        super().__init__(prim_path, None, cfg, [0, 0, 0],
                         instance_config.get('default_pos', [0, 0, 0]),
                         instance_config.get('default_ori', [0, 0, 0, 1]),
                         instance_config.get('scale', [1, 1, 1]))
        material = PhysicsMaterial(prim_path + '/physics_material', config=cfg['physics'])
        bind_material(self.prim, str(material.material.GetPath()), physics=True)
        self.instance_config = instance_config
        if mdl_file_path:
            bind_material(self.prim, mdl_material(prim_path + '/Looks/table', mdl_file_path, mdl_name))
        if not instance_config.get('static', True):
            UsdPhysics.RigidBodyAPI.Apply(self.prim)
            UsdPhysics.MassAPI.Apply(self.prim).CreateMassAttr().Set(float(instance_config.get('mass', 0.5)))
            self.asset = RigidObject(RigidObjectCfg(prim_path=prim_path, spawn=None))
            self.native_assets = [self.asset]
            self._body_sources[prim_path] = (self.asset, None)
            self._root_body_key = prim_path
            self._root_from_body = np.eye(4)
            self._cache_geometry()
