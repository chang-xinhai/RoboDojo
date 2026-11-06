"""Preserve authored conveyor graphs while binding each moving body natively."""
from isaaclab.assets import RigidObject, RigidObjectCfg
import omni.kit.app
from pxr import Usd, UsdPhysics

from .scene_object import SceneObject


class DynamicObject(SceneObject):
    def __init__(self, prim_path, usd_path, inst_config, env_origin, default_pos, default_ori, scale):
        if not usd_path:
            raise ValueError('DynamicObject requires a USD asset')
        manager = omni.kit.app.get_app().get_extension_manager()
        if not manager.is_extension_enabled('isaacsim.asset.gen.conveyor'):
            if not manager.set_extension_enabled_immediate('isaacsim.asset.gen.conveyor', True):
                raise RuntimeError('Could not enable isaacsim.asset.gen.conveyor')
        super().__init__(prim_path, usd_path, inst_config, env_origin, default_pos, default_ori, scale)
        self._graph_values = []
        for prim in Usd.PrimRange(self.prim):
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                raise ValueError(f'Dynamic asset contains an articulation: {prim.GetPath()}')
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                asset = RigidObject(RigidObjectCfg(prim_path=str(prim.GetPath()), spawn=None))
                self.native_assets.append(asset)
                self._body_sources[str(prim.GetPath())] = (asset, None)
            node_type = prim.GetAttribute('node:type')
            if node_type and node_type.Get() == 'isaacsim.asset.gen.conveyor.IsaacConveyor':
                targets = prim.GetRelationship('inputs:conveyorPrim').GetTargets()
                if len(targets) != 1 or not targets[0].HasPrefix(self.prim.GetPath()):
                    raise ValueError(f'Conveyor graph target escaped its environment: {targets}')
            for attr in prim.GetAttributes():
                if attr.GetName().startswith('graph:variable:'):
                    self._graph_values.append((attr, attr.Get()))
        self._cache_geometry()

    def apply_saved_pose(self):
        super().apply_saved_pose()
        for attr, value in self._graph_values:
            attr.Set(value)
