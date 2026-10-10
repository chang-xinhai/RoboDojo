"""Keep legacy Isaac Sim prims on Torch while Lab 3 owns the PhysX lifecycle.

Both tensor views refer to the same real PhysX scene. Lab 3 and TacSL retain
their official Warp view; legacy prims require Torch arrays and WXYZ poses.
No second simulation context, physics step, or synthetic state is introduced.
"""

from importlib import import_module
import sys


def install_legacy_prim_adapter():
    """Bind only legacy modules after Lab 3 initializes its PhysxManager."""
    from isaaclab.sim.utils.stage import get_current_stage_id
    from isaaclab_physx.physics import PhysxManager
    from isaaclab_physx.physics.physx_manager import IsaacEvents
    from isaacsim.core.simulation_manager.impl.simulation_manager import SimulationManager as OriginalManager
    import omni.physics.tensors

    OriginalManager.enable_all_default_callbacks(False)

    class LegacyManagerMeta(type):
        def __getattr__(cls, name):
            # The native camera time-mapping and PhysX flush interfaces are
            # owned by the loaded Isaac Sim extension, not a second context.
            if name in ("_physics_sim_interface", "_simulation_manager_interface"):
                return getattr(OriginalManager, name)
            if hasattr(PhysxManager, name):
                return getattr(PhysxManager, name)
            raise AttributeError(f"Unsupported legacy SimulationManager API: {name}")

    class LegacySimulationManager(metaclass=LegacyManagerMeta):
        _torch_view = None
        _view_key = None

        @classmethod
        def get_backend(cls):
            return PhysxManager.get_backend()

        @classmethod
        def _get_backend_utils(cls):
            return import_module("isaacsim.core.utils." + cls.get_backend())

        @classmethod
        def get_physics_sim_view(cls):
            official_view = PhysxManager.get_physics_sim_view()
            if official_view is None:
                cls._torch_view = cls._view_key = None
                return None
            stage_id = get_current_stage_id()
            key = (id(official_view), stage_id, cls.get_backend())
            if cls._view_key != key:
                cls._torch_view = omni.physics.tensors.create_simulation_view(cls.get_backend(), stage_id=stage_id)
                if cls._torch_view is None:
                    raise RuntimeError("Unable to create the legacy prim tensor view for the current PhysX scene")
                cls._torch_view.set_subspace_roots("/")
                cls._view_key = key
            return cls._torch_view

        @classmethod
        def register_callback(cls, callback, event, **kwargs):
            # Legacy and Lab 3 export separate enum classes with the same
            # native event values. Normalize before the official dispatcher.
            try:
                event = IsaacEvents(event.value)
            except (AttributeError, ValueError):
                pass
            return PhysxManager.register_callback(callback, event=event, **kwargs)

    prefixes = ("isaacsim.core.prims", "isaacsim.sensors.camera", "env.camera_manager", "env.scene_manager")
    bound = []
    for name, module in tuple(sys.modules.items()):
        if name == "isaacsim.core.simulation_manager" or any(
            name == prefix or name.startswith(prefix + ".") for prefix in prefixes
        ):
            if module is not None and hasattr(module, "SimulationManager"):
                module.SimulationManager = LegacySimulationManager
                bound.append(name)
    return LegacySimulationManager, tuple(bound)
