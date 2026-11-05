"""Connect Isaac Sim 6.2 Core wrappers to Isaac Lab 3's PhysX lifecycle.

Lab owns warmup, stepping and native-view invalidation. Core's deprecated
wrappers still read its singleton's frontend views, even when ``initialize``
receives a view. These three private slots are the only version-pinned bridge;
no Core lifecycle callback or third-party implementation is replaced.
"""

from isaaclab.physics import PhysicsEvent
from isaaclab.sim.utils.stage import get_current_stage_id
from isaaclab_physx.physics import PhysxCfg
from isaacsim.core.simulation_manager import SimulationManager
import omni.physics.tensors


class CorePhysicsBridge:
    def __init__(self, sim):
        if not isinstance(sim.cfg.physics, PhysxCfg):
            raise RuntimeError("RoboDojo requires the Isaac Sim PhysX backend.")
        if any(SimulationManager.get_default_callback_status().values()):
            raise RuntimeError("Core lifecycle callbacks must stay disabled; Isaac Lab owns PhysX.")
        if (
            SimulationManager.get_physics_sim_view() is not None
            or SimulationManager.get_physics_simulation_view() is not None
        ):
            raise RuntimeError("Core physics views are already owned by another simulation.")

        self._sim = sim
        self._frontend_view = None
        self._warp_view = None
        self._previous_backend = SimulationManager.get_backend()
        # Lab 2.3 used Torch on both CPU readback and CUDA pipelines. Core camera
        # and articulation consumers rely on that tensor contract.
        self._frontend = "torch"
        SimulationManager.set_backend(self._frontend)
        self._callbacks = []
        try:
            # Lab dispatches this synchronously before the legacy Core-ready event.
            self._callbacks.append(
                sim.physics_manager.register_callback(self._bind, PhysicsEvent.PHYSICS_READY, order=-1000)
            )
            self._callbacks.append(sim.physics_manager.register_callback(self._unbind, PhysicsEvent.STOP, order=-1000))
            if sim.physics_sim_view is not None:
                self._bind()
        except Exception:
            self.close()
            raise

    def _bind(self, _event=None):
        warp_view = self._sim.physics_sim_view
        if warp_view is None or not warp_view.is_valid:
            raise RuntimeError("Isaac Lab has no valid PhysX view at physics-ready.")
        if warp_view is self._warp_view:
            return
        self._unbind()
        expected_ordinal = -1 if self._sim.device == "cpu" else int(self._sim.device.split(":")[-1])
        if warp_view.device_ordinal != expected_ordinal:
            raise RuntimeError(
                f"PhysX pipeline device {warp_view.device_ordinal} disagrees with Lab device {self._sim.device}."
            )
        # Creating another frontend for the existing stage does not load or step physics.
        frontend_view = omni.physics.tensors.create_simulation_view(
            self._frontend, stage_id=get_current_stage_id(), backend="physx"
        )
        frontend_view.set_subspace_roots("/")
        self._frontend_view = frontend_view
        self._warp_view = warp_view
        SimulationManager._physics_sim_view = frontend_view
        SimulationManager._physics_sim_view__warp = warp_view
        SimulationManager._simulation_view_created = True

    def _unbind(self, _event=None):
        # Never invalidate: both frontends share the native view owned by Lab.
        if self._frontend_view is not None and SimulationManager._physics_sim_view is self._frontend_view:
            SimulationManager._physics_sim_view = None
            SimulationManager._simulation_view_created = False
        if self._warp_view is not None and SimulationManager._physics_sim_view__warp is self._warp_view:
            SimulationManager._physics_sim_view__warp = None
        self._frontend_view = None
        self._warp_view = None

    def close(self):
        self._unbind()
        for callback in self._callbacks:
            callback.deregister()
        self._callbacks.clear()
        SimulationManager.set_backend(self._previous_backend)
