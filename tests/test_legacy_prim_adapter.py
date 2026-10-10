"""Verify the local legacy facade contract without importing or launching Kit."""

from enum import Enum
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from utils.legacy_prim_adapter import install_legacy_prim_adapter


class OfficialEvent(Enum):
    PHYSICS_READY = "physics_ready"
    POST_RESET = "post_reset"


class LegacyEvent(Enum):
    PHYSICS_READY = "physics_ready"
    POST_RESET = "post_reset"


class OtherEvent(Enum):
    CUSTOM = "unrecognized_custom_event"


@pytest.fixture
def runtime(monkeypatch):
    modules = {}

    def module(name, **attributes):
        if name in modules:
            result = modules[name]
        else:
            result = ModuleType(name)
            result.__path__ = []
            modules[name] = result
            monkeypatch.setitem(sys.modules, name, result)
            if "." in name:
                parent, child = name.rsplit(".", 1)
                setattr(module(parent), child, result)
        result.__dict__.update(attributes)
        return result

    official_view = SimpleNamespace(frontend="warp", native_scene="shared_physx_scene")
    state = {"stage_id": 731, "official_view": official_view, "backend": "torch"}
    original = SimpleNamespace(
        enable_all_default_callbacks=Mock(),
        _physics_sim_interface=object(),
        _simulation_manager_interface=object(),
        original_only_api=Mock(),
    )
    callback_handle = object()
    manager = SimpleNamespace(
        get_backend=Mock(side_effect=lambda: state["backend"]),
        get_physics_sim_view=Mock(side_effect=lambda: state["official_view"]),
        get_physics_sim_device=Mock(return_value="cuda:0"),
        register_callback=Mock(return_value=callback_handle),
        deregister_callback=Mock(),
        _physics_sim_interface=SimpleNamespace(flush_changes=Mock()),
        _simulation_manager_interface=SimpleNamespace(get_simulation_time_at_swh_frame=Mock()),
    )
    factory = Mock(
        side_effect=lambda backend, stage_id: SimpleNamespace(
            frontend=backend, stage_id=stage_id, set_subspace_roots=Mock()
        )
    )
    module("isaaclab.sim.utils.stage", get_current_stage_id=lambda: state["stage_id"])
    module("isaaclab_physx.physics", PhysxManager=manager)
    module("isaaclab_physx.physics.physx_manager", IsaacEvents=OfficialEvent)
    module("isaacsim.core.simulation_manager.impl.simulation_manager", SimulationManager=original)
    module("omni.physics.tensors", create_simulation_view=factory)
    helpers = {backend: module("isaacsim.core.utils." + backend) for backend in ("torch", "numpy")}
    intended = (
        "isaacsim.core.simulation_manager",
        "isaacsim.core.prims.impl.prim",
        "isaacsim.sensors.camera.camera",
        "env.camera_manager.camera_manager",
        "env.scene_manager.objects.rigid",
    )
    excluded = (
        "isaaclab.assets.rigid_object",
        "isaaclab_contrib.sensors.tacsl_sensor",
        "env.robot_manager.robot_manager",
        "isaacsim.core.prims_extra",
        "env.scene_manager_extra",
    )
    for name in intended + excluded:
        module(name, SimulationManager=original)
    facade, bound = install_legacy_prim_adapter()
    return SimpleNamespace(
        state=state,
        official_view=official_view,
        original=original,
        manager=manager,
        factory=factory,
        helpers=helpers,
        modules=modules,
        intended=intended,
        excluded=excluded,
        facade=facade,
        bound=bound,
        callback_handle=callback_handle,
    )


@pytest.mark.parametrize("backend", ["torch", "numpy"])
def test_native_frontend_uses_official_stage_without_replacing_warp_view(runtime, backend):
    runtime.state["backend"] = backend
    legacy = runtime.facade.get_physics_sim_view()
    runtime.factory.assert_called_once_with(backend, stage_id=731)
    legacy.set_subspace_roots.assert_called_once_with("/")
    assert legacy.frontend == backend
    assert runtime.facade._get_backend_utils() is runtime.helpers[backend]
    assert runtime.manager.get_physics_sim_view() is runtime.official_view
    assert runtime.official_view.frontend == "warp"
    assert legacy is not runtime.official_view
    assert runtime.facade.get_physics_sim_view() is legacy
    assert runtime.factory.call_count == 1


def test_no_native_view_or_second_context_created_before_official_physics_ready(runtime):
    runtime.state["official_view"] = None
    assert runtime.facade.get_physics_sim_view() is None
    runtime.factory.assert_not_called()
    assert not hasattr(runtime.facade, "SimulationContext")


@pytest.mark.parametrize("changed", ["official_view", "stage_id", "backend"])
def test_lifecycle_change_refreshes_native_view(runtime, changed):
    first = runtime.facade.get_physics_sim_view()
    runtime.state[changed] = {
        "official_view": object(),
        "stage_id": 983,
        "backend": "numpy",
    }[changed]
    refreshed = runtime.facade.get_physics_sim_view()
    assert refreshed is not first
    runtime.factory.assert_called_with(runtime.state["backend"], stage_id=runtime.state["stage_id"])
    assert runtime.factory.call_count == 2
    assert runtime.facade.get_physics_sim_view() is refreshed


def test_stop_discards_cache_and_restart_recreates_view_for_same_stage(runtime):
    first = runtime.facade.get_physics_sim_view()
    runtime.state["official_view"] = None
    assert runtime.facade.get_physics_sim_view() is None
    runtime.state["official_view"] = runtime.official_view
    assert runtime.facade.get_physics_sim_view() is not first
    assert runtime.factory.call_count == 2


def test_native_view_creation_failure_does_not_return_stale_state(runtime):
    runtime.factory.return_value = None
    runtime.factory.side_effect = None
    with pytest.raises(RuntimeError, match="current PhysX scene"):
        runtime.facade.get_physics_sim_view()


@pytest.mark.parametrize("event", [LegacyEvent.PHYSICS_READY, OfficialEvent.PHYSICS_READY])
def test_callbacks_use_official_enum_dispatch_and_keep_handle_for_cleanup(runtime, event):
    callback = Mock()
    handle = runtime.facade.register_callback(callback, event, order=19, name="legacy", wrap_weak_ref=False)
    assert handle is runtime.callback_handle
    runtime.manager.register_callback.assert_called_once_with(
        callback, event=OfficialEvent.PHYSICS_READY, order=19, name="legacy", wrap_weak_ref=False
    )
    runtime.facade.deregister_callback(handle)
    runtime.manager.deregister_callback.assert_called_once_with(handle)


def test_unrecognized_event_is_not_silently_remapped(runtime):
    callback = Mock()
    runtime.facade.register_callback(callback, OtherEvent.CUSTOM)
    runtime.manager.register_callback.assert_called_once_with(callback, event=OtherEvent.CUSTOM)


def test_only_requested_legacy_modules_are_rebound_and_original_callbacks_disabled(runtime):
    runtime.original.enable_all_default_callbacks.assert_called_once_with(False)
    for name in runtime.intended:
        assert runtime.modules[name].SimulationManager is runtime.facade
        assert name in runtime.bound
    for name in runtime.excluded:
        assert runtime.modules[name].SimulationManager is runtime.original, name
        assert name not in runtime.bound
    implementation = runtime.modules["isaacsim.core.simulation_manager.impl.simulation_manager"]
    assert implementation.SimulationManager is runtime.original


def test_native_interfaces_are_retained_but_unknown_original_apis_fail(runtime):
    assert runtime.facade._physics_sim_interface is runtime.original._physics_sim_interface
    assert runtime.facade._simulation_manager_interface is runtime.original._simulation_manager_interface
    assert runtime.facade.get_physics_sim_device() == "cuda:0"
    for name in ("original_only_api", "unsupported_method"):
        with pytest.raises(AttributeError, match="Unsupported legacy SimulationManager API"):
            getattr(runtime.facade, name)
