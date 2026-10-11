"""Acceptance refuses numerical PASS when the native solver reports data loss."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from scripts.build_physics_gallery import validate_report
from scripts.check_scene_physics_live import _flush_native_errors, _observe_native_errors


def install_events(monkeypatch):
    callbacks = []
    carb = ModuleType("carb")
    carb.__path__ = []
    carb.logging = ModuleType("carb.logging")
    carb.logging.LEVEL_WARN, carb.logging.LEVEL_ERROR = 30, 40
    dispatcher = ModuleType("carb.eventdispatcher")
    dispatcher.get_eventdispatcher = lambda: SimpleNamespace(observe_event=lambda **kw: callbacks.append(kw))
    omni = ModuleType("omni")
    omni.__path__ = []
    omni.kit = ModuleType("omni.kit")
    omni.kit.__path__ = []
    omni.kit.app = ModuleType("omni.kit.app")
    omni.kit.app.GLOBAL_EVENT_ERROR_LOG = "deferred-error-log"
    for name, module in {
        "carb": carb,
        "carb.logging": carb.logging,
        "carb.eventdispatcher": dispatcher,
        "omni": omni,
        "omni.kit": omni.kit,
        "omni.kit.app": omni.kit.app,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return callbacks, omni.kit.app


class Event(dict):
    def __missing__(self, key):
        return None


def test_native_contact_data_loss_is_collected_and_raised_after_queue_flush(monkeypatch):
    callbacks, app = install_events(monkeypatch)
    errors = []
    _observe_native_errors(errors)
    subscription = callbacks[0]
    assert subscription["event_name"] == "deferred-error-log"
    event = Event(level=40, message="PhysX error: Deformable surface contact buffer overflow detected")
    app.get_app = lambda: SimpleNamespace(update=lambda: subscription["on_event"](event))
    with pytest.raises(RuntimeError, match="Native physics errors invalidate"):
        _flush_native_errors(errors)
    assert errors[0]["kind"] == "physics_capacity_or_data_loss"


def test_rtx_normal_adjacency_is_not_misclassified_as_physics_data_loss(monkeypatch):
    callbacks, _ = install_events(monkeypatch)
    errors = []
    _observe_native_errors(errors)
    callbacks[0]["on_event"](
        Event(level=40, source="rtx.scenedb.plugin", message="normal adjacency table buffer is missing")
    )
    assert errors == []


@pytest.mark.parametrize("guard", [None, {}, {"policy": "physx-known-fatal.v1", "known_fatal_count": 1, "errors": []}])
def test_gallery_refuses_unguarded_or_native_failed_numeric_pass(guard):
    report = dict(status="passed", real_runtime=True, isaacsim_version="6.1.0.0", physx_extension="omni.physx-110.3.2")
    if guard is not None:
        report["native_error_guard"] = guard
    with pytest.raises(ValueError, match="native physics error guard"):
        validate_report(report, "candidate")


def test_gallery_accepts_guarded_exact_runtime_capture():
    validate_report(
        dict(
            status="passed",
            real_runtime=True,
            isaacsim_version="6.1.0.0",
            physx_extension="omni.physx-110.3.2",
            native_error_guard=dict(policy="physx-known-fatal.v1", known_fatal_count=0, errors=[]),
        ),
        "candidate",
    )
