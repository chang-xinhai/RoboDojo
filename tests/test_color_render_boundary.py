"""Color acquisition repairs depth-sensor global state, without changing images."""

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from env.camera_manager.capture import render_sync

COLOR_DISABLE = "/rtx/sdg/force/disableColorRender"


class Settings:
    def __init__(self):
        self.values = {COLOR_DISABLE: True, "/rtx/rendermode": "RealTimePathTracing", "/rtx/post/aa/op": 0}
        self.writes = []

    def set_bool(self, key, value):
        self.writes.append((key, value))
        self.values[key] = value


@pytest.fixture
def runtime(monkeypatch):
    settings = Settings()
    carb = ModuleType("carb")
    carb.settings = ModuleType("carb.settings")
    carb.settings.get_settings = lambda: settings
    monkeypatch.setitem(sys.modules, "carb", carb)
    monkeypatch.setitem(sys.modules, "carb.settings", carb.settings)
    monkeypatch.setattr(render_sync, "zero_delay_active", lambda: True)
    monkeypatch.setattr(render_sync, "dispatcher_time", lambda: None)
    return settings


def payload(color):
    return [{"rgb": [{"data": color}], "distance_to_image_plane": [{"data": np.ones((2, 3))}]}]


def test_restore_is_idempotent_and_does_not_change_quality_or_physics_settings(runtime):
    original = dict(runtime.values)
    render_sync.ensure_camera_color_render_enabled(runtime)
    render_sync.ensure_camera_color_render_enabled(runtime)
    assert runtime.values == dict(original, **{COLOR_DISABLE: False})
    assert runtime.writes == [(COLOR_DISABLE, False)] * 2


def test_real_camera_wait_restores_color_before_any_render_or_read(runtime):
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    calls = []

    def render():
        assert runtime.values[COLOR_DISABLE] is False
        calls.append("render")

    def read():
        assert runtime.values[COLOR_DISABLE] is False
        calls.append("read")
        return payload(image)

    result, passes = render_sync.wait_for_camera_data(
        SimpleNamespace(render=render), read, physics_state=lambda: (100, 0.4)
    )
    assert calls == ["render", "read"]
    assert passes == 1
    assert result[0]["rgb"][0]["data"] is image
    np.testing.assert_array_equal(image, np.arange(18, dtype=np.uint8).reshape(2, 3, 3))


def test_late_depth_only_sensor_creation_cannot_poison_next_color_capture(runtime):
    seen = []
    env = SimpleNamespace(render=lambda: seen.append(runtime.values[COLOR_DISABLE]))
    for _ in range(2):
        # Reproduce the documented process-wide depth-only renderer side effect.
        runtime.values[COLOR_DISABLE] = True
        render_sync.wait_for_camera_data(
            env, lambda: payload(np.ones((2, 3, 3), dtype=np.uint8)), physics_state=lambda: (100, 0.4)
        )
    assert seen == [False, False]


def test_reenabling_color_does_not_turn_black_buffers_into_success(runtime):
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    env = SimpleNamespace(render=lambda: None)
    with pytest.raises(render_sync.CameraFrameNotReady, match="nonzero RGB"):
        render_sync.wait_for_camera_data(env, lambda: payload(image), physics_state=lambda: (100, 0.4), max_passes=2)
    assert runtime.values[COLOR_DISABLE] is False
    np.testing.assert_array_equal(image, np.zeros((2, 3, 3), dtype=np.uint8))
