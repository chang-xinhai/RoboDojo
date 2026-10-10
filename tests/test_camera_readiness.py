"""Real payload readiness, finite depth, timeout and frozen physics contracts."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

PATH = Path(__file__).resolve().parents[1] / "env/camera_manager/capture/render_sync.py"
spec = importlib.util.spec_from_file_location("readiness", PATH)
readiness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(readiness)


class CameraReadinessTests(unittest.TestCase):
    def setUp(self):
        self.original_color_guard = readiness.ensure_camera_color_render_enabled
        patcher = patch.object(readiness, "ensure_camera_color_render_enabled")
        self.color_guard = patcher.start()
        self.addCleanup(patcher.stop)

    def test_headless_depth_camera_color_optimization_is_cleared_before_render(self):
        native_settings = {}
        # Exercise the real helper with the same set_bool boundary Kit exposes.
        native_settings["/rtx/sdg/force/disableColorRender"] = True
        settings = SimpleNamespace(set_bool=lambda key, value: native_settings.__setitem__(key, value))
        self.color_guard.side_effect = lambda: self.original_color_guard(settings)

        def render():
            self.assertFalse(native_settings["/rtx/sdg/force/disableColorRender"])

        with (
            patch.object(readiness, "zero_delay_active", return_value=True),
            patch.object(readiness, "dispatcher_time", return_value=None),
        ):
            readiness.wait_for_camera_data(SimpleNamespace(render=render), self.payload, physics_state=lambda: 10)
        self.color_guard.assert_called_once()

    def payload(self, rgb=1, depth=1):
        return [
            {
                "rgb": [{"data": np.full((2, 3, 4), rgb, dtype=np.uint8)}],
                "distance_to_image_plane": [{"data": np.full((2, 3, 1), depth)}],
            }
        ]

    def test_alpha_alone_does_not_make_an_empty_rgb_frame_ready(self):
        data = self.payload(rgb=0)
        data[0]["rgb"][0]["data"][..., 3] = 255
        self.assertIn("RGB", readiness.pending_camera_payloads(data)[0])

    def test_partial_finite_depth_is_accepted_with_real_rgb(self):
        data = self.payload(depth=float("inf"))
        data[0]["distance_to_image_plane"][0]["data"][0, 0, 0] = 2
        self.assertEqual(readiness.pending_camera_payloads(data), [])

    def test_gpu_backed_payload_is_read_through_native_host_bridge(self):
        calls = []

        class GpuImage:
            def detach(self):
                calls.append("detach")
                return self

            def cpu(self):
                calls.append("cpu")
                return SimpleNamespace(numpy=lambda: np.ones((2, 3, 3), dtype=np.uint8))

        data = self.payload()
        data[0]["rgb"][0]["data"] = GpuImage()
        self.assertEqual(readiness.pending_camera_payloads(data), [])
        self.assertEqual(calls, ["detach", "cpu"])

    def test_render_until_requested_product_is_ready_without_physics_step(self):
        rendered = []
        env = SimpleNamespace(render=lambda: rendered.append(1))
        with (
            patch.object(readiness, "zero_delay_active", return_value=True),
            patch.object(readiness, "dispatcher_time", return_value=None),
        ):
            data, passes = readiness.wait_for_camera_data(
                env,
                lambda: self.payload(rgb=0 if len(rendered) < 3 else 1),
                physics_state=lambda: (0.4, 100),
                max_passes=5,
            )
        self.assertEqual(passes, 3)
        self.assertEqual(len(rendered), 3)
        self.assertEqual(readiness.pending_camera_payloads(data), [])

    def test_timeout_names_the_requested_channel_without_returning_a_fake_frame(self):
        env = SimpleNamespace(render=lambda: None)
        with (
            patch.object(readiness, "zero_delay_active", return_value=True),
            patch.object(readiness, "dispatcher_time", return_value=None),
        ):
            with self.assertRaisesRegex(readiness.CameraFrameNotReady, "3 frozen-physics.*distance_to_image_plane"):
                readiness.wait_for_camera_data(
                    env, lambda: self.payload(depth=float("inf")), physics_state=lambda: (0.4, 100), max_passes=3
                )

    def test_rejects_a_render_callback_that_advances_physics(self):
        state = [100]
        env = SimpleNamespace(render=lambda: state.__setitem__(0, state[0] + 1))
        with (
            patch.object(readiness, "zero_delay_active", return_value=True),
            patch.object(readiness, "dispatcher_time", return_value=None),
        ):
            with self.assertRaisesRegex(RuntimeError, "Physics advanced"):
                readiness.wait_for_camera_data(env, self.payload, physics_state=lambda: state[0])

    def test_stalled_dispatcher_cannot_pass_with_an_old_valid_buffer(self):
        rendered = []
        env = SimpleNamespace(render=lambda: rendered.append(1))
        with (
            patch.object(readiness, "zero_delay_active", return_value=True),
            patch.object(readiness, "dispatcher_time", return_value=4.0),
        ):
            with self.assertRaisesRegex(readiness.CameraFrameNotReady, "dispatcher delivered 0/1 fresh frames"):
                readiness.wait_for_camera_data(env, self.payload, physics_state=lambda: 100, max_passes=3)
        self.assertEqual(len(rendered), 3)


if __name__ == "__main__":
    unittest.main()
