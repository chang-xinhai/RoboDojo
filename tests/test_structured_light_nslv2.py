"""CPU runtime wiring contracts; model inference quality requires native runs."""

from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "structured_light_nslv2_probe", ROOT / "env/camera_manager/structured_light.py"
)
sl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sl)


class NSLv2RuntimeTests(unittest.TestCase):
    def infer(self, client, left=None, right=None, **kwargs):
        left = np.arange(12, dtype=np.uint8).reshape(3, 4) if left is None else left
        right = left[:, ::-1].copy() if right is None else right
        options = dict(fx_px=341.5, baseline_m=0.050004, stamp_ns=84000000, frame_index=21, camera="cam_left_wrist")
        options.update(kwargs)
        return sl.infer_nslv2_depth(client, left, right, **options)

    def test_original_ir_calibration_clock_and_unfiltered_left_depth(self):
        depth = np.array([[0.001, 50.0, np.nan, 1.2]] * 3, np.float32)
        provenance = {"method": "nsl_v2", "gt_input": "none", "fallback": "none"}
        client = Mock(infer=Mock(return_value=SimpleNamespace(depth_left_m=depth, provenance=provenance)))
        result, metadata = self.infer(client)
        args, kwargs = client.infer.call_args
        np.testing.assert_array_equal(args[0], np.arange(12, dtype=np.uint8).reshape(3, 4))
        np.testing.assert_array_equal(args[1], args[0][:, ::-1])
        self.assertEqual(
            kwargs, dict(fx_px=341.5, baseline_m=0.050004, stamp_ns=84000000, frame_index=21, camera="cam_left_wrist")
        )
        np.testing.assert_array_equal(result, depth)
        self.assertFalse(np.shares_memory(result, depth))
        self.assertEqual(metadata, provenance)

    def test_client_mutation_cannot_modify_recorded_ir(self):
        left, right = np.full((3, 4), 50, np.uint8), np.full((3, 4), 70, np.uint8)

        def mutate(left_input, right_input, **kwargs):
            left_input[:] = 0
            right_input[:] = 0
            return SimpleNamespace(depth_left_m=np.ones(left.shape, np.float32), provenance={})

        self.infer(SimpleNamespace(infer=mutate), left, right)
        self.assertTrue((left == 50).all())
        self.assertTrue((right == 70).all())

    def test_wrong_output_contract_hard_fails(self):
        for depth in (
            np.ones((3, 4), np.float64),
            np.ones((4, 3), np.float32),
            np.zeros((3, 4), np.float32),
            np.full((3, 4), np.inf, np.float32),
            np.full((3, 4), -1, np.float32),
        ):
            with self.subTest(dtype=depth.dtype, shape=depth.shape, value=depth.flat[0]):
                client = SimpleNamespace(
                    infer=lambda *args, **kwargs: SimpleNamespace(depth_left_m=depth, provenance={})
                )
                with self.assertRaises(ValueError):
                    self.infer(client)

    def test_invalid_calibration_and_inputs_never_reach_worker(self):
        client = Mock()
        for options in (
            {"fx_px": 0},
            {"baseline_m": float("nan")},
            {"left": np.ones((3, 4), np.float32)},
            {"right": np.ones((4, 3), np.uint8)},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.infer(client, **options)
        client.infer.assert_not_called()

    def test_runtime_factory_has_no_sgbm_backend(self):
        with patch.object(sl, "_load_native_camera_types") as native:
            with self.assertRaisesRegex(ValueError, "no fallback"):
                sl.StructuredLightCapture(None, {"backend": "sgbm"})
            native.assert_not_called()
        factory = Mock(return_value=object())
        module = SimpleNamespace(NSLv2Client=SimpleNamespace(from_config=factory))
        with patch.dict(sys.modules, {"reproduction.isaac61.nslv2.client": module}):
            options = {"weights_path": "/real/checkpoint.pt", "precision": "mixed"}
            sl.create_nslv2_client({"nsl_v2": options})
            factory.assert_called_once_with(options)

    def make_capture(self, infer):
        capture = sl.StructuredLightCapture.__new__(sl.StructuredLightCapture)
        clock = SimpleNamespace(count=21)
        clock.get_physics_step_count = lambda: clock.count
        clock.get_physics_dt = lambda: 0.004
        capture.manager = SimpleNamespace(sim=SimpleNamespace(sim=clock))
        capture.config, capture.baseline = {}, 0.05
        capture.depth_client = SimpleNamespace(infer=infer)
        capture.pattern, capture.pattern_sha256, capture.direction_texture = "pattern", "pattern-hash", "lut"
        capture.set_projectors = Mock()
        capture._render = lambda **kwargs: 2
        capture._phase_scope = lambda phase: nullcontext()
        capture._record_render_state = lambda phase: capture._phase_states.update(
            {phase: {"physics_timestamp_ns": 84000000}}
        )
        shape = (3, 4)

        def params(world, fx):
            return dict(
                cameraViewTransform=np.linalg.inv(world @ np.diag([1, -1, -1, 1])).T.reshape(-1),
                cameraProjection=np.eye(4).reshape(-1),
                cameraFocalLength=fx / shape[1],
                cameraAperture=[1.0, 0.75],
                cameraApertureOffset=[0.0, 0.0],
                renderProductResolution=shape[::-1],
                metersPerSceneUnit=1.0,
            )

        capture.rigs = {0: {}}
        for index, name in enumerate(("cam_head", "cam_left_wrist", "cam_right_wrist")):
            left = np.eye(4)
            left[:3, 3] = [0.01 * index, -0.22, 1.3]
            right = left.copy()
            right[0, 3] += 0.050004
            model_params = {
                view: SimpleNamespace(get_data=lambda value=params(world, 300 + index * 40): value)
                for view, world in (("rgb", left), ("ir_left", left), ("ir_right", right))
            }
            capture.rigs[0][name] = dict(
                shape=shape,
                cam_id=index,
                eyes={"ir_left": object(), "ir_right": object()},
                hdr={"ir_left": object(), "ir_right": object()},
                rgb=SimpleNamespace(get_data=lambda key: np.full(shape, 999, np.float32)),
                params=model_params,
                projector_intensity=1500,
                rgb_path=f"/World/{name}",
            )

        def average(views, hdr, samples):
            images = {name: np.full((*shape, 3), 22, np.uint8) for name in views}
            values = {
                name: np.full((*shape, 3), (1100 if name == "ir_left" else 2200) if not views else 0, np.float32)
                for name in hdr
            }
            return images, values, 1

        capture._average_views = average
        return capture, clock

    def test_all_three_capture_rigs_pass_actual_calibration_and_preserve_modalities(self):
        requests = []

        def infer(left, right, **kwargs):
            requests.append((left.copy(), right.copy(), kwargs))
            return SimpleNamespace(
                depth_left_m=np.full(left.shape, 2.0, np.float32),
                provenance={"method": "nsl_v2", "camera": kwargs["camera"]},
            )

        capture, clock = self.make_capture(infer)
        with patch.object(sl, "reconstruct_stereo", side_effect=AssertionError("SGBM must not run")):
            result = capture._capture()
        self.assertEqual(len(requests), 3)
        self.assertEqual(clock.count, 21)
        for index, (left, right, kwargs) in enumerate(requests):
            data = result[kwargs["camera"]]
            np.testing.assert_array_equal(data["ir_left"], left)
            np.testing.assert_array_equal(data["ir_right"], right)
            self.assertEqual(kwargs["fx_px"], 300 + 40 * index)
            self.assertAlmostEqual(kwargs["baseline_m"], 0.050004, places=12)
            self.assertEqual(kwargs["stamp_ns"], 84000000)
            self.assertEqual(kwargs["frame_index"], 21)
            self.assertTrue((data["depth_gt"] == 999).all())
            self.assertTrue((data["rgb"] == 22).all())
            self.assertTrue((data["depth_sl"] == 2).all())
            self.assertEqual(data["calibration"]["depth_sl_frame"], "ir_left")
            self.assertEqual(data["provenance"]["depth_sl_method"], "nsl_v2")

    def test_worker_failure_propagates_and_restores_projectors(self):
        def failed(*args, **kwargs):
            raise RuntimeError("NSLv2 actual service failure")

        capture, _ = self.make_capture(failed)
        with (
            patch.object(sl, "reconstruct_stereo", side_effect=AssertionError("fallback must not run")),
            self.assertRaisesRegex(RuntimeError, "NSLv2 actual service failure"),
        ):
            capture._capture()
        self.assertEqual(capture.set_projectors.call_args.args, (False,))

    def test_inference_cannot_advance_frozen_physics(self):
        def infer(left, right, **kwargs):
            clock.count += 1
            return SimpleNamespace(depth_left_m=np.ones(left.shape, np.float32), provenance={})

        capture, clock = self.make_capture(infer)
        with self.assertRaisesRegex(RuntimeError, "Physics advanced"):
            capture._capture()

    def test_renderer_cleanup_error_still_releases_model_owner(self):
        capture = sl.StructuredLightCapture.__new__(sl.StructuredLightCapture)
        capture.set_projectors = Mock()
        capture._delivery_fence = SimpleNamespace(close=Mock(side_effect=RuntimeError("fence teardown")))
        client = Mock()
        capture.depth_client = client
        with self.assertRaisesRegex(RuntimeError, "fence teardown"):
            capture.destroy()
        client.close.assert_called_once()
        self.assertIsNone(capture.depth_client)


if __name__ == "__main__":
    unittest.main()
