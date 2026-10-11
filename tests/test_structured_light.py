"""CPU contracts for illumination isolation, stereo geometry and LUT assets."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

MODULE = Path(__file__).resolve().parents[1] / "env/camera_manager/structured_light.py"
spec = importlib.util.spec_from_file_location("structured_light", MODULE)
sl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sl)


class StructuredLightTests(unittest.TestCase):
    def test_native_camera_dependencies_enable_extension_before_import(self):
        from types import SimpleNamespace

        enabled = []
        camera, sensor, rtx_camera = object(), object(), object()
        original_import = __import__

        def import_with_optional_extension(name, *args, **kwargs):
            if name == "isaacsim.core.utils.extensions":
                return SimpleNamespace(enable_extension=enabled.append)
            if name in ("isaacsim.core.experimental.objects", "isaacsim.sensors.experimental.rtx"):
                # Reproduce the canonical experience: native modules become
                # importable only after Kit enables their optional extension.
                if "isaacsim.sensors.experimental.rtx" not in enabled:
                    raise ModuleNotFoundError(name)
                return SimpleNamespace(Camera=camera, CameraSensor=sensor, RtxCamera=rtx_camera)
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=import_with_optional_extension):
            self.assertEqual(sl._load_native_camera_types(), (camera, sensor, rtx_camera))
        self.assertEqual(enabled, ["isaacsim.sensors.experimental.rtx"])

    def test_physics_timestamp_accepts_lab3_and_legacy_sim(self):
        from types import SimpleNamespace

        modern = SimpleNamespace(get_physics_step_count=lambda: 25, get_physics_dt=lambda: 0.004)
        legacy = SimpleNamespace(current_time=0.1)
        self.assertEqual(sl.physics_timestamp_ns(modern), 100000000)
        self.assertEqual(sl.physics_timestamp_ns(legacy), 100000000)

    def test_ir_uses_projected_radiance_and_removes_ambient(self):
        off = np.full((10, 20, 3), 5.0, dtype=np.float32)
        on = off.copy()
        on[3:5, 8:11] += 2.0
        image = sl.active_ir(on, off, 4.0)
        self.assertEqual(image.dtype, np.uint8)
        self.assertEqual(int(image.sum()), 6 * 128)
        np.testing.assert_array_equal(sl.active_ir(off, off, 4.0), np.zeros((10, 20), np.uint8))

    def test_ir_does_not_amplify_negative_radiance(self):
        self.assertFalse(sl.active_ir(np.zeros((2, 3, 3)), np.ones((2, 3, 3)), 1).any())
        for invalid in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                sl.active_ir(np.zeros((2, 3, 3)), np.ones((2, 3, 3)), invalid)

    def test_intrinsics_preserve_both_focal_axes(self):
        k = sl.stereo_intrinsics((640, 480), 10, 20, 15)
        np.testing.assert_allclose(k, [[320, 0, 320], [0, 320, 240], [0, 0, 1]])

    def test_render_view_recovers_moving_optical_pose(self):
        world = np.array([[0.0, 0.0, 1.0, 0.3], [0.0, 1.0, 0.0, -0.2], [-1.0, 0.0, 0.0, 1.4], [0.0, 0.0, 0.0, 1.0]])
        params = dict(
            cameraViewTransform=np.linalg.inv(world).T.reshape(-1),
            cameraProjection=np.eye(4).reshape(-1),
            cameraFocalLength=10.0,
            cameraAperture=[20.0, 15.0],
            cameraApertureOffset=[0.0, 0.0],
            renderProductResolution=[640, 480],
            metersPerSceneUnit=1.0,
        )
        model = sl.render_camera_model(params, (480, 640))
        np.testing.assert_allclose(model["T_world_from_optical"], world @ np.diag([1.0, -1.0, -1.0, 1.0]))
        np.testing.assert_allclose(model["K"], [[320, 0, 320], [0, 320, 240], [0, 0, 1]])
        params["metersPerSceneUnit"] = 0.01
        with self.assertRaisesRegex(ValueError, "metric"):
            sl.render_camera_model(params, (480, 640))

    def test_render_calibration_rejects_wrong_resolution(self):
        params = dict(
            cameraViewTransform=np.eye(4).reshape(-1),
            cameraProjection=np.eye(4).reshape(-1),
            cameraFocalLength=10.0,
            cameraAperture=[20.0, 15.0],
            cameraApertureOffset=[0.0, 0.0],
            renderProductResolution=[640, 480],
            metersPerSceneUnit=1.0,
        )
        with self.assertRaisesRegex(ValueError, "resolution"):
            sl.render_camera_model(params, (240, 320))

    def test_known_disparity_recovers_metric_depth(self):
        rng = np.random.default_rng(4)
        left = rng.integers(0, 256, size=(100, 320), dtype=np.uint8)
        right = np.zeros_like(left)
        # A 12-pixel horizontal disparity gives z=300*0.05/12=1.25 m.
        right[:, :-12] = left[:, 12:]
        depth = sl.reconstruct_stereo(left, right, 300, 0.05, max_disparity=64)
        # Both independent SGBM searches need a full disparity-search border.
        interior = depth[10:-10, 80:-80]
        self.assertGreater(np.isfinite(interior).mean(), 0.9)
        self.assertAlmostEqual(float(np.nanmedian(interior)), 1.25, places=2)

    def test_untextured_stereo_is_invalid_not_ground_truth(self):
        image = np.zeros((64, 160), np.uint8)
        self.assertTrue(np.isnan(sl.reconstruct_stereo(image, image, 300, 0.05, max_disparity=32)).all())

    def test_occluded_eye_region_is_rejected_without_filling_known_depth(self):
        rng = np.random.default_rng(17)
        left = rng.integers(0, 256, (96, 384), dtype=np.uint8)
        right = np.zeros_like(left)
        right[:, :-12] = left[:, 12:]
        # An actual occluder replaces this right-eye region, so there is no
        # corresponding measurement for left columns 140 through 180.
        right[:, 128:168] = 0
        depth = sl.reconstruct_stereo(left, right, 300, 0.05, max_disparity=64)
        self.assertGreater(np.isnan(depth[8:-8, 145:175]).mean(), 0.98)
        visible = depth[8:-8, 200:270]
        self.assertGreater(np.isfinite(visible).mean(), 0.95)
        self.assertAlmostEqual(float(np.nanmedian(visible)), 1.25, places=2)

    def test_independent_mismatched_eyes_do_not_become_dense_plausible_depth(self):
        rng = np.random.default_rng(19)
        left = rng.integers(0, 256, (96, 384), dtype=np.uint8)
        right = rng.integers(0, 256, left.shape, dtype=np.uint8)
        depth = sl.reconstruct_stereo(left, right, 300, 0.05, max_disparity=64)
        self.assertLess(np.isfinite(depth).mean(), 0.01)

    def test_invalid_reverse_endpoint_cannot_interpolate_into_a_consistent_match(self):
        from types import SimpleNamespace

        left = np.full((32, 160), 80, np.uint8)
        forward = np.full(left.shape, -16, np.int16)
        reverse = np.full(left.shape, -129 * 16, np.int16)
        forward[16, 100] = int(65.5 * 16)
        # At xR=34.5, averaging invalid -129 and valid -2 produces -65.5.
        # That falsely agrees exactly unless BOTH interpolation endpoints
        # have real right-to-left measurements.
        reverse[16, 35] = -2 * 16
        forward[16, 120] = 16 * 16
        reverse[16, 104] = -16 * 16
        solvers = [SimpleNamespace(compute=lambda *args: forward), SimpleNamespace(compute=lambda *args: reverse)]
        with patch("cv2.StereoSGBM_create", side_effect=solvers):
            depth = sl.reconstruct_stereo(
                left,
                left,
                300,
                0.05,
                max_disparity=128,
                min_texture_std_dn=0,
                min_patch_correlation=-1,
            )
        self.assertTrue(np.isnan(depth[16, 100]))
        self.assertEqual(depth[16, 120], 300 * 0.05 / 16)

    def test_invalid_stereo_inputs_fail(self):
        image = np.zeros((64, 160), np.uint8)
        with self.assertRaises(ValueError):
            sl.reconstruct_stereo(image, image, 300, 0)
        with self.assertRaises(ValueError):
            sl.reconstruct_stereo(image.astype(float), image, 300, 0.05)
        for key, value in (
            ("lr_max_diff_px", 0),
            ("lr_max_diff_px", float("nan")),
            ("min_texture_std_dn", -1),
            ("min_patch_correlation", 1.1),
            ("min_patch_correlation", float("nan")),
            ("min_depth_m", 0),
            ("min_depth_m", float("nan")),
            ("max_depth_m", 0.01),
            ("max_depth_m", float("inf")),
            ("max_disparity", float("nan")),
        ):
            with self.assertRaises(ValueError):
                sl.reconstruct_stereo(image, image, 300, 0.05, **{key: value})
        for focal, baseline in ((float("nan"), 0.05), (300, float("nan")), (float("inf"), 0.05), (300, -0.05)):
            with self.assertRaises(ValueError):
                sl.reconstruct_stereo(image, image, focal, baseline)

    def test_projector_assets_are_deterministic_and_directional(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            pattern_a, lut_a = sl.generate_projector_assets(first, resolution=(80, 64))
            pattern_b, lut_b = sl.generate_projector_assets(second, resolution=(80, 64))
            self.assertEqual(pattern_a.read_bytes(), pattern_b.read_bytes())
            self.assertEqual(lut_a.read_bytes(), lut_b.read_bytes())
            import cv2

            pattern = cv2.imread(str(pattern_a), cv2.IMREAD_GRAYSCALE)
            self.assertGreater(np.count_nonzero(pattern), 0)
            self.assertLess(np.count_nonzero(pattern), pattern.size // 2)
            lut = cv2.imread(str(lut_a), cv2.IMREAD_UNCHANGED)
            self.assertEqual(lut.shape, (512, 512, 3))
            self.assertGreater((lut[..., 2] == -1).mean(), 0.5)

    def test_capture_cleanup_runs_on_render_failure(self):
        from types import SimpleNamespace

        capture = sl.StructuredLightCapture.__new__(sl.StructuredLightCapture)
        capture.manager = SimpleNamespace(sim=SimpleNamespace(sim=SimpleNamespace(current_time=0.5)))
        calls = []
        capture.set_projectors = lambda enabled, **kwargs: calls.append(enabled)

        def broken_render():
            raise RuntimeError("renderer failed")

        capture._render = broken_render
        with self.assertRaisesRegex(RuntimeError, "renderer failed"):
            capture.capture()
        self.assertEqual(calls[-1], False)


if __name__ == "__main__":
    unittest.main()
