"""CPU contracts for illumination isolation, stereo geometry and LUT assets."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

MODULE = Path(__file__).resolve().parents[1] / "env/camera_manager/structured_light.py"
spec = importlib.util.spec_from_file_location("structured_light", MODULE)
sl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sl)


class StructuredLightTests(unittest.TestCase):
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

    def test_known_disparity_recovers_metric_depth(self):
        rng = np.random.default_rng(4)
        left = rng.integers(0, 256, size=(100, 320), dtype=np.uint8)
        right = np.zeros_like(left)
        # A 12-pixel horizontal disparity gives z=300*0.05/12=1.25 m.
        right[:, :-12] = left[:, 12:]
        depth = sl.reconstruct_stereo(left, right, 300, 0.05, max_disparity=64)
        interior = depth[10:-10, 80:-20]
        self.assertGreater(np.isfinite(interior).mean(), 0.9)
        self.assertAlmostEqual(float(np.nanmedian(interior)), 1.25, places=2)

    def test_untextured_stereo_is_invalid_not_ground_truth(self):
        image = np.zeros((64, 160), np.uint8)
        self.assertTrue(np.isnan(sl.reconstruct_stereo(image, image, 300, 0.05, max_disparity=32)).all())

    def test_invalid_stereo_inputs_fail(self):
        image = np.zeros((64, 160), np.uint8)
        with self.assertRaises(ValueError):
            sl.reconstruct_stereo(image, image, 300, 0)
        with self.assertRaises(ValueError):
            sl.reconstruct_stereo(image.astype(float), image, 300, 0.05)

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
