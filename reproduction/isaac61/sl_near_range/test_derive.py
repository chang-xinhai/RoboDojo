"""Synthetic CPU boundary tests; never native capture evidence."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("near_range_derived_candidate", Path(__file__).parent / "derive.py")
derived = importlib.util.module_from_spec(spec)
spec.loader.exec_module(derived)


class DerivedTests(unittest.TestCase):
    def test_known_near_shift_recovers_metric_depth_without_gt_input(self):
        left = np.random.default_rng(31).integers(0, 256, (64, 640), dtype=np.uint8)
        right = np.zeros_like(left)
        right[:, :-320] = left[:, 320:]
        original = np.full(left.shape, np.nan, np.float32)
        z, mask = derived.derive(left, right, original, 397.0412731487769, 0.05)
        self.assertGreater(mask[:, 360:600].mean(), 0.65)
        self.assertLess(np.percentile(abs(z[mask] - 397.0412731487769 * 0.05 / 320), 95), 1e-4)
        self.assertTrue(np.isnan(original).all())

    def test_independent_mismatched_eyes_do_not_create_dense_foreground(self):
        rng = np.random.default_rng(4)
        left = rng.integers(0, 256, (64, 640), dtype=np.uint8)
        right = rng.integers(0, 256, left.shape, dtype=np.uint8)
        original = np.full(left.shape, np.nan, np.float32)
        z, mask = derived.derive(left, right, original, 397.0412731487769, 0.05)
        self.assertLess(mask.mean(), 0.001)
        self.assertEqual(np.isfinite(z).sum(), mask.sum())

    def test_preserves_original_finite_bitwise_and_does_not_fill_untextured_ir(self):
        left = np.zeros((64, 640), np.uint8)
        right = left.copy()
        original = np.full(left.shape, np.nan, np.float32)
        original[10:30, 250:400] = np.float32(0.198765)
        before = original.copy()
        z, mask = derived.derive(left, right, original, 397.0412731487769, 0.05)
        np.testing.assert_array_equal(z, before)
        np.testing.assert_array_equal(original, before)
        self.assertFalse(mask.any())

    def test_near_surface_occluded_in_right_eye_stays_missing(self):
        left = np.random.default_rng(31).integers(0, 256, (64, 640), dtype=np.uint8)
        right = np.zeros_like(left)
        right[:, :-320] = left[:, 320:]
        right[:, 70:120] = 0
        original = np.full(left.shape, np.nan, np.float32)
        z, mask = derived.derive(left, right, original, 397.0412731487769, 0.05)
        self.assertLess(mask[:, 400:430].mean(), 0.01)
        self.assertGreater(mask[10:50, 470:550].mean(), 0.6)
        self.assertTrue(np.isnan(z[:, 400:430]).mean() > 0.99)


if __name__ == "__main__":
    unittest.main()
