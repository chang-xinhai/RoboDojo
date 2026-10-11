"""CPU contracts; these fixtures are not neural-model inference evidence."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from reproduction.isaac61.nslv2.client import NSLv2Prediction, resolve_config
from reproduction.isaac61.nslv2.export import replace_frame
from reproduction.isaac61.nslv2.runtime import decode, prepare
from tests.test_multimodal import fixture


class RuntimeTests(unittest.TestCase):
    def pair(self):
        return np.arange(640 * 480, dtype=np.uint8).reshape(480, 640)

    def test_signed_disparity_uses_horizontal_resize_scale_only(self):
        image = self.pair()
        left, right, transform = prepare(image, image, 720, 320, "resize")
        self.assertEqual(left.dtype, np.uint8)
        self.assertGreater(left.max(), 1)
        np.testing.assert_array_equal(left, right)
        disparity = np.full((320, 720), -225.0, np.float32)
        confidence = np.ones_like(disparity)
        depth, native_disparity, _, mask = decode(disparity, confidence, transform, 400.0, 0.05, None)
        np.testing.assert_allclose(depth, 0.1, rtol=1e-6)
        np.testing.assert_allclose(native_disparity, -200.0)
        self.assertTrue(mask.all())

    def test_invalid_predictions_remain_nan_without_confidence_filter_by_default(self):
        image = np.zeros((320, 720), np.uint8)
        _, _, transform = prepare(image, image, 720, 320, "resize")
        disparity = np.full(image.shape, -50.0, np.float32)
        confidence = np.zeros_like(disparity)
        disparity[0, :3] = [0.0, np.nan, np.inf]
        confidence[0, 3] = np.nan
        depth, _, _, mask = decode(disparity, confidence, transform, 400.0, 0.05, None)
        self.assertTrue(np.isnan(depth[0, :4]).all())
        self.assertTrue(mask[1:].all())
        self.assertFalse(np.isinf(depth).any())
        filtered, _, _, filtered_mask = decode(disparity, confidence, transform, 400.0, 0.05, 0.2)
        self.assertFalse(filtered_mask.any())
        self.assertTrue(np.isnan(filtered).all())

    def test_nonfinite_calibration_and_threshold_rejected(self):
        image = np.zeros((16, 32), np.uint8)
        _, _, transform = prepare(image, image, 32, 16, "resize")
        disparity = np.full(image.shape, -10.0, np.float32)
        for fx, baseline, threshold in (
            (np.nan, 0.05, None),
            (400.0, np.inf, None),
            (0.0, 0.05, None),
            (400.0, 0.05, np.nan),
        ):
            with self.subTest(fx=fx, baseline=baseline, threshold=threshold), self.assertRaises(ValueError):
                decode(disparity, np.ones_like(disparity), transform, fx, baseline, threshold)

    def test_ir_contract_and_explicit_no_fallback_configuration(self):
        image = self.pair()
        with self.assertRaises(ValueError):
            prepare(image.astype(np.float32), image, 720, 320, "resize")
        with self.assertRaises(ValueError):
            prepare(image, image[:, :-1], 720, 320, "resize")
        with self.assertRaises(ValueError):
            resolve_config({"fallback": "sgbm"})

    def test_full_depth_replacement_preserves_ir_gt_and_tactile(self):
        frame = fixture(1)
        inputs = []
        expected = {}
        for name, camera in frame["vision"].items():
            camera["calibration"] = {
                "K_by_view": {"ir_left": [[400.0, 0.0, 0.0], [0.0, 400.0, 0.0], [0.0, 0.0, 1.0]]},
                "baseline_m": 0.05,
            }
            expected[name] = {key: camera[key].copy() for key in ("rgb", "depth_gt", "ir_left", "ir_right")}
            camera["depth_sl"][:] = 9.0

        class FakeClient:
            def infer(self, left, right, **kwargs):
                inputs.append((left, right, kwargs))
                depth = np.full(left.shape, 0.1, np.float32)
                depth[0, 0] = np.nan
                return NSLv2Prediction(depth, depth, depth, np.isfinite(depth), depth, depth, {"method": "nsl_v2"})

        tactile = frame["tactile"]
        sync = frame["sync"].copy()
        replace_frame(frame, FakeClient(), 77)
        self.assertEqual(len(inputs), 3)
        self.assertIs(frame["tactile"], tactile)
        self.assertEqual(frame["sync"], sync)
        for name, camera in frame["vision"].items():
            self.assertEqual(camera["provenance"]["depth_sl_method"], "nsl_v2")
            self.assertTrue(np.isnan(camera["depth_sl"][0, 0]))
            np.testing.assert_allclose(camera["depth_sl"][1:], 0.1)
            for key, value in expected[name].items():
                np.testing.assert_array_equal(camera[key], value)
        self.assertTrue(
            all(set(kwargs) == {"fx_px", "baseline_m", "stamp_ns", "frame_index", "camera"} for _, _, kwargs in inputs)
        )


if __name__ == "__main__":
    unittest.main()
