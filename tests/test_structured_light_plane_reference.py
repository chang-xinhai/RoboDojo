"""Independent analytic oracle for the live tilted-target reconstruction test."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np

MODULE = Path(__file__).resolve().parents[1] / "scripts/check_structured_light_live.py"
spec = importlib.util.spec_from_file_location("sl_live_validation", MODULE)
live = importlib.util.module_from_spec(spec)
spec.loader.exec_module(live)


class PlaneReferenceTests(unittest.TestCase):
    def setUp(self):
        self.intrinsics = np.array([[2.0, 0.0, 2.5], [0.0, 2.0, 1.5], [0.0, 0.0, 1.0]])
        self.pose = np.diag([1.0, -1.0, -1.0, 1.0])
        self.pose[2, 3] = 1.0

    def test_frontoparallel_plane_keeps_optical_depth_constant(self):
        depth = live.plane_reference_depth((3, 5), self.intrinsics, self.pose, 0)
        np.testing.assert_allclose(depth, np.ones((3, 5)))

    def test_tilted_surface_depth_changes_across_the_image(self):
        depth = live.plane_reference_depth((3, 5), self.intrinsics, self.pose, 30)
        self.assertAlmostEqual(depth[1, 2], 1.0)
        self.assertAlmostEqual(depth[1, 0], 1 / (1 + np.tan(np.pi / 6)))
        self.assertAlmostEqual(depth[1, 4], 1 / (1 - np.tan(np.pi / 6)))
        np.testing.assert_allclose(depth[0], depth[2])
        self.assertGreater(float(np.nanstd(depth)), 0.0)

    def test_rolled_camera_rotates_the_surface_gradient_into_rows(self):
        pose = self.pose.copy()
        pose[:3, :3] = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]
        depth = live.plane_reference_depth((3, 5), self.intrinsics, pose, 30)
        np.testing.assert_allclose(depth[:, 0], depth[:, 4])
        self.assertGreater(abs(depth[0, 2] - depth[2, 2]), 0.1)


if __name__ == "__main__":
    unittest.main()
