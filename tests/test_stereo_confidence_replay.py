"""CLI acceptance boundaries; synthetic inputs are not native runtime evidence."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from tests.test_multimodal import fixture
from utils.multimodal_hdf5 import HDF5Writer

ROOT = Path(__file__).resolve().parents[1]


def calibration():
    k = [[80.0, 0.0, 256.0], [0.0, 80.0, 16.0], [0.0, 0.0, 1.0]]
    pose = np.diag([1.0, -1.0, -1.0, 1.0])
    pose[2, 3] = 1.0
    return {
        "K_by_view": {name: k for name in ("rgb", "ir_left", "ir_right")},
        "T_world_from_optical": {name: pose.tolist() for name in ("rgb", "ir_left", "ir_right")},
        "baseline_m": 0.2,
    }


def write_hdf5(directory):
    frame = fixture()
    for data in frame["vision"].values():
        data.update(
            rgb=np.ones((32, 512, 3), np.uint8),
            ir_left=np.zeros((32, 512), np.uint8),
            ir_right=np.zeros((32, 512), np.uint8),
            depth_gt=np.ones((32, 512), np.float32),
            depth_sl=np.full((32, 512), np.nan, np.float32),
            calibration=calibration(),
        )
    with HDF5Writer(directory, {"unit_test_fixture": True}) as writer:
        writer.append(frame)
    return writer.path


def write_plane(directory, textured=False):
    directory.mkdir()
    (directory / "report.json").write_text(
        json.dumps(
            {"sensors": {"head": {"calibration": calibration()}}, "config": {"max_disparity": 128}, "plane_tilt_deg": 0}
        )
    )
    left = np.zeros((32, 512), np.uint8)
    if textured:
        left = np.random.default_rng(23).integers(0, 256, left.shape, dtype=np.uint8)
    right = np.zeros_like(left)
    right[:, :-16] = left[:, 16:]
    np.savez(directory / "head.npz", ir_left=left, ir_right=right, depth_sl=np.full(left.shape, np.nan, np.float32))


class StereoReplayCliTests(unittest.TestCase):
    def replay(self, directory, *arguments):
        output = directory / "replay"
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/check_stereo_confidence.py"),
                "--output",
                str(output),
                *map(str, arguments),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertTrue((output / "report.json").exists(), result.stderr)
        return result.returncode, json.loads((output / "report.json").read_text())

    def test_hdf5_only_all_nan_scene_is_metrics_only_not_plane_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            code, report = self.replay(directory, "--hdf5", write_hdf5(directory / "episode"))
            self.assertEqual(code, 0)
            self.assertEqual(report["status"], "metrics_only")
            self.assertIsNone(report["passed"])
            self.assertEqual(report["plane_acceptance"]["status"], "not_evaluated")
            self.assertEqual(len(report["scene"]), 3)
            self.assertTrue(all(camera["confidence"]["matched_pixels"] == 0 for camera in report["scene"]))

    def test_empty_plane_inventory_cannot_be_rescued_by_scene_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            empty = directory / "empty-plane"
            empty.mkdir()
            (empty / "report.json").write_text('{"sensors": {}}')
            code, report = self.replay(directory, "--plane-dir", empty, "--hdf5", write_hdf5(directory / "episode"))
            self.assertEqual(code, 1)
            self.assertFalse(report["passed"])
            self.assertEqual(report["plane_acceptance"]["status"], "failed")
            self.assertIn("no sensor inventory", report["failure_reasons"][0])

    def test_empty_hdf5_input_fails_instead_of_reporting_empty_metrics_success(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            writer = HDF5Writer(directory / "empty-episode", {"unit_test_fixture": True})
            writer.fail("Synthetic empty input for CLI rejection test")
            code, report = self.replay(directory, "--hdf5", writer.partial_path)
            self.assertEqual(code, 1)
            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["passed"])
            self.assertEqual(report["plane_acceptance"]["status"], "not_evaluated")
            self.assertEqual(report["scene"], [])
            self.assertIn("no camera observations", report["failure_reasons"][0])

    def test_untextured_plane_has_no_depth_and_fails_acceptance(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            plane = directory / "plane"
            write_plane(plane)
            code, report = self.replay(directory, "--plane-dir", plane)
            self.assertEqual(code, 1)
            self.assertFalse(report["passed"])
            metrics = report["planes"][str(plane.resolve())]["head"]["confidence"]
            self.assertEqual(metrics["matched_pixels"], 0)
            self.assertEqual(metrics["valid_fraction"], 0)

    def test_known_shift_plane_still_passes_real_error_and_coverage_checks(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            plane = directory / "plane"
            write_plane(plane, textured=True)
            code, report = self.replay(directory, "--plane-dir", plane)
            self.assertEqual(code, 0)
            self.assertTrue(report["passed"])
            self.assertEqual(report["plane_acceptance"]["status"], "passed")
            metrics = report["planes"][str(plane.resolve())]["head"]["confidence"]
            self.assertGreater(metrics["valid_fraction"], 0.25)
            self.assertLess(metrics["p95_relative_error"], 0.05)


if __name__ == "__main__":
    unittest.main()
