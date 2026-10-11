"""Viewer display and HDF5-only export checks on declared CPU fixtures."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from reproduction.convert_multimodal_hdf5 import convert
from reproduction.visualize_multimodal import _taxels, letterbox, metric_color, visualize
from tests.test_multimodal import fixture
from utils.multimodal import MultimodalWriter
from utils.multimodal_hdf5 import HDF5Reader, HDF5Writer


class VisualizationTests(unittest.TestCase):
    def test_letterbox_preserves_portrait_geometry(self):
        image = np.full((320, 240, 3), 255, np.uint8)
        rendered = np.asarray(letterbox(image, (280, 160)))
        white = np.all(rendered == 255, axis=-1)
        self.assertEqual(white.sum(), 120 * 160)
        self.assertEqual(white[:, :80].sum(), 0)
        self.assertTrue(white[:, 80:200].all())

    def test_declared_official_taxel_axes_display_z_rows_and_x_columns(self):
        data = {
            "normal_force": np.arange(6, dtype=np.float32),
            "metadata": {"array_size": [2, 3], "taxel_array_axes": ["sensor_local_x", "sensor_local_z"]},
        }
        np.testing.assert_equal(_taxels(data, "normal_force"), [[0, 3], [1, 4], [2, 5]])
        del data["metadata"]["taxel_array_axes"]
        np.testing.assert_equal(_taxels(data, "normal_force"), [[0, 1, 2], [3, 4, 5]])

    def test_metric_scale_fixed_and_invalid_distinct(self):
        a = metric_color(np.array([[1.0, 1.5], [np.nan, 2.0]]), 0.1, 2.5)
        b = metric_color(np.array([[0.0, 1.0], [2.5, 20.0]]), 0.1, 2.5)
        np.testing.assert_equal(a[0, 0], b[0, 1])
        np.testing.assert_equal(a[1, 0], [130, 0, 130])

    def test_direct_hdf5_no_npz_selects_measured_contact_and_exports_video(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            with HDF5Writer(directory, {"synthetic_cpu_fixture": True}) as writer:
                for index in range(1, 4):
                    frame = fixture(index)
                    sample = frame["tactile"]["left"]["finger_0"]
                    sample["normal_force"][0] = 0.2 if index == 2 else 0
                    sample["optical_depth_delta"] = np.full((2, 4, 1), -0.001, np.float32)
                    sample["metadata"] = {"array_size": [2, 4]}
                    frame["vision"]["head"]["provenance"]["untrusted"] = "</script><script>alert(1)</script>"
                    writer.append(frame)
            report = visualize(writer.path)
            self.assertEqual(report["source_format"], "hdf5")
            self.assertEqual(report["selected_frame"]["physics_frame"], 2)
            self.assertEqual(report["frames"], 3)
            self.assertFalse(list(directory.glob("*.npz")))
            self.assertTrue(Path(report["video"]).stat().st_size > 100)
            html = Path(report["html"]).read_text()
            self.assertIn("normal_force", html)
            self.assertIn("const selectedPreviewIndex=1", html)
            self.assertNotIn("</script><script>alert", html)
            self.assertIn("\\u003c/script", html)
            self.assertEqual(json.loads((directory / "visualization.json").read_text())["selected_frame"]["index"], 1)

    def test_legacy_conversion_preserves_original_runtime_evidence_and_null_forces(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = MultimodalWriter(directory, {"source_commit": "original-fixture", "real_runtime": False})
            frame = fixture()
            for samples in frame["tactile"].values():
                for sample in samples.values():
                    for field in ("normal_force", "shear_force", "penetration_depth", "points_pos_w"):
                        sample[field] = None
                    sample["provenance"].update(force_field_available=False, force_unavailable_reason="fixture cloth")
            writer.append(frame)
            writer.close()
            report = convert(directory)
            with HDF5Reader(report["file"]) as reader:
                self.assertIsNone(reader.read_frame(0)["tactile"]["left"]["finger_0"]["shear_force"])
                metadata = reader.metadata["run"]
                self.assertEqual(metadata["source_commit"], "original-fixture")
                self.assertTrue(metadata["format_conversion"]["conversion_only"])
                self.assertTrue(metadata["format_conversion"]["original_files"]["frame_000000.npz"])
            self.assertTrue((Path(directory) / "frame_000000.npz").exists())
