"""Complete-only derived viewer contracts on CPU synthetic fixtures."""

import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_resume import PrefixTests, source_frame
import visualize_derived as viewer


class ViewerTests(PrefixTests):
    def test_standalone_cli_bootstrap_without_pythonpath_or_repository_cwd(self):
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        for name in ("sidecar.py", "resume_sidecar.py", "visualize_derived.py"):
            with self.subTest(script=name):
                script = Path(__file__).resolve().parent / name
                completed = subprocess.run(
                    [sys.executable, str(script), "--help"],
                    cwd=self.root,
                    env=environment,
                    text=True,
                    capture_output=True,
                    timeout=30,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn("usage:", completed.stdout)

    def complete_source_and_sidecar(self):
        self.native.append(source_frame(2))
        self.native.complete()
        viewer.resume.process(self.output, self.native.path, final=True)

    def test_complete_native_and_sidecar_stream_png_and_mp4_without_mutation(self):
        self.complete_source_and_sidecar()
        source_before = viewer.resume.base.file_sha(self.native.path)
        sidecar_before = viewer.resume.base.file_sha(self.output)
        report = viewer.visualize(
            self.native.path, self.output, self.root / "video", preview_frame=1, tile_width=180, tile_height=150
        )
        self.assertEqual(report["frames"], 2)
        self.assertAlmostEqual(report["fps_from_source_physics_clock"], 25.0)
        self.assertIn("near-range derived", report["mode"])
        self.assertFalse(report["task_complete_claimed"])
        for artifact in report["artifacts"].values():
            self.assertTrue(Path(artifact).stat().st_size > 0)
        self.assertEqual(source_before, viewer.resume.base.file_sha(self.native.path))
        self.assertEqual(sidecar_before, viewer.resume.base.file_sha(self.output))

    def test_partial_sidecar_rejected_before_any_input_hdf5_open(self):
        self.native.complete()
        with (
            patch.object(viewer, "HDF5Reader", side_effect=AssertionError("do not read for incomplete input")),
            patch.object(viewer.h5py, "File", side_effect=AssertionError("do not open partial sidecar")),
        ):
            with self.assertRaisesRegex(ValueError, "never partial"):
                viewer.visualize(self.native.path, self.output.with_suffix(".hdf5.partial"), self.root / "video")

    def test_complete_filename_does_not_hide_incomplete_sidecar_status(self):
        self.start()
        self.native.complete()
        self.output.with_suffix(".hdf5.partial").rename(self.output)
        with patch.object(viewer, "VideoStreamWriter", side_effect=AssertionError("do not encode incomplete result")):
            with self.assertRaisesRegex(ValueError, "sidecar is incomplete"):
                viewer.visualize(self.native.path, self.output, self.root / "video")
        self.assertFalse((self.root / "video").exists())

    def test_wrong_final_source_hash_is_rejected_before_video_export(self):
        self.complete_source_and_sidecar()
        with h5py.File(self.output, "r+") as handle:
            handle.attrs["final_original_hdf5_sha256"] = "wrong"
        with patch.object(viewer, "VideoStreamWriter", side_effect=AssertionError("do not encode mismatched source")):
            with self.assertRaisesRegex(ValueError, "source SHA differs"):
                viewer.visualize(self.native.path, self.output, self.root / "video", preview_frame=1)

    def test_output_video_hardlink_to_source_is_rejected_before_open(self):
        self.complete_source_and_sidecar()
        out = self.root / "video"
        out.mkdir()
        import os

        os.link(self.native.path, out / "near-range-derived.mp4")
        with patch.object(viewer, "HDF5Reader", side_effect=AssertionError("must fail preflight before source open")):
            with self.assertRaisesRegex(ValueError, "output aliases"):
                viewer.visualize(self.native.path, self.output, out)

    def test_stride_does_not_hide_corrupt_unsampled_payload(self):
        self.complete_source_and_sidecar()
        with h5py.File(self.output, "r+") as handle:
            handle["frames/00000001/head/depth_sl_derived"][6, 305] = 0.123
        with patch.object(
            viewer, "VideoStreamWriter", side_effect=AssertionError("verify every frame before encoding")
        ):
            with self.assertRaisesRegex(ValueError, "hash changed"):
                viewer.visualize(self.native.path, self.output, self.root / "video", frame_stride=2, preview_frame=0)

    def test_existing_artifacts_rejected_before_source_open(self):
        self.complete_source_and_sidecar()
        out = self.root / "video"
        out.mkdir()
        report = out / "near-range-derived.json"
        video = out / "near-range-derived.mp4"
        report.write_text('{"previous_run":true}')
        video.write_bytes(b"previous-video")
        with patch.object(viewer, "HDF5Reader", side_effect=AssertionError("existing artifacts must fail preflight")):
            with self.assertRaises(FileExistsError):
                viewer.visualize(self.native.path, self.output, out, preview_frame=0)
        self.assertEqual(report.read_text(), '{"previous_run":true}')
        self.assertEqual(video.read_bytes(), b"previous-video")


if __name__ == "__main__":
    unittest.main()
