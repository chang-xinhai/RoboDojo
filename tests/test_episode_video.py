"""Decode explicit CPU-fixture videos and verify complete HDF5 source timing."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from reproduction.export_episode_video import export_episode_video, rgb_preview
from tests.test_contact_conditioned_recording import contact_fixture
from utils.multimodal_hdf5 import HDF5Writer


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "CPU video encoding/decoding tools required")
class EpisodeVideoTests(unittest.TestCase):
    def test_all_five_source_frames_decode_at_actual_25_hz_with_contact_and_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            with HDF5Writer(directory / "source", {"synthetic_cpu_fixture": True}) as writer:
                for index in range(5):
                    frame = contact_fixture(index in (1, 2), index + 1)
                    frame["vision"]["head"]["rgb"][:] = (30 + 35 * index, 60, 80)
                    writer.append(frame)
            original_sha = hashlib.sha256(writer.path.read_bytes()).hexdigest()
            video = directory / "preview.mp4"
            report = export_episode_video(writer.path, video, width=600)
            self.assertEqual(report["committed_frames"], 5)
            self.assertEqual(report["encoded_frames"], 5)
            self.assertEqual(report["video_fps"], 25)
            self.assertAlmostEqual(report["timestamp_span_s"], 0.16)
            self.assertAlmostEqual(report["video_duration_s"], 0.2)
            self.assertEqual(report["source_hdf5_sha256"], original_sha)
            self.assertFalse(report["simulation_rerun"])
            self.assertFalse(report["task_success_claimed_by_exporter"])
            self.assertEqual(set(report["contact_active_frames_by_pad"].values()), {2})
            self.assertEqual(json.loads(video.with_suffix(".json").read_text()), report)
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-count_frames",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=nb_read_frames,r_frame_rate,width,height,duration",
                    "-of",
                    "json",
                    str(video),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            stream = json.loads(probe.stdout)["streams"][0]
            self.assertEqual(int(stream["nb_read_frames"]), 5)
            self.assertEqual(stream["r_frame_rate"], "25/1")
            self.assertAlmostEqual(float(stream["duration"]), 0.2)
            # Decode actual compressed output and verify chronological RGB changes.
            decoded = subprocess.run(
                ["ffmpeg", "-v", "error", "-i", str(video), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                check=True,
                capture_output=True,
            ).stdout
            images = np.frombuffer(decoded, np.uint8).reshape(5, report["height"], report["width"], 3)
            reds = images[:, 180, 200, 0].astype(int)
            self.assertTrue((np.diff(reds) > 20).all(), reds)
            # Main tactile panel stays black in idle frames, and renders the
            # unmodified 179-intensity contact image when contact is active.
            y = 64 + 300 + 30 + 40
            self.assertLess(images[0, y, 70].max(), 5)
            self.assertGreater(images[1, y, 70].min(), 160)
            self.assertLess(images[4, y, 70].max(), 5)
            self.assertEqual(hashlib.sha256(writer.path.read_bytes()).hexdigest(), original_sha)
            with self.assertRaises(FileExistsError):
                export_episode_video(writer.path, video, width=600)

    def test_failed_partial_and_nonuniform_timeline_never_publish_video(self):
        for case in ("partial", "nonuniform"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                writer = HDF5Writer(directory / "source", {"synthetic_cpu_fixture": True})
                for index in (1, 2, 4) if case == "nonuniform" else (1, 2, 3):
                    writer.append(contact_fixture(False, index))
                if case == "partial":
                    writer.fail("deliberate CPU fixture failure")
                    source = writer.partial_path
                else:
                    source = writer.complete()
                output = directory / "refused.mp4"
                with self.assertRaises(ValueError):
                    export_episode_video(source, output, width=600)
                self.assertFalse(output.exists())
                self.assertFalse(output.with_suffix(".json").exists())
                self.assertFalse(list(directory.glob("*.partial.mp4")))

    def test_rgb_preview_preserves_raw_intensities_and_portrait_aspect(self):
        frame = contact_fixture(True)
        for fingers in frame["tactile"].values():
            for data in fingers.values():
                data["rgb"] = np.full((320, 240, 3), 179, np.uint8)
        picture = np.asarray(rgb_preview(frame, 0, 5, frame["sync"]["timestamp_ns"], width=1200))
        pad = picture[64 + 600 + 30 : 64 + 600 + 30 + 300, :300]
        mask = (pad == 179).all(axis=-1)
        self.assertEqual(mask.sum(), 225 * 300)
        self.assertTrue(mask[:, 37:262].all())
        self.assertFalse(mask[:, :37].any())


if __name__ == "__main__":
    unittest.main()
