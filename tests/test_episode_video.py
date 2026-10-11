"""Decode explicit CPU-fixture videos and verify complete HDF5 source timing."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from reproduction.export_episode_video import export_episode_video, rgb_preview, unified_preview
from reproduction.visualize_multimodal import metric_color
from tests.test_contact_conditioned_recording import contact_fixture
from utils.multimodal_hdf5 import HDF5Writer


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "CPU video encoding/decoding tools required")
class EpisodeVideoTests(unittest.TestCase):
    def test_unified_video_keeps_every_camera_modality_and_live_gel_synchronized(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            expected = []
            with HDF5Writer(
                directory / "source", {"synthetic_cpu_fixture": True, "nslv2": {"synthetic_cpu_fixture": True}}
            ) as writer:
                for index in range(3):
                    frame = contact_fixture(index == 1, index + 1)
                    row_colors = []
                    for row, camera in enumerate(frame["vision"].values()):
                        camera["provenance"]["depth_sl_method"] = "nsl_v2"
                        camera["rgb"][:] = (50 + index * 25, 80 + row * 30, 110)
                        camera["depth_gt"][:] = 0.2 + row * 0.6
                        camera["ir_left"][:] = 30 + row * 40 + index * 10
                        camera["ir_right"][:] = 80 + row * 40 + index * 10
                        camera["depth_sl"][:] = 0.3 + row * 0.6
                        camera["depth_sl"][:, :2] = np.nan
                        row_colors.append(
                            [
                                camera["rgb"][0, 0],
                                metric_color(np.full((2, 2), camera["depth_gt"][0, 0]), 0.03, 3.0)[0, 0],
                                np.repeat(camera["ir_left"][0, 0], 3),
                                np.repeat(camera["ir_right"][0, 0], 3),
                                metric_color(np.full((2, 2), camera["depth_sl"][0, -1]), 0.03, 3.0)[0, 0],
                            ]
                        )
                    gel_color = (75 + index * 35, 135, 180)
                    for fingers in frame["tactile"].values():
                        for data in fingers.values():
                            data["raw_optical_rgb"][:] = gel_color
                            if index == 1:
                                data["rgb"] = data["raw_optical_rgb"].copy()
                    writer.append(frame)
                    expected.append((row_colors, gel_color))
            original_sha = hashlib.sha256(writer.path.read_bytes()).hexdigest()
            video = directory / "unified.mp4"
            report = export_episode_video(writer.path, video, width=1000, layout="unified", require_nslv2=True)
            self.assertEqual(report["decoded_frames"], 3)
            self.assertEqual(report["video_fps"], 25)
            self.assertEqual(len(report["views"]), 19)
            self.assertEqual(report["video_encoder"]["crf"], 18)
            self.assertEqual(report["tactile_rgb_field"], "raw_optical_rgb")
            self.assertEqual(report["depth_display_range_m"], [0.03, 3.0])
            self.assertTrue(report["nslv2_required_and_verified_every_frame"])
            self.assertEqual(report["structured_light_source"], {"synthetic_cpu_fixture": True})
            decoded = subprocess.check_output(
                ["ffmpeg", "-v", "error", "-i", str(video), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
            )
            images = np.frombuffer(decoded, np.uint8).reshape(3, report["height"], report["width"], 3)
            for index, (colors, gel_color) in enumerate(expected):
                for row, modalities in enumerate(colors):
                    y = 100 + row * 210 + 36 + 75
                    for col, color in enumerate(modalities):
                        self.assertLess(
                            np.abs(images[index, y, col * 200 + (150 if col == 4 else 100)].astype(int) - color).max(),
                            # The tiny SL fixture has a hard NaN/valid color edge;
                            # aspect-preserving resizing and lossy H.264 can ring.
                            12 if col == 4 else 8,
                        )
                    # SL invalid pixels remain visibly invalid; GT is never copied into SL.
                    self.assertLess(np.abs(images[index, y, 850].astype(int) - (130, 0, 130)).max(), 8)
                for pad in range(4):
                    self.assertLess(
                        np.abs(images[index, 746 + 36 + 100, pad * 250 + 125].astype(int) - gel_color).max(), 8
                    )
            self.assertEqual(hashlib.sha256(writer.path.read_bytes()).hexdigest(), original_sha)

    def test_unified_preview_rejects_invalid_geometry_range_and_missing_live_gel(self):
        frame = contact_fixture(False)
        for kwargs in ({"width": 990}, {"depth_range": (1, 1)}, {"depth_range": (0, np.inf)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                unified_preview(frame, 0, 2, frame["sync"]["timestamp_ns"], **kwargs)
        del frame["tactile"]["right"]["finger_1"]["raw_optical_rgb"]
        with self.assertRaisesRegex(ValueError, "recorded raw_optical_rgb"):
            unified_preview(frame, 0, 2, frame["sync"]["timestamp_ns"])

    def test_required_nslv2_rejects_mixed_camera_methods_without_publishing_partial_video(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            with HDF5Writer(directory / "source", {"synthetic_cpu_fixture": True}) as writer:
                for index in range(3):
                    frame = contact_fixture(False, index + 1)
                    for camera in frame["vision"].values():
                        camera["provenance"]["depth_sl_method"] = "nsl_v2"
                    if index == 1:
                        frame["vision"]["right_wrist"]["provenance"]["depth_sl_method"] = "synthetic_other_backend"
                    writer.append(frame)
            video = directory / "unified.mp4"
            with self.assertRaisesRegex(ValueError, "NSLv2 depth required.*frame 1"):
                export_episode_video(writer.path, video, width=1000, layout="unified", require_nslv2=True)
            self.assertFalse(video.exists())
            self.assertFalse(video.with_suffix(".json").exists())
            self.assertFalse(video.with_name("unified.partial.mp4").exists())

    def test_required_nslv2_refuses_a_completed_commissioning_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            with HDF5Writer(
                directory / "subset",
                {
                    "synthetic_cpu_fixture": True,
                    "scope": "keyframe_subset_commissioning",
                    "source_capture": {"frames": 1228},
                },
            ) as writer:
                for index in range(2):
                    frame = contact_fixture(False, index + 1)
                    for camera in frame["vision"].values():
                        camera["provenance"]["depth_sl_method"] = "nsl_v2"
                    writer.append(frame)
            video = directory / "unified.mp4"
            with self.assertRaisesRegex(ValueError, "commissioning subset"):
                export_episode_video(writer.path, video, width=1000, layout="unified", require_nslv2=True)
            self.assertFalse(video.exists())

    def test_continuous_gel_survives_idle_contact_and_release_without_mutating_source(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            colors = []
            with HDF5Writer(directory / "source", {"synthetic_cpu_fixture": True}) as writer:
                for index in range(5):
                    active = index in (1, 2)
                    frame = contact_fixture(active, index + 1)
                    color = (40 + 25 * index, 120 + index, 170 - index)
                    colors.append(color)
                    for fingers in frame["tactile"].values():
                        for data in fingers.values():
                            data["raw_optical_rgb"][:] = color
                            if active:
                                data["rgb"] = data["raw_optical_rgb"].copy()
                    writer.append(frame)
            original_sha = hashlib.sha256(writer.path.read_bytes()).hexdigest()
            video = directory / "continuous.mp4"
            report = export_episode_video(writer.path, video, width=600, tactile_view="continuous")
            self.assertEqual(report["decoded_frames"], 5)
            self.assertEqual(report["video_fps"], 25)
            self.assertEqual(report["tactile_view"], "continuous")
            self.assertEqual(report["tactile_rgb_field"], "raw_optical_rgb")
            self.assertEqual(set(report["contact_active_frames_by_pad"].values()), {2})
            decoded = subprocess.run(
                ["ffmpeg", "-v", "error", "-i", str(video), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                check=True,
                capture_output=True,
            ).stdout
            images = np.frombuffer(decoded, np.uint8).reshape(5, report["height"], report["width"], 3)
            for index, color in enumerate(colors):
                for column in range(4):
                    # All four live gel panels include the current recorded
                    # idle baseline as well as the contact and release frames.
                    actual = images[index, 434, column * 150 + 70].astype(int)
                    self.assertLess(np.abs(actual - color).max(), 8)
            self.assertEqual(hashlib.sha256(writer.path.read_bytes()).hexdigest(), original_sha)

    def test_continuous_view_requires_recorded_raw_gel_and_rejects_unknown_modes(self):
        frame = contact_fixture(False)
        del frame["tactile"]["left"]["finger_0"]["raw_optical_rgb"]
        with self.assertRaisesRegex(ValueError, "recorded raw_optical_rgb"):
            rgb_preview(frame, 0, 2, frame["sync"]["timestamp_ns"], tactile_view="continuous")
        with self.assertRaisesRegex(ValueError, "contact or continuous"):
            rgb_preview(frame, 0, 2, frame["sync"]["timestamp_ns"], tactile_view="invented")

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
            self.assertEqual(report["video_encoder"]["crf"], 23)
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
