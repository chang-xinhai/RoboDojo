"""CPU transaction and typed roundtrip checks; fixtures are not runtime evidence."""

from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from tests.test_multimodal import fixture
from utils.multimodal import validate_multimodal
from utils.multimodal_hdf5 import TACTILE_COMPONENT_SCHEMA, HDF5Reader, HDF5Writer, validate_hdf5


class HDF5Tests(unittest.TestCase):
    def test_lossless_all_fields_unicode_paths_null_and_signed_proxy(self):
        with tempfile.TemporaryDirectory() as directory:
            original = fixture()
            original["state"] = {
                "joints": np.array(["joint1", "关节2"], dtype="U6"),
                "empty": np.array([], dtype="U3"),
                "scalar": np.array("中文"),
                "numpy_scalar": np.str_("关节"),
                "/World/env_0/body": (1, None, np.int32(4)),
            }
            original["vision"]["head"]["ir_left_active_radiance"] = np.full((2, 4), np.nan, np.float32)
            original["tactile"]["left"]["finger_0"]["optical_depth_delta"] = np.full((2, 4, 1), -0.001, np.float32)
            with HDF5Writer(directory, {"fixture": True}) as writer:
                writer.append(original)
                writer.append(fixture(2))
                writer.seal()
            self.assertFalse((Path(directory) / "episode.hdf5.partial").exists())
            with HDF5Reader(Path(directory) / "episode.hdf5") as reader:
                restored = reader.read_frame(0)
                for name, value in original["state"].items():
                    if isinstance(value, np.ndarray):
                        np.testing.assert_equal(restored["state"][name], value)
                        self.assertEqual(restored["state"][name].dtype, value.dtype)
                    else:
                        self.assertEqual(restored["state"][name], value)
                np.testing.assert_equal(
                    restored["vision"]["head"]["ir_left_active_radiance"],
                    original["vision"]["head"]["ir_left_active_radiance"],
                )
                self.assertLess(restored["tactile"]["left"]["finger_0"]["optical_depth_delta"].max(), 0)
                self.assertEqual(len(reader), 2)
            self.assertEqual(validate_hdf5(Path(directory) / "episode.hdf5")["validated_frames"], 2)

    def test_bad_append_retains_only_committed_frame_and_explicit_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = HDF5Writer(directory)
            writer.append(fixture(2))
            with self.assertRaisesRegex(ValueError, "strictly increase"):
                writer.append(fixture(1))
            self.assertFalse(writer.path.exists())
            with HDF5Reader(writer.partial_path) as reader:
                self.assertEqual(len(reader), 1)
                self.assertEqual(reader.metadata["status"], "failed")
                self.assertEqual(reader.read_frame(0)["sync"]["physics_frame"], 2)
            with self.assertRaisesRegex(ValueError, "failed"):
                validate_hdf5(writer.partial_path)
            with self.assertRaisesRegex(RuntimeError, "failed"):
                writer.complete()

    def test_serialization_failure_rolls_back_staged_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = HDF5Writer(directory)
            writer.append(fixture())
            bad = fixture(2)
            bad["state"]["unsupported"] = np.array([object()], dtype=object)
            with self.assertRaisesRegex(ValueError, "Object arrays"):
                writer.append(bad)
            with HDF5Reader(writer.partial_path) as reader:
                self.assertEqual(len(reader), 1)
                self.assertEqual(len(reader.handle["pending"]), 0)
                self.assertEqual(len(reader.handle["frames"]), 1)

    def test_late_failure_withdraws_completed_file(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = HDF5Writer(directory)
            writer.append(fixture())
            writer.complete()
            writer.fail("preview failed")
            self.assertFalse(writer.path.exists())
            with HDF5Reader(writer.partial_path) as reader:
                self.assertFalse(reader.metadata["complete"])
                self.assertEqual(reader.metadata["failure_reason"], "preview failed")

    def test_corrupt_metadata_does_not_hide_capture_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = HDF5Writer(directory, {"safe": True})
            writer.append(fixture())
            writer.run_metadata["bad"] = object()
            writer.fail("physics failure")
            with HDF5Reader(writer.partial_path) as reader:
                self.assertEqual(reader.metadata["run"], {"safe": True})
                self.assertEqual(reader.metadata["status"], "failed")

    def test_failed_completion_metadata_keeps_readable_failed_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = HDF5Writer(directory)
            writer.append(fixture())
            with self.assertRaises(TypeError):
                writer.complete(validation={"invalid": object()})
            self.assertFalse(writer.path.exists())
            with HDF5Reader(writer.partial_path) as reader:
                self.assertEqual(reader.metadata["status"], "failed")
                self.assertIsNone(reader.metadata["validation"])
                self.assertEqual(len(reader), 1)

    def test_component_schema_is_explicit_without_fabricated_vision(self):
        with tempfile.TemporaryDirectory() as directory:
            sample = fixture()
            del sample["vision"]
            sample["sync"]["schema"] = TACTILE_COMPONENT_SCHEMA
            with HDF5Writer(directory, observation_schema=TACTILE_COMPONENT_SCHEMA) as writer:
                writer.append(sample)
            report = validate_hdf5(writer.path)
            self.assertEqual(report["observation_schema"], TACTILE_COMPONENT_SCHEMA)
            with HDF5Reader(writer.path) as reader:
                self.assertNotIn("vision", reader.read_frame(0))

    def test_uncommitted_frames_are_invisible_and_timeline_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with HDF5Writer(directory) as writer:
                writer.append(fixture())
            with h5py.File(writer.path, "r+") as handle:
                handle["pending"].create_group("00000001")
                handle["time/timestamp_ns"][0] = 0
            with HDF5Reader(writer.path) as reader:
                self.assertEqual(len(reader), 1)
                with self.assertRaisesRegex(ValueError, "timeline"):
                    reader.read_frame(0)

    def test_invalid_header_fields_are_rejected(self):
        for key, value in (
            ("observation_schema", "unknown.schema"),
            ("committed_frame_count", 1.9),
            ("complete", "false"),
        ):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                with HDF5Writer(directory) as writer:
                    writer.append(fixture())
                with h5py.File(writer.path, "r+") as handle:
                    del handle.attrs[key]
                    handle.attrs[key] = value
                with self.assertRaises(ValueError):
                    validate_hdf5(writer.path)

    def test_invalid_boundary_payloads_rejected_and_signed_normal_accepted(self):
        cases = {
            "shear_force": np.full((8, 3), np.nan, np.float32),
            "points_pos_w": np.full((8, 3), np.nan, np.float32),
            "depth": np.ones((2, 4, 2), np.float32),
            "penetration_depth": np.full(8, -0.1, np.float32),
        }
        for field, value in cases.items():
            with self.subTest(field=field):
                data = fixture()
                data["tactile"]["left"]["finger_0"][field] = value
                with self.assertRaises(ValueError):
                    validate_multimodal(data)
        data = fixture()
        data["tactile"]["left"]["finger_0"]["depth"][:] = np.nan
        with self.assertRaises(ValueError):
            validate_multimodal(data)
        data = fixture()
        data["sync"]["physics_frame"] = True
        with self.assertRaises(ValueError):
            validate_multimodal(data)
        data = fixture()
        data["tactile"]["left"]["finger_0"]["normal_force"][:] = -0.2
        validate_multimodal(data)
