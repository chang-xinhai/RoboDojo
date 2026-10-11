"""Actual typed-HDF5 synthetic transaction and negative tests; not native evidence."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tests.test_multimodal import fixture

import resume_sidecar as resume
from utils.multimodal_hdf5 import HDF5Writer


def source_frame(index):
    frame = fixture(index)
    frame["sync"]["physics_held_fixed"] = True
    for data in frame["vision"].values():
        left = np.random.default_rng(index).integers(0, 256, (64, 640), dtype=np.uint8)
        right = np.zeros_like(left)
        right[:, :-320] = left[:, 320:]
        old = np.full(left.shape, np.nan, np.float32)
        old[5:10, 300:320] = np.float32(0.7123456)
        data.update(
            rgb=np.zeros((64, 640, 3), np.uint8),
            depth_gt=np.ones(left.shape, np.float32),
            depth_sl=old,
            ir_left=left,
            ir_right=right,
            calibration={
                "K_by_view": {"ir_left": [[397.0, 0.0, 320.0], [0.0, 397.0, 32.0], [0.0, 0.0, 1.0]]},
                "baseline_m": 0.05,
            },
        )
    return frame


class PrefixTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.native = HDF5Writer(self.root / "native", {"unit_test_fixture": True})
        self.native.append(source_frame(1))
        self.output = self.root / "derived.hdf5"

    def tearDown(self):
        if self.native.handle is not None:
            self.native.fail("test fixture cleanup")
        self.temp.cleanup()

    def copy(self, label, mutate=None):
        target = self.root / f"{label}.hdf5.partial"
        shutil.copyfile(self.native.partial_path, target)
        if mutate:
            with h5py.File(target, "r+") as handle:
                mutate(handle)
        with h5py.File(target, "r") as handle:
            count = int(handle.attrs["committed_frame_count"])
        receipt = self.root / f"{label}.json"
        receipt.write_text(
            json.dumps(
                {
                    "status": "immutable_prefix_copy",
                    "copy_hdf5": str(target),
                    "original_active_recording": str(self.native.partial_path),
                    "copy_source_size_and_mtime_unchanged": True,
                    "copied_committed_frames": count,
                    "copy_hdf5_sha256": resume.base.file_sha(target),
                }
            )
        )
        return target, receipt

    def start(self):
        source, receipt = self.copy("prefix1")
        resume.process(self.output, source, receipt)
        return source, receipt

    def test_two_prefixes_resume_only_new_and_sealed_source_can_finalize(self):
        self.start()
        self.assertFalse(self.output.exists())
        partial = self.output.with_suffix(".hdf5.partial")
        with h5py.File(partial, "r") as handle:
            self.assertEqual(handle.attrs["status"], "incomplete")
            self.assertFalse(handle.attrs["complete"])
            first_hash = handle["frames/00000000/left_wrist"].attrs["derived_depth_sl_sha256"]
        self.native.append(source_frame(2))
        source, receipt = self.copy("prefix2")
        resume.process(self.output, source, receipt)
        self.assertFalse(self.output.exists())
        with h5py.File(partial, "r") as handle:
            self.assertEqual(handle.attrs["frame_count"], 2)
            self.assertEqual(first_hash, handle["frames/00000000/left_wrist"].attrs["derived_depth_sl_sha256"])
        self.native.complete()
        resume.process(self.output, self.native.path, final=True)
        self.assertFalse(partial.exists())
        with h5py.File(self.output, "r") as handle:
            self.assertTrue(handle.attrs["complete"])
            self.assertEqual(handle.attrs["status"], "complete")
            self.assertEqual(handle.attrs["final_original_hdf5_sha256"], resume.base.file_sha(self.native.path))

    def test_active_original_cannot_be_passed_as_immutable_copy(self):
        source, receipt = self.copy("wrongpath")
        r = json.loads(receipt.read_text())
        r["copy_hdf5"] = str(self.native.partial_path)
        receipt.write_text(json.dumps(r))
        with patch.object(resume, "HDF5Reader", side_effect=AssertionError("must not open active")):
            with self.assertRaisesRegex(ValueError, "original active"):
                resume.process(self.output, self.native.partial_path, receipt)
        self.assertFalse(self.output.exists())

    def test_hardlink_active_alias_is_rejected_before_hash_or_source_open(self):
        target = self.root / "hardlink-copy.hdf5.partial"
        os.link(self.native.partial_path, target)
        receipt = self.root / "hardlink-receipt.json"
        receipt.write_text(
            json.dumps(
                {
                    "status": "immutable_prefix_copy",
                    "copy_hdf5": str(target),
                    "original_active_recording": str(self.native.partial_path),
                    "copy_source_size_and_mtime_unchanged": True,
                    "copied_committed_frames": 1,
                    "copy_hdf5_sha256": "not-read",
                }
            )
        )
        with (
            patch.object(resume.base, "file_sha", side_effect=AssertionError("must not hash active alias")),
            patch.object(resume, "HDF5Reader", side_effect=AssertionError("must not open active alias")),
        ):
            with self.assertRaisesRegex(ValueError, "same-inode"):
                resume.process(self.output, target, receipt)

    def test_final_symlink_to_active_alias_rejected_before_hash_or_read(self):
        alias = self.root / "sealed-alias.hdf5"
        alias.symlink_to(self.native.partial_path)
        with (
            patch.object(resume.base, "file_sha", side_effect=AssertionError("must not hash active alias")),
            patch.object(resume, "HDF5Reader", side_effect=AssertionError("must not open active alias")),
        ):
            with self.assertRaisesRegex(ValueError, "active .partial"):
                resume.process(self.output, alias, final=True)

    def test_first_import_rejects_changed_candidate_before_loading_it(self):
        spec = importlib.util.spec_from_file_location("initial_pin_negative", Path(__file__).parent / "sidecar.py")
        module = importlib.util.module_from_spec(spec)
        with (
            patch("hashlib.sha256") as digest,
            patch(
                "importlib.util.spec_from_file_location",
                side_effect=AssertionError("candidate must not load before hardpin"),
            ),
        ):
            digest.return_value.hexdigest.return_value = "changed"
            with self.assertRaisesRegex(ValueError, "derivation source hash mismatch"):
                spec.loader.exec_module(module)

    def test_first_import_rejects_changed_reconstruction_before_candidate_load(self):
        spec = importlib.util.spec_from_file_location(
            "initial_reconstruction_pin_negative", Path(__file__).parent / "sidecar.py"
        )
        module = importlib.util.module_from_spec(spec)
        with (
            patch("hashlib.sha256") as digest,
            patch(
                "importlib.util.spec_from_file_location",
                side_effect=AssertionError("candidate must not load before hardpin"),
            ),
        ):
            digest.return_value.hexdigest.side_effect = [resume.base.EXPECTED_DERIVE_SHA, "changed"]
            with self.assertRaisesRegex(ValueError, "reconstruction source hash mismatch"):
                spec.loader.exec_module(module)

    def test_output_cannot_point_to_original_active_or_snapshot_copy(self):
        source, receipt = self.copy("collision-copy")
        for output in (self.native.path, source):
            with (
                self.subTest(output=output),
                patch.object(resume.h5py, "File", side_effect=AssertionError("must not open protected output")),
                patch.object(resume.base, "file_sha", side_effect=AssertionError("must fail before source hash")),
            ):
                with self.assertRaisesRegex(ValueError, "output aliases"):
                    resume.process(output, source, receipt)

    def test_output_hardlink_to_active_original_is_rejected_before_open(self):
        source, receipt = self.copy("outputalias-copy")
        output = self.root / "derived-alias.hdf5"
        os.link(self.native.partial_path, output.with_suffix(".hdf5.partial"))
        with (
            patch.object(resume.h5py, "File", side_effect=AssertionError("must not open hardlinked active output")),
            patch.object(resume.base, "file_sha", side_effect=AssertionError("must fail before source hash")),
        ):
            with self.assertRaisesRegex(ValueError, "output aliases"):
                resume.process(output, source, receipt)

    def test_wrong_copy_hash_is_rejected_before_source_read(self):
        source, receipt = self.copy("wronghash")
        r = json.loads(receipt.read_text())
        r["copy_hdf5_sha256"] = "bad"
        receipt.write_text(json.dumps(r))
        with patch.object(resume, "HDF5Reader", side_effect=AssertionError("must not open mismatched copy")):
            with self.assertRaisesRegex(ValueError, "hash"):
                resume.process(self.output, source, receipt)

    def test_changed_previous_ir_cannot_resume_under_new_copy_hash(self):
        self.start()
        self.native.append(source_frame(2))

        def mutate(handle):
            data = handle["frames/00000000/vision/head/ir_left"]
            data[0, 0] = int(data[0, 0]) ^ 1

        source, receipt = self.copy("changedir", mutate)
        with self.assertRaisesRegex(ValueError, "original IR/SL changed"):
            resume.process(self.output, source, receipt)
        self.assertFalse(self.output.exists())
        with h5py.File(self.output.with_suffix(".hdf5.partial"), "r") as handle:
            self.assertEqual(handle.attrs["status"], "failed")

    def test_changed_previous_calibration_cannot_resume(self):
        self.start()
        self.native.append(source_frame(2))

        def mutate(handle):
            handle["frames/00000000/vision/head/calibration/baseline_m/value"][()] = "0.04"

        source, receipt = self.copy("changedcalib", mutate)
        with self.assertRaisesRegex(ValueError, "calibration changed"):
            resume.process(self.output, source, receipt)

    def test_changed_previous_frozen_clock_cannot_resume(self):
        self.start()
        self.native.append(source_frame(2))

        def mutate(handle):
            stamp = 41000000
            handle["time/timestamp_ns"][0] = stamp
            handle["frames/00000000/sync/timestamp_ns/value"][()] = str(stamp)
            for name in ("head", "left_wrist", "right_wrist"):
                handle[f"frames/00000000/vision/{name}/timestamp_ns/value"][()] = str(stamp)

        source, receipt = self.copy("changedclock", mutate)
        with self.assertRaisesRegex(ValueError, "physical clock changed"):
            resume.process(self.output, source, receipt)
        self.assertFalse(self.output.exists())

    def test_corrupted_derived_payload_blocks_finalization(self):
        self.start()
        with h5py.File(self.output.with_suffix(".hdf5.partial"), "r+") as handle:
            handle["frames/00000000/head/depth_sl_derived"][20, 500] = 2.0
        self.native.complete()
        with self.assertRaisesRegex(ValueError, "derived data hash changed"):
            resume.process(self.output, self.native.path, final=True)
        self.assertFalse(self.output.exists())

    def test_candidate_hash_change_cannot_resume(self):
        source, receipt = self.start()
        with h5py.File(self.output.with_suffix(".hdf5.partial"), "r+") as handle:
            handle.attrs["derivation_module_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "algorithm hash changed"):
            resume.process(self.output, source, receipt)

    def test_finalization_never_opens_partial_source(self):
        with patch.object(resume, "HDF5Reader", side_effect=AssertionError("must not open active")):
            with self.assertRaisesRegex(ValueError, "active .partial"):
                resume.process(self.output, self.native.partial_path, final=True)


if __name__ == "__main__":
    unittest.main()
