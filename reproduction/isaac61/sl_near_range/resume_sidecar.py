"""Resumable offline prefix derivation from explicit immutable COPY receipts.

Stable prefixes stay .partial/incomplete. Only a completed original source,
full per-frame hash verification and the same frozen algorithm can finalize.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sidecar as base
from utils.multimodal_hdf5 import HDF5Reader, _read


def capture_identity(path):
    return str(Path(path).resolve()).removesuffix(".partial")


def source_signature(path):
    stat = Path(path).stat()
    return stat.st_size, stat.st_mtime_ns


def selected_frame(reader, index):
    group = reader.handle[f"frames/{index:08d}"]
    if not group.attrs.get("committed"):
        raise ValueError("Uncommitted original source frame")
    sync = _read(group["sync"])
    if any(sync[k] != v for k, v in reader.sync(index).items()):
        raise ValueError("Source clock disagrees with original timeline")
    vision = {}
    for name in base.CAMERAS:
        camera = group["vision"][name]
        vision[name] = {
            key: _read(camera[key])
            for key in ("timestamp_ns", "calibration", "provenance", "depth_sl", "ir_left", "ir_right")
        }
        if int(vision[name]["timestamp_ns"]) != int(sync["timestamp_ns"]):
            raise ValueError("Source camera clock is inconsistent")
    return sync, vision


def reject_output_alias(output, protected):
    final = Path(output)
    partial = final.with_suffix(final.suffix + ".partial")
    for target in (final, partial):
        for original in protected:
            original = Path(original)
            if target.resolve() == original.resolve() or (
                target.exists() and original.exists() and target.samefile(original)
            ):
                raise ValueError("Derived output aliases protected original capture or source COPY")


def open_prefix_writer(output, identity):
    output = Path(output)
    partial = output.with_suffix(output.suffix + ".partial")
    if output.exists():
        raise FileExistsError("Complete derived sidecar already exists")
    if not partial.exists():
        writer = base.SidecarWriter(output, {"original_capture_identity": identity}, "incomplete_prefix_derivation")
        writer.handle.attrs["status"] = "incomplete"
        return writer
    handle = h5py.File(partial, "r+")
    try:
        if (
            handle.attrs.get("schema") != base.SCHEMA
            or handle.attrs.get("complete")
            or handle.attrs.get("status") != "incomplete"
        ):
            raise ValueError("Cannot resume an invalid, failed or completed sidecar")
        if handle.attrs.get("mode") != "incomplete_prefix_derivation":
            raise ValueError("Sidecar is not a resumable contiguous prefix")
        if handle.attrs.get("derivation_module_sha256") != base.DERIVE_SHA:
            raise ValueError("Frozen candidate algorithm hash changed")
        if handle["parameters_json"].asstr()[()] != base.json_text(base.candidate.parameters()):
            raise ValueError("Frozen candidate parameters changed")
        if json.loads(handle["source_json"].asstr()[()])["original_capture_identity"] != identity:
            raise ValueError("Original capture identity changed")
        count = int(handle.attrs["frame_count"])
        if sorted(handle["frames"]) != [f"{i:08d}" for i in range(count)] or any(
            not group.attrs.get("committed") for group in handle["frames"].values()
        ):
            raise ValueError("Sidecar has an incomplete or noncontiguous committed prefix")
        writer = object.__new__(base.SidecarWriter)
        writer.path = output
        writer.partial = partial
        writer.handle = handle
        writer.count = count
        writer.last_index = count - 1
        writer.last_time = int(handle["frames"][f"{count - 1:08d}"].attrs["timestamp_ns"]) if count else -1
        writer.last_physics = int(handle["frames"][f"{count - 1:08d}"].attrs["physics_frame"]) if count else -1
        return writer
    except BaseException:
        handle.close()
        raise


def verify_old_frame(writer, index, sync, vision):
    group = writer.handle["frames"][f"{index:08d}"]
    if group.attrs["source_sync_sha256"] != hashlib.sha256(base.json_text(sync).encode()).hexdigest():
        raise ValueError("Previously processed physical clock changed")
    if int(group.attrs["timestamp_ns"]) != int(sync["timestamp_ns"]) or int(group.attrs["physics_frame"]) != int(
        sync["physics_frame"]
    ):
        raise ValueError("Stored derived clock changed")
    for name, data in vision.items():
        camera = group[name]
        for key, attr in (
            ("depth_sl", "original_depth_sl_sha256"),
            ("ir_left", "original_ir_left_sha256"),
            ("ir_right", "original_ir_right_sha256"),
        ):
            if camera.attrs[attr] != base.raw_sha(data[key]):
                raise ValueError("Previously processed original IR/SL changed")
        if (
            camera.attrs["calibration_sha256"]
            != hashlib.sha256(base.json_text(data["calibration"]).encode()).hexdigest()
        ):
            raise ValueError("Previously processed calibration changed")
        if (
            camera.attrs["original_capture_provenance_sha256"]
            != hashlib.sha256(base.json_text(data["provenance"]).encode()).hexdigest()
        ):
            raise ValueError("Previously processed original capture provenance changed")
        old = camera["depth_sl_original"][()]
        z = camera["depth_sl_derived"][()]
        mask = camera["added_mask"][()]
        if camera.attrs["derived_depth_sl_sha256"] != base.raw_sha(z) or camera.attrs[
            "added_mask_sha256"
        ] != base.raw_sha(mask):
            raise ValueError("Stored derived data hash changed")
        if base.raw_sha(old) != base.raw_sha(data["depth_sl"]):
            raise ValueError("Stored original SL changed")
        finite = np.isfinite(old)
        if not np.array_equal(z[finite].view(np.uint32), old[finite].view(np.uint32)):
            raise ValueError("Stored finite original SL was modified")
        if mask.dtype != np.bool_ or not np.array_equal(mask, np.isfinite(z) & np.isnan(old)) or np.isinf(z).any():
            raise ValueError("Stored derived mask/depth contract violated")


def process(output, source, receipt=None, final=False):
    source = Path(source)
    if final:
        if source.suffix != ".hdf5" or source.resolve().suffix != ".hdf5":
            raise ValueError("Finalization never opens an active .partial source or alias")
        identity = capture_identity(source)
        reference = {"kind": "completed_original_capture", "source_hdf5": str(source.resolve())}
        reject_output_alias(output, (source, Path(identity + ".partial")))
    else:
        if receipt is None:
            raise ValueError("Explicit stable-copy receipt is required")
        receipt = json.loads(Path(receipt).read_text())
        if (
            receipt.get("status") != "immutable_prefix_copy"
            or receipt.get("copy_source_size_and_mtime_unchanged") is not True
        ):
            raise ValueError("Source copy stability was not verified")
        original = Path(receipt["original_active_recording"]).resolve()
        if (
            source.resolve() != Path(receipt["copy_hdf5"]).resolve()
            or source.resolve() == original
            or (original.exists() and source.samefile(original))
        ):
            raise ValueError("Refusing to open original active source or wrong copy, including same-inode hardlinks")
        identity = capture_identity(original)
        reference = {"kind": "verified_immutable_prefix_copy", "source_copy": str(source.resolve()), "receipt": receipt}
        reject_output_alias(output, (source, original, Path(identity)))
    if final:
        partial = Path(output).with_suffix(Path(output).suffix + ".partial")
        if partial.exists():
            with h5py.File(partial, "r") as derived:
                expected = json.loads(derived["source_json"].asstr()[()])["original_capture_identity"]
            if identity != expected:
                raise ValueError("Final source identity differs from original prefix capture")
            expected_active = Path(expected + ".partial")
            if expected_active.exists() and source.samefile(expected_active):
                raise ValueError("Final source hardlink aliases original active capture")
    before = source_signature(source)
    source_hash = base.file_sha(source)
    if not final and source_hash != receipt["copy_hdf5_sha256"]:
        raise ValueError("Immutable COPY hash does not match receipt")
    reference["source_hdf5_sha256"] = source_hash
    with HDF5Reader(source) as reader:
        if final:
            if not reader.metadata["complete"] or reader.metadata["status"] != "complete":
                raise ValueError("Original source recording is not completed")
        elif (
            reader.metadata["status"] not in ("recording", "captured")
            or reader.metadata["complete"]
            or len(reader) != receipt["copied_committed_frames"]
        ):
            raise ValueError("COPY is not the declared incomplete committed prefix")
        if not len(reader):
            raise ValueError("Cannot process an empty source prefix")
        writer = open_prefix_writer(output, identity)
        try:
            if writer.count > len(reader):
                raise ValueError("Source prefix regressed")
            for index in range(len(reader)):
                sync, vision = selected_frame(reader, index)
                if index < writer.count:
                    verify_old_frame(writer, index, sync, vision)
                else:
                    writer.append(index, sync, vision, reference)
                if index % 25 == 0:
                    print(
                        base.json_text(
                            {"checked_source_frames": index + 1, "sidecar_frames": writer.count, "complete": False}
                        ),
                        flush=True,
                    )
            if before != source_signature(source):
                raise ValueError("Source copy or completed source changed during processing")
            if final:
                writer.handle.attrs.update(
                    final_original_hdf5_sha256=source_hash,
                    final_original_hdf5=str(source.resolve()),
                    final_source_frames=len(reader),
                )
                writer.complete()
                print(
                    base.json_text(
                        {
                            "derived_complete": True,
                            "sidecar": str(writer.path.resolve()),
                            "frames": writer.count,
                            "source_sha256": source_hash,
                            "task_complete_claimed": False,
                        }
                    ),
                    flush=True,
                )
            else:
                writer.handle.attrs.update(
                    status="incomplete",
                    complete=False,
                    last_verified_copy_sha256=source_hash,
                    last_verified_prefix_frames=len(reader),
                )
                writer.handle.flush()
                writer.handle.close()
                writer.handle = None
                print(
                    base.json_text(
                        {
                            "derived_complete": False,
                            "sidecar_partial": str(writer.partial.resolve()),
                            "frames": writer.count,
                            "source_copy_sha256": source_hash,
                            "task_complete_claimed": False,
                        }
                    ),
                    flush=True,
                )
        except BaseException as error:
            writer.fail(error)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--stable-copy", type=Path)
    group.add_argument("--final-source", type=Path)
    parser.add_argument("--copy-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stable_copy and not args.copy_receipt:
        parser.error("--copy-receipt is required for prefix copies")
    process(args.output, args.final_source or args.stable_copy, args.copy_receipt, final=bool(args.final_source))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
