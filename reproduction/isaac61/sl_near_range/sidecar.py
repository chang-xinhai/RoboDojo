"""Independent offline derived-SL sidecar. Never opens a .partial source.

The candidate reconstructor takes only IR, original SL and calibration. No GT
is stored in or passed to reconstruction. Source capture fields remain intact.
"""

import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import time

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
EXPECTED_DERIVE_SHA = "291fdeb748be893e3ceaf513c37770540dcc1eeb8109148ed61c5a9a484906fd"
EXPECTED_RECONSTRUCTION_SHA = "523087ace2a9ae291a089a5b6b0e9be6ed17b40c83f1d367fff91bcde37fea10"
DERIVE_SHA = hashlib.sha256((HERE / "derive.py").read_bytes()).hexdigest()
if DERIVE_SHA != EXPECTED_DERIVE_SHA:
    raise ValueError("First import frozen derivation source hash mismatch")
if (
    hashlib.sha256((ROOT / "env/camera_manager/structured_light.py").read_bytes()).hexdigest()
    != EXPECTED_RECONSTRUCTION_SHA
):
    raise ValueError("First import frozen reconstruction source hash mismatch")
spec = importlib.util.spec_from_file_location("frozen_near_range_derive", HERE / "derive.py")
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)
SCHEMA = "robodojo.structured_light_derived.hdf5.v1"
CAMERAS = ("head", "left_wrist", "right_wrist")


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while data := stream.read(16 * 1024 * 1024):
            digest.update(data)
    return digest.hexdigest()


def json_text(value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)

    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=convert)


def raw_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


class SidecarWriter:
    def __init__(self, path, source, mode):
        self.path = Path(path)
        self.partial = self.path.with_suffix(self.path.suffix + ".partial")
        if self.path.exists() or self.partial.exists():
            raise FileExistsError("Derived output or partial already exists")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = h5py.File(self.partial, "x")
        self.handle.attrs.update(
            schema=SCHEMA,
            status="processing",
            complete=False,
            task_complete_claimed=False,
            mode=mode,
            frame_count=0,
            derivation_module_sha256=DERIVE_SHA,
        )
        self.handle.create_dataset("source_json", data=json_text(source), dtype=h5py.string_dtype("utf-8"))
        self.handle.create_dataset(
            "parameters_json", data=json_text(candidate.parameters()), dtype=h5py.string_dtype("utf-8")
        )
        self.handle.create_group("frames")
        self.count = 0
        self.last_index = -1
        self.last_time = -1
        self.last_physics = -1

    def append(self, index, sync, vision, source_reference):
        if hashlib.sha256((HERE / "derive.py").read_bytes()).hexdigest() != DERIVE_SHA:
            raise ValueError("Candidate source changed during processing")
        stamp = int(sync["timestamp_ns"])
        physics = int(sync["physics_frame"])
        if not sync.get("physics_held_fixed"):
            raise ValueError("Source physics must be frozen during original capture")
        if index <= self.last_index or stamp <= self.last_time or physics <= self.last_physics:
            raise ValueError("Source frame index and physical clock must strictly increase")
        if set(vision) != set(CAMERAS):
            raise ValueError("All three original vision cameras are required")
        results = {}
        for name, data in vision.items():
            if int(data["timestamp_ns"]) != stamp:
                raise ValueError("Source camera timestamp does not match physical clock")
            c = data["calibration"]
            old = data["depth_sl"]
            before = raw_sha(old)
            start = time.perf_counter()
            z, added = candidate.derive(
                data["ir_left"], data["ir_right"], old, c["K_by_view"]["ir_left"][0][0], c["baseline_m"]
            )
            wall = time.perf_counter() - start
            finite = np.isfinite(old)
            if raw_sha(old) != before:
                raise ValueError("Candidate mutated original depth")
            if not np.array_equal(z[finite].view(np.uint32), old[finite].view(np.uint32)):
                raise ValueError("Candidate changed finite original depth")
            if added.dtype != np.bool_ or not np.array_equal(np.isfinite(z) & np.isnan(old), added):
                raise ValueError("Added mask must exactly identify new finite samples")
            if np.isinf(z).any() or np.any(added & finite):
                raise ValueError("Invalid derived depth or provenance mask")
            results[name] = (data, z, added, wall)
        group = self.handle["frames"].create_group(f"{index:08d}")
        group.attrs.update(
            source_frame_index=index,
            timestamp_ns=stamp,
            physics_frame=physics,
            source_reference_json=json_text(source_reference),
            sync_json=json_text(sync),
            source_sync_sha256=hashlib.sha256(json_text(sync).encode()).hexdigest(),
        )
        for name, (data, z, added, wall) in results.items():
            camera = group.create_group(name)
            camera.attrs.update(
                derive_wall_time_s=wall,
                added_pixels=int(added.sum()),
                derived_depth_sl_sha256=raw_sha(z),
                added_mask_sha256=raw_sha(added),
                original_depth_sl_sha256=raw_sha(data["depth_sl"]),
                original_ir_left_sha256=raw_sha(data["ir_left"]),
                original_ir_right_sha256=raw_sha(data["ir_right"]),
                calibration_json=json_text(data["calibration"]),
                calibration_sha256=hashlib.sha256(json_text(data["calibration"]).encode()).hexdigest(),
                original_capture_provenance_json=json_text(data.get("provenance", {})),
                original_capture_provenance_sha256=hashlib.sha256(
                    json_text(data.get("provenance", {})).encode()
                ).hexdigest(),
            )
            for key, array in (("depth_sl_derived", z), ("added_mask", added), ("depth_sl_original", data["depth_sl"])):
                camera.create_dataset(
                    key, data=array, compression="gzip", compression_opts=4, shuffle=True, fletcher32=True
                )
        group.attrs["committed"] = True
        self.count += 1
        self.last_index = index
        self.last_time = stamp
        self.last_physics = physics
        self.handle.attrs["frame_count"] = self.count
        self.handle.flush()

    def complete(self):
        if not self.count:
            raise ValueError("Cannot complete an empty derived inventory")
        self.handle.attrs.update(status="complete", complete=True)
        self.handle.flush()
        self.handle.close()
        self.handle = None
        os.replace(self.partial, self.path)

    def fail(self, error):
        if self.handle is not None:
            self.handle.attrs.update(status="failed", complete=False, failure_reason=str(error))
            self.handle.flush()
            self.handle.close()
            self.handle = None


def append_export(writer, stem):
    stem = Path(stem)
    npz = stem.with_suffix(".npz")
    meta = stem.with_suffix(".json")
    nb = npz.read_bytes()
    mb = meta.read_bytes()
    info = json.loads(mb)
    if info.get("status") != "immutable_native_frame_from_quiescent_copy":
        raise ValueError("Expected explicitly immutable native export from stable copy")
    if not info.get("copy_source_size_and_mtime_unchanged"):
        raise ValueError("Source copy stability was not verified")
    data = np.load(io.BytesIO(nb))
    vision = {}
    for name in CAMERAS:
        camera = info["calibration"][name]
        vision[name] = {
            "timestamp_ns": camera["timestamp_ns"],
            "calibration": camera["calibration"],
            "provenance": camera["provenance"],
            "depth_sl": data[name + "__depth_sl"],
            "ir_left": data[name + "__ir_left"],
            "ir_right": data[name + "__ir_right"],
        }
    reference = {
        "native_export_npz": str(npz.resolve()),
        "native_export_npz_sha256": hashlib.sha256(nb).hexdigest(),
        "native_export_json_sha256": hashlib.sha256(mb).hexdigest(),
        "snapshot_hdf5_sha256": info["snapshot_hdf5_sha256"],
        "original_recording": info["original_recording"],
        "copied_committed_frames": info["copied_committed_frames"],
    }
    writer.append(info["frame_index"], info["sync"], vision, reference)
    if (
        file_sha(npz) != reference["native_export_npz_sha256"]
        or file_sha(meta) != reference["native_export_json_sha256"]
    ):
        raise ValueError("Immutable native export changed while processing")


def append_complete_episode(writer, path):
    from utils.multimodal_hdf5 import HDF5Reader, _read

    path = Path(path)
    if path.suffix != ".hdf5" or path.resolve().suffix != ".hdf5":
        raise ValueError("Only a sealed completed .hdf5 source is accepted; never a .partial target")
    before = path.stat()
    with HDF5Reader(path) as reader:
        if not reader.metadata["complete"] or reader.metadata["status"] != "complete":
            raise ValueError("Source recording is not complete")
        source_sha = file_sha(path)
        for index in range(len(reader)):
            group = reader.handle[f"frames/{index:08d}"]
            if not group.attrs.get("committed"):
                raise ValueError("Uncommitted source frame")
            sync = _read(group["sync"])
            if any(sync[k] != v for k, v in reader.sync(index).items()):
                raise ValueError("Source clock does not match HDF5 timeline")
            vision = {}
            for name in CAMERAS:
                cg = group["vision"][name]
                vision[name] = {
                    key: _read(cg[key])
                    for key in ("timestamp_ns", "calibration", "provenance", "depth_sl", "ir_left", "ir_right")
                }
            writer.append(index, sync, vision, {"source_hdf5": str(path.resolve()), "source_hdf5_sha256": source_sha})
            if index % 25 == 0:
                print(json.dumps({"processed_source_frames": index + 1, "sidecar_frames": writer.count}), flush=True)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Completed source changed during derived processing")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--native-export", type=Path, action="append")
    sources.add_argument("--source-hdf5", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.source_hdf5 and (args.source_hdf5.suffix != ".hdf5" or args.source_hdf5.resolve().suffix != ".hdf5"):
        parser.error("Never open active .partial sources or aliases")
    mode = "commissioning_native_exports" if args.native_export else "complete_source_episode_offline_derivation"
    writer = SidecarWriter(
        args.output,
        {
            "native_exports": [str(p.resolve()) for p in args.native_export or []],
            "source_hdf5": str(args.source_hdf5.resolve()) if args.source_hdf5 else None,
        },
        mode,
    )
    try:
        if args.native_export:
            for stem in args.native_export:
                append_export(writer, stem)
        else:
            append_complete_episode(writer, args.source_hdf5)
        writer.complete()
    except BaseException as error:
        writer.fail(error)
        raise
    print(
        json.dumps(
            {
                "sidecar": str(writer.path.resolve()),
                "frames": writer.count,
                "mode": mode,
                "task_complete_claimed": False,
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
