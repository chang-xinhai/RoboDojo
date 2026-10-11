"""NSLv2-only IR inference into a full multimodal HDF5, preserving the source."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from reproduction.isaac61.nslv2.client import NSLv2Client
from reproduction.isaac61.nslv2.runtime import canonical_json, file_sha

from utils.multimodal_hdf5 import HDF5Reader, HDF5Writer

CAMERAS = ("head", "left_wrist", "right_wrist")


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    return value


def replace_frame(frame, client, index):
    """Network input is exclusively the two IR arrays, never GT or old SL."""
    for name in CAMERAS:
        camera = frame["vision"][name]
        calibration = camera["calibration"]
        fx = float(np.asarray(calibration["K_by_view"]["ir_left"])[0, 0])
        baseline = float(calibration["baseline_m"])
        prediction = client.infer(
            camera["ir_left"],
            camera["ir_right"],
            fx_px=fx,
            baseline_m=baseline,
            stamp_ns=int(frame["sync"]["timestamp_ns"]),
            frame_index=index,
            camera=name,
        )
        prediction.provenance["source_calibration_sha256"] = hashlib.sha256(
            canonical_json(json_value(calibration)).encode()
        ).hexdigest()
        prediction.provenance["source_physics_frame"] = int(frame["sync"]["physics_frame"])
        prediction.provenance["source_stereo_confidence_ignored"] = True
        prediction.provenance["source_sync_sha256"] = hashlib.sha256(
            canonical_json(json_value(frame["sync"])).encode()
        ).hexdigest()
        original_provenance = camera["provenance"]
        camera["depth_sl"] = prediction.depth_left_m
        camera["provenance"] = {
            "depth_sl_method": "nsl_v2",
            "depth_sl": "nsl_v2_learned_metric_depth",
            "nslv2": prediction.provenance,
            "source_capture_provenance": original_provenance,
        }
        camera["nslv2_prediction"] = {
            "disparity_left_model": prediction.disparity_left_model,
            "covisibility_left_model": prediction.covisibility_left_model,
            "valid_mask": prediction.valid_mask,
        }
    return frame


def process(source, directory, config, *, indices=None, expected_source_sha=None):
    source, directory = Path(source).resolve(), Path(directory).resolve()
    if source.suffix != ".hdf5":
        raise ValueError("NSLv2 processing requires a sealed .hdf5 source, never partial")
    final, partial = directory / "episode.hdf5", directory / "episode.hdf5.partial"
    for candidate in (final, partial):
        if candidate == source or (candidate.exists() and candidate.samefile(source)):
            raise ValueError("NSLv2 output must not overwrite or alias the native source")
        if candidate.exists():
            raise FileExistsError(candidate)
    before = (source.stat().st_size, source.stat().st_mtime_ns)
    source_sha = file_sha(source)
    if expected_source_sha is not None and source_sha != expected_source_sha:
        raise ValueError("Sealed native source SHA differs from its acquisition receipt")
    started = time.monotonic()
    writer = None
    with HDF5Reader(source) as reader:
        metadata = reader.metadata
        if not metadata["complete"] or metadata["status"] != "complete" or not len(reader):
            raise ValueError("NSLv2 requires a complete nonempty native source")
        selected = list(range(len(reader))) if indices is None else list(indices)
        if not selected or selected != sorted(set(selected)) or selected[0] < 0 or selected[-1] >= len(reader):
            raise ValueError("NSLv2 source indices must be nonempty, unique, increasing and in range")
        with NSLv2Client.from_config(config) as client:
            model_identity = client.identity
            source_head = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
            ).stdout.strip()
            source_status = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=all", "--", "reproduction/isaac61/nslv2"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            module_hashes = {p.name: file_sha(p) for p in Path(__file__).parent.glob("*.py")}
            run = {
                "mode": "nslv2_ir_only_offline_reprocessing",
                "scope": "full_source_episode" if indices is None else "keyframe_subset_commissioning",
                "source_capture": {"hdf5": str(source), "sha256": source_sha, "frames": len(reader)},
                "source_capture_run": metadata["run"],
                "nslv2": model_identity,
                "postprocess_source_head": source_head,
                "postprocess_source_status": source_status,
                "postprocess_source_files_sha256": module_hashes,
                "selected_source_indices": selected,
                "native_physics_replayed": False,
                "gt_input": "none",
                "fallback": "none",
                "original_capture_modified": False,
            }
            try:
                writer = HDF5Writer(directory, run)
                for position, index in enumerate(selected):
                    frame_started = time.monotonic()
                    frame = reader.read_frame(index)
                    replace_frame(frame, client, index)
                    writer.append(frame)
                    if position % 25 == 0 or position == len(selected) - 1:
                        print(
                            canonical_json(
                                {
                                    "processed_frames": position + 1,
                                    "requested_frames": len(selected),
                                    "source_frame_index": index,
                                    "model_predictions": 3 * (position + 1),
                                    "elapsed_seconds": time.monotonic() - started,
                                    "last_frame_wall_seconds": time.monotonic() - frame_started,
                                    "valid_fractions": {
                                        name: float(np.isfinite(frame["vision"][name]["depth_sl"]).mean())
                                        for name in CAMERAS
                                    },
                                    "method": "nsl_v2",
                                    "fallback": "none",
                                    "complete": False,
                                }
                            ),
                            flush=True,
                        )
                if before != (source.stat().st_size, source.stat().st_mtime_ns):
                    raise ValueError("Sealed native source changed during inference")
                if file_sha(source) != source_sha:
                    raise ValueError("Sealed native source SHA changed during inference")
                validation = {
                    "source_native_task_validation": metadata["validation"],
                    "postprocessing": {
                        "method": "nsl_v2",
                        "frames": len(selected),
                        "model_predictions": 3 * len(selected),
                        "covers_all_source_frames": indices is None,
                        "gt_input": "none",
                        "fallback": "none",
                        "new_physics_task_success_claimed": False,
                        "original_source_sha256": source_sha,
                    },
                }
                writer.complete(validation)
            except BaseException as error:
                if writer is not None:
                    writer.fail(error)
                raise
    report = {
        "method": "nsl_v2",
        "source_hdf5": str(source),
        "source_sha256": source_sha,
        "output_hdf5": str(final),
        "output_sha256": file_sha(final),
        "output_frames": len(selected),
        "scope": run["scope"],
        "source_indices": selected,
        "model_predictions": 3 * len(selected),
        "nslv2": model_identity,
        "elapsed_seconds": time.monotonic() - started,
        "original_source_unmodified": True,
        "invalid_encoding": "NaN",
        "gt_input": "none",
        "fallback": "none",
        "source_head": source_head,
        "source_files_sha256": module_hashes,
    }
    (directory / "nslv2-processing.json").write_text(json.dumps(report, indent=2) + "\n")
    print(canonical_json(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Fresh directory; writes a complete episode.hdf5")
    parser.add_argument("--source-sha256")
    parser.add_argument("--frames", type=int, nargs="+", help="Optional commissioning subset; omit for the full source")
    parser.add_argument("--config-json", type=Path)
    args = parser.parse_args()
    config = {} if args.config_json is None else json.loads(args.config_json.read_text())
    process(args.source, args.output, config, indices=args.frames, expected_source_sha=args.source_sha256)


if __name__ == "__main__":
    main()
