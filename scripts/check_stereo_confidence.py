"""Audit IR-only confidence by reprocessing preserved native RTX camera inputs.

This is a CPU replay, not a new renderer run. Analytic references and renderer
GT are used only for reporting errors after reconstruction; neither is an
input to confidence or depth. Original recording files are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def error_metrics(depth, reference):
    valid = np.isfinite(depth) & np.isfinite(reference) & (reference > 0)
    if not valid.any():
        return {"valid_fraction": float(np.isfinite(depth).mean()), "matched_pixels": 0}
    absolute = np.abs(depth[valid] - reference[valid])
    relative = absolute / reference[valid]
    return {
        "valid_fraction": float(np.isfinite(depth).mean()),
        "matched_pixels": int(valid.sum()),
        "median_relative_error": float(np.median(relative)),
        "p95_relative_error": float(np.percentile(relative, 95)),
        "median_abs_error_m": float(np.median(absolute)),
        "p95_abs_error_m": float(np.percentile(absolute, 95)),
    }


def registered_scene_error(depth, gt, calibration):
    """Evaluate a left-IR point against independent GT at its RGB projection."""
    left_k = np.asarray(calibration["K_by_view"]["ir_left"])
    rgb_k = np.asarray(calibration["K_by_view"]["rgb"])
    poses = calibration["T_world_from_optical"]
    rgb_from_left = np.linalg.inv(np.asarray(poses["rgb"])) @ np.asarray(poses["ir_left"])
    rows, cols = np.indices(depth.shape, dtype=np.float64)
    xyz = np.stack(
        ((cols + 0.5 - left_k[0, 2]) / left_k[0, 0] * depth, (rows + 0.5 - left_k[1, 2]) / left_k[1, 1] * depth, depth),
        axis=-1,
    )
    xyz = xyz @ rgb_from_left[:3, :3].T + rgb_from_left[:3, 3]
    with np.errstate(invalid="ignore", divide="ignore"):
        u = xyz[..., 0] / xyz[..., 2] * rgb_k[0, 0] + rgb_k[0, 2] - 0.5
        v = xyz[..., 1] / xyz[..., 2] * rgb_k[1, 1] + rgb_k[1, 2] - 0.5
    reference = cv2.remap(
        gt,
        np.nan_to_num(u, nan=-1).astype(np.float32),
        np.nan_to_num(v, nan=-1).astype(np.float32),
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float("nan"),
    )
    metric = error_metrics(xyz[..., 2], reference)
    metric["evaluation_frame"] = "predicted IR-left point projected into native RGB optical GT"
    return metric


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plane-dir", type=Path, action="append", default=[])
    parser.add_argument("--hdf5", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.plane_dir and not args.hdf5:
        parser.error("Provide preserved native plane directories or HDF5")
    args.output.mkdir(parents=True, exist_ok=True)
    source = ROOT / "env/camera_manager/structured_light.py"
    sl = load_source("confidence_replay", source)
    plane = load_source("plane_reference", ROOT / "scripts/check_structured_light_live.py")
    confidence = {"lr_max_diff_px": 0.5, "min_texture_std_dn": 2.0, "min_patch_correlation": 0.5}
    report = {
        "mode": "CPU reprocessing preserved native IR; not a new RTX run",
        "confidence": confidence,
        "reconstruction_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "planes": {},
        "scene": [],
        "failure_reasons": [],
    }
    for directory in args.plane_dir:
        original = json.loads((directory / "report.json").read_text())
        sensors = {}
        inventory = original.get("sensors")
        if not isinstance(inventory, dict) or not inventory:
            report["failure_reasons"].append(f"Plane control has no sensor inventory: {directory.resolve()}")
            report["planes"][str(directory.resolve())] = sensors
            continue
        for name, sensor in inventory.items():
            data = np.load(directory / f"{name}.npz")
            c = sensor["calibration"]
            depth = sl.reconstruct_stereo(
                data["ir_left"],
                data["ir_right"],
                c["K_by_view"]["ir_left"][0][0],
                c["baseline_m"],
                max_disparity=original["config"]["max_disparity"],
                **confidence,
            )
            reference = plane.plane_reference_depth(
                depth.shape, c["K_by_view"]["ir_left"], c["T_world_from_optical"]["ir_left"], original["plane_tilt_deg"]
            )
            sensors[name] = {
                "original": error_metrics(data["depth_sl"], reference),
                "confidence": error_metrics(depth, reference),
            }
            np.savez_compressed(
                args.output / f"{directory.name}-{name}.npz",
                depth_sl=depth,
                depth_sl_original=data["depth_sl"],
                ir_left=data["ir_left"],
                ir_right=data["ir_right"],
            )
        report["planes"][str(directory.resolve())] = sensors
    if args.hdf5:
        from utils.multimodal_hdf5 import HDF5Reader

        with HDF5Reader(args.hdf5) as reader:
            for index in range(len(reader)):
                frame = reader.read_frame(index)
                for name, data in frame["vision"].items():
                    c = data["calibration"]
                    depth = sl.reconstruct_stereo(
                        data["ir_left"],
                        data["ir_right"],
                        c["K_by_view"]["ir_left"][0][0],
                        c["baseline_m"],
                        **confidence,
                    )
                    report["scene"].append(
                        {
                            "frame": index,
                            "camera": name,
                            "original": registered_scene_error(data["depth_sl"], data["depth_gt"], c),
                            "confidence": registered_scene_error(depth, data["depth_gt"], c),
                        }
                    )
        if not report["scene"]:
            report["failure_reasons"].append("HDF5 input contains no camera observations")
    report["plane_acceptance"] = {"p95_relative_error": 0.05, "minimum_valid_fraction": 0.25}
    if report["failure_reasons"]:
        report["passed"] = False
        report["status"] = "failed"
        report["plane_acceptance"]["status"] = "failed" if args.plane_dir else "not_evaluated"
    elif not args.plane_dir:
        # Native scene GT is a diagnostic reference, not an analytic plane
        # control. A metrics-only replay never certifies depth acceptance.
        report["passed"] = None
        report["status"] = "metrics_only"
        report["plane_acceptance"]["status"] = "not_evaluated"
    else:
        report["passed"] = all(
            s["confidence"].get("p95_relative_error", 1) <= 0.05 and s["confidence"]["valid_fraction"] >= 0.25
            for sensors in report["planes"].values()
            for s in sensors.values()
        )
        report["status"] = "passed" if report["passed"] else "failed"
        report["plane_acceptance"]["status"] = report["status"]
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
