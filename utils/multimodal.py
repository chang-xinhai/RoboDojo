"""Lossless, strict synchronized recording independent of the simulator runtime.

Camera depth_gt is rendered image-plane distance in metres, whereas depth_sl
is reconstructed in the left IR camera frame. They are never interchangeable.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import uuid

import numpy as np

CAMERAS = ("head", "left_wrist", "right_wrist")
SIDES = ("left", "right")
SCHEMA_VERSION = "robodojo.multimodal.v1"


def numpy_copy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value).copy()


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    return value


def _validate_pad_projection(data, image_only, label):
    """Check optional native pose-derived force coordinates without changing them."""
    fields = ("force_body_xyz", "pad_normal_force", "pad_shear_force_xz", "taxel_orientation_w_xyzw", "gel_pose_w_xyzw")
    if not any(field in data for field in fields):
        return
    if any(field not in data for field in fields):
        raise ValueError(f"{label}: incomplete pad force projection fields")
    if image_only:
        if any(data[field] is not None for field in fields):
            raise ValueError(f"{label}: unavailable pad force projection must be null")
        return
    count = np.asarray(data["normal_force"]).size
    shapes = ((count, 3), (count,), (count, 2), (count, 4), (7,))
    arrays = []
    for field, shape in zip(fields, shapes, strict=True):
        value = np.asarray(data[field])
        if value.shape != shape or value.dtype.kind != "f" or not np.isfinite(value).all():
            raise ValueError(f"{label}: invalid shape or finite values for {field}")
        arrays.append(value)
    body, normal, shear, taxel, pose = arrays
    for quaternion in (taxel, pose[3:]):
        if not np.allclose(np.linalg.norm(quaternion, axis=-1), 1, atol=1e-3, rtol=0):
            raise ValueError(f"{label}: force projection quaternion must have unit norm")
    original_shear = np.asarray(data["shear_force"])
    if original_shear.shape != (count, 2):
        raise ValueError(f"{label}: pad projection requires official XY shear")
    original = np.column_stack((original_shear, data["normal_force"]))

    def rotate(quaternion, vector):
        q = np.asarray(quaternion, dtype=np.float64)
        q = q / np.linalg.norm(q, axis=-1, keepdims=True)
        v = np.asarray(vector, dtype=np.float64)
        first = np.cross(q[..., :3], v)
        return v + 2 * (q[..., 3:] * first + np.cross(q[..., :3], first))

    inverse_gel = pose[3:].copy()
    inverse_gel[:3] *= -1
    expected = rotate(inverse_gel, rotate(taxel, original))
    if (
        not np.allclose(body, expected, atol=1e-5, rtol=1e-4)
        or not np.allclose(normal, -body[:, 1], atol=1e-5, rtol=1e-4)
        or not np.allclose(shear, body[:, [0, 2]], atol=1e-5, rtol=1e-4)
    ):
        raise ValueError(f"{label}: pad force projection disagrees with actual orientations and original forces")


def validate_multimodal(observation, *, tactile_component=False):
    """Reject missing sensors, stale clocks and ambiguous depth before writing."""
    sync = observation.get("sync", {})
    expected_schema = "robodojo.tactile_component.v1" if tactile_component else SCHEMA_VERSION
    if sync.get("schema") != expected_schema:
        raise ValueError(f"Expected schema {expected_schema}")
    frame = sync.get("physics_frame")
    stamp = sync.get("timestamp_ns")
    if (
        isinstance(frame, bool)
        or not isinstance(frame, int)
        or frame < 0
        or isinstance(stamp, bool)
        or not isinstance(stamp, int)
        or stamp < 0
    ):
        raise ValueError("A nonnegative physics_frame and simulation timestamp_ns are required")
    if sync.get("physics_held_fixed") is False:
        raise ValueError("Capture did not hold physics fixed")
    if "simulation_time_s" in sync and (
        not np.isfinite(sync["simulation_time_s"]) or abs(sync["simulation_time_s"] * 1e9 - stamp) > 1.0
    ):
        raise ValueError("Simulation time does not match timestamp")
    vision = observation.get("vision", {})
    for name in () if tactile_component else CAMERAS:
        if name not in vision:
            raise ValueError(f"Missing required camera: {name}")
        camera = vision[name]
        for field in ("rgb", "depth_gt", "depth_sl", "ir_left", "ir_right", "calibration", "provenance"):
            if field not in camera:
                raise ValueError(f"Missing {name}.{field}")
        rgb = np.asarray(camera["rgb"])
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError(f"{name}.rgb must be uint8 HWC RGB")
        for field in ("ir_left", "ir_right"):
            arr = np.asarray(camera[field])
            if arr.ndim != 2 or arr.size == 0 or arr.dtype != np.uint8:
                raise ValueError(f"{name}.{field} must be uint8 HW infrared intensity")
        if camera["ir_left"].shape != camera["ir_right"].shape:
            raise ValueError(f"{name}: stereo IR shapes differ")
        for field, expected in (("depth_gt", rgb.shape[:2]), ("depth_sl", camera["ir_left"].shape)):
            depth = np.asarray(camera[field])
            if depth.shape != expected or depth.dtype != np.float32:
                raise ValueError(f"{name}.{field} must be float32 HW in its camera frame")
            if np.any(np.isfinite(depth) & (depth < 0)):
                raise ValueError(f"{name}.{field} has negative metric depth")
        if not np.any(np.isfinite(camera["depth_gt"]) & (camera["depth_gt"] > 0)):
            raise ValueError(f"{name}: no valid rendered ground truth depth")
        if camera.get("timestamp_ns") != stamp:
            raise ValueError(f"{name}: camera timestamp does not match frozen physics time")
        if not camera["calibration"] or not camera["provenance"]:
            raise ValueError(f"{name}: calibration and provenance must be present")
        if camera["calibration"].get("depth_unit", "m") != "m":
            raise ValueError(f"{name}: camera depth calibration must declare metres")
    tactile = observation.get("tactile", {})
    for side in SIDES:
        if side not in tactile or not tactile[side]:
            raise ValueError(f"Missing required TacSL gripper: {side}")
        if not {"finger_0", "finger_1"}.issubset(tactile[side]):
            raise ValueError(f"Missing required TacSL gripper fingers: {side}")
        for finger, data in tactile[side].items():
            if not isinstance(data, dict):
                raise ValueError(f"Invalid tactile finger {side}.{finger}")
            for field in ("rgb", "depth", "normal_force", "shear_force", "penetration_depth", "points_pos_w"):
                if field not in data:
                    raise ValueError(f"Missing TacSL field {side}.{finger}.{field}")
            tactile_rgb = np.asarray(data["rgb"])
            tactile_depth = np.asarray(data["depth"])
            if tactile_rgb.ndim != 3 or tactile_rgb.shape[-1] not in (3, 4) or tactile_rgb.dtype != np.uint8:
                raise ValueError(f"{side}.{finger}: TacSL RGB must be uint8 HWC")
            if (
                tactile_depth.shape not in (tactile_rgb.shape[:2], (*tactile_rgb.shape[:2], 1))
                or tactile_depth.dtype != np.float32
            ):
                raise ValueError(f"{side}.{finger}: TacSL depth must match image dimensions in float32 metres")
            conditioned = data.get("provenance", {}).get("contact_conditioned_observations") is True
            if (
                not np.all(np.isfinite(tactile_depth))
                or np.any(tactile_depth < 0)
                or (not conditioned and not np.any(tactile_depth > 0))
            ):
                raise ValueError(f"{side}.{finger}: invalid TacSL optical depth")
            if "optical_depth_delta" in data:
                delta = np.asarray(data["optical_depth_delta"])
                if delta.shape != tactile_depth.shape or delta.dtype != np.float32 or not np.all(np.isfinite(delta)):
                    raise ValueError(f"{side}.{finger}: invalid signed optical depth delta")
            if conditioned:
                label = f"{side}.{finger}"
                required = (
                    "contact_active",
                    "optical_depth_delta",
                    "raw_optical_rgb",
                    "raw_optical_depth",
                    "raw_optical_depth_delta",
                )
                if any(field not in data for field in required):
                    raise ValueError(f"{label}: incomplete contact-conditioned optical fields")
                active = np.asarray(data.get("contact_active"))
                if active.shape != () or active.dtype != np.bool_:
                    raise ValueError(f"{label}: contact_active must be a scalar bool")
                if data.get("provenance", {}).get("force_field_available") is False:
                    raise ValueError(f"{label}: contact conditioning requires actual native force fields")
                penetration = np.asarray(data["penetration_depth"])
                if bool(active) != bool(np.any(penetration > 0)):
                    raise ValueError(f"{label}: contact_active disagrees with native taxel penetration")
                fields = (
                    ("rgb", tactile_rgb),
                    ("depth", tactile_depth),
                    ("optical_depth_delta", np.asarray(data.get("optical_depth_delta"))),
                )
                for field, value in fields:
                    raw = np.asarray(data.get("raw_optical_" + field.removeprefix("optical_")))
                    if raw.shape != value.shape or raw.dtype != value.dtype or not np.all(np.isfinite(raw)):
                        raise ValueError(f"{label}: invalid raw optical {field}")
                    if field == "depth" and (np.any(raw < 0) or not np.any(raw > 0)):
                        raise ValueError(f"{label}: raw optical depth must retain positive metric depth")
                    if bool(active):
                        if not np.array_equal(value, raw):
                            raise ValueError(f"{label}: active contact image differs from original optical data")
                    elif np.any(value != 0):
                        raise ValueError(f"{label}: inactive contact image must contain only zero sentinels")
            for key, units in (
                ("depth_units", "m"),
                ("penetration_units", "m"),
                ("force_units", "N"),
                ("optical_depth_delta_units", "m"),
            ):
                if data.get("metadata", {}).get(key, units) != units:
                    raise ValueError(f"{side}.{finger}: inconsistent metric units for {key}")
            force_fields = ("normal_force", "shear_force", "penetration_depth", "points_pos_w")
            image_only = data.get("provenance", {}).get("force_field_available") is False
            if image_only:
                if not data.get("provenance", {}).get("force_unavailable_reason"):
                    raise ValueError(f"{side}.{finger}: image-only TacSL must declare unavailable force reason")
                if any(data[field] is not None for field in force_fields):
                    raise ValueError(f"{side}.{finger}: unavailable force fields must be null, never fabricated")
            else:
                if any(data[field] is None or not np.asarray(data[field]).size for field in force_fields):
                    raise ValueError(f"{side}.{finger}: missing available TacSL force data")
                force = np.asarray(data["normal_force"])
                penetration = np.asarray(data["penetration_depth"])
                if not np.all(np.isfinite(force)) or not np.all(np.isfinite(penetration)):
                    raise ValueError(f"{side}.{finger}: invalid TacSL force or penetration")
                points = np.asarray(data["points_pos_w"])
                shear = np.asarray(data["shear_force"])
                if not np.all(np.isfinite(points)) or not np.all(np.isfinite(shear)) or np.any(penetration < 0):
                    raise ValueError(f"{side}.{finger}: invalid TacSL shear, world points or penetration")
                if (
                    points.ndim != 2
                    or force.ndim != 1
                    or penetration.ndim != 1
                    or shear.ndim != 2
                    or points.shape[-1] != 3
                    or shear.shape[-1] not in (2, 3)
                    or points.shape[0] != force.shape[0]
                    or shear.shape[0] != force.shape[0]
                    or penetration.shape[0] != force.shape[0]
                ):
                    raise ValueError(f"{side}.{finger}: tactile force samples do not match taxel positions")
            _validate_pad_projection(data, image_only, f"{side}.{finger}")
            if "physics_frame" in data and (isinstance(data["physics_frame"], bool) or data["physics_frame"] != frame):
                raise ValueError(f"{side}.{finger}: tactile physics frame mismatch")
            if data.get("timestamp_ns") != stamp:
                raise ValueError(f"{side}.{finger}: tactile timestamp mismatch")
            if data.get("provenance", {}).get("backend") != "isaaclab_tacsl":
                raise ValueError(f"{side}.{finger}: expected actual Isaac Lab TacSL backend")
    return observation


def _pack(value, prefix, arrays):
    if hasattr(value, "detach"):
        value = numpy_copy(value)
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ValueError(f"Object arrays cannot be recorded: {prefix}")
        arrays[prefix] = value
        return {"__array__": prefix}
    if isinstance(value, dict):
        return {str(k): _pack(v, f"{prefix}/{k}" if prefix else str(k), arrays) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_pack(v, f"{prefix}/{i}", arrays) for i, v in enumerate(value)]
    return json_value(value)


def _unpack(value, arrays):
    if isinstance(value, dict):
        if set(value) == {"__array__"}:
            return arrays[value["__array__"]].copy()
        return {k: _unpack(v, arrays) for k, v in value.items()}
    if isinstance(value, list):
        return [_unpack(v, arrays) for v in value]
    return value


def read_frame(path):
    with np.load(path, allow_pickle=False) as arrays:
        metadata = json.loads(str(arrays["__metadata__"]))
        return _unpack(metadata, arrays)


class MultimodalWriter:
    """One atomic NPZ per frame; manifest appears only after successful close.

    Existing output directories containing a manifest or frames are rejected.
    A failed run retains readable frames but no completed manifest.
    """

    def __init__(self, directory, run_metadata=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if (self.directory / "manifest.json").exists() or any(self.directory.glob("frame_*.npz")):
            raise FileExistsError(f"Recording directory already contains data: {self.directory}")
        self.frames = []
        self.run_metadata = json_value(run_metadata or {})
        self.closed = False

    def append(self, observation):
        if self.closed:
            raise RuntimeError("Cannot append after close")
        validate_multimodal(observation)
        sync = observation["sync"]
        if self.frames and (
            sync["physics_frame"] <= self.frames[-1]["physics_frame"]
            or sync["timestamp_ns"] <= self.frames[-1]["timestamp_ns"]
        ):
            raise ValueError("Recording physics frames and timestamps must strictly increase")
        arrays = {}
        metadata = _pack(observation, "", arrays)
        arrays["__metadata__"] = np.asarray(json.dumps(metadata, allow_nan=False))
        filename = f"frame_{len(self.frames):06d}.npz"
        destination = self.directory / filename
        temporary = self.directory / f".{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        self.frames.append(
            {"file": filename, "physics_frame": sync["physics_frame"], "timestamp_ns": sync["timestamp_ns"]}
        )
        return destination

    def close(self):
        if self.closed:
            return
        if not self.frames:
            raise ValueError("Cannot finalize an empty recording")
        manifest = {
            "schema": SCHEMA_VERSION,
            "complete": True,
            "run": self.run_metadata,
            "frames": self.frames,
            "depth_units": "metres",
            "depth_gt_frame": "rgb_optical",
            "depth_sl_frame": "left_ir_optical",
            "invalid_structured_light_depth": "NaN",
            "synchronization": "physics held fixed during RGB, projector-on/off IR and TacSL reads",
        }
        tmp = self.directory / ".manifest.tmp"
        tmp.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        os.replace(tmp, self.directory / "manifest.json")
        self.closed = True
        return manifest


def write_failed_capture(observation, directory, reason, frame_index=0):
    """Preserve raw sensor payloads for a failed quality check, without a pass manifest."""
    destination = Path(directory) / "failed-frame" / f"frame_{frame_index:06d}"
    destination.mkdir(parents=True, exist_ok=True)
    diagnostics = {"status": "failed", "reason": str(reason), "sync": observation.get("sync", {}), "cameras": {}}
    for name, camera in observation.get("vision", {}).items():
        if name not in CAMERAS:
            raise ValueError(f"Unexpected camera diagnostic filename: {name}")
        arrays = {
            field: numpy_copy(camera[field])
            for field in ("rgb", "depth_gt", "depth_sl", "ir_left", "ir_right")
            if field in camera
        }
        np.savez_compressed(destination / f"{name}.npz", **arrays)
        metrics = {}
        for field, array in arrays.items():
            finite = np.isfinite(array)
            metrics[field] = {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "finite_fraction": float(finite.mean()),
                "nonzero_fraction": float((array != 0).mean()),
                "finite_min": float(array[finite].min()) if finite.any() else None,
                "finite_max": float(array[finite].max()) if finite.any() else None,
            }
            if field.startswith("depth"):
                metrics[field]["valid_positive_fraction"] = float((finite & (array > 0)).mean())
        diagnostics["cameras"][name] = {
            "raw_file": f"{name}.npz",
            "metrics": metrics,
            "calibration": camera.get("calibration", {}),
            "provenance": camera.get("provenance", {}),
        }
    (destination / "diagnostic.json").write_text(json.dumps(json_value(diagnostics), indent=2, allow_nan=False) + "\n")
    return destination
