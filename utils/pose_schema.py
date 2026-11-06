"""Typed readers for historical RoboDojo WXYZ assets and versioned XYZW layouts.

Unmarked metadata and Eval_Layout JSON were produced by the scalar-first
RoboDojo pipeline. Only their documented pose fields are converted; numeric
values are never used to guess a quaternion convention. Shared assets stay intact.
"""
from copy import deepcopy

import numpy as np

from utils.rotations import quat_wxyz_to_xyzw

POSE_SCHEMA = {"version": 2, "quaternion_order": "xyzw", "pose_order": "xyz+xyzw"}
OBJECT_TYPES = ("Rigid", "Dynamic", "Geometry", "Articulation", "Garment", "Fluid")


def _legacy(data):
    schema = data.get("pose_schema")
    if schema is None:
        return True
    if schema != POSE_SCHEMA:
        raise ValueError(f"Unsupported RoboDojo pose schema: {schema!r}")
    return False


def _convert(value, width):
    """Convert a field known by its schema to contain quaternion(s) or pose(s)."""
    arr = np.asarray(value)
    if arr.size == 0:
        return deepcopy(value)
    if arr.ndim == 0 or arr.shape[-1] != width:
        raise ValueError(f"Expected pose field with final dimension {width}, got {arr.shape}")
    arr = arr.copy()
    arr[..., -4:] = quat_wxyz_to_xyzw(arr[..., -4:])
    return arr.tolist()


def encode_layout(layout):
    """Return a serializable copy of an INTERNAL XYZW layout with its format marker.

    Pass this result to the ordinary JSON writer when exporting a layout.
    """
    result = deepcopy(layout)
    result["pose_schema"] = dict(POSE_SCHEMA)
    return result


def decode_layout(layout):
    """Read legacy Eval_Layout or version-2 layout into marked internal XYZW data."""
    if layout is None:
        return None
    result = deepcopy(layout)
    if _legacy(result):
        for kind in OBJECT_TYPES:
            for records in result.get(kind, {}).values():
                for record in records:
                    # At this specific level qpos is a placement orientation,
                    # not an articulation's nested joint-position state.
                    for field in ("default_ori", "qpos"):
                        if record.get(field) is not None:
                            record[field] = _convert(record[field], 4)
        for kind in ("Room", "Table", "Ground"):
            fixture = result.get(kind, {})
            if fixture.get("default_ori") is not None:
                fixture["default_ori"] = _convert(fixture["default_ori"], 4)
    result["pose_schema"] = dict(POSE_SCHEMA)
    return result


def decode_object_metadata(metadata):
    """Read typed metadata pose fields; geometry, directions and joint values are unchanged."""
    result = deepcopy(metadata)
    if not _legacy(result):
        return result
    geometry = result.get("geometry", {})
    for name in ("aligned_bbox", "oriented_bbox"):
        bbox = geometry.get(name, {})
        if "frame" in bbox:
            bbox["frame"] = _convert(bbox["frame"], 7)

    def visit_records(records, field_widths):
        if not isinstance(records, dict):
            return
        for field, value in records.items():
            if field in field_widths and not isinstance(value, dict):
                records[field] = _convert(value, field_widths[field])
            elif isinstance(value, dict):
                visit_records(value, field_widths)

    for role in ("active", "passive"):
        data = result.get(role, {})
        # Place records can have a second level of numbered candidate records.
        visit_records(data.get("place", {}), {"orientation": 4, "center": 7})
        visit_records(data.get("support", {}), {"center": 7})
        # Garment functional qpos contains orientations paired with vertex IDs.
        visit_records(data.get("functional", {}), {"frame": 7, "qpos": 4})
    result["pose_schema"] = dict(POSE_SCHEMA)
    return result
