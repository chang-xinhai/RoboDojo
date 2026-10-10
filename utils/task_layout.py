"""Load canonical task layouts and reject smoke captures of empty task scenes."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from pathlib import Path

OBJECT_STORAGES = {
    "Rigid": "_rigid_and_dynamic_objects",
    "Dynamic": "_rigid_and_dynamic_objects",
    "Articulation": "_articulation_objects",
    "Garment": "_garment_objects",
    "Geometry": "_geometry_objects",
    "Fluid": "_fluid_objects",
}


WRAPPER_PHYSICAL_TYPES = {f"{physical_type}Object": physical_type for physical_type in OBJECT_STORAGES}


def expected_object_counts(layout):
    counts = {}
    for object_type in OBJECT_STORAGES:
        categories = layout.get(object_type, {})
        if not isinstance(categories, Mapping):
            raise ValueError(f"Layout {object_type} must map categories to instance lists")
        count = 0
        for category, instances in categories.items():
            if not isinstance(instances, (list, tuple)):
                raise ValueError(f"Layout {object_type}.{category} must be an instance list")
            count += len(instances)
        counts[object_type] = count
    if not sum(counts.values()):
        raise ValueError("Task layout has no task objects; fixtures alone cannot validate a task capture")
    return counts


def load_task_layout(root, task, layout_config="arx_x5", layout_id=0, layout_file=None):
    """Load the fixed-seed eval layout before the first environment reset.

    RGBD and multimodal observation profiles use the arx_x5 embodiment layout;
    an explicit file can be supplied for another embodiment or custom scene.
    """
    if layout_id < 0:
        raise ValueError("layout_id must be nonnegative")
    path = (
        Path(layout_file)
        if layout_file is not None
        else (Path(root) / "Assets" / "Eval_Layout" / "RoboDojo" / layout_config / "0" / f"{task}_{layout_id}.json")
    )
    path = path.resolve()
    raw = path.read_bytes()
    layout = json.loads(raw)
    if not isinstance(layout, dict):
        raise ValueError(f"Task layout must be a JSON object: {path}")
    expected = expected_object_counts(layout)
    provenance = {
        "file": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "layout_config": layout_config,
        "layout_id": int(layout_id),
        "task": task,
        "expected_object_counts": expected,
    }
    return layout, provenance


def verify_task_scene(scene_manager, layout, env_id=0):
    """Check live spawned object counts and USD prims against the saved layout."""
    expected = expected_object_counts(layout)
    expected_by_storage = {}
    for object_type, storage in OBJECT_STORAGES.items():
        expected_by_storage[storage] = expected_by_storage.get(storage, 0) + expected[object_type]
    actual = {}
    prim_paths = {}
    physical_type_counts = {physical_type: 0 for physical_type in OBJECT_STORAGES}
    wrapper_types = {}
    physics_migration = {}
    for storage, expected_count in expected_by_storage.items():
        batches = getattr(scene_manager, storage, None)
        objects = batches[env_id] if batches is not None else {}
        actual[storage.removeprefix("_")] = len(objects)
        if len(objects) < expected_count:
            raise RuntimeError(
                f"Task scene has {len(objects)} {storage} objects but saved layout requires {expected_count}; "
                "ensure the layout is installed before first reset"
            )
        paths = []
        for name, obj in objects.items():
            path = getattr(obj, "prim_path", None) or getattr(obj, "_prim_path", None)
            if not isinstance(path, str) or not path:
                raise RuntimeError(f"Task object {name} has no live USD prim path")
            stage = getattr(scene_manager, "stage", None)
            if stage is None or not stage.GetPrimAtPath(path).IsValid():
                raise RuntimeError(f"Task object USD prim is absent: {path}")
            wrapper = type(obj)
            physical_type = next(
                (
                    WRAPPER_PHYSICAL_TYPES[base.__name__]
                    for base in wrapper.__mro__
                    if base.__name__ in WRAPPER_PHYSICAL_TYPES
                ),
                None,
            )
            if physical_type is None:
                raise RuntimeError(f"Task object {path} has an unrecognized physics wrapper: {wrapper.__name__}")
            expected_storage = OBJECT_STORAGES[physical_type]
            if expected_storage != storage:
                raise RuntimeError(
                    f"Task object {path} wrapper {wrapper.__name__} is registered in wrong storage {storage}"
                )
            layout_manager = getattr(scene_manager, "layout_manager", None)
            instances = getattr(layout_manager, "instance_type_by_env", None)
            declared = instances[env_id].get(name) if instances is not None else None
            if declared is not None and declared != physical_type.lower():
                raise RuntimeError(
                    f"Task object {path} layout declares {declared} but live wrapper is {wrapper.__name__}"
                )
            physical_type_counts[physical_type] += 1
            prim = stage.GetPrimAtPath(path)
            wrapper_types[path] = {
                "class": f"{wrapper.__module__}.{wrapper.__qualname__}",
                "physical_type": physical_type,
                "instance_name": name,
                "layout_type": declared,
                "applied_schemas": list(prim.GetAppliedSchemas()) if hasattr(prim, "GetAppliedSchemas") else [],
            }
            migration = getattr(obj, "physics_migration", None)
            if migration:
                physics_migration[path] = deepcopy(dict(migration))
            paths.append(path)
        prim_paths[storage.removeprefix("_")] = paths
    for physical_type, expected_count in expected.items():
        if physical_type_counts[physical_type] < expected_count:
            raise RuntimeError(
                f"Task scene has {physical_type_counts[physical_type]} {physical_type}Object wrappers "
                f"but saved layout requires {expected_count}; shared storage count cannot prove physical type"
            )
    return {
        "env_id": int(env_id),
        "expected_object_counts": expected,
        "actual_object_counts": actual,
        "object_prim_paths": prim_paths,
        "actual_physical_type_counts": physical_type_counts,
        "wrapper_types": wrapper_types,
        "physics_migration": physics_migration,
    }
