"""Execute real layout/reset methods with real USD path occupancy, without Kit."""

import ast
from copy import deepcopy
from dataclasses import dataclass
import os
from pathlib import Path
import re
from types import MethodType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
from omegaconf import DictConfig, OmegaConf
from pxr import Usd, UsdGeom
import pytest

from utils.replay_inventory import retained_asset_key
from utils.task_layout import verify_task_scene

ROOT = Path(__file__).resolve().parents[1]
TYPES = ("Rigid", "Dynamic", "Geometry", "Articulation", "Garment", "Fluid")


def actual_methods(path, classname, names, namespace):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == classname)
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / path), "exec"), namespace)
    return {name: namespace[name] for name in names}


class DynamicObject(SimpleNamespace):
    pass


@pytest.mark.parametrize("device,physical", [("cuda", "Rigid"), ("cuda", "Dynamic"), ("cpu", "Articulation")])
@pytest.mark.parametrize(
    "change",
    [
        "poses",
        "unknown_model",
        "increased_count",
        "decreased_count",
        "unsupported_scale",
        "unsupported_mass",
        "preloaded_variant",
    ],
)
def test_reset_preserves_existing_names_and_refreshes_labels_poses_and_scale(device, physical, change):
    stage = Usd.Stage.CreateInMemory()
    namespace = {
        "deepcopy": deepcopy,
        "os": os,
        "DictConfig": DictConfig,
        "OmegaConf": OmegaConf,
        "OBJECTS_PATH": "/readonly/cpu-fixture",
        "re": re,
        "load_object_metadata": lambda *args: {},
        "load_desc_info": lambda *args: None,
        "is_prim_path_valid": lambda path: stage.GetPrimAtPath(path).IsValid(),
        "retained_asset_key": retained_asset_key,
    }
    tree = ast.parse((ROOT / "env/scene_manager/layout_manager.py").read_text())
    record = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ObjectTypeRecords")
    namespace.update(dataclass=dataclass, Any=Any)
    exec(compile(ast.Module(body=[record], type_ignores=[]), "actual_object_records", "exec"), namespace)
    methods = actual_methods(
        "env/scene_manager/layout_manager.py",
        "LayoutManager",
        {
            "load_saved_layout",
            "clear_layout_state",
            "clear_object_records",
            "clear_fixture_records",
            "_generate_object_paths",
            "_get_next_object_id",
            "decode_inst_name",
        },
        namespace,
    )
    original = [
        {
            "category_idx": 7,
            "label": f"original_{i}",
            "default_pos": [i * 0.1, 0, 0.8],
            "default_ori": [1, 0, 0, 0],
            "scale": [1, 1, 1],
        }
        for i in range(2)
    ]
    if change == "preloaded_variant":
        original[1].update(scale=[1.25, 1.25, 1.25], physics={"mass": 3.0})
    layout = SimpleNamespace(
        num_envs=1,
        saved_layouts=[{physical: {"same_model": original}}],
        object_records_by_type={kind: namespace["ObjectTypeRecords"].create(1) for kind in TYPES},
        instance_type_by_env=[{}],
        layout_valid=[True],
        cluttered_generator={},
        rooms=[[]],
        tables=[[]],
        grounds=[[]],
        lights=[[]],
        table_info=[{}],
        ground_info=[{}],
        env_roots=["/World/envs/env_0"],
        _object_id_counters=0,
        _set_env_seed=lambda _: None,
        cluttered_generator_init=lambda _: None,
    )
    for name, method in methods.items():
        if name != "decode_inst_name":
            setattr(layout, name, MethodType(method, layout))
    first = layout.load_saved_layout(0)[physical]["same_model"]
    objects = {}
    for instance in first:
        UsdGeom.Xform.Define(stage, instance["prim_path"])
        wrapper = DynamicObject if physical == "Dynamic" else SimpleNamespace
        objects[instance["inst_name"]] = wrapper(
            _prim_path=instance["prim_path"],
            instance_name=instance["inst_name"],
            default_pos=instance["default_pos"],
            default_ori=instance["default_ori"],
            scale=instance["scale"],
            instance_config=deepcopy(instance),
        )
    names = list(objects)
    updated = deepcopy(original)
    if change == "preloaded_variant":
        updated.reverse()
    for index, item in enumerate(updated):
        item.update(
            label=f"changed_label_{index}",
            default_pos=[0.4 + index * 0.2, -0.1, 0.9],
            default_ori=[0, 0, 0, 1],
        )
    if change == "unknown_model":
        updated[0]["category_idx"] = 8
    elif change == "increased_count":
        updated.append(deepcopy(updated[0]))
    elif change == "decreased_count":
        updated = updated[:1]
    elif change == "unsupported_scale":
        updated[0]["scale"] = [1.25, 1.25, 1.25]
    elif change == "unsupported_mass":
        updated[0]["physics"] = {"mass": 3.0}
    layout.saved_layouts = [{physical: {"same_model": updated}}]
    scene = SimpleNamespace(
        setup_scene=False,
        device=SimpleNamespace(type=device),
        layout_manager=layout,
        config={},
        _articulation_objects=[objects if physical == "Articulation" else {}],
        _rigid_and_dynamic_objects=[objects if physical != "Articulation" else {}],
        _inactive_scene_object_names=[set()],
        reload_lights=lambda _: None,
        reload_ground=lambda _: None,
        reload_room=lambda _: None,
        reload_table=lambda _: None,
        relocate_stale_objects=Mock(),
        clear_scene_objects=Mock(),
        spawn_scene_objects=Mock(),
    )
    method = actual_methods(
        "env/scene_manager/scene_manager.py",
        "SceneManager",
        {"reload_env_scene"},
        {
            "List": list,
            "DynamicObject": DynamicObject,
            "LayoutManager": SimpleNamespace(decode_inst_name=methods["decode_inst_name"]),
            "retained_asset_key": retained_asset_key,
        },
    )["reload_env_scene"]
    if change in ("unknown_model", "increased_count", "unsupported_scale", "unsupported_mass"):
        with pytest.raises(RuntimeError, match="fresh TaskEnv"):
            method(scene, 0)
        assert layout.instance_type_by_env[0] == dict.fromkeys(names, physical.lower())
        assert [item["label"] for item in layout.object_records_by_type[physical].layout_records_by_env[0]] == [
            "original_0",
            "original_1",
        ]
        return
    method(scene, 0)
    active_names = names[: len(updated)]
    if change == "preloaded_variant":
        active_names.reverse()
    refreshed = scene.config["env0"][physical]["same_model"]
    assert [item["inst_name"] for item in refreshed] == active_names
    assert [item["prim_path"] for item in refreshed] == [objects[name]._prim_path for name in active_names]
    assert layout.instance_type_by_env[0] == dict.fromkeys(active_names, physical.lower())
    records = layout.object_records_by_type[physical]
    assert set(records.metadata_by_env[0]) == set(active_names)
    assert [item["label"] for item in records.layout_records_by_env[0]] == [
        f"changed_label_{i}" for i in range(len(updated))
    ]
    for index, name in enumerate(active_names):
        obj = objects[name]
        assert obj.default_pos == [0.4 + index * 0.2, -0.1, 0.9]
        assert obj.default_ori == [0, 0, 0, 1]
        assert obj.scale == updated[index]["scale"]
    if change == "decreased_count":
        assert scene._inactive_scene_object_names == [{names[1]}]
        calls = []
        for name, obj in objects.items():
            obj.apply_saved_pose = lambda object_name=name: calls.append(object_name)
            obj.prim_path = obj._prim_path
            obj.get_world_pose = lambda body=obj: (np.asarray(body.default_pos), np.asarray([1, 0, 0, 0]))
            obj.sample_mesh_vertices = lambda body=obj: (np.asarray([body.default_pos]), None, None, None)
        scene._tables = [None]
        scene._garment_objects = [{}]
        scene._geometry_objects = [{}]
        scene._fluid_objects = [{}]
        scene.sim = SimpleNamespace(sim_step=Mock())
        apply_poses = actual_methods(
            "env/scene_manager/scene_manager.py",
            "SceneManager",
            {"apply_saved_poses"},
            {
                "List": list,
            },
        )["apply_saved_poses"]
        apply_poses(scene, [0])
        assert calls == [names[0]]
        # Give the CPU stand-ins their actual physical wrapper names for verifier classification.
        if physical != "Dynamic":
            wrapper = type(f"{physical}Object", (SimpleNamespace,), {})
            for name, obj in list(objects.items()):
                objects[name] = wrapper(**vars(obj))
        scene.stage = stage
        scene.env_origins = [np.zeros(3)]
        report = verify_task_scene(scene, layout.saved_layouts[0])
        assert report["actual_physical_type_counts"][physical] == 1
        assert report["inactive_object_names"] == [names[1]]
        assert list(report["live_placements"]) == [objects[names[0]].prim_path]
    assert original[0]["label"] == "original_0"
    # A fresh generated identity would differ because real USD paths remain occupied.
    assert layout._generate_object_paths(0, "same_model", 7, type=physical.lower())[1] not in names
