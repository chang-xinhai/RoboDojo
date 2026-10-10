"""Canonical layout provenance and live-scene presence checks without Kit."""

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from utils.task_layout import load_task_layout, verify_task_scene


# CPU stand-ins only; names exercise wrapper classification without Isaac.
class RigidObject(SimpleNamespace):
    pass


class DynamicObject(SimpleNamespace):
    pass


class GarmentObject(SimpleNamespace):
    pass


class TaskLayoutTests(unittest.TestCase):
    def test_load_records_exact_file_hash_and_expected_instances(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Assets/Eval_Layout/RoboDojo/arx_x5/0/stack_bowls_0.json"
            path.parent.mkdir(parents=True)
            raw = json.dumps({"Rigid": {"bowl": [{}, {}, {}]}, "Table": []}).encode()
            path.write_bytes(raw)
            layout, provenance = load_task_layout(directory, "stack_bowls")
            self.assertEqual(provenance["sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(provenance["expected_object_counts"]["Rigid"], 3)
            self.assertEqual(len(layout["Rigid"]["bowl"]), 3)

    def test_missing_layout_is_not_replaced_with_empty_scene(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_task_layout(directory, "stack_bowls")
            path = Path(directory) / "empty.json"
            path.write_text(json.dumps({"Table": [], "Light": []}))
            with self.assertRaisesRegex(ValueError, "no task objects"):
                load_task_layout(directory, "stack_bowls", layout_file=path)

    def test_empty_or_invalid_spawned_scene_is_rejected(self):
        layout = {"Rigid": {"bowl": [{}, {}, {}]}}
        scene = SimpleNamespace(_rigid_and_dynamic_objects=[{}])
        with self.assertRaisesRegex(RuntimeError, "requires 3"):
            verify_task_scene(scene, layout)
        objects = {f"bowl_{index}": RigidObject(prim_path=f"/World/bowl_{index}") for index in range(3)}
        scene._rigid_and_dynamic_objects = [objects]
        scene.stage = SimpleNamespace(GetPrimAtPath=lambda path: SimpleNamespace(IsValid=lambda: False))
        with self.assertRaisesRegex(RuntimeError, "prim is absent"):
            verify_task_scene(scene, layout)
        scene.stage = SimpleNamespace(GetPrimAtPath=lambda path: SimpleNamespace(IsValid=lambda: True))
        report = verify_task_scene(scene, layout)
        self.assertEqual(report["actual_object_counts"]["rigid_and_dynamic_objects"], 3)
        self.assertEqual(len(report["object_prim_paths"]["rigid_and_dynamic_objects"]), 3)

    def test_shared_storage_never_masks_wrong_dynamic_wrapper(self):
        layout = {"Rigid": {"base": [{}]}, "Dynamic": {"toy": [{}]}}
        scene = SimpleNamespace(
            _rigid_and_dynamic_objects=[
                {"base": RigidObject(prim_path="/World/base"), "toy": RigidObject(prim_path="/World/toy")}
            ],
            stage=SimpleNamespace(GetPrimAtPath=lambda path: SimpleNamespace(IsValid=lambda: True)),
        )
        with self.assertRaisesRegex(RuntimeError, "0 DynamicObject wrappers"):
            verify_task_scene(scene, layout)
        scene._rigid_and_dynamic_objects[0]["toy"] = DynamicObject(prim_path="/World/toy")
        report = verify_task_scene(scene, layout)
        self.assertEqual(report["actual_physical_type_counts"]["Dynamic"], 1)
        self.assertEqual(report["wrapper_types"]["/World/toy"]["physical_type"], "Dynamic")
        scene.layout_manager = SimpleNamespace(instance_type_by_env=[{"toy": "rigid"}])
        with self.assertRaisesRegex(RuntimeError, "layout declares rigid"):
            verify_task_scene(scene, layout)

    def test_garment_physics_migration_is_preserved_as_non_equivalent(self):
        migration = {
            "model": "surface_xpbd_fem",
            "benchmark_equivalent": False,
            "adhesion_removed": {"replacement": "dry_surface_contact"},
        }
        garment = GarmentObject(prim_path="/World/cloth", physics_migration=migration)
        scene = SimpleNamespace(
            _garment_objects=[{"cloth": garment}],
            stage=SimpleNamespace(
                GetPrimAtPath=lambda path: SimpleNamespace(
                    IsValid=lambda: True, GetAppliedSchemas=lambda: ["PhysxSurfaceDeformableBodyAPI"]
                )
            ),
        )
        report = verify_task_scene(scene, {"Garment": {"cloth": [{}]}})
        self.assertEqual(report["actual_physical_type_counts"]["Garment"], 1)
        self.assertFalse(report["physics_migration"]["/World/cloth"]["benchmark_equivalent"])
        migration["benchmark_equivalent"] = True
        self.assertFalse(report["physics_migration"]["/World/cloth"]["benchmark_equivalent"])


if __name__ == "__main__":
    unittest.main()
