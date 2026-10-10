"""Canonical layout provenance and live-scene presence checks without Kit."""

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from pxr import Gf, Usd, UsdGeom

from utils.task_layout import live_object_placement, load_task_layout, verify_task_scene


# CPU stand-ins only; names exercise wrapper classification without Isaac.
class RigidObject(SimpleNamespace):
    def get_world_pose(self):
        return np.asarray(getattr(self, "position", [0, 0, 0])), np.asarray([1, 0, 0, 0])


class DynamicObject(SimpleNamespace):
    def sample_mesh_vertices(self):
        return np.asarray(getattr(self, "points", [[0, 0, 0]])), None, None, None


class GarmentObject(DynamicObject):
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

    def test_staged_offscreen_body_cannot_certify_a_live_task_scene(self):
        staged = RigidObject(prim_path="/World/envs/env_0/body", position=[100000, 100000, 100000])
        scene = SimpleNamespace(
            _rigid_and_dynamic_objects=[{"body": staged}],
            stage=SimpleNamespace(GetPrimAtPath=lambda path: SimpleNamespace(IsValid=lambda: True)),
        )
        with self.assertRaisesRegex(RuntimeError, "offscreen/unplaced"):
            verify_task_scene(scene, {"Rigid": {"body": [{}]}})

    def test_empty_nonfinite_and_missing_runtime_placement_are_rejected(self):
        for points in ([], [[float("nan"), 0, 0]]):
            with self.assertRaisesRegex(RuntimeError, "finite live placement"):
                live_object_placement(GarmentObject(prim_path="/World/cloth", points=points), "Garment", np.zeros(3))
        with self.assertRaises((AttributeError, RuntimeError)):
            live_object_placement(SimpleNamespace(prim_path="/World/body"), "Rigid", np.zeros(3))

    def test_live_world_and_environment_centers_keep_nonzero_origin_once(self):
        body = RigidObject(prim_path="/World/envs/env_1/body", position=[21, 32, 43])
        result = live_object_placement(body, "Rigid", np.asarray([20, 30, 40]))
        self.assertEqual(result["world_center"], [21, 32, 43])
        self.assertEqual(result["relative_center"], [1, 2, 3])
        self.assertAlmostEqual(result["max_distance_from_env_origin_m"], np.sqrt(14))

    def test_live_body_99m_away_is_rejected_even_inside_emergency_boundary(self):
        body = RigidObject(prim_path="/World/body", position=[99, 0, 0.8], default_pos=[0, 0, 0.8])
        with self.assertRaisesRegex(RuntimeError, "saved|layout"):
            live_object_placement(body, "Rigid", np.zeros(3))

    def test_real_usd_fluid_local_particles_receive_world_transform_once(self):
        stage = Usd.Stage.CreateInMemory()
        env = UsdGeom.Xform.Define(stage, "/World/envs/env_1")
        env.AddTranslateOp().Set(Gf.Vec3d(10, 20, 30))
        root = UsdGeom.Xform.Define(stage, "/World/envs/env_1/fluid")
        root.AddTranslateOp().Set(Gf.Vec3d(1, 2, 3))
        root.AddRotateZOp().Set(90)
        instancer = UsdGeom.PointInstancer.Define(stage, "/World/envs/env_1/fluid/particles")
        fluid = SimpleNamespace(
            prim_path=str(root.GetPath()),
            point_instancer=instancer,
            init_pos=[1, 2, 3],
            get_particle_positions=lambda: (np.asarray([[0.25, 0, 0], [0.75, 0, 0]]), None, None),
        )
        placement = live_object_placement(fluid, "Fluid", [10, 20, 30])
        np.testing.assert_allclose(placement["world_center"], [11, 22.5, 33])
        np.testing.assert_allclose(placement["relative_center"], [1, 2.5, 3])
        self.assertEqual(placement["state_source"], "native_pbd_particle_readback")
        self.assertAlmostEqual(placement["center_distance_from_saved_position_m"], 0.5)
        root.GetOrderedXformOps()[0].Set(Gf.Vec3d(100000, 0, 0))
        with self.assertRaisesRegex(RuntimeError, "offscreen/unplaced"):
            live_object_placement(fluid, "Fluid", [10, 20, 30])


if __name__ == "__main__":
    unittest.main()
