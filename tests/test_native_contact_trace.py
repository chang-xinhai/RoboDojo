"""Disk-backed diagnostics on analytical CPU sensors; no simulator evidence."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from utils.native_contact_trace import NativeContactTrace, native_task_inventory


class StoredSensor:
    def __init__(self, path):
        pose = np.array([[0, 0, 0, 0, 0, 0, 1]], np.float32)
        view = SimpleNamespace(count=1, get_transforms=lambda: pose.copy())
        self._contact_object_body_view = self._elastomer_body_view = view
        self.cfg = SimpleNamespace(contact_object_prim_path_expr=path)
        self._data = SimpleNamespace(
            tactile_normal_force=np.zeros((1, 8), np.float32),
            tactile_shear_force=np.zeros((1, 8, 2), np.float32),
            penetration_depth=np.zeros((1, 8), np.float32),
            tactile_points_pos_w=np.zeros((1, 8, 3), np.float32),
        )
        self.sdf = np.linspace(-0.001, 0.005, 8, dtype=np.float32)

    @property
    def data(self):
        raise AssertionError("Diagnostic access must not trigger lazy sensing")


def analytical_env():
    context = SimpleNamespace(frame=1)
    context.get_physics_step_count = lambda: context.frame
    context.get_physics_dt = lambda: 0.04
    paths = [f"/World/tube{i}/mesh" for i in range(3)]
    tactile = SimpleNamespace(
        contact_paths_by_env=[paths],
        sensors={
            side: {finger: {0: [StoredSensor(path) for path in paths]} for finger in ("finger_0", "finger_1")}
            for side in ("left", "right")
        },
    )
    scene = SimpleNamespace(
        layout_manager=SimpleNamespace(get_instance_name=lambda env_id, label: label),
        _rigid_and_dynamic_objects=[{f"tube{i}": SimpleNamespace(prim_path=f"/World/tube{i}") for i in range(3)}],
    )
    return SimpleNamespace(sim=SimpleNamespace(sim=context), tactile_manager=tactile, scene_manager=scene)


def inventory(env):
    return {"synthetic_cpu_fixture": True, "physics_frame": env.sim.sim.frame}


def sync(env):
    return {"physics_frame": env.sim.sim.frame, "timestamp_ns": env.sim.sim.frame * 40_000_000}


class NativeTraceTests(unittest.TestCase):
    def test_streamed_force_clones_clocks_unicode_and_native_pose_shapes_roundtrip(self):
        env = analytical_env()
        with tempfile.TemporaryDirectory() as directory:
            trace = NativeContactTrace(env, directory, 5, query=lambda sensor: sensor.sdf, inventory=inventory)
            trace.append(sync(env))
            sensor = env.tactile_manager.sensors["left"]["finger_0"][0][0]
            sensor._data.tactile_normal_force[0, 0] = -0.5
            sensor._data.tactile_shear_force[0, 0] = [0.1, -0.2]
            env.sim.sim.frame = 2
            trace.append(sync(env))
            self.assertTrue(all(isinstance(value, np.memmap) for value in trace._arrays.values()))
            path = trace.complete()
            self.assertEqual(trace.complete(), path)
            self.assertFalse((Path(directory) / ".native-contact-trace").exists())
            with np.load(path, allow_pickle=False) as data:
                self.assertEqual(data["sdf_m"].shape, (2, 4, 3, 8))
                self.assertEqual(data["normal_force_N"].dtype, np.float32)
                self.assertEqual(data["shear_force_N"].shape, (2, 4, 3, 8, 2))
                self.assertEqual(data["normal_force_N"][0, 0, 0, 0], 0)
                self.assertEqual(data["normal_force_N"][1, 0, 0, 0], -0.5)
                self.assertEqual(data["contact_body_pose_w_xyzw"].shape, (2, 4, 3, 7))
                self.assertEqual(data["taxel_points_pos_w_m"].shape, (2, 4, 3, 8, 3))
                np.testing.assert_equal(data["physics_frame"], [1, 2])
                np.testing.assert_equal(data["timestamp_ns"], [40_000_000, 80_000_000])
                np.testing.assert_equal(data["collider_labels"], ["tube0", "tube1", "tube2"])
                self.assertEqual(data["collider_labels"].dtype.kind, "U")
            metadata = json.loads((Path(directory) / "native-contact-trace.json").read_text())
            self.assertFalse(metadata["task_success_claimed"])
            self.assertEqual(env.sim.sim.frame, 2)

    def test_stale_clock_mutated_force_and_hidden_step_are_rejected(self):
        def mutate(sensor):
            sensor._data.tactile_normal_force[0, 0] += 1
            return sensor.sdf

        for case in ("clock", "force", "physics"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                env = analytical_env()

                def query(sensor):
                    if case == "force":
                        return mutate(sensor)
                    if case == "physics":
                        env.sim.sim.frame += 1
                    return sensor.sdf

                trace = NativeContactTrace(env, directory, 2, query=query, inventory=inventory)
                clock = sync(env)
                if case == "clock":
                    clock["timestamp_ns"] += 1
                with self.assertRaises(RuntimeError):
                    trace.append(clock)
                self.assertEqual(trace.count, 0)
                self.assertFalse(trace.path.exists())

    def test_colliders_must_match_live_native_tube_labels_and_sensor_order(self):
        for case in ("foreign", "order", "body_count"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                env = analytical_env()
                if case == "foreign":
                    env.tactile_manager.contact_paths_by_env[0][0] = "/World/probe/mesh"
                else:
                    sensor = env.tactile_manager.sensors["left"]["finger_0"][0][0]
                    if case == "order":
                        sensor.cfg.contact_object_prim_path_expr = "/World/tube2/mesh"
                    else:
                        sensor._contact_object_body_view.count = 0
                with self.assertRaises(RuntimeError):
                    NativeContactTrace(env, directory, 2, query=lambda sensor: sensor.sdf, inventory=inventory)

    def test_actual_usd_inventory_allows_canonical_stand_and_preserves_original_collision(self):
        env, stage = usd_inventory_env()
        report = native_task_inventory(env)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["ambient_geometry"][0]["role"], "original_canonical_camera_stand")
        self.assertFalse(report["objects"]["tube0"]["kinematic_enabled"])
        self.assertEqual(
            report["objects"]["tube0"]["collision_approximations"][0]["approximation"], "convexDecomposition"
        )
        self.assertTrue(stage.GetPrimAtPath("/World/camera_stand").IsValid())

    def test_actual_usd_inventory_rejects_kinematic_tube_probe_and_nonfixed_attachment(self):
        try:
            from pxr import UsdGeom, UsdPhysics
        except ImportError:
            self.skipTest("USD Python SDK is unavailable; no simulator is launched by these CPU checks")
        for corruption in ("kinematic", "probe", "nut", "revolute_attachment"):
            with self.subTest(corruption=corruption):
                env, stage = usd_inventory_env()
                if corruption == "kinematic":
                    UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath("/World/tube0")).CreateKinematicEnabledAttr(True)
                elif corruption == "revolute_attachment":
                    joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/illegal_attachment")
                    joint.CreateBody0Rel().SetTargets(["/World/tube0"])
                    joint.CreateBody1Rel().SetTargets(["/World/robot"])
                else:
                    UsdGeom.Xform.Define(stage, "/World/" + corruption)
                with self.assertRaises(RuntimeError):
                    native_task_inventory(env)


def usd_inventory_env():
    try:
        from pxr import Usd, UsdGeom, UsdPhysics
    except ImportError as error:
        raise unittest.SkipTest("USD SDK unavailable; this is a CPU inventory contract") from error
    env = analytical_env()
    stage = Usd.Stage.CreateInMemory()
    env.tactile_manager.stage = stage
    for index in range(3):
        root = UsdGeom.Xform.Define(stage, f"/World/tube{index}")
        UsdPhysics.RigidBodyAPI.Apply(root.GetPrim()).CreateKinematicEnabledAttr(False)
        mesh = UsdGeom.Mesh.Define(stage, f"/World/tube{index}/mesh")
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexDecomposition")
    for name in ("slot", "camera_stand"):
        UsdGeom.Xform.Define(stage, "/World/" + name)
    env.scene_manager._geometry_objects = [
        {
            "slot": SimpleNamespace(prim_path="/World/slot"),
            "camera_stand_0_5": SimpleNamespace(prim_path="/World/camera_stand"),
        }
    ]
    return env, stage


if __name__ == "__main__":
    unittest.main()
