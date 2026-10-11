"""Acceptance runner tests use ordinary subprocesses, never start Isaac/Kit."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from reproduction.isaac61 import check_tasks as runner

FIXTURE_DIRECTORY = tempfile.TemporaryDirectory()
FIXTURE_LAYOUT = Path(FIXTURE_DIRECTORY.name) / "stack_bowls_0.json"
FIXTURE_LAYOUT.write_text(
    json.dumps({"Rigid": {"bowl": [{"category_idx": 10, "default_pos": [i * 0.1, 0, 0.8]} for i in range(3)]}})
)
FIXTURE_LOG = Path(FIXTURE_DIRECTORY.name) / "clean-runtime.log"
FIXTURE_LOG.write_text("[INFO] Native task initialized\n")
SURFACE_OVERFLOW = (
    "2026-10-11T04:29:39Z [8,183ms] [Error] [omni.physx.plugin] PhysX error: "
    "Deformable surface contact buffer overflow detected, please increase "
    "PxGpuDynamicsMemoryConfig::maxDeformableSurfaceContacts to at least 1048576"
)


def evidence():
    expected = dict.fromkeys(runner.OBJECT_STORAGES, 0)
    expected["Rigid"] = 3
    actual = dict.fromkeys(runner.OBJECT_STORAGES.values(), 0)
    actual["rigid_and_dynamic_objects"] = 3
    paths = {storage: [] for storage in actual}
    paths["rigid_and_dynamic_objects"] = [f"/World/envs/env_0/rigid/bowl/bowl_10_{i}" for i in range(3)]
    return {
        "status": "passed",
        "task": "stack_bowls",
        "env_cfg": "arx_x5_rgbd",
        "num_envs": 1,
        "checks": sorted(runner.REQUIRED_CHECKS),
        "policy_evaluation": False,
        "real_runtime": True,
        "physics_steps": 10,
        "packages": {"isaacsim": "6.1.0.0", "isaaclab": "3.0.0"},
        "isaaclab_source_commit": "ea_pin",
        "task_runtime": {
            "instruction": {"env_id": 0, "templates": ["Stack the three bowls."]},
            "reward": {
                "registered": True,
                "executed": True,
                "step_executed": True,
                "harness_success_is_episode_validity": True,
                "parser_state_restored_between_extra_calls": True,
                "predicate_evaluation_mode": "isolated_extra_calls_before_normal_reward_step",
                "reward_values": [0.0],
                "score_values": [0.0],
                "predicate_count": 2,
                "predicate_evaluations": [
                    {"env_id": 0, "name": "object_upright", "value": 1.0},
                    {"env_id": 0, "name": "object_stacked", "value": 0.0},
                ],
            },
            "physics_step_before": 10,
            "physics_step_after": 10,
        },
        "layout": {
            "task": "stack_bowls",
            "file": str(FIXTURE_LAYOUT),
            "sha256": hashlib.sha256(FIXTURE_LAYOUT.read_bytes()).hexdigest(),
            "expected_object_counts": expected,
        },
        "scene": {
            "env_id": 0,
            "expected_object_counts": expected,
            "actual_object_counts": actual,
            "object_prim_paths": paths,
            "actual_physical_type_counts": expected.copy(),
            "env_origin": [0, 0, 0],
            "placement_verified": True,
            "live_placements": {
                path: {
                    "state_source": "native_body_pose",
                    "world_center": [i * 0.1, 0, 0.8],
                    "relative_center": [i * 0.1, 0, 0.8],
                    "saved_relative_position": [i * 0.1, 0, 0.8],
                    "max_distance_from_env_origin_m": (i * i * 0.01 + 0.64) ** 0.5,
                    "center_distance_from_saved_position_m": 0,
                }
                for i, path in enumerate(paths["rigid_and_dynamic_objects"])
            },
            "wrapper_types": {
                path: {
                    "class": "env.scene_manager.objects.rigid.RigidObject",
                    "physical_type": "Rigid",
                    "instance_name": f"bowl_10_{i}",
                    "layout_type": "rigid",
                }
                for i, path in enumerate(paths["rigid_and_dynamic_objects"])
            },
        },
        "camera_shapes": {"cam_" + name: {"rgb": [48, 64, 3], "depth": [48, 64]} for name in runner.CAMERAS},
    }


def context():
    return {
        "git_commit": "current_commit",
        "source_sha256": "current_source",
        "isaaclab_source_commit": "ea_pin",
        "packages": {"isaacsim": "6.1.0.0", "isaaclab": "3.0.0"},
    }


class AcceptanceTests(unittest.TestCase):
    def test_inventory_contains_all_54_tasks_and_random_variants(self):
        tasks = runner.inventory_tasks()
        self.assertEqual(len(tasks), 54)
        self.assertEqual(len(set(tasks)), 54)
        self.assertEqual(sum(task.endswith("_random") for task in tasks), 12)

    def test_full_rgbd_evidence_is_accepted(self):
        sample = evidence()
        self.assertIs(runner.check_evidence(sample, "stack_bowls", "arx_x5_rgbd"), sample)

    def test_legacy_color_alias_is_accepted_without_fabricating_rgb(self):
        sample = evidence()
        for camera in sample["camera_shapes"].values():
            camera["color"] = camera.pop("rgb")
        self.assertIs(runner.check_evidence(sample, "stack_bowls", "arx_x5_rgbd"), sample)

    def test_task_api_evidence_accepts_unsuccessful_policy_and_finite_negative_reward(self):
        sample = evidence()
        reward = sample["task_runtime"]["reward"]
        reward["reward_values"] = [-0.25]
        reward["predicate_evaluations"][0]["value"] = -0.5
        self.assertIs(runner.check_evidence(sample, "stack_bowls", "arx_x5_rgbd"), sample)
        self.assertFalse(sample["policy_evaluation"])

    def test_old_scene_smoke_cannot_resume_without_task_api_execution(self):
        sample = evidence()
        del sample["task_runtime"]
        with self.assertRaisesRegex(ValueError, "instruction generation"):
            runner.check_evidence(sample, "stack_bowls", "arx_x5_rgbd")
        record = {
            "status": "passed",
            "context": context(),
            "process_exit_code": 0,
            "real_runtime": True,
            "evidence": sample,
        }
        self.assertFalse(runner.can_resume(record, context(), "stack_bowls", "arx_x5_rgbd"))

    def test_empty_invalid_or_wrong_environment_instructions_are_rejected(self):
        for field, value in (
            ("env_id", 1),
            ("env_id", False),
            ("templates", []),
            ("templates", "Stack the bowls."),
            ("templates", [" "]),
            ("templates", ["Stack the bowls.", None]),
        ):
            sample = evidence()
            sample["task_runtime"]["instruction"][field] = value
            with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, "instruction"):
                runner.check_evidence(sample, "stack_bowls", "arx_x5_rgbd")

    def test_registered_rewards_without_real_step_and_finite_results_are_rejected(self):
        for field, value in (
            ("executed", False),
            ("registered", False),
            ("step_executed", False),
            ("step_executed", 1),
            ("reward_values", []),
            ("reward_values", [0, 1]),
            ("reward_values", [float("nan")]),
            ("score_values", [float("inf")]),
            ("score_values", [True]),
        ):
            sample = evidence()
            sample["task_runtime"]["reward"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                runner.check_evidence(sample, "stack_bowls", "arx_x5_rgbd")

    def test_reward_probe_preserves_frozen_physics_and_states_its_validity_scope(self):
        for field, value in (
            ("physics_step_before", None),
            ("physics_step_before", -1),
            ("physics_step_before", False),
            ("physics_step_before", 10.0),
            ("physics_step_after", float("nan")),
            ("physics_step_after", 11),
        ):
            sample = evidence()
            sample["task_runtime"][field] = value
            with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, "frozen physics"):
                runner.check_evidence(sample, "stack_bowls", "arx_x5_rgbd")
        for field, value in (
            ("harness_success_is_episode_validity", False),
            ("parser_state_restored_between_extra_calls", False),
            ("predicate_evaluation_mode", "policy_success"),
        ):
            sample = evidence()
            sample["task_runtime"]["reward"][field] = value
            with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, "episode validity"):
                runner.check_evidence(sample, "stack_bowls", "arx_x5_rgbd")

    def test_short_circuited_unidentified_and_nonfinite_predicate_results_are_rejected(self):
        for field, value in (
            ("predicate_count", 0),
            ("predicate_count", True),
            ("predicate_count", 3),
            ("predicate_evaluations", []),
            ("predicate_evaluations", [{"env_id": 0, "name": "check", "value": 0}]),
        ):
            sample = evidence()
            sample["task_runtime"]["reward"][field] = value
            with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, "predicate"):
                runner.check_evidence(sample, "stack_bowls", "arx_x5_rgbd")
        for field, value in (("env_id", 1), ("env_id", False), ("name", ""), ("value", True), ("value", float("nan"))):
            sample = evidence()
            sample["task_runtime"]["reward"]["predicate_evaluations"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, "predicate"):
                runner.check_evidence(sample, "stack_bowls", "arx_x5_rgbd")

    def test_missing_actual_runtime_and_unstepped_scene_are_rejected(self):
        for field, value in (("real_runtime", False), ("physics_steps", 0), ("isaaclab_source_commit", None)):
            sample = dict(evidence(), **{field: value})
            with self.assertRaises(ValueError):
                runner.check_evidence(sample, "stack_bowls", "arx_x5_rgbd")

    def test_missing_camera_and_misaligned_depth_are_rejected(self):
        missing = evidence()
        del missing["camera_shapes"]["cam_left_wrist"]
        with self.assertRaisesRegex(ValueError, "both wrist"):
            runner.check_evidence(missing, "stack_bowls", "arx_x5_rgbd")
        wrong = evidence()
        wrong["camera_shapes"]["cam_head"]["depth"] = [24, 32]
        with self.assertRaisesRegex(ValueError, "depth shape"):
            runner.check_evidence(wrong, "stack_bowls", "arx_x5_rgbd")

    def test_empty_scene_wrong_layout_and_wrong_environment_are_rejected(self):
        for field in ("layout", "scene"):
            missing = evidence()
            del missing[field]
            with self.assertRaises(ValueError):
                runner.check_evidence(missing, "stack_bowls", "arx_x5_rgbd")
        wrong = evidence()
        wrong["layout"]["sha256"] = "stale_layout"
        with self.assertRaisesRegex(ValueError, "SHA256"):
            runner.check_evidence(wrong, "stack_bowls", "arx_x5_rgbd")
        wrong = evidence()
        wrong["scene"]["actual_object_counts"]["rigid_and_dynamic_objects"] = 0
        with self.assertRaisesRegex(ValueError, "Missing live"):
            runner.check_evidence(wrong, "stack_bowls", "arx_x5_rgbd")
        wrong = evidence()
        wrong["scene"]["object_prim_paths"]["rigid_and_dynamic_objects"][0] = "/World/envs/env_1/Rigid/bowl"
        with self.assertRaisesRegex(ValueError, "outside"):
            runner.check_evidence(wrong, "stack_bowls", "arx_x5_rgbd")

    def test_source_hash_ignores_reports_but_includes_new_source_and_json_locks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "-c",
                    "user.name=test",
                    "-c",
                    "user.email=test@local",
                    "commit",
                    "-qm",
                    "test",
                    "--allow-empty",
                ],
                check=True,
            )
            (root / "reproduction").mkdir()
            (root / "module.py").write_text("value = 1\n")
            # Only HEAD resolution is stubbed; git ls-files and byte hashing are real.
            with patch.object(runner, "git", return_value="fixed_head"):
                initial = runner.source_identity(root)
                (root / "reproduction/asset-download.log").write_text("downloading\n")
                (root / "reproduction/task-smoke.json").write_text('{"status":"passed"}')
                self.assertEqual(runner.source_identity(root), initial)
                (root / "reproduction/asset-lock.json").write_text('{"source":"new_pin"}')
                with_lock = runner.source_identity(root)
                self.assertNotEqual(with_lock["source_sha256"], initial["source_sha256"])
                (root / "module.py").write_text("value = 2\n")
                self.assertNotEqual(runner.source_identity(root)["source_sha256"], with_lock["source_sha256"])

    def test_shared_storage_and_count_labels_cannot_mask_wrong_physical_wrapper(self):
        wrong = evidence()
        path = wrong["scene"]["object_prim_paths"]["rigid_and_dynamic_objects"][0]
        wrong["scene"]["wrapper_types"][path]["physical_type"] = "Dynamic"
        with self.assertRaisesRegex(ValueError, "Saved physical type"):
            runner.check_evidence(wrong, "stack_bowls", "arx_x5_rgbd")
        wrong["scene"]["wrapper_types"][path]["layout_type"] = "dynamic"
        with self.assertRaisesRegex(ValueError, "counts disagree"):
            runner.check_evidence(wrong, "stack_bowls", "arx_x5_rgbd")
        wrong["scene"]["actual_physical_type_counts"].update(Rigid=2, Dynamic=1)
        with self.assertRaisesRegex(ValueError, "Shared storage"):
            runner.check_evidence(wrong, "stack_bowls", "arx_x5_rgbd")

    def test_counted_objects_and_rgbd_background_cannot_replace_live_placement(self):
        for field in ("live_placements", "placement_verified", "env_origin"):
            missing = evidence()
            del missing["scene"][field]
            with self.assertRaises(ValueError):
                runner.check_evidence(missing, "stack_bowls", "arx_x5_rgbd")
        offscreen = evidence()
        placement = next(iter(offscreen["scene"]["live_placements"].values()))
        placement.update(
            world_center=[99, 0, 0.8],
            relative_center=[99, 0, 0.8],
            max_distance_from_env_origin_m=99.01,
            center_distance_from_saved_position_m=99,
        )
        with self.assertRaisesRegex(ValueError, "too far"):
            runner.check_evidence(offscreen, "stack_bowls", "arx_x5_rgbd")

    def test_placement_nonfinite_frames_sources_and_fabricated_distance_are_rejected(self):
        cases = (
            ("world_center", [float("nan"), 0, 0.8]),
            ("relative_center", [0, 0, 1.8]),
            ("state_source", "usd_authoring_transform"),
            ("center_distance_from_saved_position_m", 0.2),
            ("max_distance_from_env_origin_m", 0),
            ("saved_relative_position", [0, 0, 1.2]),
        )
        for field, value in cases:
            sample = evidence()
            next(iter(sample["scene"]["live_placements"].values()))[field] = value
            with self.assertRaises(ValueError):
                runner.check_evidence(sample, "stack_bowls", "arx_x5_rgbd")

    def test_live_positions_use_environment_origin_once(self):
        sample = evidence()
        origin = [10, 20, 30]
        sample["scene"]["env_origin"] = origin
        for placement in sample["scene"]["live_placements"].values():
            placement["world_center"] = [a + b for a, b in zip(placement["relative_center"], origin)]
        self.assertIs(runner.check_evidence(sample, "stack_bowls", "arx_x5_rgbd"), sample)

    def test_duplicate_saved_position_cannot_cover_distinct_layout_instances(self):
        sample = evidence()
        placements = list(sample["scene"]["live_placements"].values())
        placements[1].update(saved_relative_position=[0, 0, 0.8], center_distance_from_saved_position_m=0.1)
        with self.assertRaisesRegex(ValueError, "cover all objects"):
            runner.check_evidence(sample, "stack_bowls", "arx_x5_rgbd")

    def test_extra_camera_stand_fixture_has_placement_but_cannot_replace_bowls(self):
        sample = evidence()
        scene = sample["scene"]
        path = "/World/envs/env_0/geometry/camera_stand/camera_stand_0_11"
        scene["actual_object_counts"]["geometry_objects"] = 1
        scene["object_prim_paths"]["geometry_objects"] = [path]
        scene["actual_physical_type_counts"]["Geometry"] = 1
        scene["wrapper_types"][path] = {
            "class": "env.scene_manager.objects.geometry.GeometryObject",
            "physical_type": "Geometry",
            "instance_name": "camera_stand_0_11",
            "layout_type": "geometry",
        }
        scene["live_placements"][path] = {
            "state_source": "static_geometry_pose",
            "world_center": [0, 0, 0.8],
            "relative_center": [0, 0, 0.8],
            "saved_relative_position": [0, 0, 0.8],
            "center_distance_from_saved_position_m": 0,
            "max_distance_from_env_origin_m": 0.8,
        }
        self.assertIs(runner.check_evidence(sample, "stack_bowls", "arx_x5_rgbd"), sample)
        del scene["live_placements"][path]
        with self.assertRaisesRegex(ValueError, "Every actual task object"):
            runner.check_evidence(sample, "stack_bowls", "arx_x5_rgbd")

    def test_resume_rejects_wrong_revision_runtime_exit_and_evidence(self):
        ctx = context()
        record = {
            "status": "passed",
            "context": deepcopy(ctx),
            "process_exit_code": 0,
            "real_runtime": True,
            "evidence": evidence(),
            "log": str(FIXTURE_LOG),
            "native_log_validation": runner.audit_native_log(FIXTURE_LOG),
        }
        self.assertTrue(runner.can_resume(record, ctx, "stack_bowls", "arx_x5_rgbd"))
        stale = deepcopy(ctx)
        stale["git_commit"] = "new_commit"
        self.assertFalse(runner.can_resume(record, stale, "stack_bowls", "arx_x5_rgbd"))
        stale = deepcopy(ctx)
        stale["source_sha256"] = "new_uncommitted_contents"
        self.assertFalse(runner.can_resume(record, stale, "stack_bowls", "arx_x5_rgbd"))
        for field, value in (("process_exit_code", 1), ("real_runtime", False), ("evidence", {})):
            changed = dict(record, **{field: value})
            self.assertFalse(runner.can_resume(changed, ctx, "stack_bowls", "arx_x5_rgbd"))
        older = deepcopy(ctx)
        older["packages"]["isaacsim"] = "5.1.0.0"
        self.assertFalse(runner.can_resume(dict(record, context=older), older, "stack_bowls", "arx_x5_rgbd"))
        changed_lab = deepcopy(record)
        changed_lab["evidence"]["isaaclab_source_commit"] = "different_lab_pin"
        self.assertFalse(runner.can_resume(changed_lab, ctx, "stack_bowls", "arx_x5_rgbd"))

    def test_exit_zero_and_passed_report_cannot_hide_native_physics_data_loss(self):
        native_errors = (
            SURFACE_OVERFLOW,
            SURFACE_OVERFLOW.replace("[Error]", "[Warning]"),
            "[Warning] [omni.physx.plugin] PxValidateTriangleMesh for PxDeformableSurface failed, /World/Cloth",
            "[Error] [omni.physx.plugin] PhysX error: CUDA kernel launch failed",
            "[Error] [omni.physics.tensors.plugin] Assertion failed: buffer length matches body count",
            "[Error] [omni.physx.tensors.plugin] Pattern '/World/Cloth' did not match any surface deformable bodies",
        )
        actual_execute = runner.execute_process
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(output_dir=root, env_cfg="arx_x5_rgbd", environment=root, timeout=3)
            for native_line in native_errors:
                with self.subTest(native_line=native_line):
                    # This stand-in exits zero and creates a fresh valid report.
                    # It proves subprocess acceptance behavior, not Isaac physics.
                    def execute_standin(command, log, timeout, **kwargs):
                        code = (
                            "import pathlib,sys; print(sys.argv[1],flush=True); "
                            "pathlib.Path(sys.argv[2]).write_text(sys.argv[3])"
                        )
                        return actual_execute(
                            [
                                sys.executable,
                                "-c",
                                code,
                                native_line,
                                str(root / "stack_bowls.json"),
                                json.dumps(evidence()),
                            ],
                            log,
                            timeout,
                        )

                    with patch.object(runner, "execute_process", side_effect=execute_standin):
                        record = runner.run_task("stack_bowls", args, context())
                    self.assertEqual(record["process_exit_code"], 0)
                    self.assertEqual(record["evidence"]["status"], "passed")
                    self.assertEqual(record["status"], "failed")
                    self.assertIn("Native physics failure", record["error"])
                    self.assertEqual(record["native_log_validation"]["known_fatal_count"], 1)
                    self.assertFalse(runner.can_resume(record, context(), "stack_bowls", "arx_x5_rgbd"))

    def test_supported_collision_fallback_and_unrelated_renderer_errors_are_not_native_fatal(self):
        lines = (
            "[Warning] [omni.physx.plugin] GPU convex cooking failed; using CPU collision fallback",
            "[Error] [omni.hydra.rtx] Cannot create smooth normal adjacency buffer for /World/Cloth",
            "RuntimeWarning: overflow encountered in reduce",
            "[INFO] [omni.physx.plugin] PxGpuDynamicsMemoryConfig::maxDeformableSurfaceContacts = 1048576",
        )
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "ordinary-warnings.log"
            log.write_text("\n".join(lines))
            audit = runner.audit_native_log(log)
            self.assertEqual(audit["known_fatal_count"], 0)
            self.assertEqual(audit["sha256"], hashlib.sha256(log.read_bytes()).hexdigest())

    def test_resume_requires_unchanged_real_log_with_current_native_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "runtime.log"
            log.write_text("[INFO] Scene initialized\n")
            record = {
                "status": "passed",
                "context": context(),
                "process_exit_code": 0,
                "real_runtime": True,
                "evidence": evidence(),
                "log": str(log),
                "native_log_validation": runner.audit_native_log(log),
            }
            self.assertTrue(runner.can_resume(record, context(), "stack_bowls", "arx_x5_rgbd"))
            log.write_text(log.read_text() + SURFACE_OVERFLOW + "\n")
            self.assertFalse(runner.can_resume(record, context(), "stack_bowls", "arx_x5_rgbd"))
            # A recomputed hash still cannot certify a known fatal native error.
            record["native_log_validation"] = runner.audit_native_log(log)
            self.assertFalse(runner.can_resume(record, context(), "stack_bowls", "arx_x5_rgbd"))
            log.unlink()
            self.assertFalse(runner.can_resume(record, context(), "stack_bowls", "arx_x5_rgbd"))

    def test_zero_exit_without_fresh_output_cannot_reuse_stale_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "stack_bowls.json").write_text('{"status":"passed"}')
            args = SimpleNamespace(output_dir=root, env_cfg="arx_x5_rgbd", environment=root, timeout=180)
            with patch.object(runner, "execute_process", return_value=(0, False, 0.1)):
                record = runner.run_task("stack_bowls", args, context())
            self.assertEqual(record["status"], "failed")
            self.assertIn("fresh runtime evidence", record["error"])
            self.assertFalse((root / "stack_bowls.json").exists())

    def test_timeout_terminates_real_ordinary_subprocess(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "ordinary-process.log"
            code, timed_out, duration = runner.execute_process(
                [sys.executable, "-c", "import time; time.sleep(10)"], log, timeout=0.05
            )
            self.assertTrue(timed_out)
            self.assertLess(code, 0)
            self.assertLess(duration, 5)


if __name__ == "__main__":
    unittest.main()
