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
FIXTURE_LAYOUT.write_text(json.dumps({"Rigid": {"bowl": [{}, {}, {}]}}))


def evidence():
    expected = dict.fromkeys(runner.OBJECT_STORAGES, 0)
    expected["Rigid"] = 3
    actual = dict.fromkeys(runner.OBJECT_STORAGES.values(), 0)
    actual["rigid_and_dynamic_objects"] = 3
    paths = {storage: [] for storage in actual}
    paths["rigid_and_dynamic_objects"] = [f"/World/envs/env_0/Rigid/bowl/bowl_{i}" for i in range(3)]
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
            "wrapper_types": {
                path: {
                    "class": "env.scene_manager.objects.rigid.RigidObject",
                    "physical_type": "Rigid",
                    "instance_name": f"bowl_{i}",
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

    def test_resume_rejects_wrong_revision_runtime_exit_and_evidence(self):
        ctx = context()
        record = {
            "status": "passed",
            "context": deepcopy(ctx),
            "process_exit_code": 0,
            "real_runtime": True,
            "evidence": evidence(),
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
