"""Acceptance runner tests use ordinary subprocesses, never start Isaac/Kit."""

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from reproduction.isaac61 import check_tasks as runner


def evidence():
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
