"""Execute production seed propagation and description selection without Kit."""

import ast
from collections.abc import Sequence
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def actual_class(path, name, methods, namespace, base=None):
    source = ast.parse((ROOT / path).read_text())
    original = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == name)
    cls = ast.ClassDef(
        name=name,
        bases=[ast.Name(id="Parent", ctx=ast.Load())] if base else [],
        keywords=[],
        body=[node for node in original.body if isinstance(node, ast.FunctionDef) and node.name in methods],
        decorator_list=[],
    )
    scope = {"Parent": base, **namespace}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), path, "exec"), scope)
    return scope[name]


def seed_env(seeds):
    calls = []

    def seed_everywhere(seed):
        calls.append(seed)
        np.random.seed(seed)

    namespace = {"Any": Any, "Sequence": Sequence, "SequenceABC": Sequence, "seed_everywhere": seed_everywhere}
    base = actual_class(
        "env/environment/base_env.py",
        "BaseEnv",
        {
            "update_seed",
            "_process_seed_input",
            "_set_global_seed",
            "_set_env_seeds_full",
            "_is_sequence_seed",
            "get_env_seed",
            "_set_env_seed",
        },
        namespace,
    )
    task = actual_class("env/environment/task_env.py", "TaskEnv", {"update_seed"}, namespace, base)
    env = task()
    env.num_envs = len(seeds)
    env.config = SimpleNamespace(scene=SimpleNamespace(num_envs=len(seeds)))
    env.env_seeds = list(seeds)
    env.env_seed_list = env.env_seeds
    env._seed_list = list(seeds)
    env._global_seed = seeds[0]
    dispatched = []
    for name in ("scene_manager", "robot_manager", "camera_manager"):
        setattr(
            env,
            name,
            SimpleNamespace(update_env_seeds=lambda values, name=name: dispatched.append((name, list(values)))),
        )
    return env, calls, dispatched


class EpisodeSeedPropagationTests(unittest.TestCase):
    def test_scalar_replay_rebinds_alias_and_dispatches_current_episode(self):
        env, calls, dispatched = seed_env([0, 1])
        stale = env.env_seed_list
        env.update_seed(7)
        self.assertIs(env.env_seed_list, env.env_seeds)
        self.assertEqual(stale, [0, 1])
        self.assertEqual(env.env_seed_list, [7, 7])
        self.assertEqual([values for _, values in dispatched], [[7, 7]] * 3)
        self.assertEqual(calls, [7])
        self.assertEqual([env.get_env_seed(i) for i in range(2)], [7, 7])

    def test_list_seeds_and_subset_selection_preserve_distinct_environment_seed(self):
        env, calls, _ = seed_env([0, 0, 0])
        env.update_seed([3, 7, 11])
        self.assertIs(env.env_seed_list, env.env_seeds)
        self.assertEqual(env.env_seed_list, [3, 7, 11])
        env._set_env_seed(2)
        env._set_env_seed(1)
        self.assertEqual(calls, [3, 11, 7])
        with self.assertRaises(IndexError):
            env._set_env_seed(3)

    def test_none_keeps_active_episode_without_reseeding(self):
        env, calls, dispatched = seed_env([4, 8])
        active = env.env_seed_list
        env.update_seed(None)
        self.assertIs(env.env_seed_list, active)
        self.assertEqual(calls, [])
        self.assertEqual([values for _, values in dispatched], [[4, 8]] * 3)

    def test_invalid_list_does_not_mutate_active_seed_metadata(self):
        env, calls, dispatched = seed_env([4, 8])
        with self.assertRaises(ValueError):
            env.update_seed([7])
        self.assertEqual(env.env_seed_list, [4, 8])
        self.assertEqual(calls, [])
        self.assertEqual(dispatched, [])

    def test_actual_description_manager_refreshes_seed_and_active_layout_on_replay(self):
        env, _, _ = seed_env([0])
        env.scene_manager.layout_manager = SimpleNamespace()
        env.gen_instruction = lambda env_idx: [f"layout-{env.get_env_seed(env_idx)} choice-{i}" for i in range(12)]
        fake_seed = ModuleType("env.seeding")
        fake_seed.seed_everywhere = np.random.seed
        with patch.dict(sys.modules, {"env.seeding": fake_seed}):
            module = importlib.import_module("env.observation_manager.obs_manager")
            with patch("env.description_manager.desc_manager.seed_everywhere", np.random.seed):
                obs = module.ObsManager({"collect_freq": 25}, 1, 0.01, "replay", {"seen": 1}, seeds_per_env=[0])
                obs.initialize(env)
                self.assertTrue(str(obs.instruction[0]).startswith("layout-0"))
                env.update_seed(7)
                obs.initialize(env)
                obs.reset()  # The production smoke replay follows the evaluator lifecycle.
                self.assertEqual(obs.desc_manager._seeds_per_env, [7])
                templates = env.gen_instruction(0)
                self.assertEqual(str(obs.instruction[0]), str(np.random.RandomState(7).choice(templates)))
                self.assertTrue(str(obs.instruction[0]).startswith("layout-7"))

    def test_smoke_replay_reinitializes_then_resets_real_description_templates(self):
        tree = ast.parse((ROOT / "reproduction/smoke_task.py").read_text())
        loop = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.For)
            and isinstance(node.target, ast.Tuple)
            and "replay_layout" in ast.unparse(node.target)
        )
        calls = [
            ast.unparse(node.value)
            for node in loop.body
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        ]
        self.assertLess(calls.index("reset_task_episode(env, replay_profile)"), calls.index("obs.initialize(env)"))
        self.assertLess(calls.index("obs.initialize(env)"), calls.index("obs.reset()"))
        self.assertLess(calls.index("obs.reset()"), calls.index("obs.render_for_capture()"))


if __name__ == "__main__":
    unittest.main()
