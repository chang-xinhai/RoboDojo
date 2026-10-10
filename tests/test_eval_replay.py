"""Execute evaluator lifecycle methods without starting Kit or a policy server."""

import ast
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def lifecycle_class(base, globals_dict):
    """Compile actual nested EvalEnv methods with their real super() closure."""
    module = ast.parse((ROOT / "src/eval_client/eval_env.py").read_text())
    factory = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "create_eval_env")
    evaluator = next(node for node in factory.body if isinstance(node, ast.ClassDef))
    selected = [
        node
        for node in evaluator.body
        if isinstance(node, ast.FunctionDef) and node.name in {"reset", "_post_setup_scene"}
    ]
    cls = ast.ClassDef(
        name="EvalLifecycle",
        bases=[ast.Name(id="TaskBase", ctx=ast.Load())],
        keywords=[],
        body=selected,
        decorator_list=[],
    )
    namespace = {"TaskBase": base, **globals_dict}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), "eval_env.py", "exec"), namespace)
    return namespace["EvalLifecycle"]


def block_instruction():
    source = ROOT / "task/RoboDojo/tasks/stack_blocks_by_language.py"
    cls = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.ClassDef))
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "gen_instruction")
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(source), "exec"), namespace)
    return namespace["gen_instruction"]


class EvalReplayTests(unittest.TestCase):
    def test_language_observation_binds_active_layout_after_native_pool_reset(self):
        # Avoid unrelated GPU seeding imports; observations and instruction
        # generation execute their production implementation on CPU fixtures.
        np.random.seed(0)
        seed_module = ModuleType("env.seeding")
        seed_module.seed_everywhere = lambda seed: None
        with patch.dict(sys.modules, {"env.seeding": seed_module}):
            module = importlib.import_module("env.observation_manager.obs_manager")
            obs = module.ObsManager({"collect_freq": 25}, 1, 0.01, "stack_blocks_by_language", {"seen": 1})

        class TaskBase:
            def _post_setup_scene(self, sim):
                self.pool_initialized = True

            def reset(self, seed=None, options=None):
                self._post_setup_scene(None)
                # Real core reloads pooled body metadata before returning.
                self.active_models = self.current_layout["models"]

        cls = lifecycle_class(TaskBase, {})
        cls.gen_instruction = block_instruction()
        env = cls()
        env.num_envs = 1
        env.active_models = [9, 9, 9]  # Parked pool metadata must never become the instruction.
        env.current_layout = None

        def set_layout(env_id, layout):
            env.current_layout = layout

        env.scene_manager = SimpleNamespace(layout_manager=SimpleNamespace(set_saved_layout=set_layout))
        layouts = {0: {"models": [1, 2, 3]}, 1: {"models": [7, 8, 9]}}
        env.seed_manager = SimpleNamespace(get_seed_scene_info=layouts.__getitem__)
        env.obs_manager = obs
        env._abort_video_writers = lambda: None
        env.setup_scene = lambda: None
        env.robot_manager = SimpleNamespace(set_origin_endpose=lambda: None, set_robot_init_state=lambda: None)
        parser = SimpleNamespace(get_label_cat_index=lambda labels: [[model] for model in env.active_models])
        env.reward_manager = SimpleNamespace(func_parser=parser, init_state=lambda: None)
        env.model_client = SimpleNamespace(call=lambda **kwargs: None)

        env._post_setup_scene(None)
        self.assertTrue(env.pool_initialized)
        self.assertFalse(hasattr(obs, "env"))
        env.reset(seed=[0])
        self.assertIn("blue, green, and red", str(obs.instruction[0]))
        env.reset(seed=[1])
        self.assertIn("yellow, orange, and cyan", str(obs.instruction[0]))
        self.assertNotIn("blue", str(obs.instruction[0]))


if __name__ == "__main__":
    unittest.main()
