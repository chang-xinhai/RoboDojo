"""Exercise collector process status with a cleanup API that uses os._exit.

All sensor/task modules are deterministic CPU fixtures; this is not runtime
capture evidence. The actual collector, NPZ writer and shutdown code run.
"""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = r"""
import os
from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace
sys.path.insert(0, str(Path.cwd() / "tests"))
from test_multimodal import fixture

def install(name, **exports):
    module = ModuleType(name)
    module.__dict__.update(exports)
    sys.modules[name] = module

class App:
    def close(self, exit_code=0, **kwargs):
        os._exit(exit_code)

class Launcher:
    def __init__(self, args):
        self.app = App()
    @staticmethod
    def add_app_launcher_args(parser):
        parser.add_argument("--kit_args", default="")

install("isaaclab.app", AppLauncher=Launcher)

class OmegaConf:
    @staticmethod
    def create(value):
        def convert(item):
            if isinstance(item, dict):
                return SimpleNamespace(**{key: convert(child) for key, child in item.items()})
            return item
        return convert(value)

install("omegaconf", OmegaConf=OmegaConf)

class RigidObject(SimpleNamespace):
    def __init__(self, prim_path):
        super().__init__(prim_path=prim_path, default_pos=[0.1, 0.2, 0.8], position=[100000.0, 100000.0, 100000.0])
    def get_world_pose(self):
        return self.position, [1.0, 0.0, 0.0, 0.0]

class PhysicsContext:
    frame = 0
    def get_physics_step_count(self): return self.frame
    def get_physics_dt(self): return .04

class Env:
    dt = .04
    env_seed_list = [0]
    def __init__(self, cfg, app, **kwargs):
        assert cfg.tactile["enabled"], "TacSL config was not passed to environment"
        assert cfg.saved_layouts[0]["Rigid"]["test_object"], "Layout must exist before task construction"
        objects = {} if mode == "empty_scene" else {"test": RigidObject(prim_path="/World/test")}
        self.scene_manager = SimpleNamespace(
            _rigid_and_dynamic_objects=[objects],
            stage=SimpleNamespace(GetPrimAtPath=lambda path: SimpleNamespace(IsValid=lambda: True)),
            apply_saved_poses=self.apply_saved_poses,
        )
        self.reset_complete = False
        self.placed = False
        self.sim = SimpleNamespace(sim=PhysicsContext())
    def reset(self, **kwargs):
        self.reset_complete = True
    def apply_saved_poses(self, env_idx_list):
        assert self.reset_complete, "Saved poses must be applied after reset staging"
        assert env_idx_list == [0]
        if mode == "placement_failure":
            raise RuntimeError("deliberate task placement failure")
        if mode != "offscreen_not_restored":
            for obj in self.scene_manager._rigid_and_dynamic_objects[0].values():
                obj.position = list(obj.default_pos)
        self.placed = True
    def sim_step(self, **kwargs): self.sim.sim.frame += 1
    def close(self):
        if mode == "close_failure":
            raise RuntimeError("deliberate environment cleanup failure")

class Obs:
    collect_freq = 25
    collect_interval = 1
    def __init__(self, *args, **kwargs): pass
    def initialize(self, env):
        self.env = env
    def get_obs(self):
        assert self.env.placed, "Objects were left in reset offscreen staging"
        frame = fixture(self.env.sim.sim.frame)
        if mode == "escaped_mid_record":
            for obj in self.env.scene_manager._rigid_and_dynamic_objects[0].values():
                obj.position = [100000., 100000., 100000.]
        for camera in frame["vision"].values():
            camera["ir_left"][:] = 1
            camera["depth_sl"][:] = 1
        return {0: frame}

install("env.observation_manager.obs_manager", ObsManager=Obs)
install("task.RoboDojo.task_registry", load_task_class=lambda task: (task, Env))

def load_yaml(path):
    if "/env_cfg/arx_x5" in str(path):
        return {
            "config": {key: "test" for key in ("sim", "scene", "camera", "robot")},
            "observation": {"collect_freq":25, "multimodal": True},
            "tactile": {"enabled": True},
        }
    return {"scene": {"num_envs":1}}

install("utils.load_file", load_yaml=load_yaml)

def visualize(path):
    raise RuntimeError("deliberate visualization failure")

install("reproduction.visualize_multimodal", visualize=visualize)
mode, output = sys.argv[1:]
import utils.isaaclab_compat as compat

def enable_extensions():
    if mode == "extension_failure":
        raise RuntimeError("deliberate legacy extension initialization failure")

compat.enable_legacy_isaacsim_extensions = enable_extensions
layout_file = Path(output) / "test-layout.json"
layout_file.write_text('{"Rigid":{"test_object":[{}]}}')
sys.argv = ["collect_multimodal.py", "--frames", "1", "--warmup-steps", "0", "--output", output,
            "--layout-file", str(layout_file)]
if mode == "visualization_failure":
    sys.argv.append("--visualize")
runpy.run_path("reproduction/collect_multimodal.py", run_name="__main__")
"""


class CollectorExitTests(unittest.TestCase):
    def run_collector(self, mode, directory):
        return subprocess.run(
            [sys.executable, "-c", BOOTSTRAP, mode, directory],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_successful_capture_reports_zero_and_writes_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_collector("success", directory)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((Path(directory) / "validation.json").exists())
            validation = json.loads((Path(directory) / "validation.json").read_text())
            self.assertEqual(validation["scene"]["actual_object_counts"]["rigid_and_dynamic_objects"], 1)
            self.assertEqual(len(validation["layout"]["sha256"]), 64)
            self.assertTrue(validation["scene"]["saved_poses_applied"])
            self.assertTrue(validation["scene"]["placement_verified"])
            self.assertEqual(validation["scene_final"]["physics_frame"], validation["frames"][-1]["physics_frame"])
            self.assertLess(validation["scene_initial"]["physics_frame"], validation["scene_final"]["physics_frame"])
            self.assertEqual(validation["scene"]["live_placements"]["/World/test"]["world_center"], [0.1, 0.2, 0.8])

    def test_missing_task_objects_never_produce_successful_recording(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_collector("empty_scene", directory)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("saved layout requires 1", result.stderr)
            self.assertFalse((Path(directory) / "validation.json").exists())
            self.assertFalse((Path(directory) / "manifest.json").exists())

    def test_extension_startup_failure_survives_direct_process_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_collector("extension_failure", directory)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("deliberate legacy extension initialization failure", result.stderr)
            self.assertFalse((Path(directory) / "validation.json").exists())

    def test_task_placement_failure_never_produces_successful_recording(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_collector("placement_failure", directory)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("deliberate task placement failure", result.stderr)
            self.assertFalse((Path(directory) / "validation.json").exists())
            self.assertFalse((Path(directory) / "manifest.json").exists())

    def test_offscreen_objects_are_rejected_even_after_placement_callback_returns(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_collector("offscreen_not_restored", directory)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("offscreen/unplaced", result.stderr)
            self.assertFalse((Path(directory) / "validation.json").exists())
            self.assertFalse((Path(directory) / "manifest.json").exists())

    def test_object_escape_after_initial_verification_prevents_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_collector("escaped_mid_record", directory)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("offscreen/unplaced", result.stderr)
            self.assertTrue((Path(directory) / "frame_000000.npz").exists())
            self.assertFalse((Path(directory) / "manifest.json").exists())
            self.assertFalse((Path(directory) / "validation.json").exists())

    def test_environment_cleanup_failure_survives_direct_process_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_collector("close_failure", directory)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("deliberate environment cleanup failure", result.stderr)
            self.assertFalse((Path(directory) / "validation.json").exists())

    def test_visualization_failure_survives_direct_process_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_collector("visualization_failure", directory)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("deliberate visualization failure", result.stderr)
            self.assertFalse((Path(directory) / "validation.json").exists())


if __name__ == "__main__":
    unittest.main()
