"""Exercise collector process status with a cleanup API that uses os._exit.

All sensor/task modules are deterministic CPU fixtures; this is not runtime
capture evidence. The actual collector, NPZ writer and shutdown code run.
"""

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

class Env:
    dt = .04
    env_seed_list = [0]
    def __init__(self, cfg, app, **kwargs):
        assert cfg.tactile["enabled"], "TacSL config was not passed to environment"
    def reset(self, **kwargs): pass
    def sim_step(self, **kwargs): pass
    def close(self):
        if mode == "close_failure":
            raise RuntimeError("deliberate environment cleanup failure")

class Obs:
    collect_freq = 25
    collect_interval = 1
    def __init__(self, *args, **kwargs): pass
    def initialize(self, env): pass
    def get_obs(self):
        frame = fixture()
        for camera in frame["vision"].values():
            camera["ir_left"][:] = 1
            camera["depth_sl"][:] = 1
        return {0: frame}

install("env.observation_manager.obs_manager", ObsManager=Obs)
install("task.RoboDojo.task_registry", load_task_class=lambda task: (None, Env))

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
sys.argv = ["collect_multimodal.py", "--frames", "1", "--warmup-steps", "0", "--output", output]
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
