"""Exercise collector process status with a cleanup API that uses os._exit.

All sensor/task modules are deterministic CPU fixtures; this is not runtime
capture evidence. The actual collector, HDF5 writer and shutdown code run.
"""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = r"""
import json
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
    num_envs = 1
    env_seed_list = [3]
    def __init__(self, cfg, app, **kwargs):
        assert cfg.tactile["enabled"], "TacSL config was not passed to environment"
        if mode.startswith("contact_"):
            assert cfg.tactile.compliance_stiffness == 100.0
            assert list(cfg.tactile.array_size) == [20, 25]
            assert cfg.sim.particle_readback is True
            assert cfg.sim.host_readback_reason == "kinematic_contact_calibration"
        else:
            assert cfg.tactile.get("compliance_stiffness", 1500.) == 1500.
        self.contact_fixtures = SimpleNamespace(config=contact_config if mode.startswith("contact_") else None)
        self.tactile_manager = SimpleNamespace()
        assert cfg.saved_layouts[0]["Rigid"]["test_object"], "Layout must exist before task construction"
        assert list(cfg.sim.seed) == [3] and kwargs["seed"] == 3
        assert cfg.eval_cfg.seed == 2 and cfg.eval_cfg.episode_seed == 3
        expected = [("x5", "target"), ("x5", "target")]
        if not mode.startswith("expert_"):
            expected.append(("franka", "support"))
        assert [(robot.robot_name, robot.get("type", "target")) for robot in cfg.robot.robots] == expected
        if mode.startswith("expert_"):
            assert cfg.tactile.contact_conditioned_observations is True
            assert list(cfg.tactile.contact_object_labels) == ["tube0", "tube1", "tube2"]
            class Rewards:
                def init_state(self): pass
                def call_func_parser(self, check, env_idx):
                    return not (mode == "expert_official_failure" and check[0] == "axis_up")
                def is_A_depth_in_B(self, **kwargs): return ("depth", kwargs)
                def is_A_in_B(self, **kwargs): return ("in_slot", kwargs)
                def is_axis_up(self, **kwargs): return ("axis_up", kwargs)
                def all_robot_back_to_origin(self): return ("origin", {})
            self.reward_manager = Rewards()
        assert all(robot.need_planner is False for robot in cfg.robot.robots)
        assert cfg.camera.default_frequency == 25
        self.lifecycle = []
        self.robot_manager = SimpleNamespace(
            set_origin_endpose=self.set_origin_endpose, set_robot_init_state=self.set_robot_init_state)
        objects = {} if mode == "empty_scene" else {"test": RigidObject(prim_path="/World/test")}
        if mode.startswith("expert_"):
            import numpy as np
            objects = {f"test_{i}": RigidObject(prim_path=f"/World/test_{i}") for i in range(3)}
            paths = [f"/World/test_{i}/mesh" for i in range(3)]
            def sensor(path):
                pose = np.array([[0., 0., 0., 0., 0., 0., 1.]], dtype=np.float32)
                view = SimpleNamespace(count=1, get_transforms=lambda: pose.copy())
                data = SimpleNamespace(tactile_normal_force=np.zeros((1, 8), np.float32),
                    tactile_shear_force=np.zeros((1, 8, 2), np.float32),
                    penetration_depth=np.zeros((1, 8), np.float32),
                    tactile_points_pos_w=np.zeros((1, 8, 3), np.float32))
                return SimpleNamespace(cfg=SimpleNamespace(contact_object_prim_path_expr=path),
                    _data=data, _contact_object_body_view=view, _elastomer_body_view=view,
                    raw_sdf=np.ones(8, np.float32))
            self.tactile_manager = SimpleNamespace(contact_paths_by_env=[paths], sensors={
                side: {finger: {0: [sensor(path) for path in paths]} for finger in ("finger_0", "finger_1")}
                for side in ("left", "right")})
        self.scene_manager = SimpleNamespace(
            _rigid_and_dynamic_objects=[objects],
            stage=SimpleNamespace(GetPrimAtPath=lambda path: SimpleNamespace(IsValid=lambda: True)),
            apply_saved_poses=self.apply_saved_poses,
        )
        if mode.startswith("expert_"):
            self.scene_manager.layout_manager = SimpleNamespace(
                get_instance_name=lambda env_idx, label: "test_" + label[-1])
        self.reset_complete = False
        self.placed = False
        self.sim = SimpleNamespace(sim=PhysicsContext())
    def record_lifecycle(self, *event):
        self.lifecycle.append(list(event))
        (Path(output) / "fixture-lifecycle.json").write_text(json.dumps(self.lifecycle))
    def reset(self, seed=None, **kwargs):
        assert seed == 3 and self.eval_seed == 2
        self.record_lifecycle("reset", seed, self.eval_seed)
        self.reset_complete = True
    def set_origin_endpose(self):
        assert self.placed, "Robot origin must follow saved placement"
        self.record_lifecycle("origin")
    def set_robot_init_state(self):
        assert self.lifecycle[-1] == ["origin"], "Robot history must follow origin snapshot"
        self.record_lifecycle("history")
    def apply_saved_poses(self, env_idx_list):
        assert self.reset_complete, "Saved poses must be applied after reset staging"
        assert env_idx_list == [0]
        self.record_lifecycle("place", env_idx_list)
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
        assert self.env.lifecycle[-2:] == [["origin"], ["history"]], "Robot episode history not initialized"
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

def visualize(path):
    raise RuntimeError("deliberate visualization failure")

install("reproduction.visualize_multimodal", visualize=visualize)
mode, output = sys.argv[1:]
import utils.isaaclab_compat as compat

def enable_extensions():
    if mode == "extension_failure":
        raise RuntimeError("deliberate legacy extension initialization failure")

compat.enable_legacy_isaacsim_extensions = enable_extensions
if mode.startswith("contact_"):
    import utils.contact_demo as contact_module
    contact_config = contact_module.ContactDemoConfig()
    class ContactController:
        def __init__(self, fixtures, manager, sim):
            self.sim = sim
            self.config = contact_config
            self.phases = set()
        def configuration(self):
            return {"enabled": True, "synthetic_cpu_fixture": True}
        def apply(self, elapsed, frame):
            self.command = {**contact_module.phase_command(self.config, elapsed),
                            "issued_before_physics_frame": frame}
        def sample(self):
            assert self.sim.frame > self.command["issued_before_physics_frame"]
            self.phases.add(self.command["phase"])
            return {"physics_frame": self.sim.frame + (1 if mode == "contact_clock_failure" else 0),
                    "timestamp_ns": round(self.sim.frame * self.sim.get_physics_dt() * 1e9),
                    "last_command": dict(self.command)}
        def validate(self):
            assert self.phases == set(contact_module.PHASES), self.phases
            return {"phases_observed": sorted(self.phases), "synthetic_cpu_fixture": True}
        def close(self): pass
    contact_module.ContactMotionDemo = ContactController
if mode.startswith("expert_"):
    class ExpertController:
        def __init__(self, env, args):
            self.env = env
            assert args.expert_trajectory.name == "official-commands.npz"
            self.command = None
        @property
        def finished(self): return self.env.sim.sim.frame >= 3 and mode != "expert_timeout"
        def configuration(self): return {"synthetic_cpu_fixture": True, "native_task_commands_only": True}
        def apply(self, elapsed, physics_frame):
            self.command = {"elapsed_time_s": elapsed, "issued_before_physics_frame": physics_frame}
            if mode == "expert_advances_physics": self.env.sim_step()
        def sample(self):
            return {"physics_frame": self.env.sim.sim.frame,
                    "timestamp_ns": round(self.env.sim.sim.frame * .04 * 1e9),
                    "last_command": self.command}
        def validate(self): return {"passed": mode != "expert_validation_failure"}
    install("_test_expert", Controller=ExpertController)
    import utils.native_contact_trace as trace_module
    trace_module.query_native_sdf = lambda sensor: sensor.raw_sdf
    trace_module.native_task_inventory = lambda env: {"synthetic_cpu_fixture": True,
        "physics_frame": env.sim.sim.frame, "no_gpu_evidence": True}
layout_file = Path(output) / "test-layout.json"
layout_file.write_text(json.dumps({"Rigid": {"test_object": [{}] * (3 if mode.startswith("expert_") else 1)}}))
sys.argv = ["collect_multimodal.py", "--task", "make_kong", "--frames", "1", "--warmup-steps", "0", "--output", output,
            "--layout-file", str(layout_file), "--layout-id", "3", "--eval-seed", "2"]
if mode.startswith("contact_"):
    sys.argv[sys.argv.index("--frames") + 1] = "2" if mode == "contact_short" else "100"
    sys.argv.append("--contact-demo")
if mode.startswith("expert_"):
    sys.argv[sys.argv.index("--task") + 1] = "insert_tubes"
    sys.argv[sys.argv.index("--frames") + 1] = "6"
    sys.argv += ["--expert-controller", "_test_expert:Controller",
                 "--expert-trajectory", str(Path(output) / "official-commands.npz")]
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

    def test_expert_episode_initial_and_terminal_capture_and_official_validation(self):
        from utils.multimodal_hdf5 import HDF5Reader

        with tempfile.TemporaryDirectory() as directory:
            result = self.run_collector("expert_success", directory)
            self.assertEqual(result.returncode, 0, result.stderr)
            with HDF5Reader(Path(directory) / "episode.hdf5") as reader:
                self.assertEqual(len(reader), 4)
                first, last = reader.read_frame(0), reader.read_frame(3)
                self.assertEqual(first["sync"]["physics_frame"], 0)
                self.assertIsNone(first["action"]["expert_controller"]["last_command"])
                self.assertEqual(last["sync"]["physics_frame"], 3)
                validation = reader.metadata["validation"]["expert_episode"]
                self.assertTrue(validation["passed"])
                self.assertTrue(validation["official_terminal_predicates"]["all_robot_back_to_origin"])
                self.assertEqual(
                    sum(
                        len(validation["official_terminal_predicates"][label]) for label in ("tube0", "tube1", "tube2")
                    ),
                    9,
                )
                self.assertFalse(reader.metadata["run"]["contact_demo"]["enabled"])

    def test_expert_timeout_unqualified_result_and_hidden_physics_step_reject_completion(self):
        for mode, reason in (
            ("expert_timeout", "maximum recording frames"),
            ("expert_validation_failure", "successful complete episode"),
            ("expert_official_failure", "terminal predicates failed"),
            ("expert_advances_physics", "advanced physics outside"),
        ):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                result = self.run_collector(mode, directory)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn(reason, result.stderr)
                self.assertFalse((Path(directory) / "episode.hdf5").exists())
                self.assertFalse((Path(directory) / "validation.json").exists())

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
            self.assertTrue(Path(validation["frames"][0]["file"]).exists())
            self.assertTrue(validation["frames"][0]["file"].endswith("episode.hdf5"))
            self.assertLess(validation["scene_initial"]["physics_frame"], validation["scene_final"]["physics_frame"])
            self.assertEqual(validation["scene"]["live_placements"]["/World/test"]["world_center"], [0.1, 0.2, 0.8])
            manifest = json.loads((Path(directory) / "manifest.json").read_text())
            profile = manifest["run"]["runtime_profile"]
            self.assertEqual(profile["canonical_task"], "make_kong")
            self.assertEqual(profile["effective_components"]["robot"], "dual_x5_and_franka_competition")
            self.assertEqual((profile["eval_seed"], profile["episode_seed"]), (2, 3))
            lifecycle = json.loads((Path(directory) / "fixture-lifecycle.json").read_text())
            self.assertEqual(lifecycle, [["reset", 3, 2], ["place", [0]], ["origin"], ["history"]])

    def test_contact_profile_and_all_phases_are_saved_with_matching_clocks(self):
        from utils.multimodal_hdf5 import HDF5Reader

        with tempfile.TemporaryDirectory() as directory:
            result = self.run_collector("contact_success", directory)
            self.assertEqual(result.returncode, 0, result.stderr)
            with HDF5Reader(Path(directory) / "episode.hdf5") as reader:
                self.assertEqual(len(reader), 100)
                self.assertEqual(
                    reader.metadata["run"]["resolved_configuration"]["tactile"]["compliance_stiffness"], 100.0
                )
                for index in range(len(reader)):
                    frame = reader.read_frame(index)
                    self.assertEqual(frame["action"]["contact_demo"]["physics_frame"], frame["sync"]["physics_frame"])
                self.assertEqual(len(reader.metadata["validation"]["contact_demo"]["phases_observed"]), 10)

    def test_contact_insufficient_duration_rejected_before_recording(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_collector("contact_short", directory)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("requires at least 100 frames", result.stderr)
            self.assertFalse((Path(directory) / "episode.hdf5").exists())

    def test_contact_sensor_clock_mismatch_marks_failed_partial(self):
        from utils.multimodal_hdf5 import HDF5Reader

        with tempfile.TemporaryDirectory() as directory:
            result = self.run_collector("contact_clock_failure", directory)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("same physics clock", result.stderr)
            with HDF5Reader(Path(directory) / "episode.hdf5.partial") as reader:
                self.assertEqual(reader.metadata["status"], "failed")
            self.assertFalse((Path(directory) / "manifest.json").exists())

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
            from utils.multimodal_hdf5 import HDF5Reader

            with HDF5Reader(Path(directory) / "episode.hdf5.partial") as reader:
                self.assertEqual(len(reader), 1)
                self.assertEqual(reader.metadata["status"], "failed")
            self.assertFalse((Path(directory) / "episode.hdf5").exists())
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
