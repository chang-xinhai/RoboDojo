"""CPU contracts for task profiles, real replay seeds and support-arm authoring."""

import ast
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest

from env.global_configs import ROOT_DIR
from env.robot_manager.robot_class.franka import Franka
from utils.isaaclab_compat import as_torch, set_joint_target, wxyz_to_xyzw
from utils.load_file import load_yaml

ROOT = Path(ROOT_DIR).resolve()
CANDIDATE = Path(__file__).resolve().parents[1]


def load_candidate(name, relative):
    spec = spec_from_file_location(name, CANDIDATE / relative)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runtime = load_candidate("task_runtime_config_under_test", "utils/task_runtime_config.py")
layouts = load_candidate("task_layout_under_test", "utils/task_layout.py")


def configuration(profile):
    requested = load_yaml(str(ROOT / "env_cfg" / (profile + ".yml")))
    cfg = OmegaConf.create(
        {
            key: load_yaml(str(ROOT / "env_cfg" / key / (requested["config"][key] + ".yml")))
            for key in ("sim", "scene", "robot", "camera")
        }
    )
    cfg.tactile = requested.get("tactile", {})
    return cfg, requested


@pytest.mark.parametrize("task", ["imitate_sorting_sequence", "make_kong", "play_tic_tac_toe"])
@pytest.mark.parametrize("profile", ["arx_x5_rgbd", "arx_x5_multimodal"])
def test_canonical_support_profiles_retain_target_sensors_and_disable_all_planners(task, profile):
    cfg, requested = configuration(profile)
    before_camera = OmegaConf.to_container(cfg.camera, resolve=True)
    before_camera.pop("default_frequency", None)
    before_tactile = OmegaConf.to_container(cfg.tactile, resolve=True)
    proof = runtime.configure_task_runtime(cfg, task, profile, requested, {"eval_seed": 0, "layout_id": 3})
    assert [(r.robot_name, r.get("type", "target")) for r in cfg.robot.robots] == [
        ("x5", "target"),
        ("x5", "target"),
        ("franka", "support"),
    ]
    assert all(robot.need_planner is False for robot in cfg.robot.robots)
    after_camera = OmegaConf.to_container(cfg.camera, resolve=True)
    assert after_camera.pop("default_frequency") == 25
    assert after_camera == before_camera
    assert OmegaConf.to_container(cfg.tactile, resolve=True) == before_tactile
    assert proof["effective_components"]["robot"] == "dual_x5_and_franka_competition"
    assert proof["effective_components"]["camera"] == requested["config"]["camera"]
    assert proof["eval_seed"] == 0 and proof["episode_seed"] == 3
    assert list(cfg.sim.seed) == [3]
    assert cfg.sim.scene.num_envs == 1 and cfg.sim.device == "cuda:0"
    assert len(proof["effective_configuration_sha256"]) == 64


def test_enable_planners_retains_canonical_support_no_planner_setting():
    cfg, requested = configuration("arx_x5_rgbd")
    runtime.configure_task_runtime(cfg, "make_kong", "arx_x5_rgbd", requested, {"layout_id": 0}, enable_planner=True)
    assert [r.get("need_planner", True) for r in cfg.robot.robots] == [True, True, False]


@pytest.mark.parametrize("task", ["match_and_pick_from_conveyor", "pick_from_conveyor_by_image"])
def test_conveyor_task_scene_is_selected_before_any_native_scene_creation(task):
    cfg, requested = configuration("arx_x5_rgbd")
    proof = runtime.configure_task_runtime(cfg, task, "arx_x5_rgbd", requested, {"layout_id": 0})
    assert proof["effective_components"]["scene"] == "conveyor"
    expected = load_yaml(str(ROOT / "env_cfg/scene/conveyor.yml"))
    assert OmegaConf.to_container(cfg.scene, resolve=True) == expected
    assert proof["effective_components"]["camera"] == "camera_config_rgbd"


def test_random_variant_uses_exact_production_task_rules_and_teleop_stabilization():
    cfg, requested = configuration("arx_x5_multimodal")
    proof = runtime.configure_task_runtime(cfg, "make_toast_random", "arx_x5_multimodal", requested, {"layout_id": 0})
    assert cfg.sim.physx.enable_stabilization is True
    assert all(robot.enabled_self_collisions is False for robot in cfg.robot.robots)
    # The random task's rules use common interval 10, whereas its base uses 5.
    assert cfg.sim.render_interval == 10
    assert proof["canonical_task"] == "make_toast_random"
    assert proof["task_overrides"]["robot_self_collision"] is False


def test_layout_cohort_and_episode_seeds_load_the_corresponding_real_path(tmp_path):
    import json

    for cohort in (0, 2):
        directory = tmp_path / f"Assets/Eval_Layout/RoboDojo/arx_x5/{cohort}"
        directory.mkdir(parents=True)
        (directory / "imitate_sorting_sequence_3.json").write_text(
            json.dumps({"Rigid": {"bowl": [{"category_idx": cohort}]}})
        )
    row, proof = layouts.load_task_layout(tmp_path, "imitate_sorting_sequence", layout_id=3, eval_seed=2)
    assert row["Rigid"]["bowl"][0]["category_idx"] == 2
    assert proof["eval_seed"] == 2 and proof["episode_seed"] == 3
    assert "/arx_x5/2/" in proof["file"]
    override = tmp_path / "arbitrary-custom-name.json"
    override.write_text(json.dumps(row))
    _, custom = layouts.load_task_layout(tmp_path, "imitate_sorting_sequence", "arx_x5", 3, override, eval_seed=2)
    assert custom["eval_seed"] == 2 and custom["episode_seed"] == 3
    assert custom["file"] == str(override.resolve())


@pytest.mark.parametrize("field,value", [("layout_id", -1), ("layout_id", True), ("eval_seed", -1), ("eval_seed", 2.0)])
def test_invalid_layout_seed_is_rejected_before_reading_files(tmp_path, field, value):
    with pytest.raises(ValueError, match=field):
        layouts.load_task_layout(tmp_path, "make_kong", **{field: value})


def test_replay_and_repeat_use_individual_episode_seed_and_preserve_original_profile():
    initial = {"eval_seed": 2, "episode_seed": 3}
    replay = runtime.episode_profile(initial, {"eval_seed": 2, "layout_id": 7})
    events = []
    env = SimpleNamespace(
        num_envs=1,
        reset=lambda seed: events.append(("reset", seed, env.eval_seed)),
        scene_manager=SimpleNamespace(apply_saved_poses=lambda env_idx_list: events.append(("place", env_idx_list))),
        robot_manager=SimpleNamespace(
            set_origin_endpose=lambda: events.append(("origin",)),
            set_robot_init_state=lambda: events.append(("history",)),
        ),
    )
    runtime.reset_task_episode(env, replay)
    runtime.reset_task_episode(env, initial)
    assert events == [
        ("reset", 7, 2),
        ("place", [0]),
        ("origin",),
        ("history",),
        ("reset", 3, 2),
        ("place", [0]),
        ("origin",),
        ("history",),
    ]
    assert initial == {"eval_seed": 2, "episode_seed": 3}
    with pytest.raises(ValueError, match="cohort"):
        runtime.episode_profile(initial, {"eval_seed": 0, "layout_id": 7})


def test_actual_support_wrapper_and_scene_cfg_author_canonical_root_and_preserve_home_joints():
    cfg = load_yaml(str(ROOT / "env_cfg/robot/dual_x5_and_franka_competition.yml"))["robots"][2]
    robot = Franka(cfg)
    source = ROOT / "env/robot_manager/robot_manager.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RobotManager")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_attach_scene_cfg")

    class Cfg(SimpleNamespace):
        def replace(self, **kwargs):
            return Cfg(**{**vars(self), **kwargs})

    home = {"panda_joint1": 0, "panda_joint4": -2.617993877991494, "panda_finger_joint1": 0.04}
    model = Cfg(init_state=Cfg(joint_pos=home))
    namespace = {
        "ArticulationCfg": SimpleNamespace(InitialStateCfg=lambda **kwargs: Cfg(**kwargs)),
        "ENV_REGEX_NAMESPACE": "{ENV_REGEX_NS}",
        "wxyz_to_xyzw": wxyz_to_xyzw,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    namespace[method.name](SimpleNamespace(_get_robot_cfg=lambda robot_name: model), "franka", robot, 2, cfg)
    assert robot.SceneCfg.prim_path == "{ENV_REGEX_NS}/robot2"
    np.testing.assert_array_equal(robot.SceneCfg.init_state.pos, [0, 0.6, 0.765])
    np.testing.assert_array_equal(robot.SceneCfg.init_state.rot, [0, 0, -0.707, 0.707])
    assert robot.SceneCfg.init_state.joint_pos == home
    assert robot.entity_origin_pose == [0, 0.6, 0.765, 0.707, 0, 0, -0.707]


def test_actual_support_default_joints_are_restored_with_zero_native_velocity_and_drive_targets():
    import torch

    source = ROOT / "env/robot_manager/robot_config/franka.py"
    function = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef))

    def construct(**kwargs):
        return SimpleNamespace(**kwargs)

    construct.InitialStateCfg = construct
    namespace = {
        "ArticulationCfg": construct,
        "ImplicitActuatorCfg": construct,
        "sim_utils": SimpleNamespace(
            UsdFileCfg=construct, RigidBodyPropertiesCfg=construct, ArticulationRootPropertiesCfg=construct
        ),
        "ROBOTS_PATH": str(ROOT / "Assets/Robots"),
        "wxyz_to_xyzw": wxyz_to_xyzw,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    config = namespace[function.name]()
    assert Path(config.spawn.usd_path).is_file()
    home = list(config.init_state.joint_pos.values())
    np.testing.assert_allclose(home, [0, np.pi / 16, 0, -5 * np.pi / 6, 0, 2.941592653589793, np.pi / 4, 0.04, 0.04])
    source = ROOT / "env/robot_manager/robot_manager.py"
    cls = next(
        n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "RobotManager"
    )
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "set_robot_init_pose")
    ns = {"as_torch": as_torch, "set_joint_target": set_joint_target}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), ns)

    class Arm:
        def __init__(self, values):
            self.data = SimpleNamespace(
                default_joint_pos=torch.tensor([values]), default_joint_vel=torch.zeros((1, len(values)))
            )
            self.position = torch.full((1, len(values)), 9.0)
            self.velocity = torch.full((1, len(values)), 2.0)
            self.actuators = SimpleNamespace(
                target_command=SimpleNamespace(
                    set_position_index=lambda value, **kwargs: setattr(self, "position_target", value.clone()),
                    set_velocity_index=lambda value, **kwargs: setattr(self, "velocity_target", value.clone()),
                )
            )
            self.events = []

        def write_joint_state_to_sim_index(self, *, position, velocity):
            self.position, self.velocity = position.clone(), velocity.clone()
            self.events.append("write_live_state")

        def reset(self):
            assert self.events == ["write_live_state"]
            self.events.append("reset_buffers")

    arms = [Arm([0.1, 0.2]), Arm([0.3, 0.4]), Arm(home)]
    manager = SimpleNamespace(
        robot_key=arms,
        robot_list=[SimpleNamespace(robot_type="arm", type=role) for role in ["target", "target", "support"]],
    )
    ns[method.name](manager)
    for arm in arms:
        assert torch.equal(arm.position, arm.data.default_joint_pos)
        assert torch.equal(arm.position_target, arm.data.default_joint_pos)
        assert torch.count_nonzero(arm.velocity) == torch.count_nonzero(arm.velocity_target) == 0
        assert arm.events == ["write_live_state", "reset_buffers"]


@pytest.mark.parametrize("script", ["smoke_task.py", "collect_multimodal.py"])
def test_entry_points_install_profile_before_constructing_scene_and_record_effective_provenance(script):
    tree = ast.parse((CANDIDATE / "reproduction" / script).read_text())
    profile_calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "configure_task_runtime"
    ]
    constructors = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "task_class"
    ]
    assert len(profile_calls) == 1 and profile_calls[0].lineno < min(n.lineno for n in constructors)
    assert '"runtime_profile"' in (CANDIDATE / "reproduction" / script).read_text()
    assert not any(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "reset"
        and any(k.arg == "seed" and isinstance(k.value, ast.Constant) and k.value.value == 0 for k in n.keywords)
    )
