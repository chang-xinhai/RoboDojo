"""Resolve task-specific evaluation settings before creating native scenes."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf

from env.global_configs import ROOT_DIR
from utils.load_file import load_yaml
from utils.pipeline_utils import process_config


def episode_profile(profile, layout_provenance):
    """Bind the replay to the same actual cohort and its individual scene seed."""
    cohort = layout_provenance.get("eval_seed", 0)
    scene_seed = layout_provenance["layout_id"]
    for name, value in (("eval_seed", cohort), ("layout_id", scene_seed)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"Loaded layout {name} must be a nonnegative integer")
    if cohort != profile["eval_seed"]:
        raise ValueError("A replay must use the same actual layout/trajectory cohort")
    return {**profile, "episode_seed": scene_seed}


def configure_task_runtime(cfg, task_name, requested_env_cfg, eval_cfg, layout_provenance, *, enable_planner=False):
    """Apply the production eval configuration while retaining requested sensors.

    The caller has already resolved the task class via its registry, including
    random task names. Use that exact name, as the production main does, rather
    than silently inheriting different settings by removing its suffix.
    """
    cfg, eval_num = process_config(cfg, task_name=task_name)
    if not enable_planner:
        # Apply this after process_config installs any support-arm profile.
        for robot in cfg.robot.robots:
            robot.need_planner = False
    profile = episode_profile({"eval_seed": layout_provenance.get("eval_seed", 0)}, layout_provenance)
    # Reproduction processes are deliberately one environment on the active GPU.
    # Set their runtime controls after all task profile changes are resolved.
    cfg.sim.scene.num_envs = 1
    cfg.sim.device = "cuda:0"
    cfg.sim.seed = [profile["episode_seed"]]
    cfg.camera.default_frequency = eval_cfg["observation"]["collect_freq"]
    cfg.eval_cfg = {
        "task_name": task_name,
        "config_name": requested_env_cfg,
        "seed": profile["eval_seed"],
        "episode_seed": profile["episode_seed"],
    }
    rules = load_yaml(str(Path(ROOT_DIR) / "task/RoboDojo/config/_task.yml"))
    task_rules, common = rules["tasks"].get(task_name, {}), rules.get("common", {})
    components = {}
    for kind, default in (("robot", "dual_x5"), ("scene", "default"), ("camera", "camera_config")):
        task_setting = task_rules.get(kind + "_config", common.get(kind + "_config", default))
        components[kind] = task_setting if task_setting != default else eval_cfg["config"][kind]
    effective = {kind: OmegaConf.to_container(cfg[kind], resolve=True) for kind in ("robot", "scene", "camera")}
    profile.update(
        canonical_task=task_name,
        requested_env_cfg=requested_env_cfg,
        effective_components=components,
        effective_configuration_sha256=hashlib.sha256(
            json.dumps(effective, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest(),
        robot_roles=[
            {
                "robot_name": robot.robot_name,
                "type": robot.get("type", "target"),
                "need_planner": robot.get("need_planner", True),
                "default_root_pos": list(robot.default_root_pos),
                "default_root_rot_wxyz": list(robot.default_root_rot),
            }
            for robot in cfg.robot.robots
        ],
        task_overrides=deepcopy(task_rules),
        native_eval_num=eval_num,
        num_envs=1,
        device="cuda:0",
    )
    return profile


def reset_task_episode(env, profile):
    """Follow the task reset, saved placement and robot history lifecycle."""
    env.eval_seed = profile["eval_seed"]
    env.reset(seed=profile["episode_seed"])
    # Bare TaskEnv stages CUDA bodies offscreen and calibrates empty TacSL gels.
    # EvalEnv.setup_scene places them afterward, retaining the actual baseline.
    env.scene_manager.apply_saved_poses(env_idx_list=list(range(env.num_envs)))
    env.robot_manager.set_origin_endpose()
    env.robot_manager.set_robot_init_state()
