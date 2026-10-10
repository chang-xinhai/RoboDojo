"""Initialize/reset one real RoboDojo task without a policy server."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "XPolicyLab")]
from isaaclab.app import AppLauncher

from utils.isaaclab_compat import add_app_launcher_args

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="stack_bowls")
parser.add_argument("--env-cfg", default="arx_x5_rgbd")
parser.add_argument("--output", type=Path, default=ROOT / "reproduction/task-smoke.json")
parser.add_argument("--layout-config", default="arx_x5")
parser.add_argument("--layout-id", type=int, default=0)
parser.add_argument("--layout-file", type=Path)
parser.add_argument("--enable-planner", action="store_true", help="Also initialize production planners")
add_app_launcher_args(parser, AppLauncher)
args = parser.parse_args()
args.output.unlink(missing_ok=True)
args.enable_cameras = True
from env.camera_manager.capture.render_sync import add_zero_delay_kit_args

add_zero_delay_kit_args(args)
launcher = AppLauncher(args)
app = launcher.app
success = False
try:
    from utils.isaaclab_compat import enable_legacy_isaacsim_extensions

    enable_legacy_isaacsim_extensions()
    import numpy as np
    from omegaconf import OmegaConf

    from env.observation_manager.obs_manager import ObsManager
    from task.RoboDojo.task_registry import load_task_class
    from utils.load_file import load_yaml
    from utils.task_layout import load_task_layout, verify_task_scene

    eval_cfg = load_yaml(str(ROOT / "env_cfg" / (args.env_cfg + ".yml")))
    cfg = OmegaConf.create(
        {
            key: load_yaml(str(ROOT / "env_cfg" / key / (eval_cfg["config"][key] + ".yml")))
            for key in ("sim", "scene", "camera", "robot")
        }
    )
    cfg.task_env = load_yaml(str(ROOT / "task/RoboDojo/config" / (args.task + ".yml")))
    cfg.sim.scene.num_envs = 1
    cfg.sim.device = "cuda:0"
    cfg.sim.seed = [0]
    if not args.enable_planner:
        for robot in cfg.robot.robots:
            robot.need_planner = False
    cfg.camera.default_frequency = eval_cfg["observation"]["collect_freq"]
    layout, layout_provenance = load_task_layout(
        ROOT, args.task, layout_config=args.layout_config, layout_id=args.layout_id, layout_file=args.layout_file
    )
    cfg.saved_layouts = [layout]
    _, task_class = load_task_class(args.task)
    env = task_class(cfg, app, seed=0)
    env.reset(seed=0)
    scene_evidence = verify_task_scene(env.scene_manager, layout)
    obs_cfg = eval_cfg["observation"]
    obs_cfg["vision"].update(depth=True, intrinsic_matrix=True, extrinsic_matrix=True)
    obs = ObsManager(obs_cfg, 1, env.dt, args.task, {}, seeds_per_env=env.env_seed_list)
    obs.initialize(env)
    for _ in range(10):
        env.sim_step(render=False)
    obs.render_for_capture()
    data = obs.get_obs()
    shapes = {}
    for name, camera in data[0]["vision"].items():
        # The policy observation API retains its historical RGB alias.
        rgb = camera.get("rgb", camera.get("color"))
        assert rgb is not None, (name, list(camera))
        assert rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[-1] == 3, (name, rgb.shape, rgb.dtype)
        assert "depth" in camera, (name, camera.keys())
        depth = camera["depth"]
        assert depth.shape == rgb.shape[:2], (name, depth.shape, rgb.shape)
        assert np.any(np.isfinite(depth) & (depth > 0)), name
        shapes[name] = {key: list(value.shape) for key, value in camera.items() if hasattr(value, "shape")}
    assert set(shapes) == {"cam_head", "cam_left_wrist", "cam_right_wrist"}, shapes
    import importlib.metadata as metadata
    import subprocess

    result = {
        "status": "passed",
        "real_runtime": True,
        "physics_steps": 10,
        "packages": {name: metadata.version(name) for name in ("isaacsim", "isaaclab", "torch", "warp-lang")},
        "isaaclab_source_commit": subprocess.check_output(
            ["git", "-C", str(ROOT / "third_party/IsaacLab3"), "rev-parse", "HEAD"], text=True
        ).strip(),
        "task": args.task,
        "env_cfg": args.env_cfg,
        "num_envs": 1,
        "checks": ["task scene initialization", "robot reset", "RGBD observation capture"],
        "camera_shapes": shapes,
        "policy_evaluation": False,
        "planner_initialized": args.enable_planner,
        "layout": layout_provenance,
        "scene": scene_evidence,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    env.close()
    success = True
except Exception:
    import traceback

    traceback.print_exc()
finally:
    app.close(wait_for_replicator=False, skip_cleanup=True, exit_code=0 if success else 1)
