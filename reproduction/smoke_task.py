"""Initialize/reset one real RoboDojo task without a policy server."""

import argparse
from copy import deepcopy
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
parser.add_argument("--eval-seed", type=int, default=0, help="Actual saved-layout and support-trajectory cohort.")
parser.add_argument("--layout-file", type=Path)
parser.add_argument(
    "--replay-layout-ids", type=int, nargs="*", default=[], help="Preload and reset a finite replay cohort"
)
parser.add_argument("--repeat-envs", type=int, default=1, help="Construct and close multiple tasks in the same Kit app")
parser.add_argument("--check-reset", action="store_true", help="Probe real GPU body/joint motion and exact resets")
parser.add_argument("--probe-mass-variant", action="store_true", help="Replay a separately pooled native mass variant")
parser.add_argument("--enable-planner", action="store_true", help="Also initialize production planners")
parser.add_argument("--disable-nslv2", action="store_true", help="Capture raw stereo IR without starting NSLv2 inference")
add_app_launcher_args(parser, AppLauncher)
args = parser.parse_args()
if args.repeat_envs < 1:
    parser.error("repeat-envs must be positive")
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
    from utils.task_runtime_config import configure_task_runtime, episode_profile, reset_task_episode

    eval_cfg = load_yaml(str(ROOT / "env_cfg" / (args.env_cfg + ".yml")))
    cfg = OmegaConf.create(
        {
            key: load_yaml(str(ROOT / "env_cfg" / key / (eval_cfg["config"][key] + ".yml")))
            for key in ("sim", "scene", "camera", "robot")
        }
    )
    if args.disable_nslv2:
        cfg.camera.structured_light.nsl_v2.enabled = False
    task_name, task_class = load_task_class(args.task)
    cfg.tactile = eval_cfg.get("tactile", {})
    cfg.task_env = load_yaml(str(ROOT / "task/RoboDojo/config" / (task_name + ".yml")))
    layout, layout_provenance = load_task_layout(
        ROOT,
        task_name,
        layout_config=args.layout_config,
        layout_id=args.layout_id,
        layout_file=args.layout_file,
        eval_seed=args.eval_seed,
    )
    cfg.saved_layouts = [layout]
    replay_layouts = [
        load_task_layout(
            ROOT, task_name, layout_config=args.layout_config, layout_id=layout_id, eval_seed=args.eval_seed
        )
        for layout_id in args.replay_layout_ids
    ]
    if args.probe_mass_variant:
        variant_layout = deepcopy(layout)
        category, instances = next(iter(variant_layout.get("Rigid", {}).items()), (None, []))
        if not instances:
            raise ValueError("Mass variant probe requires a rigid object in the canonical layout")
        physics = instances[0].setdefault("physics", {})
        source_mass = min(float(physics.get("mass", 0.5)), 0.5)
        physics["mass"] = source_mass * 0.5
        replay_layouts.append(
            (variant_layout, {**layout_provenance, "probe": "native_mass_variant", "variant_mass_kg": physics["mass"]})
        )
    inventory_audit = None
    if replay_layouts:
        from utils.replay_inventory import build_retained_inventory

        inventory, inventory_audit = build_retained_inventory([layout, *[row for row, _ in replay_layouts]])
        cfg.retained_layout_inventory = inventory
    runtime_profile = configure_task_runtime(
        cfg, task_name, args.env_cfg, eval_cfg, layout_provenance, enable_planner=args.enable_planner
    )
    original_cfg = deepcopy(cfg)
    env = task_class(cfg, app, seed=runtime_profile["episode_seed"])
    reset_task_episode(env, runtime_profile)

    def native_rigid_properties():
        properties = {}
        inactive = env.scene_manager._inactive_scene_object_names[0]
        for name, body in env.scene_manager._rigid_and_dynamic_objects[0].items():
            if name in inactive or not hasattr(body, "_rigid_prim_view"):
                continue
            view = body._rigid_prim_view
            assert view.is_physics_handle_valid(), name
            # Query the actual PhysX tensor view, avoiding the legacy getter's
            # authored-USD fallback when native handles are unavailable.
            mass = view._physics_view.get_masses().reshape(-1)
            actual_mass = float(mass.detach().cpu().numpy()[0])
            expected_mass = min(float(body.instance_config.get("physics", {}).get("mass", 0.5)), 0.5)
            np.testing.assert_allclose(actual_mass, expected_mass, rtol=1e-5, atol=1e-7)
            properties[name] = {"mass_kg": actual_mass, "expected_mass_kg": expected_mass, "prim_path": body._prim_path}
        return properties

    initial_properties = native_rigid_properties() if replay_layouts else None
    reset_evidence = None
    if args.check_reset:
        from reproduction.isaac61.reset_probe import probe_motion_and_reset

        reset_evidence = probe_motion_and_reset(env)
    obs_cfg = eval_cfg["observation"]
    obs_cfg["vision"].update(depth=True, intrinsic_matrix=True, extrinsic_matrix=True)
    obs = ObsManager(obs_cfg, 1, env.dt, args.task, {}, seeds_per_env=env.env_seed_list)
    obs.initialize(env)
    for _ in range(10):
        env.sim_step(render=False)
    from reproduction.isaac61.task_runtime_probe import validate_task_runtime

    task_runtime = validate_task_runtime(env)
    scene_evidence = verify_task_scene(env.scene_manager, layout)
    obs.render_for_capture()
    data = obs.get_obs()
    shapes = {}
    for name, camera in data[0]["vision"].items():
        # The policy observation API retains its historical RGB alias.
        rgb = camera.get("rgb", camera.get("color"))
        assert rgb is not None, (name, list(camera))
        assert rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[-1] == 3, (name, rgb.shape, rgb.dtype)
        depth = camera.get("depth", camera.get("depth_gt"))
        assert depth is not None, (name, camera.keys())
        assert depth.shape == rgb.shape[:2], (name, depth.shape, rgb.shape)
        assert np.any(np.isfinite(depth) & (depth > 0)), name
        shapes[name] = {key: list(value.shape) for key, value in camera.items() if hasattr(value, "shape")}
    assert set(shapes) == {"head", "left_wrist", "right_wrist"}, shapes
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
        "checks": [
            "task scene initialization",
            "robot reset",
            "task instruction generation",
            "task reward evaluation",
            "RGBD observation capture",
        ],
        "task_runtime": task_runtime,
        "runtime_profile": runtime_profile,
        "camera_shapes": shapes,
        "policy_evaluation": False,
        "planner_initialized": args.enable_planner,
        "nslv2_inference_enabled": not args.disable_nslv2,
        "layout": layout_provenance,
        "scene": scene_evidence,
        "motion_and_reset": reset_evidence,
        "native_rigid_properties": initial_properties,
    }
    result["retained_inventory"] = inventory_audit
    result["replayed_layouts"] = []
    for replay_layout, replay_provenance in replay_layouts:
        env.scene_manager.layout_manager.set_saved_layout(0, replay_layout)
        replay_profile = episode_profile(runtime_profile, replay_provenance)
        reset_task_episode(env, replay_profile)
        obs.initialize(env)
        obs.reset()
        for _ in range(10):
            env.sim_step(render=False)
        replay_runtime = validate_task_runtime(env)
        replay_scene = verify_task_scene(env.scene_manager, replay_layout)
        obs.render_for_capture()
        replay_frame = obs.get_obs()[0]
        for name, camera in replay_frame["vision"].items():
            rgb = camera.get("rgb", camera.get("color"))
            assert rgb is not None and rgb.any(), (replay_provenance, name)
            assert np.any(np.isfinite(camera["depth"]) & (camera["depth"] > 0)), (replay_provenance, name)
        result["replayed_layouts"].append(
            {
                "layout": replay_provenance,
                "scene": replay_scene,
                "task_runtime": replay_runtime,
                "runtime_profile": replay_profile,
                "native_rigid_properties": native_rigid_properties(),
            }
        )
    stage_ids = [str(env.sim.sim.stage.GetRootLayer().identifier)]
    repeated_scenes = [scene_evidence]
    env.close()
    for repeat in range(1, args.repeat_envs):
        env = task_class(deepcopy(original_cfg), app, seed=runtime_profile["episode_seed"])
        reset_task_episode(env, runtime_profile)
        validate_task_runtime(env)
        repeated_scenes.append(verify_task_scene(env.scene_manager, layout))
        repeated_obs = ObsManager(deepcopy(obs_cfg), 1, env.dt, args.task, {}, seeds_per_env=env.env_seed_list)
        repeated_obs.initialize(env)
        repeated_obs.render_for_capture()
        repeated_frame = repeated_obs.get_obs()[0]
        assert set(repeated_frame["vision"]) == {"cam_head", "cam_left_wrist", "cam_right_wrist"}
        for name, camera in repeated_frame["vision"].items():
            rgb = camera.get("rgb", camera.get("color"))
            assert rgb is not None and rgb.any(), (repeat, name)
            assert np.any(np.isfinite(camera["depth"]) & (camera["depth"] > 0)), (repeat, name)
        stage_ids.append(str(env.sim.sim.stage.GetRootLayer().identifier))
        assert len(set(stage_ids)) == len(stage_ids), "Sequential tasks reused the closed USD stage"
        env.close()
    result["sequential_environments"] = {
        "count": args.repeat_envs,
        "distinct_stages": stage_ids,
        "scenes": repeated_scenes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    success = True
except Exception:
    import traceback

    failure = {"status": "failed", "real_runtime": True, "task": args.task, "traceback": traceback.format_exc()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(failure, indent=2) + "\n")
    print(json.dumps(failure), flush=True)
    traceback.print_exc()
finally:
    app.close(wait_for_replicator=False, skip_cleanup=True, exit_code=0 if success else 1)
