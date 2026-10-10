"""Capture actual RoboDojo RGB/depth/stereo IR and bilateral TacSL frames.

Run via reproduction/isaac61/run.sh collect so simulator imports resolve.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "XPolicyLab")]
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="stack_bowls")
parser.add_argument("--env-cfg", default="arx_x5_multimodal")
parser.add_argument("--layout-config", default="arx_x5")
parser.add_argument("--layout-id", type=int, default=0)
parser.add_argument("--layout-file", type=Path, help="Override the canonical saved eval layout.")
parser.add_argument("--frames", type=int, default=5)
parser.add_argument("--warmup-steps", type=int, default=10)
parser.add_argument("--output", type=Path, default=ROOT / "reproduction" / "isaac61" / "outputs" / "multimodal-capture")
parser.add_argument("--visualize", action="store_true")
parser.add_argument(
    "--motion-demo", action="store_true", help="Command both wrist joints with a small real sine motion."
)
parser.add_argument("--motion-amplitude", type=float, default=0.03, help="Wrist amplitude in rad, at most 0.05.")
parser.add_argument("--motion-frequency", type=float, default=0.5, help="Wrist sine frequency in Hz, at most 1.")
parser.add_argument(
    "--min-sl-valid-fraction",
    type=float,
    default=0.001,
    help="Minimum measured active stereo depth coverage per camera (0 permits all invalid depth).",
)
from utils.isaaclab_compat import add_app_launcher_args

add_app_launcher_args(parser, AppLauncher)
args = parser.parse_args()
if args.frames < 1 or args.warmup_steps < 0 or not 0 <= args.min_sl_valid_fraction <= 1:
    parser.error("frames must be positive, warmup-steps nonnegative and min-sl-valid-fraction in [0,1]")
if args.motion_demo and (
    args.frames < 2 or not 0 < args.motion_amplitude <= 0.05 or not 0 < args.motion_frequency <= 1.0
):
    parser.error("motion-demo requires frames>=2, amplitude in (0,0.05] rad and frequency in (0,1] Hz")
args.enable_cameras = True
from env.camera_manager.capture.render_sync import add_zero_delay_kit_args

add_zero_delay_kit_args(args)
launcher = AppLauncher(args)
app = launcher.app
success = False
try:
    from utils.isaaclab_compat import enable_legacy_isaacsim_extensions, simulation_snapshot

    enable_legacy_isaacsim_extensions()
    import numpy as np
    from omegaconf import OmegaConf

    from env.observation_manager.obs_manager import ObsManager
    from task.RoboDojo.task_registry import load_task_class
    from utils.load_file import load_yaml
    from utils.multimodal import MultimodalWriter, json_value, write_failed_capture
    from utils.task_layout import load_task_layout, verify_task_scene

    eval_cfg = load_yaml(str(ROOT / "env_cfg" / (args.env_cfg + ".yml")))
    cfg = OmegaConf.create(
        {
            key: load_yaml(str(ROOT / "env_cfg" / key / (eval_cfg["config"][key] + ".yml")))
            for key in ("sim", "scene", "camera", "robot")
        }
    )
    cfg.tactile = eval_cfg.get("tactile", {})
    cfg.task_env = load_yaml(str(ROOT / "task/RoboDojo/config" / (args.task + ".yml")))
    cfg.sim.scene.num_envs = 1
    cfg.sim.device = "cuda:0"
    cfg.sim.seed = [0]
    cfg.camera.default_frequency = eval_cfg["observation"]["collect_freq"]
    task_name, task_class = load_task_class(args.task)
    layout, layout_provenance = load_task_layout(
        ROOT, task_name, layout_config=args.layout_config, layout_id=args.layout_id, layout_file=args.layout_file
    )
    cfg.saved_layouts = [layout]
    env = task_class(cfg, app, seed=0)
    env.reset(seed=0)
    # TaskEnv stages CUDA objects offscreen during reset and calibrates the
    # empty TacSL gel there. Place the canonical task objects/table afterward,
    # following EvalEnv.setup_scene, without recalibrating away contact.
    env.scene_manager.apply_saved_poses(env_idx_list=[0])
    scene_validation = verify_task_scene(env.scene_manager, layout)
    scene_validation["saved_poses_applied"] = True
    initial_time, initial_frame = simulation_snapshot(env.sim.sim)
    scene_validation.update(physics_frame=initial_frame, timestamp_ns=round(initial_time * 1e9))
    obs_cfg = eval_cfg["observation"]
    if not obs_cfg.get("multimodal"):
        raise ValueError("Collector requires observation.multimodal=true; use arx_x5_multimodal")
    obs = ObsManager(obs_cfg, 1, env.dt, args.task, {}, seeds_per_env=env.env_seed_list)
    obs.initialize(env)
    for _ in range(args.warmup_steps):
        env.sim_step(render=False)
    motion = None
    motion_validation = {"enabled": False}
    if args.motion_demo:
        from utils.motion_demo import WristMotionDemo

        motion = WristMotionDemo(env.robot_manager, args.motion_amplitude, args.motion_frequency)
        motion_start_time, _ = simulation_snapshot(env.sim.sim)
    writer = MultimodalWriter(
        args.output,
        {
            "task": args.task,
            "env_cfg": args.env_cfg,
            "seed": 0,
            "requested_collect_frequency_hz": obs.collect_freq,
            "layout": layout_provenance,
            "scene": scene_validation,
            "motion_demo": motion.configuration() if motion else {"enabled": False},
        },
    )
    reports = []
    for index in range(args.frames):
        # Integer stepping preserves requested simulated collection frequency.
        for _ in range(int(obs.collect_interval)):
            if motion:
                current_time, current_frame = simulation_snapshot(env.sim.sim)
                motion.apply(current_time + env.dt - motion_start_time, current_frame)
            env.sim_step(render=False)
        frame = obs.get_obs()[0]
        if motion:
            frame["action"]["motion_demo"] = motion.sample()
        quality = {}
        for name, camera in frame["vision"].items():
            valid = np.isfinite(camera["depth_sl"]) & (camera["depth_sl"] > 0)
            fraction = float(valid.mean())
            failure = None
            if fraction < args.min_sl_valid_fraction:
                failure = (
                    f"{name}: structured-light reconstruction valid fraction {fraction:.6f} "
                    f"below required {args.min_sl_valid_fraction}"
                )
            elif not camera["ir_left"].any() or not camera["ir_right"].any():
                failure = f"{name}: projector IR capture has no intensity signal"
            if failure:
                diagnostic = write_failed_capture(frame, args.output, failure, frame_index=index)
                print(json.dumps({"status": "failed", "diagnostic": str(diagnostic), "reason": failure}), flush=True)
                raise RuntimeError(failure)
            quality[name] = {
                "structured_light_valid_fraction": fraction,
                "ir_left_nonzero_fraction": float((camera["ir_left"] > 0).mean()),
                "ir_right_nonzero_fraction": float((camera["ir_right"] > 0).mean()),
            }
        path = writer.append(frame)
        report = {
            "index": index,
            "file": str(path),
            **frame["sync"],
            "modalities": {
                name: {k: list(v.shape) for k, v in cam.items() if hasattr(v, "shape")}
                for name, cam in frame["vision"].items()
            },
            "quality": quality,
            "motion_demo": json_value(frame["action"].get("motion_demo", {"enabled": False})),
            "tactile_fingers": {side: list(fingers) for side, fingers in frame["tactile"].items()},
        }
        reports.append(report)
        print(json.dumps(report), flush=True)
    if motion:
        motion_validation = motion.validate_motion()
    # Recheck native placement against the same final physics state captured
    # by the sensors, before publishing a completed recording manifest.
    final_time, final_frame = simulation_snapshot(env.sim.sim)
    if final_frame != frame["sync"]["physics_frame"] or round(final_time * 1e9) != frame["sync"]["timestamp_ns"]:
        raise RuntimeError("Physics advanced after final sensor capture")
    final_scene_validation = verify_task_scene(env.scene_manager, layout)
    if simulation_snapshot(env.sim.sim) != (final_time, final_frame):
        raise RuntimeError("Physics advanced while validating final task placement")
    final_scene_validation.update(
        saved_poses_applied=True, physics_frame=final_frame, timestamp_ns=round(final_time * 1e9)
    )
    writer.run_metadata.update(
        scene_initial=scene_validation, scene_final=final_scene_validation, scene=final_scene_validation
    )
    writer.close()
    env.close()
    if args.visualize:
        from reproduction.visualize_multimodal import visualize

        print(json.dumps(visualize(args.output)), flush=True)
    (args.output / "validation.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "real_runtime": True,
                "frames": reports,
                "layout": layout_provenance,
                "scene": final_scene_validation,
                "scene_initial": scene_validation,
                "scene_final": final_scene_validation,
                "motion_demo": motion_validation,
            },
            indent=2,
        )
        + "\n"
    )
    success = True
except BaseException:
    # SimulationApp may exit the process directly during cleanup; print the
    # active error before close and preserve its failure status explicitly.
    import traceback

    traceback.print_exc()
    raise
finally:
    app.close(exit_code=0 if success else 1, wait_for_replicator=False, skip_cleanup=True)
