"""Record a real continuous nut contact calibration on all four X5 TacSL pads.

This component benchmark contains mounted sensors and prescribed native
kinematic probes. It does not claim a grasping policy or three-camera data.
Run through the pinned IsaacLab environment with --viz none.
"""

import argparse
from copy import deepcopy
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import traceback
from types import SimpleNamespace

from isaaclab.app import AppLauncher

from utils.isaaclab_compat import add_app_launcher_args

parser = argparse.ArgumentParser(description=__doc__)
add_app_launcher_args(parser, AppLauncher)
parser.add_argument("--output", default="reproduction/isaac61/outputs/tactile-nut-contact-61")
parser.add_argument("--shape", choices=("official_nut", "annulus"), default="official_nut")
parser.add_argument("--asset-scale", type=float, default=0.6)
parser.add_argument("--indentation", type=float, default=0.0006)
parser.add_argument("--phase-duration", type=float, default=0.4)
parser.add_argument("--frequency", type=float, default=25.0)
parser.add_argument("--compliance-stiffness", type=float, default=100.0)
parser.add_argument("--normal-stiffness", type=float, default=1500.0)
parser.add_argument(
    "--particle-readback",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Required native host kinematic targets and real mesh SDF queries (enabled by default)",
)
args = parser.parse_args()
if not 0 < args.frequency <= 120:
    parser.error("Frequency must be in (0,120] Hz")
args.enable_cameras = True
args.kit_args = (getattr(args, "kit_args", "") or "") + (
    " --/app/hydraEngine/waitIdle=true --/app/updateOrder/checkForHydraRenderComplete=1000"
    " --/app/asyncRendering=false --/app/asyncRenderingLowLatency=false"
    " --/omni/replicator/asyncRendering=false"
)
launcher = AppLauncher(args)
app = launcher.app
exit_code = 1
writer = None

try:
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    import isaaclab.sim as sim_utils
    from isaaclab.utils.version import get_isaac_sim_version
    from isaaclab_physx.physics import PhysxCfg
    from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update
    import numpy as np
    from PIL import Image

    from env.robot_manager.robot_config.x5 import get_robot_config
    from env.tactile_manager import TactileManager
    from env.tactile_manager.contact_metrics import contact_frame_metrics
    from utils.contact_demo import (
        PADS,
        ContactDemoConfig,
        ContactFixtures,
        ContactMotionDemo,
        close_contact_physics,
        configure_contact_physics,
    )
    from utils.isaaclab_compat import as_torch, simulation_snapshot
    from utils.multimodal import json_value
    from utils.multimodal_hdf5 import TACTILE_COMPONENT_SCHEMA, HDF5Writer, validate_hdf5
    from utils.recording_provenance import capture_provenance

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob("episode.hdf5*")) or (output / "report.json").exists():
        raise FileExistsError("Choose a fresh output directory to preserve previous evidence")
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device="cuda:0", physics=PhysxCfg()))
    if args.particle_readback:
        configure_contact_physics(sim)
    sim.set_setting("/isaaclab/render/rtx_sensors", True)
    sim.add_render_callback("robodojo_tacsl_rtx_pump", lambda _: ensure_isaac_rtx_render_update(force=True))
    light = sim_utils.DomeLightCfg(intensity=1500)
    light.func("/World/Light", light)
    scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=False)
    for index in range(2):
        cfg = get_robot_config()
        cfg.prim_path = f"{{ENV_REGEX_NS}}/robot{index}"
        cfg.init_state.pos = (-0.3 if index == 0 else 0.3, 0.0, 0.5)
        cfg.init_state.rot = (0.0, 0.0, 0.0, 1.0)
        setattr(scene_cfg, f"robot{index}", cfg)
    scene = InteractiveScene(scene_cfg)
    config = ContactDemoConfig(
        shape=args.shape,
        asset_scale=args.asset_scale,
        indentation_m=args.indentation,
        phase_duration_s=args.phase_duration,
    )
    fixtures = ContactFixtures(config, num_envs=1)
    fixtures.spawn(sim_utils.get_current_stage())
    manager = TactileManager(
        num_envs=1,
        config={
            "enabled": True,
            "array_size": [20, 25],
            "normal_stiffness": args.normal_stiffness,
            "compliance_stiffness": args.compliance_stiffness,
        },
        robot_manager=SimpleNamespace(
            robot_list=[SimpleNamespace(robot_name="x5", type="target", arm_index=i) for i in range(2)]
        ),
        device="cuda:0",
    )
    manager.initialize(
        SimpleNamespace(sim=sim),
        SimpleNamespace(
            _rigid_and_dynamic_objects=[{}],
            _articulation_objects=[{}],
            _tactile_calibration_objects=fixtures.objects,
        ),
    )
    sim.reset()
    sim.set_setting("/rtx/sdg/force/disableColorRender", False)
    dt = sim.get_physics_dt()
    # ArticulationCfg records joint defaults; a standalone scene must actually
    # apply those states and actuator targets, as TaskEnv's robot reset does.
    for robot_index in range(2):
        robot = scene[f"robot{robot_index}"]
        robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
        robot.set_joint_position_target(robot.data.default_joint_pos)
        robot.set_joint_velocity_target(robot.data.default_joint_vel)

    def step():
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt)

    for _ in range(20):
        step()
    manager.calibrate()
    motion = ContactMotionDemo(fixtures, manager, sim)
    # The no-contact targets first settle through actual physics steps.
    motion.apply(0.0, simulation_snapshot(sim)[1])
    for _ in range(6):
        step()

    def unbatch(observations):
        result = {}
        for side, finger in PADS:
            result.setdefault(side, {})[finger] = {
                key: as_torch(value)[0].detach().cpu().numpy().copy() if hasattr(value, "shape") else deepcopy(value)
                for key, value in observations[side][finger].items()
            }
        return result

    baseline = unbatch(manager.get_observations())
    diagnostics = {
        "native_probe_poses": motion.sample(),
        "native_robot_joint_positions": {
            f"robot{i}": as_torch(scene[f"robot{i}"].data.joint_pos)[0].detach().cpu().tolist() for i in range(2)
        },
        "per_pad": {},
    }
    for side, finger in PADS:
        sensors = manager.sensors[side][finger][0]
        diagnostics["per_pad"][side + "_" + finger] = {
            "points_mean_w": baseline[side][finger]["points_pos_w"].mean(axis=0).tolist(),
            "optical_depth_delta_max_m": float(baseline[side][finger]["optical_depth_delta"].max()),
            "contact_sources": [
                {
                    "path": sensor.cfg.contact_object_prim_path_expr,
                    "normal_max_N": float(sensor.data.tactile_normal_force[0].max()),
                    "penetration_max_m": float(sensor.data.penetration_depth[0].max()),
                }
                for sensor in sensors
            ],
        }
    (output / "baseline-diagnostic.json").write_text(json.dumps(json_value(diagnostics), indent=2) + "\n")
    for side, finger in PADS:
        force = baseline[side][finger]["normal_force"]
        if float(np.max(np.abs(force))) > 1e-5:
            raise RuntimeError(f"{side}.{finger}: baseline probe is already in contact")
    profile = {
        **capture_provenance(Path(__file__).resolve().parents[2]),
        "purpose": "four mounted X5 pads, prescribed kinematic calibration, not policy grasping",
        "isaac_sim": str(get_isaac_sim_version()),
        "isaac_sim_distribution": version("isaacsim"),
        "isaaclab_distribution": version("isaaclab"),
        "sampling_frequency_hz": args.frequency,
        "physics_dt_s": dt,
        "force_model": "official TacSL per-taxel penalty estimate, not calibrated load-cell force",
        "array_size": [20, 25],
        "calibration_compliant_contact_stiffness": args.compliance_stiffness,
        "normal_penalty_stiffness": args.normal_stiffness,
        "particle_readback": args.particle_readback,
        "contact_demo": fixtures.configuration(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "baseline_diagnostic": diagnostics,
        "baseline_rgb": {
            side: {finger: baseline[side][finger]["rgb"] for s, finger in PADS if s == side}
            for side in ("left", "right")
        },
    }
    writer = HDF5Writer(output, profile, observation_schema=TACTILE_COMPONENT_SCHEMA)
    records = []
    next_capture = 0.0
    best_support = -1.0
    steps = math.ceil(config.duration_s / dt)
    for index in range(steps + 1):
        elapsed = min(index * dt, config.duration_s)
        motion.apply(elapsed, simulation_snapshot(sim)[1])
        step()
        if elapsed + 1e-9 < next_capture and index != steps:
            continue
        next_capture += 1 / args.frequency
        snapshot = simulation_snapshot(sim)
        native = motion.sample()
        tactile = unbatch(manager.get_observations())
        if simulation_snapshot(sim) != snapshot:
            raise RuntimeError("Physics advanced during contact capture")
        metrics = {
            side: {
                finger: contact_frame_metrics(tactile[side][finger], baseline[side][finger]["rgb"])
                for s, finger in PADS
                if s == side
            }
            for side in ("left", "right")
        }
        frame = {
            "sync": {
                "schema": TACTILE_COMPONENT_SCHEMA,
                "physics_frame": snapshot[1],
                "timestamp_ns": round(snapshot[0] * 1e9),
                "simulation_time_s": snapshot[0],
                "physics_held_fixed": True,
            },
            "tactile": tactile,
            "contact_demo": native,
            "contact_metrics": metrics,
            "robot_joint_positions": {
                f"robot{i}": as_torch(scene[f"robot{i}"].data.joint_pos)[0].detach().cpu().numpy().copy()
                for i in range(2)
            },
        }
        writer.append(frame)
        record = {
            "index": writer.count - 1,
            "phase": native["last_command"]["phase"],
            "physics_frame": snapshot[1],
            "timestamp_ns": round(snapshot[0] * 1e9),
            "metrics": metrics,
        }
        records.append(record)
        support = sum(metrics[side][finger]["largest_component_pixel_fraction"] for side, finger in PADS)
        if support > best_support:
            best_support = support
            Image.fromarray(
                np.concatenate([tactile[side][finger]["rgb"][..., :3] for side, finger in PADS], axis=1)
            ).save(output / "live-raw-rgb.png")
            (output / "live-raw-rgb.json").write_text(json.dumps(json_value(record), indent=2) + "\n")
        print(json.dumps(json_value(record)), flush=True)

    validation = {"native_motion": motion.validate(), "pads": {}}
    for side, finger in PADS:
        pad_metrics = [record["metrics"][side][finger] for record in records]
        strongest = max(range(len(records)), key=lambda i: pad_metrics[i]["largest_component_pixel_fraction"])
        peak = pad_metrics[strongest]
        released = pad_metrics[-1]
        label = side + "_" + finger
        if max(max(abs(m["normal_force_max_N"]), abs(m["normal_force_min_N"])) for m in pad_metrics) <= 0.01:
            raise RuntimeError(f"{label}: no measured official contact force")
        if peak["largest_component_pixel_fraction"] < 0.02 or peak["rgb_mean_abs_change_0_255"] < 2:
            raise RuntimeError(f"{label}: no substantial measured optical contact")
        if max(m["closed_contour_hole_count"] for m in pad_metrics) < 1:
            raise RuntimeError(f"{label}: no measured closed nut contact contour")
        if released["contact_pixel_fraction"] > max(0.01, peak["contact_pixel_fraction"] * 0.2):
            raise RuntimeError(f"{label}: optical contact did not release")
        if max(abs(released["normal_force_max_N"]), abs(released["normal_force_min_N"])) > 1e-5:
            raise RuntimeError(f"{label}: official force did not release")
        if max(abs(m["pad_normal_force_min_N"]) for m in pad_metrics) <= 0.01:
            raise RuntimeError(f"{label}: no actual inward pad-force projection")
        if max(abs(released["pad_normal_force_max_N"]), abs(released["pad_normal_force_min_N"])) > 1e-5:
            raise RuntimeError(f"{label}: pad force did not release")
        validation["pads"][label] = {"selected_frame": strongest, "peak": peak, "released": released}
    writer.seal()
    motion.close()
    manager.destroy()
    close_contact_physics(sim)
    path = writer.complete(validation)
    validate_hdf5(path)
    report = {
        "status": "passed",
        "real_runtime": True,
        "profile": json_value({key: value for key, value in profile.items() if key != "baseline_rgb"}),
        "frame_count": len(records),
        "frames": records,
        "validation": validation,
        "source_hdf5": str(path),
        "source_hdf5_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (output / "report.json").write_text(json.dumps(json_value(report), indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "passed", "source_hdf5": str(path), "frames": len(records)}), flush=True)
    exit_code = 0
except BaseException as error:
    traceback.print_exc()
    if writer is not None:
        writer.fail(error)
finally:
    app.close(wait_for_replicator=False, skip_cleanup=True, exit_code=exit_code)
