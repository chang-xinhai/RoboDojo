"""Validate a real pouring task with CUDA robots, PBD readback and TacSL.

Run in a fresh process and serialize GPU access with other smoke jobs. A native
PBD position perturbation probes solver motion, followed by a real saved reset.
"""

import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "XPolicyLab")]


def main():
    from isaaclab.app import AppLauncher

    from utils.isaaclab_compat import add_app_launcher_args

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="pour_liquid_into_cup")
    parser.add_argument("--layout-id", type=int, default=0)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--no-tactile", action="store_true", help="Isolate robot/PBD compatibility first.")
    parser.add_argument("--allow-isaac60-debug", action="store_true", help="Permit clearly labelled 6.0.1 debugging.")
    parser.add_argument("--output", type=Path, default=ROOT / "reproduction/isaac61/outputs/fluid-task")
    add_app_launcher_args(parser, AppLauncher)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    report_file = args.output / "report.json"
    report_file.unlink(missing_ok=True)
    args.enable_cameras = True
    from env.camera_manager.capture.render_sync import add_zero_delay_kit_args

    add_zero_delay_kit_args(args)
    app = AppLauncher(args).app
    env = None
    success = False
    result = {"status": "failed", "real_runtime": True, "task": args.task, "checks": []}
    try:
        from utils.isaaclab_compat import enable_legacy_isaacsim_extensions

        enable_legacy_isaacsim_extensions()
        import numpy as np
        from omegaconf import OmegaConf
        from omni.physx import get_physx_interface
        from PIL import Image
        import torch

        from env.observation_manager.obs_manager import ObsManager
        from task.RoboDojo.task_registry import load_task_class
        from utils.isaaclab_compat import as_torch, set_joint_target
        from utils.load_file import load_yaml
        from utils.task_layout import load_task_layout, verify_task_scene

        result["packages"] = {name: metadata.version(name) for name in ("isaacsim", "torch", "warp-lang")}
        try:
            result["packages"]["isaaclab"] = metadata.version("isaaclab")
        except metadata.PackageNotFoundError:
            result["packages"]["isaaclab"] = "source checkout (not pip installed)"
        result["isaaclab_source_commit"] = subprocess.check_output(
            ["git", "-C", str(ROOT / "third_party/IsaacLab3"), "rev-parse", "HEAD"], text=True
        ).strip()
        target_runtime = result["packages"]["isaacsim"] == "6.1.0.0"
        if not target_runtime:
            assert args.allow_isaac60_debug and result["packages"]["isaacsim"] == "6.0.1.0", result["packages"]
        result["target_runtime"] = target_runtime
        cfg_yaml = load_yaml(str(ROOT / "env_cfg/arx_x5_multimodal.yml"))
        cfg = OmegaConf.create(
            {
                key: load_yaml(str(ROOT / "env_cfg" / key / (cfg_yaml["config"][key] + ".yml")))
                for key in ("sim", "scene", "camera", "robot")
            }
        )
        cfg.task_env = load_yaml(str(ROOT / "task/RoboDojo/config" / (args.task + ".yml")))
        assert cfg.task_env.get("Fluid"), f"{args.task} is not a fluid task"
        cfg.sim.scene.num_envs = 1
        cfg.sim.device = "cuda:0"
        cfg.sim.seed = [0]
        cfg.camera.default_frequency = 25
        # Keep this focused on native particle/tensor compatibility. Full
        # structured-light quality is independently covered by the collector.
        cfg.camera.structured_light.enabled = False
        cfg.tactile = cfg_yaml["tactile"]
        cfg.tactile.enabled = not args.no_tactile
        for robot in cfg.robot.robots:
            robot.need_planner = False
        layout, provenance = load_task_layout(ROOT, args.task, layout_id=args.layout_id)
        cfg.saved_layouts = [layout]
        _, task_class = load_task_class(args.task)
        env = task_class(cfg, app)
        env.reset(seed=0)
        env.scene_manager.apply_saved_poses(list(range(env.num_envs)))
        result["layout"] = provenance
        result["scene"] = verify_task_scene(env.scene_manager, layout)
        assert not get_physx_interface().is_readback_suppressed(), "PBD task is still using suppressed GPU readback"
        assert env.sim.sim.device == "cuda:0"
        result["checks"].append("real canonical task objects and GPU dynamics with PBD host readback")
        before_joints = []
        commands = []
        seen = set()
        for articulation in env.robot_manager.robot_key:
            if id(articulation) in seen:
                continue
            seen.add(id(articulation))
            joint_pos = as_torch(articulation.data.joint_pos)
            link_pose = as_torch(articulation.data.body_link_pose_w)
            assert joint_pos.device.type == "cuda" and link_pose.device.type == "cuda", (
                joint_pos.device,
                link_pose.device,
            )
            assert torch.isfinite(joint_pos).all() and torch.isfinite(link_pose).all()
            before_joints.append(joint_pos.clone())
            target = joint_pos.clone()
            target[:, articulation.joint_names.index("joint1")] += 0.03
            set_joint_target(articulation, target)
            commands.append(articulation)
        assert len(commands) == 2, len(commands)
        fluids = list(env.scene_manager._fluid_objects[0].values())
        assert fluids, "Canonical fluid task produced no PBD objects"
        particle_before = []
        for fluid in fluids:
            positions = fluid.get_particle_positions()[0]
            assert len(positions) and np.isfinite(positions).all()
            # Move actual native particles 3 cm upward and let real gravity/PBD
            # evolve them. This catches stale USD buffers with a static bottle.
            probe = positions.copy()
            probe[:, 2] += 0.03
            fluid.set_particle_positions(probe)
            particle_before.append(probe.copy())
        for _ in range(args.steps):
            env.sim_step(render=False)
        joint_motion = [
            float(torch.max(torch.abs(as_torch(arm.data.joint_pos) - before)).item())
            for arm, before in zip(commands, before_joints, strict=True)
        ]
        assert all(value > 1e-4 for value in joint_motion), joint_motion
        result["joint_motion_rad"] = joint_motion
        result["checks"].append("both X5 CUDA articulation tensors and native joint target commands advance")
        particle_motion = []
        arrays = {}
        for index, (fluid, before) in enumerate(zip(fluids, particle_before, strict=True)):
            after = fluid.get_particle_positions()[0]
            assert after.shape == before.shape and np.isfinite(after).all()
            delta = float(np.max(np.linalg.norm(after - before, axis=1)))
            assert delta > 0.001, f"PBD solver output is stale: {delta} m"
            particle_motion.append(delta)
            arrays[f"fluid_{index}_before"] = before
            arrays[f"fluid_{index}_after"] = after.copy()
            fluid.apply_saved_pose()
            restored = fluid.get_particle_positions()[0]
            np.testing.assert_allclose(restored, fluid.init_position, atol=1e-5)
        result["fluid_motion_m"] = particle_motion
        result["checks"].append("native PBD solver motion readback and exact saved particle reset")
        # Verify the live scene after the motion probe and immediately before
        # frozen capture; an initial layout check cannot detect later blowups.
        result["scene"] = verify_task_scene(env.scene_manager, layout)

        obs_cfg = OmegaConf.to_container(OmegaConf.create(cfg_yaml["observation"]), resolve=True)
        obs_cfg["multimodal"] = False
        obs = ObsManager(obs_cfg, 1, env.dt, args.task, {}, seeds_per_env=env.env_seed_list)
        obs.initialize(env)
        obs.render_for_capture()
        frame = obs.get_obs()[0]
        assert set(frame["vision"]) == {"cam_head", "cam_left_wrist", "cam_right_wrist"}
        for name, camera in frame["vision"].items():
            rgb = camera.get("rgb", camera.get("color"))
            depth = camera["depth"]
            assert rgb is not None and rgb.any() and np.any(np.isfinite(depth) & (depth > 0))
            Image.fromarray(rgb).save(args.output / f"{name}.png")
            arrays[name + "_rgb"] = rgb
            arrays[name + "_depth"] = depth
        result["checks"].append("head and both wrist RTX RGBD frames in the readback pipeline")
        if not args.no_tactile:
            tactile = env.tactile_manager.get_observations()
            assert set(tactile) == {"left", "right"}
            result["tactile_query_backends"] = {}
            for side, fingers in tactile.items():
                assert set(fingers) == {"finger_0", "finger_1"}
                for finger, values in fingers.items():
                    assert values["provenance"]["force_field_available"], (side, finger, values["provenance"])
                    backend = values["provenance"]["contact_query_backend"]
                    assert backend == "warp_triangle_mesh_signed_distance", (side, finger, backend)
                    result["tactile_query_backends"][f"{side}/{finger}"] = backend
                    for name in ("normal_force", "shear_force", "depth", "penetration_depth"):
                        value = values[name]
                        assert torch.isfinite(value).all(), (side, finger, name)
                        arrays[f"{side}_{finger}_{name}"] = value.detach().cpu().numpy()
            result["checks"].append(
                "all four official TacSL force/depth fields use real triangle-mesh queries in host readback mode"
            )
        np.savez_compressed(args.output / "capture.npz", **arrays)
        result.update(
            status="passed" if target_runtime else "debug_passed", steps=args.steps, tactile_enabled=not args.no_tactile
        )
        env.close()
        env = None
        success = True
    except BaseException as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        report_file.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        app.close(exit_code=0 if success else 1, wait_for_replicator=False, skip_cleanup=True)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
