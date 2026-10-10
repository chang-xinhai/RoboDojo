"""Validate all four ARX TacSL mounts with a physical SDF sphere contact.

The test moves a real PhysX kinematic mesh probe into each mounted compliant gel,
then checks official TacSL force, penetration and RTX/Taxim image changes. It
saves measured arrays and an image gallery; no tactile signals are synthesized.
"""

import argparse
import json
from pathlib import Path
import traceback
from types import SimpleNamespace

from isaaclab.app import AppLauncher

from utils.isaaclab_compat import add_app_launcher_args, as_torch

parser = argparse.ArgumentParser()
add_app_launcher_args(parser, AppLauncher)
parser.add_argument("--output", default="reproduction/isaac61/outputs/tactile-contact")
args = parser.parse_args()
args.enable_cameras = True
launcher = AppLauncher(args)
app = launcher.app
exit_code = 1

try:
    import cv2
    from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    import isaaclab.sim as sim_utils
    from isaaclab_physx.physics import PhysxCfg
    import numpy as np
    from pxr import Gf, UsdGeom, UsdPhysics
    import torch
    import trimesh

    from env.robot_manager.robot_config.x5 import get_robot_config
    from env.tactile_manager import TactileManager

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").unlink(missing_ok=True)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device="cuda:0", physics=PhysxCfg()))
    scene = InteractiveScene(InteractiveSceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=False))
    light = sim_utils.DomeLightCfg(intensity=1500)
    light.func("/World/Light", light)
    robots = []
    for index in range(2):
        cfg = get_robot_config()
        cfg.prim_path = f"/World/envs/env_0/robot{index}"
        cfg.init_state.pos = (-0.3 if index == 0 else 0.3, 0.0, 0.5)
        cfg.init_state.rot = (0.0, 0.0, 0.0, 1.0)
        robots.append(Articulation(cfg))

    # A triangulated sphere, rather than a USD Sphere primitive, is necessary
    # because PhysX SDF shape views support collision meshes.
    stage = sim_utils.get_current_stage()
    path = "/World/envs/env_0/Probe"
    root = UsdGeom.Xform.Define(stage, path)
    root.AddTranslateOp().Set(Gf.Vec3d(10.0, 0.0, 1.0))
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim()).CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(0.02)
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.005)
    mesh = UsdGeom.Mesh.Define(stage, path + "/mesh")
    mesh.CreatePointsAttr(sphere.vertices.tolist())
    mesh.CreateFaceVertexCountsAttr([3] * len(sphere.faces))
    mesh.CreateFaceVertexIndicesAttr(sphere.faces.flatten().tolist())
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    probe_cfg = RigidObjectCfg(prim_path=path, spawn=None)
    probe_cfg.init_state.pos = (10.0, 0.0, 1.0)
    probe = RigidObject(probe_cfg)
    manager = TactileManager(
        num_envs=1,
        config={"enabled": True, "array_size": [12, 16], "normal_stiffness": 1500.0},
        robot_manager=SimpleNamespace(
            robot_list=[SimpleNamespace(robot_name="x5", type="target", arm_index=i) for i in range(2)]
        ),
        device="cuda:0",
    )
    manager.initialize(
        SimpleNamespace(sim=sim),
        SimpleNamespace(
            _rigid_and_dynamic_objects=[{"probe": SimpleNamespace(prim_path=path)}], _articulation_objects=[{}]
        ),
    )
    sim.reset()
    sim.set_setting("/rtx/sdg/force/disableColorRender", False)
    dt = sim.get_physics_dt()

    def step(count):
        for _ in range(count):
            for robot in robots:
                robot.write_data_to_sim()
            sim.step()
            for robot in robots:
                robot.update(dt)
            probe.update(dt)

    def move_probe(position):
        pose = torch.tensor([[*position, 0.0, 0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32)
        probe.write_root_pose_to_sim_index(root_pose=pose)
        step(8)

    step(20)
    manager.calibrate()
    baseline = manager.get_observations()
    results = {}
    rows = []
    for side in ("left", "right"):
        for finger in ("finger_0", "finger_1"):
            primary = manager.sensors[side][finger][0]
            primary._update_force_field(slice(None))
            points = primary.data.tactile_points_pos_w[0]
            centre = points.mean(dim=0)
            # Place sphere centre 4 mm outward from the gel face so a 5 mm
            # radius sphere produces approximately 1 mm physical indentation.
            local_outward = torch.tensor([[0.0, -1.0, 0.0]], device="cuda")
            from isaaclab.utils.math import quat_apply
            import warp as wp

            transform = wp.to_torch(primary._elastomer_body_view.get_transforms())
            outward = quat_apply(transform[:, 3:], local_outward)[0]
            move_probe((centre + 0.004 * outward).cpu().tolist())
            measured = manager.get_observations()[side][finger]
            force = measured["normal_force"][0]
            penetration = measured["penetration_depth"][0]
            depth = measured["depth"][0]
            rgb = measured["rgb"][0]
            before = baseline[side][finger]["rgb"][0]
            delta = (rgb.float() - before.float()).abs().mean().item()
            assert torch.isfinite(force).all() and torch.isfinite(penetration).all()
            assert force.max().item() > 0.01, f"{side}.{finger}: no official SDF force"
            assert penetration.max().item() > 0.0001, f"{side}.{finger}: no gel indentation"
            assert torch.isfinite(depth).all(), f"{side}.{finger}: non-finite RTX depth"
            assert delta > 0.001, f"{side}.{finger}: tactile RGB did not change during contact"
            label = f"{side}_{finger}"
            arrays = {k: as_torch(v)[0].detach().cpu().numpy() for k, v in measured.items() if hasattr(v, "shape")}
            np.savez_compressed(output / f"{label}.npz", **arrays)
            before_np = before.cpu().numpy()
            after_np = rgb.cpu().numpy()
            cv2.imwrite(str(output / f"{label}.png"), cv2.cvtColor(after_np, cv2.COLOR_RGB2BGR))
            pair = np.concatenate((before_np, after_np), axis=1)
            cv2.putText(
                pair,
                label + "  no contact / physical contact",
                (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
            )
            rows.append(pair)
            results[label] = {
                "max_normal_force_N": force.max().item(),
                "max_penetration_m": penetration.max().item(),
                "rgb_mean_abs_change": delta,
                "physics_frame": measured["physics_frame"],
                "timestamp_ns": measured["timestamp_ns"],
            }
            move_probe([10.0, 0.0, 1.0])
    cv2.imwrite(str(output / "gallery.png"), cv2.cvtColor(np.concatenate(rows, axis=0), cv2.COLOR_RGB2BGR))
    result = {
        "status": "passed",
        "physics": "PhysX SDF + real kinematic mesh contact",
        "tactile": "official IsaacLab TacSL R1.5",
        "fingers": results,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    exit_code = 0
except BaseException:
    traceback.print_exc()
finally:
    app.close(wait_for_replicator=False, skip_cleanup=True, exit_code=exit_code)
