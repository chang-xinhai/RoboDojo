"""Validate all four ARX TacSL mounts with a physical SDF sphere contact.

The test moves a real PhysX kinematic mesh probe into each mounted compliant gel,
then checks official TacSL force, penetration and RTX/Taxim image changes. It
saves measured arrays and an image gallery; no tactile signals are synthesized.
"""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import traceback
from types import SimpleNamespace

from isaaclab.app import AppLauncher

from utils.isaaclab_compat import add_app_launcher_args, as_torch

parser = argparse.ArgumentParser()
add_app_launcher_args(parser, AppLauncher)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--output", default="reproduction/isaac61/outputs/tactile-contact")
parser.add_argument("--particle-readback", action="store_true")
args = parser.parse_args()
args.enable_cameras = True
# Match Isaac Sim zero-delay rendering; baseline/contact frames must reflect
# the current physical geometry rather than an asynchronous earlier frame.
args.kit_args = (getattr(args, "kit_args", "") or "") + (
    " --/app/hydraEngine/waitIdle=true --/app/updateOrder/checkForHydraRenderComplete=1000"
    " --/app/asyncRendering=false --/app/asyncRenderingLowLatency=false"
    " --/omni/replicator/asyncRendering=false"
)
launcher = AppLauncher(args)
app = launcher.app
exit_code = 1

try:
    import cv2
    import isaaclab
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    import isaaclab.sim as sim_utils
    from isaaclab.utils.version import get_isaac_sim_version
    from isaaclab_physx.physics import PhysxCfg
    from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update
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
    if args.particle_readback:
        from utils.particle_readback_bridge import install_particle_readback_bridge

        sim.set_setting("/physics/suppressReadback", False)
        sim.set_setting("/physics/updateToUsd", True)
        sim.set_setting("/physics/updateParticlesToUsd", True)
        sim.set_setting("/isaaclab/robodojo/particle_readback", True)
        install_particle_readback_bridge(sim.physics_manager)
    sim.set_setting("/isaaclab/render/rtx_sensors", True)
    sim.add_render_callback("robodojo_tacsl_rtx_pump", lambda _: ensure_isaac_rtx_render_update(force=True))
    light = sim_utils.DomeLightCfg(intensity=1500)
    light.func("/World/Light", light)
    scene_cfg = InteractiveSceneCfg(num_envs=args.num_envs, env_spacing=2.0, replicate_physics=False)
    for index in range(2):
        cfg = get_robot_config()
        cfg.prim_path = f"{{ENV_REGEX_NS}}/robot{index}"
        cfg.init_state.pos = (-0.3 if index == 0 else 0.3, 0.0, 0.5)
        cfg.init_state.rot = (0.0, 0.0, 0.0, 1.0)
        setattr(scene_cfg, f"robot{index}", cfg)
    scene = InteractiveScene(scene_cfg)
    robots = [scene[f"robot{index}"] for index in range(2)]

    # Author different real probe meshes after USD cloning. This exercises
    # heterogeneous env-scoped SDF paths and the upstream env_0 wildcard hazard
    # without an AssetBase default state replicating identical probe geometry.
    stage = sim_utils.get_current_stage()
    probe_paths = []
    probe_radii = []
    probe_mesh_offsets = []
    probe_scales = []
    for env_index in range(args.num_envs):
        path = f"/World/envs/env_{env_index}/probe"
        probe_paths.append(path)
        radius = 0.005 + env_index * 0.0005
        scale = 1.25 if env_index else 1.0
        probe_scales.append(scale)
        probe_radii.append(radius * scale)
        offset = (0.02 * scale if env_index else 0.0, 0.0, 0.0)
        probe_mesh_offsets.append(offset)
        root = UsdGeom.Xform.Define(stage, path)
        root.AddTranslateOp().Set(Gf.Vec3d(10.0 + env_index, 0.0, 1.0))
        root.AddScaleOp().Set(Gf.Vec3d(scale))
        UsdPhysics.RigidBodyAPI.Apply(root.GetPrim()).CreateKinematicEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(0.02)
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=radius)
        mesh = UsdGeom.Mesh.Define(stage, path + "/mesh")
        mesh.AddTranslateOp().Set(Gf.Vec3d(0.02 if env_index else 0.0, 0.0, 0.0))
        mesh.AddRotateZOp().Set(45 if env_index else 0)
        mesh.CreatePointsAttr(sphere.vertices.tolist())
        mesh.CreateFaceVertexCountsAttr([3] * len(sphere.faces))
        mesh.CreateFaceVertexIndicesAttr(sphere.faces.flatten().tolist())
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    manager = TactileManager(
        num_envs=args.num_envs,
        config={"enabled": True, "array_size": [12, 16], "normal_stiffness": 1500.0},
        robot_manager=SimpleNamespace(
            robot_list=[SimpleNamespace(robot_name="x5", type="target", arm_index=i) for i in range(2)]
        ),
        device="cuda:0",
    )
    manager.initialize(
        SimpleNamespace(sim=sim),
        SimpleNamespace(
            _rigid_and_dynamic_objects=[{"probe": SimpleNamespace(prim_path=path)} for path in probe_paths],
            _articulation_objects=[{} for _ in probe_paths],
        ),
    )
    sim.reset()
    sim.set_setting("/rtx/sdg/force/disableColorRender", False)
    dt = sim.get_physics_dt()
    import warp as wp

    physics_view = sim.physics_manager.get_physics_sim_view()
    probe_views = [physics_view.create_rigid_body_view([path]) for path in probe_paths]
    probe_indices = wp.array([0], dtype=wp.int32, device="cuda:0")
    sdf_validation = []
    for env_index in range(args.num_envs):
        sdf = manager.sensors["left"]["finger_0"][env_index][0]._contact_object_sdf_view
        shape_points = torch.zeros((1, 192, 3), device="cuda")
        shape_points[0, :, 1] = probe_radii[env_index] + 0.001
        shape_points[0, 1, 1] = probe_radii[env_index] - 0.001
        body_points = shape_points @ sdf.rotation + sdf.translation
        samples = wp.to_torch(sdf.get_sdf_and_gradients(wp.from_torch(body_points)))
        distances = samples[0, :2, -1]
        expected = torch.tensor([0.001, -0.001], device="cuda")
        sdf_validation.append({"env_id": env_index, "distance_m": distances.cpu().tolist()})
        assert torch.allclose(distances, expected, atol=0.0002), (
            f"SDF metric/shape frame mismatch in env{env_index}: {distances.tolist()}"
        )

    def step(count):
        for _ in range(count):
            for robot in robots:
                robot.write_data_to_sim()
            sim.step()
            for robot in robots:
                robot.update(dt)

    def move_probe(env_index, position):
        body_position = [float(p) - o for p, o in zip(position, probe_mesh_offsets[env_index], strict=True)]
        pose = torch.tensor([[*body_position, 0.0, 0.0, 0.0, 1.0]], device="cuda", dtype=torch.float32)
        probe_views[env_index].set_transforms(wp.from_torch(pose), indices=probe_indices)
        step(8)

    step(20)
    manager.calibrate()
    baseline = manager.get_observations()
    results = {}
    rows = []
    for env_index in range(args.num_envs):
        for side in ("left", "right"):
            for finger in ("finger_0", "finger_1"):
                primary = manager.sensors[side][finger][env_index][0]
                primary._update_force_field(slice(None))
                points = primary.data.tactile_points_pos_w[0]
                centre = points.mean(dim=0)
                # Place sphere centre 4 mm outward from the gel face so a 5 mm
                # radius sphere produces approximately 1 mm physical indentation.
                local_outward = torch.tensor([[0.0, -1.0, 0.0]], device="cuda")
                from isaaclab.utils.math import quat_apply

                transform = wp.to_torch(primary._elastomer_body_view.get_transforms())
                outward = quat_apply(transform[:, 3:], local_outward)[0]
                move_probe(env_index, (centre + (probe_radii[env_index] - 0.001) * outward).cpu().tolist())
                measured = manager.get_observations()[side][finger]
                for other in range(args.num_envs):
                    if other != env_index:
                        assert measured["normal_force"][other].max().item() < 1e-5, (
                            "Cross-environment tactile force leakage"
                        )
                force = measured["normal_force"][env_index]
                penetration = measured["penetration_depth"][env_index]
                depth = measured["depth"][env_index]
                rgb = measured["rgb"][env_index]
                before = baseline[side][finger]["rgb"][env_index]
                before_depth = baseline[side][finger]["depth"][env_index]
                delta = (rgb.float() - before.float()).abs().mean().item()
                label = f"env{env_index}_{side}_{finger}"
                # Preserve genuine failed captures for optical/contact diagnosis.
                # These files alone do not certify acceptance; result.json does.
                arrays = {
                    k: as_torch(v)[env_index].detach().cpu().numpy() for k, v in measured.items() if hasattr(v, "shape")
                }
                arrays["baseline_depth"] = before_depth.cpu().numpy()
                arrays["baseline_rgb"] = before.cpu().numpy()
                np.savez_compressed(output / f"{label}.npz", **arrays)
                print(
                    json.dumps(
                        {
                            "finger": label,
                            "force_max_N": force.max().item(),
                            "penetration_max_m": penetration.max().item(),
                            "depth_min_m": depth.min().item(),
                            "depth_max_m": depth.max().item(),
                            "depth_mean_change_m": (depth - before_depth).abs().mean().item(),
                            "rgb_mean_change": delta,
                        }
                    ),
                    flush=True,
                )
                assert torch.isfinite(force).all() and torch.isfinite(penetration).all()
                assert force.max().item() > 0.01, f"{side}.{finger}: no official SDF force"
                assert penetration.max().item() > 0.0001, f"{side}.{finger}: no gel indentation"
                assert torch.isfinite(depth).all(), f"{side}.{finger}: non-finite RTX depth"
                assert delta > 0.001, f"{side}.{finger}: tactile RGB did not change during contact"
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
                move_probe(env_index, [10.0 + env_index, 0.0, 1.0])
    cv2.imwrite(str(output / "gallery.png"), cv2.cvtColor(np.concatenate(rows, axis=0), cv2.COLOR_RGB2BGR))
    result = {
        "status": "passed",
        "num_envs": args.num_envs,
        "isaac_sim_version": str(get_isaac_sim_version()),
        "isaac_sim_distribution_version": version("isaacsim"),
        "isaac_lab_distribution_version": version("isaaclab"),
        "isaac_lab_source_path": str(Path(isaaclab.__file__).resolve()),
        "contact_object_paths": probe_paths,
        "contact_mesh_body_offsets_m": probe_mesh_offsets,
        "contact_body_uniform_scales": probe_scales,
        "sdf_metric_query_validation": sdf_validation,
        "physics": "PhysX real kinematic mesh contact",
        "contact_query_backend": manager.sensors["left"]["finger_0"][0][0].contact_query_backend,
        "particle_readback": args.particle_readback,
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
