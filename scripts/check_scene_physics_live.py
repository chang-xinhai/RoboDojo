#!/usr/bin/env python3
"""Verify real surface cloth and PBD fluid motion/reset on Isaac Sim 6.1 GPU."""

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import shlex
import sys
import tempfile
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "reproduction/isaac61/outputs/scene-physics",
    )
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--skip-fluid", action="store_true", default=True, help="Cloth mode is the default")
    parser.add_argument("--fluid-only", action="store_true", help="Test fluids separately using CPU readback")
    parser.add_argument("--gravity-scale", type=float, default=1.0, help="Explicit surface acceleration calibration")
    parser.add_argument("--garment-asset", type=Path, help="Use a bundled garment USD and its adjacent metadata.json")
    parser.add_argument(
        "--cpu-readback", action="store_true", help="Keep GPU dynamics, enable host readback required by PBD fluids"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    for stale in (
        "report.json",
        "cloth_motion.npz",
        "fluid_motion.npz",
        "validated_scene.usda",
        "cloth_before.png",
        "cloth_after.png",
        "cloth_contact.png",
        "cloth_reset.png",
        "fluid_before.png",
        "fluid_after.png",
        "fluid_reset.png",
    ):
        (args.output / stale).unlink(missing_ok=True)
    from isaacsim import SimulationApp

    app = SimulationApp(
        {"headless": True, "fast_shutdown": True, "extra_args": shlex.split(os.environ.get("ROBODOJO_KIT_ARGS", ""))}
    )
    exit_code = 0
    try:
        from isaacsim.core.api import SimulationContext
        import isaacsim.core.utils.numpy as np_utils
        from isaacsim.core.utils.stage import get_current_stage
        from isaacsim.sensors.camera import Camera
        import numpy as np
        from omni.physx.scripts import physicsUtils
        from PIL import Image
        from pxr import Gf, UsdGeom, UsdLux
        import torch

        from env.scene_manager.objects.garment import GarmentObject
        from env.scene_manager.objects.primitives import PlaneMesh

        sim = SimulationContext(
            physics_dt=1 / 120, rendering_dt=1 / 60, stage_units_in_meters=1, backend="torch", device="cuda:0"
        )
        sim.get_physics_context().enable_gpu_dynamics(True)
        sim.get_physics_context().set_broadphase_type("GPU")
        stage = get_current_stage()
        physicsUtils.add_ground_plane(stage, "/World/Ground", "Z", 10, Gf.Vec3f(0), Gf.Vec3f(0.5))
        UsdLux.DomeLight.Define(stage, "/World/Light").CreateIntensityAttr(500)
        camera = Camera("/World/PhysicsCamera", resolution=(320, 240))
        eye = Gf.Vec3d(1.0, -1.2, 1.0)
        quaternion = (
            Gf.Matrix4d().SetLookAt(eye, Gf.Vec3d(0, 0, 0.4), Gf.Vec3d(0, 0, 1)).GetInverse().ExtractRotationQuat()
        )
        camera.set_world_pose(
            np.asarray(eye), np.asarray([quaternion.GetReal(), *quaternion.GetImaginary()]), camera_axes="usd"
        )
        camera.set_focal_length(12.0)
        camera.set_horizontal_aperture(20.0)
        if args.fluid_only:
            _validate_fluid_only(args, sim, stage, camera)
            return 0
        if args.cpu_readback:
            raise ValueError("Surface deformable tensor reads require the GPU pipeline; test fluids with --fluid-only")
        cloth_config = {
            "physics": {"garment_config": {"mass": 0.1}, "surface_material": {"gravity_scale": args.gravity_scale}},
            "visual": {"color": [0.8, 0.2, 0.1]},
        }
        usd_path, primitive_type = None, "Plane"
        if args.garment_asset:
            args.garment_asset = args.garment_asset.resolve()
            usd_path, primitive_type = str(args.garment_asset), None
            cloth_config = json.loads(args.garment_asset.with_name("metadata.json").read_text())
            cloth_config["visual"] = {"color": [0.8, 0.2, 0.1]}
        path = "/World/envs/env_0/objects/cloth"
        if primitive_type:
            PlaneMesh(path, Gf.Vec3f(0, 0, 0.6), dimx=12, dimy=12, scale=0.03)
        cloth = GarmentObject(
            path,
            usd_path,
            cloth_config,
            torch.zeros(3),
            (0, 0, 0.6),
            (1, 0, 0, 0),
            (1, 1, 1),
            primitive_type,
        )
        # Exercise an environment whose USD parent is translated too. World-pose
        # setters must add the configured origin once, independent of hierarchy.
        UsdGeom.Xform.Define(stage, "/World/envs/env_1").AddTranslateOp().Set(Gf.Vec3d(2.0, 0.0, 0.0))
        path_1 = "/World/envs/env_1/objects/cloth"
        if primitive_type:
            PlaneMesh(path_1, Gf.Vec3f(0, 0, 0.6), dimx=12, dimy=12, scale=0.03)
        cloth_1 = GarmentObject(
            path_1,
            usd_path,
            cloth_config,
            torch.tensor([2.0, 0.0, 0.0]),
            (0, 0, 0.6),
            (1, 0, 0, 0),
            (1, 1, 1),
            primitive_type,
        )
        sim.initialize_physics()
        sim.play()
        sim.step(render=False)
        cloth.initialize()
        camera.initialize()
        for _ in range(20):
            sim.render()
        rgb_before = camera.get_rgba()[..., :3].copy()
        before = cloth.sample_mesh_vertices()[0]
        cloth_1.initialize()
        before_1 = cloth_1.sample_mesh_vertices()[0]
        crossenv_delta = np.mean(before_1 - before, axis=0)
        np.testing.assert_allclose(crossenv_delta, [2.0, 0.0, 0.0], atol=1e-5)
        for _ in range(args.steps):
            sim.step(render=False)
        after = cloth.sample_mesh_vertices()[0]
        for _ in range(5):
            sim.render()
        rgb_after = camera.get_rgba()[..., :3].copy()

        def cloth_mask(rgb):
            return (rgb[..., 0] > 20) & (rgb[..., 0] > rgb[..., 1] * 1.3) & (rgb[..., 0] > rgb[..., 2] * 1.3)

        mask_before, mask_after = cloth_mask(rgb_before), cloth_mask(rgb_after)
        assert min(mask_before.sum(), mask_after.sum()) > 8, "Red FEM garment is not visible in both renders"
        mask_union = mask_before | mask_after
        report = {
            "physics_model": cloth.physics_model,
            "physics_migration": cloth.physics_migration,
            "surface_parameters": cloth.surface_parameters,
            "cloth_nodes": len(before),
            "cpu_readback_requested": args.cpu_readback,
            "crossenv_offset_m": crossenv_delta.tolist(),
            "cloth_drop_m": float(np.mean(before[:, 2] - after[:, 2])),
            "cloth_points_finite": bool(np.isfinite(after).all()),
            "cloth_rgb_mean_abs_change": float(
                np.mean(np.abs(rgb_after.astype(float) - rgb_before.astype(float))[mask_union])
            ),
        }
        assert report["cloth_drop_m"] > 0.002, report
        assert report["cloth_points_finite"], report
        assert report["cloth_rgb_mean_abs_change"] > 0.05, "Physics moved but the rendered scene did not update"
        Image.fromarray(rgb_before).save(args.output / "cloth_before.png")
        Image.fromarray(rgb_after).save(args.output / "cloth_after.png")
        # Continue into physical contact with the rigid ground. Freefall alone
        # would pass even if the surface collider had been authored incorrectly.
        gravity_scale = cloth.surface_parameters["gravity_scale"]
        if gravity_scale <= 0:
            raise ValueError("The freefall/contact live check requires a positive gravity calibration")
        contact_steps = max(
            60, int(np.ceil(np.sqrt(2 * max(0, before[:, 2].max()) / (9.81 * gravity_scale)) * 120)) + 20
        )
        for _ in range(max(0, contact_steps - args.steps)):
            sim.step(render=False)
        contact = cloth.sample_mesh_vertices()[0]
        report["cloth_ground_contact_min_z_m"] = float(contact[:, 2].min())
        assert report["cloth_ground_contact_min_z_m"] > -0.02, "Surface cloth penetrated the rigid ground"
        assert report["cloth_ground_contact_min_z_m"] < 0.03, "Surface cloth never reached ground contact"
        for _ in range(5):
            sim.render()
        Image.fromarray(camera.get_rgba()[..., :3]).save(args.output / "cloth_contact.png")
        cloth.init_pos = (0.1, 0.1, 0.8)
        cloth.apply_saved_pose()
        restored = cloth.sample_mesh_vertices()[0]
        report["cloth_reset_max_error_m"] = float(np.max(np.abs(restored - cloth._world_reset_points())))
        velocity = cloth._surface_view.get_simulation_nodal_velocities().detach().cpu().numpy()
        report["cloth_reset_max_velocity"] = float(np.max(np.abs(velocity)))
        assert report["cloth_reset_max_error_m"] < 1e-5, report
        assert report["cloth_reset_max_velocity"] < 1e-6, report
        sim.step(render=False)
        persisted = cloth.sample_mesh_vertices()[0]
        report["cloth_reset_one_step_max_error_m"] = float(np.max(np.abs(persisted - restored)))
        assert report["cloth_reset_one_step_max_error_m"] < 0.005, "Cloth reset did not persist into PhysX"
        reset_step_velocity = cloth._surface_view.get_simulation_nodal_velocities()[0].mean(dim=0).cpu().numpy()
        report["cloth_reset_one_step_mean_velocity_m_s"] = reset_step_velocity.tolist()
        # In one freefall step the additional gravity is integrated into
        # genuine nodal velocities; this checks acceleration, not USD motion.
        expected_z = -9.81 * cloth.surface_parameters["gravity_scale"] / 120
        np.testing.assert_allclose(reset_step_velocity, [0, 0, expected_z], atol=0.01)
        for _ in range(5):
            sim.render()
        rgb_reset = camera.get_rgba()[..., :3].copy()
        reset_pixels = np.argwhere(cloth_mask(rgb_reset))
        assert len(reset_pixels) > 8, "Reset cloth is missing from the rendered frame"
        # Camera's projection convenience method pads with its Torch backend
        # even when given NumPy inputs. Request the official matrices explicitly
        # in NumPy, so the check is independent of the simulation's frontend.
        intrinsics = camera.get_intrinsics_matrix(device="cpu", backend_utils_cls=np_utils)
        view = camera.get_view_matrix_ros(device="cpu", backend_utils_cls=np_utils)
        homogeneous = np.column_stack((persisted, np.ones(len(persisted))))
        projection = homogeneous @ (intrinsics @ view[:3]).T
        projected = projection[:, :2] / projection[:, 2:3]
        rendered_center = reset_pixels[:, [1, 0]].mean(axis=0)
        report["cloth_reset_render_center_px"] = rendered_center.tolist()
        report["cloth_reset_projected_bounds_px"] = [projected.min(axis=0).tolist(), projected.max(axis=0).tolist()]
        Image.fromarray(rgb_reset).save(args.output / "cloth_reset.png")
        assert np.all(rendered_center >= projected.min(axis=0) - 10), "Rendered cloth lags behind the physical reset"
        assert np.all(rendered_center <= projected.max(axis=0) + 10), "Rendered cloth lags behind the physical reset"
        report.update(
            status="passed",
            real_runtime=True,
            isaacsim_version=version("isaacsim"),
            physx_extension=__import__("omni.kit.app", fromlist=["get_app"])
            .get_app()
            .get_extension_manager()
            .get_enabled_extension_id("omni.physx"),
            device="cuda:0",
        )
        np.savez(args.output / "cloth_motion.npz", before=before, after=after, contact=contact, reset=restored)
        if not stage.Flatten().Export(str((args.output / "validated_scene.usda").resolve())):
            raise RuntimeError("Could not export the validated physics stage")
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        sim.stop()
    except Exception:
        exit_code = 1
        error = traceback.format_exc()
        print(error, file=sys.stderr, flush=True)
        (args.output / "report.json").write_text(
            json.dumps({"status": "failed", "real_runtime": True, "error": error}, indent=2) + "\n"
        )
    finally:
        # SimulationApp fast shutdown exits the process: preserve the failure code.
        app.close(exit_code=exit_code)
    return exit_code


def _validate_fluid_only(args, sim, stage, camera):
    import carb
    import numpy as np
    from omni.physx import get_physx_interface
    from PIL import Image
    from pxr import Gf, Usd, UsdGeom
    import torch

    from env.scene_manager.objects.fluid import FluidObject

    if not args.cpu_readback:
        raise ValueError("--fluid-only requires --cpu-readback for PBD particle state/render output")
    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "fluid.usda")
        source = Usd.Stage.CreateNew(path)
        source.SetDefaultPrim(UsdGeom.Xform.Define(source, "/Fluid").GetPrim())
        mesh = UsdGeom.Mesh.Define(source, "/Fluid/model")
        mesh.GetPointsAttr().Set(
            [Gf.Vec3f(x, y, z) for x in (-0.04, 0.04) for y in (-0.04, 0.04) for z in (-0.04, 0.04)]
        )
        source.GetRootLayer().Save()
        fluid = FluidObject(
            "/World/envs/env_0/objects/fluid",
            path,
            {"physics": {"particle_system": {"fluid_rest_offset": 0.01, "isosurface": False}}},
            torch.zeros(3),
            (0, 0, 0.6),
            (1, 0, 0, 0),
        )
        carb.settings.get_settings().set_bool("/physics/suppressReadback", False)
        sim.initialize_physics()
        sim.play()
        sim.step(render=False)
        fluid.initialize()
        before = fluid.get_particle_positions()[0].copy()
        camera.initialize()
        for _ in range(20):
            sim.render()
        rgb_before = camera.get_rgba()[..., :3].copy()
        for _ in range(args.steps):
            sim.step(render=False)
        after = fluid.get_particle_positions()[0].copy()
        for _ in range(5):
            sim.render()
        rgb_after = camera.get_rgba()[..., :3].copy()
        report = {
            "status": "passed",
            "real_runtime": True,
            "isaacsim_version": version("isaacsim"),
            "physx_extension": __import__("omni.kit.app", fromlist=["get_app"])
            .get_app()
            .get_extension_manager()
            .get_enabled_extension_id("omni.physx"),
            "gpu_dynamics": True,
            "cpu_readback_requested": True,
            "readback_suppressed": get_physx_interface().is_readback_suppressed(),
            "fluid_particles": len(before),
            "fluid_max_displacement_m": float(np.max(np.linalg.norm(after - before, axis=1))),
            "fluid_rgb_mean_abs_change": float(np.mean(np.abs(rgb_after.astype(float) - rgb_before.astype(float)))),
        }
        assert not report["readback_suppressed"], report
        assert report["fluid_max_displacement_m"] > 0.002, report
        assert report["fluid_rgb_mean_abs_change"] > 0.01, report
        fluid.apply_saved_pose()
        report["fluid_reset_max_error_m"] = float(np.max(np.abs(fluid.get_particle_positions()[0] - before)))
        assert report["fluid_reset_max_error_m"] < 1e-5, report
        sim.step(render=False)
        report["fluid_reset_one_step_max_error_m"] = float(np.max(np.abs(fluid.get_particle_positions()[0] - before)))
        # Fluid density constraints can legitimately move freshly reset nodes
        # by one maximum-speed step. Reject a stale pose or doubled transform.
        report["fluid_reset_max_allowed_step_m"] = 1.5 * 2.5 / 120
        assert report["fluid_reset_one_step_max_error_m"] < report["fluid_reset_max_allowed_step_m"], report
        for _ in range(5):
            sim.render()
        rgb_reset = camera.get_rgba()[..., :3].copy()
        report["fluid_reset_rgb_mean_abs_change"] = float(
            np.mean(np.abs(rgb_reset.astype(float) - rgb_after.astype(float)))
        )
        assert report["fluid_reset_rgb_mean_abs_change"] > 0.01, "PBD reset did not reach the renderer"
        np.savez(args.output / "fluid_motion.npz", before=before, after=after, reset=fluid.get_particle_positions()[0])
        Image.fromarray(rgb_before).save(args.output / "fluid_before.png")
        Image.fromarray(rgb_after).save(args.output / "fluid_after.png")
        Image.fromarray(rgb_reset).save(args.output / "fluid_reset.png")
        if not stage.Flatten().Export(str(args.output / "validated_scene.usda")):
            raise RuntimeError("Could not export the validated fluid stage")
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        sim.stop()


if __name__ == "__main__":
    raise SystemExit(main())
