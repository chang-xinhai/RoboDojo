#!/usr/bin/env python3
"""Verify real surface cloth and PBD fluid motion/reset on Isaac Sim 6.1 GPU."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import sys
import tempfile
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/validation/scene_physics"))
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--skip-fluid", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    for stale in ("report.json", "cloth_motion.npz", "validated_scene.usda"):
        (args.output / stale).unlink(missing_ok=True)
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "fast_shutdown": True})
    exit_code = 0
    try:
        from isaacsim.core.api import SimulationContext
        from isaacsim.core.utils.stage import get_current_stage
        import numpy as np
        from omni.physx.scripts import physicsUtils
        from pxr import Gf, Usd, UsdGeom
        import torch

        from env.scene_manager.objects.fluid import FluidObject
        from env.scene_manager.objects.garment import GarmentObject
        from env.scene_manager.objects.primitives import PlaneMesh

        sim = SimulationContext(
            physics_dt=1 / 120, rendering_dt=1 / 60, stage_units_in_meters=1, backend="torch", device="cuda:0"
        )
        sim.get_physics_context().enable_gpu_dynamics(True)
        sim.get_physics_context().set_broadphase_type("GPU")
        stage = get_current_stage()
        physicsUtils.add_ground_plane(stage, "/World/Ground", "Z", 10, Gf.Vec3f(0), Gf.Vec3f(0.5))
        path = "/World/envs/env_0/objects/cloth"
        PlaneMesh(path, Gf.Vec3f(0, 0, 0.6), dimx=12, dimy=12, scale=0.03)
        cloth = GarmentObject(
            path,
            None,
            {"physics": {"garment_config": {"mass": 0.1}}, "visual": {"color": [0.8, 0.2, 0.1]}},
            torch.zeros(3),
            (0, 0, 0.6),
            (1, 0, 0, 0),
            (1, 1, 1),
            "Plane",
        )
        # Exercise an environment whose USD parent is translated too. World-pose
        # setters must add the configured origin once, independent of hierarchy.
        UsdGeom.Xform.Define(stage, "/World/envs/env_1").AddTranslateOp().Set(Gf.Vec3d(2.0, 0.0, 0.0))
        path_1 = "/World/envs/env_1/objects/cloth"
        PlaneMesh(path_1, Gf.Vec3f(0, 0, 0.6), dimx=12, dimy=12, scale=0.03)
        cloth_1 = GarmentObject(
            path_1,
            None,
            {"physics": {"garment_config": {"mass": 0.1}}},
            torch.tensor([2.0, 0.0, 0.0]),
            (0, 0, 0.6),
            (1, 0, 0, 0),
            (1, 1, 1),
            "Plane",
        )
        fluid = None
        with tempfile.TemporaryDirectory() as directory:
            if not args.skip_fluid:
                usd_path = str(Path(directory) / "fluid.usda")
                source = Usd.Stage.CreateNew(usd_path)
                source.SetDefaultPrim(UsdGeom.Xform.Define(source, "/Fluid").GetPrim())
                mesh = UsdGeom.Mesh.Define(source, "/Fluid/model")
                mesh.GetPointsAttr().Set(
                    [Gf.Vec3f(x, y, z) for x in (-0.03, 0.03) for y in (-0.03, 0.03) for z in (-0.03, 0.03)]
                )
                source.GetRootLayer().Save()
                fluid = FluidObject(
                    "/World/envs/env_0/objects/fluid",
                    usd_path,
                    {"physics": {"particle_system": {"fluid_rest_offset": 0.01, "isosurface": False}}},
                    torch.zeros(3),
                    (0.5, 0, 0.6),
                    (1, 0, 0, 0),
                )
            sim.initialize_physics()
            sim.play()
            sim.step(render=False)
            cloth.initialize()
            before = cloth.sample_mesh_vertices()[0]
            cloth_1.initialize()
            before_1 = cloth_1.sample_mesh_vertices()[0]
            crossenv_delta = np.mean(before_1 - before, axis=0)
            np.testing.assert_allclose(crossenv_delta, [2.0, 0.0, 0.0], atol=1e-5)
            fluid_before = None
            if fluid:
                fluid.initialize()
                fluid_before = fluid.get_particle_positions()[0].copy()
            for _ in range(args.steps):
                sim.step(render=False)
            after = cloth.sample_mesh_vertices()[0]
            report = {
                "physics_model": cloth.physics_model,
                "cloth_nodes": len(before),
                "crossenv_offset_m": crossenv_delta.tolist(),
                "cloth_drop_m": float(np.mean(before[:, 2] - after[:, 2])),
                "cloth_points_finite": bool(np.isfinite(after).all()),
            }
            assert report["cloth_drop_m"] > 0.002, report
            assert report["cloth_points_finite"], report
            cloth.init_pos = (0.1, 0.1, 0.8)
            cloth.apply_saved_pose()
            restored = cloth.sample_mesh_vertices()[0]
            report["cloth_reset_max_error_m"] = float(
                np.max(np.abs(restored - cloth._world_points(cloth._rest_points)))
            )
            velocity = cloth._surface_view.get_simulation_nodal_velocities().detach().cpu().numpy()
            report["cloth_reset_max_velocity"] = float(np.max(np.abs(velocity)))
            assert report["cloth_reset_max_error_m"] < 1e-5, report
            assert report["cloth_reset_max_velocity"] < 1e-6, report
            if fluid:
                fluid_after = fluid.get_particle_positions()[0]
                report["fluid_particles"] = len(fluid_before)
                report["fluid_max_displacement_m"] = float(np.max(np.linalg.norm(fluid_after - fluid_before, axis=1)))
                assert report["fluid_max_displacement_m"] > 0.002, report
                fluid.apply_saved_pose()
                report["fluid_reset_max_error_m"] = float(
                    np.max(np.abs(fluid.get_particle_positions()[0] - fluid_before))
                )
                assert report["fluid_reset_max_error_m"] < 1e-5, report
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
            np.savez(args.output / "cloth_motion.npz", before=before, after=after, reset=restored)
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


if __name__ == "__main__":
    raise SystemExit(main())
