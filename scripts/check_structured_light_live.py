#!/usr/bin/env python3
"""Render three projected stereo rigs against a known-distance plane in Isaac Sim.

Run with the isolated Isaac Sim 6.1 Python environment. This checks the actual
RTX projector and triangulation independently from task/robot dependencies.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import sys
import traceback
from types import SimpleNamespace


def plane_reference_depth(shape, intrinsics, world_from_optical, tilt_deg):
    """Optical-axis depth for the known plane through world origin, tilted about Y.

    CameraParams uses raster coordinates spanning [0, width]/[0, height],
    so pixel (row, col) is sampled at (col + 0.5, row + 0.5).
    """
    import numpy as np

    k = np.asarray(intrinsics, dtype=np.float64)
    pose = np.asarray(world_from_optical, dtype=np.float64)
    rows, cols = np.indices(shape, dtype=np.float64)
    rays = np.stack(((cols + 0.5 - k[0, 2]) / k[0, 0], (rows + 0.5 - k[1, 2]) / k[1, 1], np.ones(shape)), axis=-1)
    tilt = np.deg2rad(tilt_deg)
    normal = np.array([np.sin(tilt), 0.0, np.cos(tilt)])
    denominator = np.einsum("i,ij,hwj->hw", normal, pose[:3, :3], rays)
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = -(normal @ pose[:3, 3]) / denominator
    depth[(depth <= 0) | ~np.isfinite(depth)] = np.nan
    return depth


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=root / "reproduction/isaac61/outputs/structured-light")
    parser.add_argument("--intensity", type=float, default=1500)
    parser.add_argument("--saturation-radiance", type=float, default=5500)
    parser.add_argument("--distance", type=float, default=1.0)
    parser.add_argument("--roll-deg", type=float, default=0.0)
    parser.add_argument("--plane-tilt-deg", type=float, default=0.0)
    parser.add_argument("--sensor-profile", type=Path, help="Camera YAML whose structured_light block is validated")
    parser.add_argument("--resolution", type=int, nargs=2, default=[320, 240], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--debug-render-fence", action="store_true", help="Save native render delivery diagnostics")
    args = parser.parse_args()
    if not -45 < args.plane_tilt_deg < 45:
        parser.error("--plane-tilt-deg must be between -45 and 45 degrees")
    sys.path.insert(0, str(root))
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    from isaacsim import SimulationApp

    from env.camera_manager.capture.render_sync import zero_delay_kit_args

    app = SimulationApp(
        {
            "headless": True,
            "fast_shutdown": True,
            "renderer": "RayTracedLighting",
            "extra_args": shlex.split(os.environ.get("ROBODOJO_KIT_ARGS", ""))
            + zero_delay_kit_args().split()
            + ["--/rtx/post/aa/op=0", "--/rtx-defaults/post/aa/op=0"],
            "width": 960,
            "height": 720,
        }
    )
    capture = None
    exit_code = 0
    try:
        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("isaacsim.sensors.experimental.rtx")
        from isaacsim.core.api import SimulationContext
        from isaacsim.core.utils.stage import get_current_stage
        from isaacsim.sensors.camera import Camera
        import numpy as np
        import omni.replicator.core as rep
        from PIL import Image, ImageDraw
        from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

        from env.camera_manager.structured_light import StructuredLightCapture

        rep.settings.set_render_rtx_realtime(antialiasing="Off")
        sim = SimulationContext(physics_dt=1 / 60, rendering_dt=1 / 60, stage_units_in_meters=1)
        stage = get_current_stage()
        # Rotate the target surface around its center at world z=0, rather
        # than changing only the camera orientation. Offset thickness along
        # the rotated normal so the visible surface center remains fixed.
        target = UsdGeom.Xform.Define(stage, "/World/target")
        half_tilt = np.deg2rad(args.plane_tilt_deg) / 2
        UsdGeom.Xformable(target).AddOrientOp().Set(
            Gf.Quatf(float(np.cos(half_tilt)), 0.0, float(np.sin(half_tilt)), 0.0)
        )
        plane = UsdGeom.Cube.Define(stage, "/World/target/surface")
        plane.CreateSizeAttr(1.0)
        xform = UsdGeom.Xformable(plane)
        xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.025))
        xform.AddScaleOp().Set(Gf.Vec3d(4, 4, 0.05))
        material = UsdShade.Material.Define(stage, "/World/target_material")
        shader = UsdShade.Shader.Define(stage, "/World/target_material/shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.55, 0.55, 0.55))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(plane.GetPrim()).Bind(material)
        dome = UsdLux.DomeLight.Define(stage, "/World/ambient")
        dome.CreateIntensityAttr(300)

        cameras = []
        names = ["cam_head", "cam_left_wrist", "cam_right_wrist"]
        for name, x in zip(names, (0.0, -0.2, 0.2)):
            camera = Camera(f"/World/{name}", resolution=tuple(args.resolution))
            half_angle = np.deg2rad(args.roll_deg) / 2
            camera.set_local_pose(
                np.array([x, 0, args.distance]),
                np.array([np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)]),
                camera_axes="usd",
            )
            camera.set_focal_length(10.0)
            camera.set_horizontal_aperture(20.0, False)
            camera.set_vertical_aperture(15.0, False)
            camera.set_clipping_range(0.01, 10)
            cameras.append(camera)

        def extrinsics(cam_id, env_id):
            del env_id
            pos, quat = cameras[cam_id].get_world_pose(camera_axes="usd")
            w, x, y, z = np.asarray(quat)
            matrix = np.eye(4)
            matrix[:3, :3] = [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
            matrix[:3, 3] = np.asarray(pos)
            return matrix

        manager = SimpleNamespace(
            num_envs=1,
            cameras=[cameras],
            camera_names=[names],
            sim=SimpleNamespace(sim=sim),
            get_camera_extrinsics=extrinsics,
        )
        config = dict(
            baseline_m=0.05,
            projector_intensity=args.intensity,
            saturation_radiance=args.saturation_radiance,
            settle_render_passes=5,
            max_render_passes=32,
            max_disparity=64,
            asset_cache=str(args.output / "assets"),
        )
        if args.sensor_profile:
            import yaml

            config.update(yaml.safe_load(args.sensor_profile.read_text())["structured_light"])
        capture = StructuredLightCapture(manager, config)
        sim.initialize_physics()
        sim.play()
        sim.step(render=False)
        # Initial graph/material/shader compilation needs more than a phase fence.
        for _ in range(30):
            sim.render()
        if args.debug_render_fence:
            import carb.settings
            import omni.graph.core as og

            from env.camera_manager.capture.render_sync import dispatcher_time

            diagnostic = []
            for phase in ("baseline", "rgb", "ir", "restored"):
                from contextlib import nullcontext

                with capture._phase_scope(phase) if phase in {"rgb", "ir"} else nullcontext():
                    for render_id in range(8):
                        sim.render()
                        node = og.get_node_by_path("/Render/PostProcess/SDGPipeline/PostProcessDispatcher")
                        record = {
                            "phase": phase,
                            "render_id": render_id,
                            "dispatcher_time": dispatcher_time(),
                            "physics_time": sim.current_time,
                            "app_update_number": app._app.get_update_number(),
                            "settings": {
                                key: carb.settings.get_settings().get(key)
                                for key in (
                                    "/rtx/post/aa/op",
                                    "/rtx-defaults/post/aa/op",
                                    "/rtx-transient/post/aa/limitedOps",
                                    "/rtx/rendermode",
                                    "/rtx/sdg/force/disableColorRender",
                                )
                            },
                            "dispatcher_products": list(node.get_attribute("outputs:renderProductPaths").get()),
                        }
                        for name, rig in capture.rigs[0].items():
                            image = capture._array(rig["rgb"].get_data("rgb"), rig["shape"], 3)
                            record[name] = {"sum": int(image.sum()), "std": float(image.std())}
                        diagnostic.append(record)
                        print("NATIVE_FENCE_DIAGNOSTIC " + json.dumps(record), flush=True)
            (args.output / "render-fence.json").write_text(json.dumps(diagnostic, indent=2))
        frames = capture.capture()
        report = dict(
            isaacsim_version=importlib.metadata.version("isaacsim"),
            antialiasing=config.get("rendering", {"rgb_antialiasing_op": 0, "ir_antialiasing_op": 0}),
            config=config,
            sensors={},
            physics_timestamp_ns=round(sim.current_time * 1e9),
            reference_distance_m=args.distance,
            reference_roll_deg=args.roll_deg,
            plane_tilt_deg=args.plane_tilt_deg,
            validation_thresholds=dict(
                stereo_median_relative_error=0.05,
                stereo_p95_relative_error=0.05,
                minimum_stereo_valid_fraction=0.25,
                stereo_center_abs_error_m=0.08,
                ground_truth_plane_median_abs_error_m=0.001,
                minimum_depth_spatial_range_m=0.02,
            ),
        )
        width, height = args.resolution
        montage = Image.new("RGB", (width * 4, height * 3 + 24))
        draw = ImageDraw.Draw(montage)
        for col, title in enumerate(
            ("RGB projector off", "IR left sensor response", "IR right sensor response", "SL depth green = 1 meter")
        ):
            draw.text((width * col + 8, 6), title, fill="white")
        failed = []
        for row, (name, frame) in enumerate(frames.items()):
            depth = frame["depth_sl"]
            finite = np.isfinite(depth)
            median = float(np.nanmedian(depth)) if finite.any() else None
            ir = frame["ir_left"]
            calibration = frame["calibration"]
            reference = plane_reference_depth(
                depth.shape,
                calibration["K_by_view"]["ir_left"],
                calibration["T_world_from_optical"]["ir_left"],
                args.plane_tilt_deg,
            )
            center = np.s_[
                depth.shape[0] // 2 - 8 : depth.shape[0] // 2 + 8, depth.shape[1] // 2 - 8 : depth.shape[1] // 2 + 8
            ]
            center_depth = float(np.nanmedian(depth[center])) if np.isfinite(depth[center]).any() else None
            center_reference = float(np.nanmedian(reference[center]))
            matched = finite & np.isfinite(reference)
            reference_error = (
                float(np.nanmedian(np.abs(depth[matched] - reference[matched]))) if matched.any() else None
            )
            relative_error = (
                float(np.nanmedian(np.abs(depth[matched] - reference[matched]) / reference[matched]))
                if matched.any()
                else None
            )
            p95_relative_error = (
                float(np.nanpercentile(np.abs(depth[matched] - reference[matched]) / reference[matched], 95))
                if matched.any()
                else None
            )
            p95_abs_error = (
                float(np.nanpercentile(np.abs(depth[matched] - reference[matched]), 95)) if matched.any() else None
            )
            sl_range = float(np.diff(np.nanpercentile(depth[finite], [10, 90]))[0]) if finite.any() else 0.0
            gt = frame["depth_gt"]
            gt_valid = np.isfinite(gt) & (gt > 0) & (gt < 10)
            gt_range = float(np.diff(np.nanpercentile(gt[gt_valid], [10, 90]))[0]) if gt_valid.any() else 0.0
            gt_reference = plane_reference_depth(
                gt.shape,
                calibration["K_by_view"]["rgb"],
                calibration["T_world_from_optical"]["rgb"],
                args.plane_tilt_deg,
            )
            gt_matched = gt_valid & np.isfinite(gt_reference)
            gt_reference_error = (
                float(np.nanmedian(np.abs(gt[gt_matched] - gt_reference[gt_matched]))) if gt_matched.any() else None
            )
            quality = dict(
                ir_nonzero_fraction=float((ir > 0).mean()),
                ir_std=float(ir.std()),
                ir_saturation_fraction=float((ir == 255).mean()),
                stereo_valid_fraction=float(finite.mean()),
                stereo_depth_median_m=median,
                depth_gt_median_m=float(np.nanmedian(frame["depth_gt"])),
                timestamp_ns=frame["timestamp_ns"],
                calibration=frame["calibration"],
                provenance=frame["provenance"],
                reference_center_depth_m=center_reference,
                stereo_center_depth_m=center_depth,
                stereo_plane_median_abs_error_m=reference_error,
                stereo_plane_median_relative_error=relative_error,
                stereo_plane_p95_relative_error=p95_relative_error,
                stereo_plane_p95_abs_error_m=p95_abs_error,
                stereo_center_abs_error_m=abs(center_depth - center_reference) if center_depth is not None else None,
                stereo_depth_spatial_range_p10_p90_m=sl_range,
                depth_gt_spatial_range_p10_p90_m=gt_range,
                depth_gt_plane_median_abs_error_m=gt_reference_error,
            )
            report["sensors"][name] = quality
            if (
                quality["ir_nonzero_fraction"] < 0.005
                or quality["ir_std"] < 1
                or quality["stereo_valid_fraction"] < 0.25
                or median is None
                or center_depth is None
                or abs(center_depth - center_reference) > 0.08
                or relative_error is None
                or relative_error > 0.05
                or p95_relative_error is None
                or p95_relative_error > 0.05
                or gt_reference_error is None
                or gt_reference_error > 0.001
            ):
                failed.append(name)
            if abs(args.plane_tilt_deg) > 1e-6 and (gt_range < 0.02 or sl_range < 0.02):
                failed.append(name)
            expected_optical = extrinsics(row, 0) @ np.diag([1.0, -1.0, -1.0, 1.0])
            optical = np.asarray(frame["calibration"]["T_world_from_optical"]["rgb"])
            quality["render_pose_max_abs_error"] = float(np.max(np.abs(optical - expected_optical)))
            if quality["render_pose_max_abs_error"] > 1e-5:
                failed.append(name)
            depth_color = np.zeros((*depth.shape, 3), np.uint8)
            depth_color[..., 1] = np.where(finite, np.clip(1 - np.abs(depth - 1), 0, 1) * 255, 0).astype(np.uint8)
            for col, pixels in enumerate((frame["rgb"], frame["ir_left"], frame["ir_right"], depth_color)):
                image = Image.fromarray(pixels).convert("RGB")
                montage.paste(image, (width * col, height * row + 24))
            draw.rectangle((0, height * row + 24, 145, height * row + 44), fill="black")
            draw.text((8, height * row + 28), name, fill="white")
            np.savez_compressed(
                args.output / f"{name}.npz",
                **{k: value for k, value in frame.items() if hasattr(value, "shape")},
            )
        montage.save(args.output / "structured_light.png")
        report["passed"] = not failed
        report["failed_sensors"] = failed
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        if failed:
            raise RuntimeError(f"Projected stereo validation failed: {failed}; inspect {args.output}")
    except Exception:
        # Kit's fast shutdown can terminate Python before the interpreter
        # displays an exception raised inside a finally block.
        traceback.print_exc()
        exit_code = 1
        raise
    finally:
        if capture is not None:
            capture.destroy()
        app.close(exit_code=exit_code)


if __name__ == "__main__":
    main()
