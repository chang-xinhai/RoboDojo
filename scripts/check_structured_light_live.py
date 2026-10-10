#!/usr/bin/env python3
"""Render three projected stereo rigs against a known-distance plane in Isaac Sim.

Run with the isolated Isaac Sim 6.1 Python environment. This checks the actual
RTX projector and triangulation independently from task/robot dependencies.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/validation/structured_light"))
    parser.add_argument("--intensity", type=float, default=1500)
    parser.add_argument("--saturation-radiance", type=float, default=5500)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
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
            "extra_args": zero_delay_kit_args().split() + ["--/rtx/post/aa/op=0", "--/rtx-defaults/post/aa/op=0"],
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
        plane = UsdGeom.Cube.Define(stage, "/World/target")
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
            camera = Camera(f"/World/{name}", resolution=(320, 240))
            camera.set_local_pose(np.array([x, 0, 1.0]), np.array([1.0, 0.0, 0.0, 0.0]), camera_axes="usd")
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
        capture = StructuredLightCapture(manager, config)
        sim.initialize_physics()
        sim.play()
        sim.step(render=False)
        # Initial graph/material/shader compilation needs more than a phase fence.
        for _ in range(30):
            sim.render()
        frames = capture.capture()
        report = dict(
            isaacsim_version=importlib.metadata.version("isaacsim"),
            antialiasing="Off",
            config=config,
            sensors={},
            physics_timestamp_ns=round(sim.current_time * 1e9),
        )
        montage = Image.new("RGB", (320 * 4, 240 * 3 + 24))
        draw = ImageDraw.Draw(montage)
        for col, title in enumerate(
            ("RGB projector off", "IR left active radiance", "IR right active radiance", "SL depth green = 1 meter")
        ):
            draw.text((320 * col + 8, 6), title, fill="white")
        failed = []
        for row, (name, frame) in enumerate(frames.items()):
            depth = frame["depth_sl"]
            finite = np.isfinite(depth)
            median = float(np.nanmedian(depth)) if finite.any() else None
            ir = frame["ir_left"]
            quality = dict(
                ir_nonzero_fraction=float((ir > 0).mean()),
                ir_std=float(ir.std()),
                ir_saturation_fraction=float((ir == 255).mean()),
                stereo_valid_fraction=float(finite.mean()),
                stereo_depth_median_m=median,
                depth_gt_median_m=float(np.nanmedian(frame["depth_gt"])),
                timestamp_ns=frame["timestamp_ns"],
            )
            report["sensors"][name] = quality
            if (
                quality["ir_nonzero_fraction"] < 0.005
                or quality["ir_std"] < 1
                or quality["stereo_valid_fraction"] < 0.1
                or median is None
                or abs(median - 1.0) > 0.15
            ):
                failed.append(name)
            depth_color = np.zeros((*depth.shape, 3), np.uint8)
            depth_color[..., 1] = np.where(finite, np.clip(1 - np.abs(depth - 1), 0, 1) * 255, 0).astype(np.uint8)
            for col, pixels in enumerate((frame["rgb"], frame["ir_left"], frame["ir_right"], depth_color)):
                image = Image.fromarray(pixels).convert("RGB")
                montage.paste(image, (320 * col, 240 * row + 24))
            draw.rectangle((0, 240 * row + 24, 145, 240 * row + 44), fill="black")
            draw.text((8, 240 * row + 28), name, fill="white")
            np.savez_compressed(
                args.output / f"{name}.npz",
                **{k: frame[k] for k in ("rgb", "ir_left", "ir_right", "depth_sl", "depth_gt")},
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
