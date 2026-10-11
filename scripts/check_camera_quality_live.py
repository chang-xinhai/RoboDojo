"""Compare real AA-off single samples and DLAA eight-sample RGB at one pose.

No spatial image filter is applied. NPZ files preserve native sensor frames;
PNG and temporal differences are separate diagnostic evidence.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "XPolicyLab")]


def temporal_metrics(images):
    import numpy as np

    pixels = np.stack(images).astype(np.float32)
    differences = np.abs(np.diff(pixels, axis=0))
    return {
        "mean_rgb_dn": float(pixels.mean()),
        "temporal_mean_abs_difference_dn": float(differences.mean()),
        "temporal_median_abs_difference_dn": float(np.median(differences)),
        "temporal_p95_abs_difference_dn": float(np.percentile(differences, 95)),
    }


def render_quality_gallery(samples, metrics, output):
    """Place original native RGB samples beside their measured temporal variation."""
    from PIL import Image, ImageDraw, ImageFont

    names = list(samples["single_off"])
    height, width = samples["single_off"][names[0]][0].shape[:2]
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 18) if Path(font_path).is_file() else ImageFont.load_default()
    gallery = Image.new("RGB", (width * 2, (height + 50) * len(names)), (16, 24, 38))
    draw = ImageDraw.Draw(gallery)
    for row, name in enumerate(names):
        for column, mode in enumerate(("single_off", "dlaa_eight")):
            origin = (column * width, row * (height + 50))
            noise = metrics[mode][name]["temporal_mean_abs_difference_dn"]
            title = f"{name} | {mode} | temporal mean abs delta: {noise:.3f} DN"
            draw.text((origin[0] + 8, origin[1] + 12), title, font=font, fill="white")
            gallery.paste(Image.fromarray(samples[mode][name][0]), (origin[0], origin[1] + 50))
    gallery.save(output)


def main():
    from isaaclab.app import AppLauncher

    from utils.isaaclab_compat import add_app_launcher_args

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reproduction/isaac61/outputs/quality-iteration/frozen-rgb"
    )
    parser.add_argument("--repeats", type=int, default=3)
    add_app_launcher_args(parser, AppLauncher)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("At least two independently rendered repetitions are required")
    args.enable_cameras = True
    args.kit_args = ((args.kit_args or "") + " " + os.environ.get("ROBODOJO_KIT_ARGS", "")).strip()
    from env.camera_manager.capture.render_sync import add_zero_delay_kit_args

    add_zero_delay_kit_args(args)
    app = AppLauncher(args).app
    passed = False
    try:
        from utils.isaaclab_compat import enable_legacy_isaacsim_extensions, simulation_snapshot

        enable_legacy_isaacsim_extensions()
        import numpy as np
        from omegaconf import OmegaConf
        from PIL import Image

        from task.RoboDojo.task_registry import load_task_class
        from utils.load_file import load_yaml
        from utils.multimodal import json_value
        from utils.task_layout import load_task_layout, verify_task_scene
        from utils.task_runtime_config import configure_task_runtime, reset_task_episode

        profile = load_yaml(str(ROOT / "env_cfg/arx_x5_multimodal.yml"))
        cfg = OmegaConf.create(
            {
                key: load_yaml(str(ROOT / "env_cfg" / key / (profile["config"][key] + ".yml")))
                for key in ("sim", "scene", "camera", "robot")
            }
        )
        cfg.tactile = profile["tactile"]
        cfg.task_env = load_yaml(str(ROOT / "task/RoboDojo/config/stack_bowls.yml"))
        cfg.sim.scene.num_envs = 1
        cfg.sim.device = "cuda:0"
        cfg.sim.seed = [0]
        layout, provenance = load_task_layout(ROOT, "stack_bowls", layout_config="arx_x5", layout_id=0)
        cfg.saved_layouts = [layout]
        _, cls = load_task_class("stack_bowls")
        runtime_profile = configure_task_runtime(cfg, "stack_bowls", "arx_x5_multimodal", profile, provenance)
        env = cls(cfg, app, seed=runtime_profile["episode_seed"])
        reset_task_episode(env, runtime_profile)
        scene = verify_task_scene(env.scene_manager, layout)
        for _ in range(10):
            env.sim_step(render=False)
        fixed_state = simulation_snapshot(env.sim.sim)
        args.output.mkdir(parents=True, exist_ok=True)
        quality_cfg = OmegaConf.to_container(cfg.camera.structured_light, resolve=True)
        baseline_cfg = deepcopy(quality_cfg)
        baseline_cfg["rendering"]["rgb_antialiasing_op"] = 0
        baseline_cfg["rendering"]["rgb_samples_per_pass"] = 1
        # Keep exposure, geometry, projector and IR identical; isolate real RGB sampling.
        samples = {}
        phase_states = {}
        for mode, capture_cfg in [("single_off", baseline_cfg), ("dlaa_eight", quality_cfg)]:
            samples[mode] = {}
            for repetition in range(args.repeats):
                if env.camera_manager._structured_light_capture is not None:
                    env.camera_manager._structured_light_capture.config = capture_cfg
                else:
                    env.camera_manager.init_config.structured_light = capture_cfg
                frame = env.camera_manager.capture_multimodal()
                for name, camera in frame.items():
                    samples[mode].setdefault(name, []).append(camera["rgb"])
                    np.savez_compressed(
                        args.output / f"{mode}-{repetition}-{name}.npz",
                        **{key: value for key, value in camera.items() if hasattr(value, "shape")},
                    )
                    Image.fromarray(camera["rgb"]).save(args.output / f"{mode}-{repetition}-{name}.png")
                    phase_states[f"{mode}/{repetition}/{name}"] = camera["provenance"]["actual_render_states"]
                if simulation_snapshot(env.sim.sim) != fixed_state:
                    raise RuntimeError("Physics advanced between controlled RGB quality samples")
        metrics = {
            mode: {name: temporal_metrics(images) for name, images in cameras.items()}
            for mode, cameras in samples.items()
        }
        improved = all(
            metrics["dlaa_eight"][name]["temporal_mean_abs_difference_dn"]
            < metrics["single_off"][name]["temporal_mean_abs_difference_dn"]
            for name in samples["single_off"]
        )
        report = {
            "status": "passed" if improved else "failed",
            "same_physics_state": True,
            "physics_state": fixed_state,
            "configuration": quality_cfg,
            "metrics": metrics,
            "actual_render_states": phase_states,
            "layout": provenance,
            "scene": scene,
            "controlled_variables": "Only RGB AA operation and true render sample count differ; exposure fixed",
        }
        (args.output / "report.json").write_text(json.dumps(json_value(report), indent=2) + "\n")
        render_quality_gallery(samples, metrics, args.output / "rgb-before-after.png")
        print(json.dumps(json_value(report)), flush=True)
        if not improved:
            raise RuntimeError("Native DLAA/eight-sample mode did not reduce temporal RGB variation for every view")
        env.close()
        passed = True
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    finally:
        app.close(exit_code=0 if passed else 1, wait_for_replicator=False, skip_cleanup=True)


if __name__ == "__main__":
    main()
