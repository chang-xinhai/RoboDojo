"""Render original component HDF5 TacSL frames as PNG and continuous MP4.

Only raw RGB and declared metric arrays are displayed. Color mapping has fixed
units and limits for every frame; RGB is never recolored or given fake markers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from reproduction.visualize_multimodal import letterbox, metric_color
from utils.multimodal_hdf5 import TACTILE_COMPONENT_SCHEMA, HDF5Reader

PADS = tuple((side, finger) for side in ("left", "right") for finger in ("finger_0", "finger_1"))


def mosaic(frame, *, delta_limit=0.002, force_limit=0.5, shear_limit=0.2, pad_normal=False):
    """Four mounted pads: raw RGB, signed optical proxy and signed model forces."""
    width, header, rgb_height, field_height = 280, 76, 382, 250
    canvas = Image.new("RGB", (4 * width, header + rgb_height + 3 * field_height), "#101827")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("DejaVuSans.ttf", 13)
    phase = frame["contact_demo"]["last_command"]["phase"]
    elapsed = frame["contact_demo"]["last_command"]["elapsed_time_s"]
    draw.text(
        (12, 9), f"Four mounted X5 TacSL pads | {phase} | contact sequence {elapsed:.3f} s", font=font, fill="white"
    )
    draw.text(
        (12, 30),
        f"Native physics frame {frame['sync']['physics_frame']} | "
        f"simulation {frame['sync']['timestamp_ns'] / 1e9:.6f} s",
        font=font,
        fill="#aec9da",
    )
    draw.text(
        (12, 51),
        "Kinematic calibration | optical change is not FEM deformation | "
        "per-taxel penalty force is not load-cell force",
        font=font,
        fill="#aec9da",
    )
    for column, (side, finger) in enumerate(PADS):
        data = frame["tactile"][side][finger]
        x, y = column * width, header
        draw.text((x + 8, y + 4), f"{side}.{finger} | original Taxim RGB", font=font, fill="white")
        canvas.paste(letterbox(data["rgb"][..., :3], (width, rgb_height - 32)), (x, y + 27))
        y += rgb_height
        fields = (
            ("optical_depth_delta", -delta_limit, delta_limit, "m signed optical proxy"),
            (
                "pad_normal_force" if pad_normal else "normal_force",
                -force_limit,
                force_limit,
                "N outward -Y / taxel" if pad_normal else "N official taxel Z / taxel",
            ),
            ("shear_force", 0.0, shear_limit, "N shear magnitude / taxel"),
        )
        for field, low, high, units in fields:
            values = np.asarray(data[field])
            if field == "shear_force":
                values = np.linalg.norm(values, axis=-1)
            if field in ("normal_force", "pad_normal_force", "shear_force"):
                shape = tuple(data["metadata"]["array_size"])
                if data["metadata"].get("taxel_array_axes") != ["sensor_local_x", "sensor_local_z"]:
                    raise ValueError("Cannot invent an undeclared taxel grid orientation")
                if values.size != np.prod(shape):
                    raise ValueError("Force samples do not match declared taxel grid")
                values = values.reshape(shape).T
            draw.text((x + 8, y + 3), field, font=font, fill="white")
            canvas.paste(letterbox(metric_color(values, low, high), (width, field_height - 66)), (x, y + 25))
            gradient = np.linspace(low, high, width - 16)[None, :].repeat(6, 0)
            canvas.paste(Image.fromarray(metric_color(gradient, low, high)), (x + 8, y + field_height - 37))
            draw.text((x + 8, y + field_height - 27), f"{low:g} .. {high:g} {units}", font=font, fill="#aec9da")
            y += field_height
    return canvas


def render(
    path, output=None, *, delta_limit=0.002, force_limit=0.5, shear_limit=0.2, pad_normal=False, make_video=True
):
    from utils.save_file import VideoStreamWriter

    path = Path(path).resolve()
    directory = Path(output).resolve() if output else path.parent
    directory.mkdir(parents=True, exist_ok=True)
    if min(delta_limit, force_limit, shear_limit) <= 0:
        raise ValueError("Display limits must be positive")
    with HDF5Reader(path) as reader:
        metadata = reader.metadata
        if metadata["observation_schema"] != TACTILE_COMPONENT_SCHEMA or metadata["status"] != "complete":
            raise ValueError("Expected completed tactile component HDF5")
        if not len(reader):
            raise ValueError("Cannot visualize an empty contact recording")

        def strength(frame):
            # Raw measured positive depth support, no synthetic force ranking.
            return sum(
                frame["contact_metrics"][side][finger]["largest_component_pixel_fraction"] for side, finger in PADS
            )

        selected, best = 0, -1.0
        for index in range(len(reader)):
            value = strength(reader.read_frame(index))
            if value > best:
                selected, best = index, value
        options = {
            "delta_limit": delta_limit,
            "force_limit": force_limit,
            "shear_limit": shear_limit,
            "pad_normal": pad_normal,
        }
        selected_frame = reader.read_frame(selected)
        image = mosaic(selected_frame, **options)
        image.save(directory / "contact.png")
        video_path = directory / "contact.mp4"
        frequency = metadata["run"]["sampling_frequency_hz"]
        if make_video:
            writer = VideoStreamWriter(str(video_path), image.height, image.width, 3, fps=frequency)
            try:
                for index in range(len(reader)):
                    pixels = np.asarray(mosaic(reader.read_frame(index), **options))
                    writer.append(pixels)
                writer.close(announce=False)
            except BaseException:
                writer.abort()
                raise
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-count_frames",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=nb_read_frames",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(video_path),
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            decoded_count = int(probe.stdout.strip())
            if decoded_count != len(reader):
                raise RuntimeError(f"MP4 has {decoded_count} frames, HDF5 has {len(reader)}")
        report = {
            "status": "passed",
            "source_hdf5": str(path),
            "source_hdf5_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "observation_schema": TACTILE_COMPONENT_SCHEMA,
            "frame_count": len(reader),
            "selected_frame": selected,
            "selected_physics_frame": selected_frame["sync"]["physics_frame"],
            "selected_phase": selected_frame["contact_demo"]["last_command"]["phase"],
            "selection": "maximum sum of actual largest positive optical-depth components over four pads; first tie",
            "display_limits": options,
            "rgb_processing": "original recorded uint8 RGB, aspect-preserving display resize only",
            "force_display": "signed pad outward -Y" if pad_normal else "signed original official taxel Z",
            "force_note": "shear magnitude from original official taxel XY; fixed metric ranges, no fabricated markers",
            "contact_png": str(directory / "contact.png"),
            "contact_mp4": str(video_path) if make_video else None,
            "video_fps": frequency if make_video else None,
            "time_note": "Source frame timestamps are authoritative; MP4 uses nominal acquisition frequency",
        }
        (directory / "visualization.json").write_text(json.dumps(report, indent=2) + "\n")
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hdf5")
    parser.add_argument("--output")
    parser.add_argument("--delta-limit", type=float, default=0.002)
    parser.add_argument("--force-limit", type=float, default=0.5)
    parser.add_argument("--shear-limit", type=float, default=0.2)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--pad-normal", action="store_true", help="Display actual rotated signed outward -Y pad force")
    args = parser.parse_args()
    print(
        json.dumps(
            render(
                args.hdf5,
                args.output,
                delta_limit=args.delta_limit,
                force_limit=args.force_limit,
                shear_limit=args.shear_limit,
                pad_normal=args.pad_normal,
                make_video=not args.no_video,
            )
        )
    )


if __name__ == "__main__":
    main()
