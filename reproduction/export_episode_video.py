"""Export every committed HDF5 frame as a clear RGB task video, without Isaac.

Example: python reproduction/export_episode_video.py /path/episode.hdf5
         --output /path/episode-rgb.mp4
The source timeline must have a uniform actual simulation interval. Frames are
never sampled, duplicated, normalized, or replaced with synthetic contact.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.multimodal import SCHEMA_VERSION
from utils.multimodal_hdf5 import HDF5Reader
from utils.save_file import VideoStreamWriter

PADS = tuple((side, finger) for side in ("left", "right") for finger in ("finger_0", "finger_1"))


@lru_cache(maxsize=32)
def _font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _fit_font(draw, text, maximum_width, preferred_size):
    """Measure the actual font so narrow previews retain complete labels."""
    for size in range(preferred_size, 0, -1):
        font = _font(size)
        if draw.textlength(text, font=font) <= maximum_width:
            return font
    raise ValueError("Preview label cannot fit the available width")


def rgb_preview(frame, index, count, initial_timestamp_ns, *, width=1200):
    """Preserve image aspect and actual uint8 intensities; display source contact flags."""
    if width < 600 or width % 12:
        raise ValueError("Video width must be at least 600 pixels and divisible by 12")
    header, label_height = 64, 30
    head_width = width * 2 // 3
    camera_height = head_width * 3 // 4
    pad_height = width // 4 + label_height
    height = header + camera_height + pad_height
    # libx264 yuv420p needs even dimensions.
    height += height % 2
    canvas = Image.new("RGB", (width, height), "#101827")
    draw = ImageDraw.Draw(canvas)
    font_size = 18 if width >= 1000 else 12
    stamp = frame["sync"]["timestamp_ns"]
    heading = (
        f"Full episode | frame {index + 1}/{count} | native physics {frame['sync']['physics_frame']} | "
        f"sim {stamp / 1e9:.3f}s | elapsed {(stamp - initial_timestamp_ns) / 1e9:.3f}s"
    )
    draw.text(
        (10, 7),
        heading,
        font=_fit_font(draw, heading, width - 20, font_size),
        fill="white",
    )
    legend = "No tube contact = zero tactile; raw gel retained in HDF5. L0/L1: left; R0/R1: right."
    draw.text(
        (10, 34),
        legend,
        font=_fit_font(draw, legend, width - 20, font_size),
        fill="#a9c9df",
    )

    def tile(rgb, box, label):
        x, y, w, h = box
        draw.text((x + 7, y + 5), label, font=_fit_font(draw, label, w - 14, font_size), fill="white")
        content = Image.new("RGB", (w, h - label_height), "black")
        image = ImageOps.contain(Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB"), content.size)
        content.paste(image, ((content.width - image.width) // 2, (content.height - image.height) // 2))
        canvas.paste(content, (x, y + label_height))

    tile(frame["vision"]["head"]["rgb"], (0, header, head_width, camera_height), "HEAD RGB")
    wrist_width = width - head_width
    for row, name in enumerate(("left_wrist", "right_wrist")):
        tile(
            frame["vision"][name]["rgb"],
            (head_width, header + row * camera_height // 2, wrist_width, camera_height // 2),
            name.upper() + " RGB",
        )
    for column, (side, finger) in enumerate(PADS):
        data = frame["tactile"][side][finger]
        conditioned = data.get("provenance", {}).get("contact_conditioned_observations") is True
        if conditioned:
            state = "CONTACT" if bool(data["contact_active"]) else "NO CONTACT (zero)"
        else:
            state = "RAW / contact unspecified"
        tile(
            data["rgb"][..., :3],
            (column * width // 4, header + camera_height, width // 4, pad_height),
            f"{side[0].upper()}{finger[-1]}: {state}",
        )
    return canvas


def _timeline(reader):
    if len(reader) < 2:
        raise ValueError("At least two committed frames are needed to derive the actual video rate")
    first, previous = reader.sync(0), reader.sync(0)
    interval = reader.sync(1)["timestamp_ns"] - first["timestamp_ns"]
    if interval <= 0:
        raise ValueError("Source timestamps must increase strictly")
    for index in range(1, len(reader)):
        current = reader.sync(index)
        if current["physics_frame"] <= previous["physics_frame"]:
            raise ValueError("Source physics frames must increase strictly")
        delta = current["timestamp_ns"] - previous["timestamp_ns"]
        if delta <= 0 or abs(delta - interval) > 1:
            raise ValueError("Nonuniform simulation timestamps require a variable-rate video; refusing retiming")
        previous = current
    return first, previous, interval


def export_episode_video(source, output, *, width=1200):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source.is_dir():
        source = source / "episode.hdf5"
    if output.suffix.lower() != ".mp4" or output == source:
        raise ValueError("Output must be a separate .mp4 file")
    report_path = output.with_suffix(".json")
    temporary = output.with_name(output.stem + ".partial.mp4")
    if any(path.exists() for path in (output, report_path, temporary)):
        raise FileExistsError("Video, report or partial output already exists; choose a new output filename")
    with source.open("rb") as stream:
        source_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    writer = None
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with HDF5Reader(source) as reader:
            if source.name.endswith(".partial") or reader.handle.attrs["status"] != "complete":
                raise ValueError("Full-episode video requires a completed recording; partial recordings are refused")
            if reader.handle.attrs["observation_schema"] != SCHEMA_VERSION:
                raise ValueError("Task video requires three RGB cameras and bilateral TacSL, not component data")
            first, last, interval = _timeline(reader)
            fps = 1e9 / interval
            contact_counts = {f"{side}.{finger}": 0 for side, finger in PADS}
            conditioned_counts = dict.fromkeys(contact_counts, 0)
            for index in range(len(reader)):
                frame = reader.read_frame(index)
                image = rgb_preview(frame, index, len(reader), first["timestamp_ns"], width=width)
                if writer is None:
                    writer = VideoStreamWriter(str(temporary), image.height, image.width, 3, fps=fps)
                writer.append(np.asarray(image))
                for side, finger in PADS:
                    data = frame["tactile"][side][finger]
                    if data.get("provenance", {}).get("contact_conditioned_observations") is True:
                        key = f"{side}.{finger}"
                        conditioned_counts[key] += 1
                        contact_counts[key] += int(bool(data["contact_active"]))
            writer.close(announce=False)
            if writer.n_frames != len(reader):
                raise RuntimeError("Video did not receive every committed source frame")
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-count_frames",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=nb_read_frames,r_frame_rate,width,height",
                    "-of",
                    "json",
                    str(temporary),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            stream_info = json.loads(probe.stdout)["streams"][0]
            decoded_count = int(stream_info["nb_read_frames"])
            numerator, denominator = map(int, stream_info["r_frame_rate"].split("/"))
            if decoded_count != len(reader) or not math.isclose(numerator / denominator, fps, rel_tol=1e-6):
                raise RuntimeError("Decoded video frame count/rate does not preserve the HDF5 episode")
            with source.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != source_sha:
                    raise RuntimeError("Source HDF5 changed during export")
            report = {
                "status": "passed",
                "display_only": True,
                "simulation_rerun": False,
                "source_hdf5": str(source),
                "source_hdf5_sha256": source_sha,
                "source_status": "complete",
                "video": str(output),
                "committed_frames": len(reader),
                "encoded_frames": writer.n_frames,
                "decoded_frames": decoded_count,
                "frame_stride": 1,
                "first_sync": first,
                "last_sync": last,
                "actual_frame_interval_ns": interval,
                "video_fps": fps,
                "timestamp_span_s": (last["timestamp_ns"] - first["timestamp_ns"]) / 1e9,
                "video_duration_s": len(reader) / fps,
                "final_sample_display_hold_s": 1 / fps,
                "width": image.width,
                "height": image.height,
                "contact_conditioned_frames_by_pad": conditioned_counts,
                "contact_active_frames_by_pad": contact_counts,
                "views": ["head.rgb", "left_wrist.rgb", "right_wrist.rgb", *contact_counts],
                "task_success_claimed_by_exporter": False,
            }
        temporary.replace(output)
        try:
            report_path.write_text(json.dumps(report, indent=2) + "\n")
        except BaseException:
            output.unlink(missing_ok=True)
            raise
        return report
    except BaseException:
        if writer is not None:
            writer.abort()
        temporary.unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1200)
    args = parser.parse_args()
    print(json.dumps(export_episode_video(args.source, args.output, width=args.width), indent=2))


if __name__ == "__main__":
    main()
