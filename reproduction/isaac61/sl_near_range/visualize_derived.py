"""Supplemental visualization of a complete native source + complete derived SL.

Only an in-memory frame's depth_sl is overridden. Original RGB, IR, GT and
TacSL streams come from the source HDF5. No source file is modified.
"""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import h5py
import numpy as np
from PIL import ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))
from reproduction.visualize_multimodal import MODES, mosaic

import resume_sidecar as resume
from utils.multimodal_hdf5 import HDF5Reader
from utils.save_file import VideoStreamWriter


def visualize(
    source,
    sidecar,
    output,
    *,
    frame_stride=1,
    preview_frame=296,
    tactile="rgb",
    depth_min=0.04,
    depth_max=2.5,
    tile_width=280,
    tile_height=210,
):
    source = Path(source)
    sidecar = Path(sidecar)
    output = Path(output)
    if (
        source.suffix != ".hdf5"
        or source.resolve().suffix != ".hdf5"
        or sidecar.suffix != ".hdf5"
        or sidecar.resolve().suffix != ".hdf5"
    ):
        raise ValueError("Supplemental viewer requires completed .hdf5 paths; never partial files or aliases")
    if frame_stride < 1 or not 0 <= depth_min < depth_max or tactile not in MODES:
        raise ValueError("Invalid view parameters")
    paths = {ext: output / ("near-range-derived." + ext) for ext in ("png", "mp4", "json")}
    for path in paths.values():
        resume.reject_output_alias(path, (source, sidecar))
    if any(path.exists() or path.is_symlink() for path in paths.values()):
        raise FileExistsError("Supplemental artifacts already exist; use a fresh output directory")
    source_before = resume.source_signature(source)
    derived_before = resume.source_signature(sidecar)
    with HDF5Reader(source) as reader, h5py.File(sidecar, "r") as derived:
        if not reader.metadata["complete"] or reader.metadata["status"] != "complete":
            raise ValueError("Native source recording is incomplete")
        if (
            derived.attrs.get("schema") != resume.base.SCHEMA
            or not derived.attrs.get("complete")
            or derived.attrs.get("status") != "complete"
        ):
            raise ValueError("Derived sidecar is incomplete")
        if derived.attrs.get("derivation_module_sha256") != resume.base.EXPECTED_DERIVE_SHA:
            raise ValueError("Derived algorithm identity mismatch")
        if derived["parameters_json"].asstr()[()] != resume.base.json_text(resume.base.candidate.parameters()):
            raise ValueError("Derived fixed parameters mismatch")
        if len(reader) != derived.attrs["frame_count"] or len(reader) != derived.attrs["final_source_frames"]:
            raise ValueError("Derived sidecar does not cover the whole native source")
        if not 0 <= preview_frame < len(reader):
            raise ValueError("Preview frame is outside native recording")
        source_sha = resume.base.file_sha(source)
        if source_sha != derived.attrs["final_original_hdf5_sha256"]:
            raise ValueError("Native source SHA differs from finalized derivation")
        sidecar_sha = resume.base.file_sha(sidecar)
        inventory = [f"{index:08d}" for index in range(len(reader))]
        if sorted(reader.handle["frames"]) != inventory or sorted(derived["frames"]) != inventory:
            raise ValueError("Source or derived frame inventory is not contiguous")
        previous_time = previous_physics = -1
        for index in range(len(reader)):
            sync, vision = resume.selected_frame(reader, index)
            group = derived["frames"][f"{index:08d}"]
            if not group.attrs.get("committed") or group.attrs.get("source_frame_index") != index:
                raise ValueError("Derived frame is uncommitted or has the wrong source index")
            if (
                not sync.get("physics_held_fixed")
                or sync["timestamp_ns"] <= previous_time
                or sync["physics_frame"] <= previous_physics
            ):
                raise ValueError("Source capture clock is not strictly increasing and frozen")
            resume.verify_old_frame(SimpleNamespace(handle=derived), index, sync, vision)
            previous_time = sync["timestamp_ns"]
            previous_physics = sync["physics_frame"]
        indices = list(range(0, len(reader), frame_stride))
        duration = (reader.sync(indices[-1])["timestamp_ns"] - reader.sync(indices[0])["timestamp_ns"]) / 1e9
        fps = (len(indices) - 1) / duration if duration > 0 else 1.0
        output.mkdir(parents=True, exist_ok=True)
        writer = None

        def picture(index):
            frame = reader.read_frame(index)
            resume.verify_old_frame(SimpleNamespace(handle=derived), index, frame["sync"], frame["vision"])
            for name in resume.base.CAMERAS:
                frame["vision"][name]["depth_sl"] = derived[f"frames/{index:08d}/{name}/depth_sl_derived"][()]
            result = mosaic(
                frame, tile_width=tile_width, tile_height=tile_height, mode=tactile, depth_range=(depth_min, depth_max)
            )
            draw = ImageDraw.Draw(result)
            draw.rectangle((0, 25, result.width, 50), fill="#111827")
            draw.text(
                (12, 28),
                "Original IR | near-range derived SL (IR-only CPU) | purple = NaN | original capture retained",
                fill="#a7c4d9",
            )
            for row, name in enumerate(resume.base.CAMERAS):
                x = 4 * tile_width
                y = 58 + row * tile_height
                draw.rectangle((x, y, x + tile_width, y + 23), fill="#111827")
                draw.text((x + 6, y + 5), name + ": near-range derived SL (IR-left)", fill="white")
            return result

        try:
            for index in indices:
                image = picture(index)
                if writer is None:
                    writer = VideoStreamWriter(str(paths["mp4"]), image.height, image.width, 3, fps=fps)
                writer.append(np.asarray(image))
                if index == preview_frame:
                    image.save(paths["png"])
            if preview_frame not in indices:
                picture(preview_frame).save(paths["png"])
            writer.close()
            writer = None
        finally:
            if writer is not None:
                writer.close()
        if source_before != resume.source_signature(source) or derived_before != resume.source_signature(sidecar):
            raise ValueError("Source or complete sidecar changed during export")
        report = {
            "mode": "supplemental near-range derived SL visualization; not original capture configuration",
            "native_source_hdf5": str(source.resolve()),
            "native_source_sha256": source_sha,
            "derived_sidecar_hdf5": str(sidecar.resolve()),
            "derived_sidecar_sha256": sidecar_sha,
            "algorithm_sha256": resume.base.EXPECTED_DERIVE_SHA,
            "frames": len(indices),
            "source_frames": len(reader),
            "all_source_frames_integrity_verified": True,
            "fps_from_source_physics_clock": fps,
            "frame_stride": frame_stride,
            "preview_source_frame": preview_frame,
            "tactile_display": tactile,
            "depth_display_range_m": [depth_min, depth_max],
            "display_only_depth_coloring": True,
            "native_rgb_ir_gt_tactile_unchanged": True,
            "only_in_memory_depth_sl_overridden": True,
            "task_complete_claimed": False,
            "artifacts": {key: str(path.resolve()) for key, path in paths.items()},
        }
        paths["json"].write_text(json.dumps(report, indent=2) + "\n")
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--preview-frame", type=int, default=296)
    parser.add_argument("--tactile", choices=MODES, default="rgb")
    parser.add_argument("--depth-min", type=float, default=0.04)
    parser.add_argument("--depth-max", type=float, default=2.5)
    args = parser.parse_args()
    print(
        json.dumps(
            visualize(
                args.source,
                args.sidecar,
                args.output,
                frame_stride=args.frame_stride,
                preview_frame=args.preview_frame,
                tactile=args.tactile,
                depth_min=args.depth_min,
                depth_max=args.depth_max,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
