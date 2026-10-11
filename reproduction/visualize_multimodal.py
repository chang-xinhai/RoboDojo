"""Stream HDF5 (or legacy NPZ) into metric PNG, MP4 and interactive HTML."""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import io
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.multimodal import CAMERAS, json_value, read_frame, validate_multimodal
from utils.multimodal_hdf5 import HDF5Reader

MODES = ("rgb", "optical_depth_delta", "normal_force", "shear_force")


def metric_color(array, low, high):
    """Fixed display-only metric scale; NaN remains purple, never interpolated."""
    array = np.asarray(array).squeeze()
    finite = np.isfinite(array)
    scaled = np.zeros(array.shape, dtype=float)
    np.subtract(array, low, out=scaled, where=finite)
    scaled = np.clip(scaled / (high - low), 0, 1)
    # Blue -> cyan -> yellow -> red; the same value always has the same color.
    stops = np.array([[35, 35, 150], [0, 190, 220], [245, 235, 50], [210, 35, 20]])
    positions = np.linspace(0, 1, len(stops))
    rgb = np.stack([np.interp(scaled, positions, stops[:, i]) for i in range(3)], -1).astype(np.uint8)
    rgb[~finite] = (130, 0, 130)
    return rgb


def letterbox(array, size):
    tile = Image.new("RGB", size, "#050a13")
    image = ImageOps.contain(Image.fromarray(np.asarray(array, dtype=np.uint8)).convert("RGB"), size)
    tile.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return tile


def _taxels(data, field):
    values = np.asarray(data[field])
    if field == "shear_force":
        values = np.linalg.norm(values, axis=-1)
    shape = data.get("metadata", {}).get("array_size") or data.get("provenance", {}).get("array_size")
    if shape is None or np.prod(shape) != values.size:
        # Unspecified layout is shown as a strip, never an invented spatial grid.
        shape = (1, values.size)
    return values.reshape(tuple(shape))


def mosaic(
    frame,
    tile_width=280,
    tile_height=210,
    *,
    mode="rgb",
    depth_range=(0.1, 2.5),
    delta_limit=0.002,
    force_limit=0.5,
    shear_limit=0.2,
):
    header = 58
    canvas = Image.new("RGB", (5 * tile_width, header + 5 * tile_height), "#111827")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (12, 8),
        f"RoboDojo | physics {frame['sync']['physics_frame']} | "
        f"simulation {frame['sync']['timestamp_ns'] / 1e9:.3f} s | tactile {mode}",
        fill="white",
    )
    draw.text(
        (12, 28),
        "Original IR exposure | metric colors only for display | purple = invalid | aspect preserved",
        fill="#a7c4d9",
    )

    def paste(row, col, array, label, scale=None):
        x, y = col * tile_width, header + row * tile_height
        draw.text((x + 6, y + 5), label, fill="white")
        canvas.paste(letterbox(array, (tile_width, tile_height - 48)), (x, y + 26))
        if scale:
            lo, hi, units = scale
            bar = metric_color(np.linspace(lo, hi, tile_width)[None, :].repeat(4, 0), lo, hi)
            canvas.paste(Image.fromarray(bar), (x, y + tile_height - 19))
            draw.text((x + 5, y + tile_height - 14), f"{lo:g} .. {hi:g} {units}", fill="white")

    for row, name in enumerate(CAMERAS):
        cam = frame["vision"][name]
        for col, key in enumerate(("rgb", "depth_gt", "ir_left", "ir_right", "depth_sl")):
            scale = None
            if key.startswith("depth"):
                array = metric_color(cam[key], *depth_range)
                scale = (*depth_range, "m")
                label = f"{name}: {'GT RGB optical' if key == 'depth_gt' else 'SL left IR optical'}"
            elif key.startswith("ir"):
                array = np.repeat(np.asarray(cam[key])[..., None], 3, axis=-1)
                label = f"{name}: {key} uint8"
            else:
                array, label = cam[key], f"{name}: RGB"
            paste(row, col, array, label, scale)
    for row, side in enumerate(("left", "right"), start=3):
        lines = []
        for index, (finger, data) in enumerate(frame["tactile"][side].items()):
            label = f"{side}.{finger}"
            paste(row, 2 * index, np.asarray(data["rgb"])[..., :3], label + " RGB")
            field = "depth" if mode == "rgb" else mode
            if field not in data or data[field] is None:
                array = np.zeros((2, 2, 3), np.uint8)
                scale = None
                description = "unavailable (see provenance)"
            else:
                values = _taxels(data, field) if field in ("normal_force", "shear_force") else data[field]
                limits = {
                    "depth": (0, 0.05, "m optical depth"),
                    "optical_depth_delta": (-delta_limit, delta_limit, "m optical proxy"),
                    "normal_force": (-force_limit, force_limit, "N"),
                    "shear_force": (0, shear_limit, "N magnitude"),
                }
                scale = limits[field]
                array = metric_color(values, *scale[:2])
                description = field
            paste(row, 2 * index + 1, array, label + " " + description, scale)
            if data["normal_force"] is None:
                lines += [
                    finger,
                    "Image-only; force unavailable",
                    str(data["provenance"].get("force_unavailable_reason")),
                    "",
                ]
            else:
                lines += [
                    finger,
                    f"max normal {np.max(data['normal_force']):.5g} N",
                    f"max shear {np.max(np.linalg.norm(data['shear_force'], axis=-1)):.5g} N",
                    "",
                ]
        draw.multiline_text((4 * tile_width + 8, header + row * tile_height + 8), "\n".join(lines), fill="#c1e6d3")
    return canvas


class LegacyReader:
    def __init__(self, directory):
        self.path = Path(directory) / "manifest.json"
        self.metadata = json.loads(self.path.read_text())
        self.frames = self.metadata["frames"]

    def __len__(self):
        return len(self.frames)

    def sync(self, index):
        return self.frames[index]

    def read_frame(self, index):
        return validate_multimodal(read_frame(self.path.parent / self.frames[index]["file"]))

    def close(self):
        pass


def _score(frame):
    forces = [
        float(np.max(np.abs(data["normal_force"])))
        for fingers in frame["tactile"].values()
        for data in fingers.values()
        if data["normal_force"] is not None
    ]
    if forces:
        return max(forces), "maximum absolute measured normal force (N)"
    deltas = [
        float(np.max(np.abs(data["optical_depth_delta"])))
        for fingers in frame["tactile"].values()
        for data in fingers.values()
        if "optical_depth_delta" in data
    ]
    return max(deltas, default=0), "optical proxy variation (m), force unavailable"


def visualize(
    source,
    output=None,
    *,
    frame_stride=1,
    max_frames=None,
    depth_min=0.1,
    depth_max=2.5,
    delta_limit=0.002,
    force_limit=0.5,
    shear_limit=0.2,
):
    source = Path(source)
    if frame_stride < 1 or max_frames is not None and max_frames < 1:
        raise ValueError("frame stride and max frames must be positive")
    if not 0 <= depth_min < depth_max or min(delta_limit, force_limit, shear_limit) <= 0:
        raise ValueError("Display metric limits must be positive and depth min < max")
    if source.is_dir():
        candidate = source / "episode.hdf5"
        if candidate.exists():
            source = candidate
    reader = LegacyReader(source) if source.is_dir() else HDF5Reader(source)
    directory = Path(output) if output else (source if source.is_dir() else source.parent)
    directory.mkdir(parents=True, exist_ok=True)
    png, video, html_path = [directory / ("multimodal." + ext) for ext in ("png", "mp4", "html")]
    writer = None
    try:
        if isinstance(reader, HDF5Reader) and reader.metadata["observation_schema"] != "robodojo.multimodal.v1":
            raise ValueError(
                "Full multimodal viewer requires the full camera+TacSL schema; component viewer is separate"
            )
        indices = range(0, len(reader), frame_stride)
        if max_frames is not None:
            indices = indices[:max_frames]
        if not indices:
            raise ValueError("Recording has no committed frames")
        duration = (reader.sync(indices[-1])["timestamp_ns"] - reader.sync(indices[0])["timestamp_ns"]) / 1e9
        fps = (len(indices) - 1) / duration if duration > 0 else 1.0
        from utils.save_file import VideoStreamWriter

        best_score, selected = -1, None
        with html_path.open("w") as document:
            document.write('<!doctype html><meta charset="utf-8">')
            document.write(
                "<details><summary>Recording status, source and runtime settings</summary><pre>"
                + html.escape(json.dumps(json_value(reader.metadata), indent=2))
                + "</pre></details>"
            )
            document.write("""<title>RoboDojo HDF5 viewer</title>
<style>body{background:#111827;color:#eef;font:16px system-ui;margin:24px}
img{width:100%;max-width:1400px}input{width:65%}pre{white-space:pre-wrap}</style>
<h1>RoboDojo synchronized multimodal recording</h1>
<p>GT depth: RGB optical frame. SL depth: left IR optical frame. IR retains original exposure.
TacSL optical delta is a signed camera-depth proxy, not FEM deformation.</p>
<button id="play">Play</button><input id="slider" type="range" min="0" value="0">
<select id="mode"><option>rgb</option><option>optical_depth_delta</option>
<option>normal_force</option><option>shear_force</option></select><span id="counter"></span>
<p><img id="picture"></p><details><summary>Per-frame calibration and provenance</summary>
<pre id="details"></pre></details><script>const frames=[\n""")
            for position, index in enumerate(indices):
                frame = reader.read_frame(index)
                picture = mosaic(
                    frame,
                    depth_range=(depth_min, depth_max),
                    delta_limit=delta_limit,
                    force_limit=force_limit,
                    shear_limit=shear_limit,
                )
                if writer is None:
                    writer = VideoStreamWriter(str(video), picture.height, picture.width, 3, fps=fps)
                writer.append(np.asarray(picture))
                score, semantics = _score(frame)
                if score > best_score:
                    best_score = score
                    selected = {"index": index, **frame["sync"], "score": score, "score_semantics": semantics}
                    picture.save(png)
                if position:
                    document.write(",\n")
                # Encode one mode at a time directly to disk; no episode-sized image list.
                document.write('{"views":{')
                for mode_index, mode in enumerate(MODES):
                    view = (
                        picture
                        if mode == "rgb"
                        else mosaic(
                            frame,
                            mode=mode,
                            depth_range=(depth_min, depth_max),
                            delta_limit=delta_limit,
                            force_limit=force_limit,
                            shear_limit=shear_limit,
                        )
                    )
                    stream = io.BytesIO()
                    view.save(stream, format="PNG")
                    encoded = "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()
                    document.write(("," if mode_index else "") + json.dumps(mode) + ":" + json.dumps(encoded))
                details = {
                    "sync": frame["sync"],
                    "vision": {
                        name: {key: cam[key] for key in ("calibration", "provenance")}
                        for name, cam in frame["vision"].items()
                    },
                    "tactile": {
                        side: {
                            finger: {key: data.get(key) for key in ("metadata", "provenance")}
                            for finger, data in fingers.items()
                        }
                        for side, fingers in frame["tactile"].items()
                    },
                }
                # Escape HTML parser termination in metadata strings, including untrusted asset names.
                document.write('},"details":' + json.dumps(json_value(details)).replace("<", "\\u003c") + "}")
            document.write("];const selectedPreviewIndex=" + str(indices.index(selected["index"])) + ";\n")
            document.write("""
const slider=document.getElementById('slider'),mode=document.getElementById('mode');
slider.max=frames.length-1;slider.value=selectedPreviewIndex;
function show(){const i=Number(slider.value);document.getElementById('picture').src=frames[i].views[mode.value];
document.getElementById('counter').textContent=(i+1)+' / '+frames.length;
document.getElementById('details').textContent=JSON.stringify(frames[i].details,null,2)}
slider.oninput=show;mode.onchange=show;let timer=null;
document.getElementById('play').onclick=()=>{if(timer){clearInterval(timer);timer=null}
else{timer=setInterval(()=>{slider.value=(Number(slider.value)+1)%frames.length;show()},250)}};show();</script>""")
        writer.close()
        report = {
            "png": str(png.resolve()),
            "video": str(video.resolve()),
            "html": str(html_path.resolve()),
            "frames": len(indices),
            "source_format": "hdf5" if isinstance(reader, HDF5Reader) else "legacy_npz",
            "source_hdf5": str(source.resolve()) if isinstance(reader, HDF5Reader) else None,
            "selected_frame": selected,
            "depth_range_m": [depth_min, depth_max],
            "optical_delta_range_m": [-delta_limit, delta_limit],
            "display_only": True,
            "recording_status": reader.metadata.get("status", "legacy_complete"),
        }
        if isinstance(reader, HDF5Reader):
            with source.open("rb") as stream:
                report["source_hdf5_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        (directory / "visualization.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    except BaseException:
        if writer:
            writer.abort()
        raise
    finally:
        reader.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="episode.hdf5 or legacy recording directory")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--depth-min", type=float, default=0.1)
    parser.add_argument("--depth-max", type=float, default=2.5)
    parser.add_argument("--delta-limit", type=float, default=0.002)
    parser.add_argument("--force-limit", type=float, default=0.5)
    parser.add_argument("--shear-limit", type=float, default=0.2)
    print(json.dumps(visualize(**vars(parser.parse_args())), indent=2))
