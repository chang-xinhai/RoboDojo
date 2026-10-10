"""Render a recorded dataset to a PNG mosaic, MP4 and standalone HTML viewer."""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.multimodal import CAMERAS, json_value, read_frame, validate_multimodal


def _intensity(array):
    array = np.asarray(array)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros((*array.shape[:2], 3), dtype=np.uint8)
    lo, hi = np.percentile(array[finite], [2, 98])
    scaled = np.zeros(array.shape, dtype=np.float64)
    np.subtract(array, lo, out=scaled, where=finite)
    scaled /= max(float(hi - lo), 1e-9)
    gray = (np.clip(scaled, 0, 1) * 255).astype(np.uint8)
    color = np.stack([gray, gray, gray], axis=-1)
    color[~finite] = (130, 0, 130)
    return color


def mosaic(frame, tile_width=280, tile_height=190):
    labels = (
        "RGB",
        "Depth GT (m, RGB frame)",
        "IR left (projector difference)",
        "IR right (projector difference)",
        "Depth SL (m, left IR frame)",
    )
    header = 50
    rows = 5
    canvas = Image.new("RGB", (5 * tile_width, header + rows * tile_height), "#111827")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (12, 10),
        f"RoboDojo | physics frame {frame['sync']['physics_frame']} | sim {frame['sync']['simulation_time_s']:.3f} s",
        fill="white",
    )
    draw.text(
        (12, 28),
        "Synchronized actual runtime | purple = invalid depth | display normalization only; NPZ retains metres",
        fill="#a7c4d9",
    )
    for row, name in enumerate(CAMERAS):
        camera = frame["vision"][name]
        for col, key in enumerate(("rgb", "depth_gt", "ir_left", "ir_right", "depth_sl")):
            array = camera[key] if key == "rgb" else _intensity(camera[key])
            tile = Image.fromarray(array).resize((tile_width, tile_height - 30))
            canvas.paste(tile, (col * tile_width, header + row * tile_height + 30))
            draw.text((col * tile_width + 8, header + row * tile_height + 7), f"{name}: {labels[col]}", fill="white")
    for row, side in enumerate(("left", "right"), start=3):
        fingers = frame["tactile"][side]
        for idx, (finger, data) in enumerate(fingers.items()):
            for offset, key in enumerate(("rgb", "depth")):
                col = idx * 2 + offset
                array = np.asarray(data[key])
                array = array[..., :3].astype(np.uint8) if key == "rgb" else _intensity(array)
                canvas.paste(
                    Image.fromarray(array).resize((tile_width, tile_height - 30)),
                    (col * tile_width, header + row * tile_height + 30),
                )
                draw.text(
                    (col * tile_width + 8, header + row * tile_height + 7),
                    f"{side}.{finger}: TacSL {key}",
                    fill="white",
                )
        force_lines = []
        for finger, data in fingers.items():
            if data["normal_force"] is None:
                force_lines.extend(
                    (
                        finger,
                        "TacSL RGB/depth available",
                        "Force field unavailable:",
                        str(data["provenance"]["force_unavailable_reason"]),
                        "",
                    )
                )
                continue
            force = np.asarray(data["normal_force"])
            force_lines.extend(
                (
                    finger,
                    f"max normal: {float(np.max(force)):.6g} N",
                    f"sum normal: {float(np.sum(force)):.6g} N",
                    f"taxels: {force.size}",
                    f"max penetration: {float(np.max(data['penetration_depth'])):.6g} m",
                    "",
                )
            )
        draw.multiline_text(
            (4 * tile_width + 10, header + row * tile_height + 8), "\n".join(force_lines), fill="#c1e6d3"
        )
    return canvas


def visualize(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    frames = [validate_multimodal(read_frame(directory / entry["file"])) for entry in manifest["frames"]]
    mosaics = [mosaic(frame) for frame in frames]
    png = directory / "multimodal.png"
    mosaics[0].save(png)
    encoded = []
    for picture in mosaics:
        stream = io.BytesIO()
        picture.save(stream, format="PNG")
        encoded.append("data:image/png;base64," + base64.b64encode(stream.getvalue()).decode())
    details = {
        "manifest": manifest,
        "camera_calibration": {
            name: {"calibration": camera["calibration"], "provenance": camera["provenance"]}
            for name, camera in frames[0]["vision"].items()
        },
        "tactile_provenance": {
            side: {finger: sample["provenance"] for finger, sample in fingers.items()}
            for side, fingers in frames[0]["tactile"].items()
        },
    }
    metadata = html.escape(json.dumps(json_value(details), indent=2))
    html_path = directory / "multimodal.html"
    template = """<!doctype html><meta charset="utf-8"><title>RoboDojo multimodal capture</title>
<style>
body{background:#111827;color:#e5edf7;font:16px system-ui;margin:24px}
img{width:100%;max-width:1400px}input{width:70%}pre{white-space:pre-wrap}button{padding:8px}
</style>
<h1>RoboDojo synchronized multimodal capture</h1>
<p>Three RGB / metric depth / binocular projected IR views plus bilateral TacSL.
Structured-light depth uses the left IR optical frame.
IR models projector intensity without NIR spectral material response.</p>
<button id="play">Play</button>
<input id="slider" type="range" min="0" max="__MAX_FRAME__" value="0">
<span id="counter"></span><p><img id="picture"></p>
<details><summary>Recording calibration and provenance</summary><pre>__METADATA__</pre></details>
<script>
const images=__IMAGES__;
const slider=document.getElementById('slider'), picture=document.getElementById('picture');
const counter=document.getElementById('counter');
function show(){
  picture.src=images[Number(slider.value)];
  counter.textContent=(Number(slider.value)+1)+' / '+images.length;
}
slider.oninput=show;
let timer=null;
document.getElementById('play').onclick=()=>{
  if(timer){clearInterval(timer);timer=null}
  else{timer=setInterval(()=>{slider.value=(Number(slider.value)+1)%images.length;show()},250)}
};
show();
</script>
"""
    # Replace trusted generated placeholders before inserting escaped metadata.
    document = template.replace("__MAX_FRAME__", str(len(frames) - 1))
    document = document.replace("__IMAGES__", json.dumps(encoded))
    document = document.replace("__METADATA__", metadata)
    html_path.write_text(document)
    from utils.save_file import VideoStreamWriter

    duration = (frames[-1]["sync"]["timestamp_ns"] - frames[0]["sync"]["timestamp_ns"]) / 1e9
    fps = (len(frames) - 1) / duration if duration > 0 else 1.0
    video = directory / "multimodal.mp4"
    writer = VideoStreamWriter(str(video), mosaics[0].height, mosaics[0].width, 3, fps=fps)
    try:
        for picture in mosaics:
            writer.append(np.asarray(picture))
        writer.close()
    except Exception:
        writer.abort()
        raise
    return {
        "png": str(png.resolve()),
        "video": str(video.resolve()),
        "html": str(html_path.resolve()),
        "frames": len(frames),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(json.dumps(visualize(parser.parse_args().directory), indent=2))
