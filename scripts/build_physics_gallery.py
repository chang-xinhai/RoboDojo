"""Build a portable gallery from passed, genuine Isaac Sim 6.1 physics captures."""

import argparse
import base64
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=root / "reproduction/isaac61/outputs")
    parser.add_argument("--output", type=Path, default=root / "reproduction/isaac61/outputs/physics-gallery")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = [
        ("scene-gravity2-61", "Surface cloth · 169 nodes · gravity 2g", "cloth"),
        ("scene-garment-00001-61", "Bundled Top_Long/00001 · 10,410 nodes · gravity 2g", "cloth"),
        ("scene-fluid-61", "Native PBD fluid · 125 particles · GPU dynamics + host readback", "fluid"),
    ]
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 19)
    small_font = ImageFont.truetype(font_path, 15)
    sheet = Image.new("RGB", (1280, 1030), "#101820")
    draw = ImageDraw.Draw(sheet)
    draw.text((16, 12), "Isaac Sim 6.1.0.0 / OmniPhysX 110.3.2 — actual RTX captures", font=font, fill="white")
    sections = []
    manifest = []
    for row, (directory, title, kind) in enumerate(cases):
        source = args.inputs / directory
        report = json.loads((source / "report.json").read_text())
        if report.get("status") != "passed" or not report.get("real_runtime"):
            raise ValueError(f"{source} is not a passed real-runtime capture")
        if report.get("isaacsim_version") != "6.1.0.0" or report.get("physx_extension") != "omni.physx-110.3.2":
            raise ValueError(f"{source} does not contain the expected actual simulator/physics versions")
        states = ["before", "after", "contact", "reset"] if kind == "cloth" else ["before", "after", "reset"]
        y = 58 + row * 319
        draw.text((16, y), title, font=font, fill="white")
        figures = []
        for col, state in enumerate(states):
            path = source / f"{kind}_{state}.png"
            with Image.open(path) as frame:
                sheet.paste(frame.convert("RGB").resize((320, 240)), (col * 320, y + 50))
            caption = {
                "before": "Before simulation",
                "after": "After 30 physics steps",
                "contact": "Rigid ground contact",
                "reset": "Live physical reset",
            }[state]
            draw.text((col * 320 + 12, y + 28), caption, font=small_font, fill="#b6c9d8")
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            figures.append(
                f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(caption)}">'
                f"<figcaption>{caption}</figcaption></figure>"
            )
        manifest.append({"source": str(source.resolve()), "report": report})
        sections.append(
            f'<section><h2>{html.escape(title)}</h2><div class="frames">{"".join(figures)}</div>'
            "<details><summary>Measured runtime report</summary>"
            f"<pre>{html.escape(json.dumps(report, indent=2))}</pre></details></section>"
        )
    sheet.save(args.output / "physics-captures.png")
    page = (
        """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RoboDojo — real Isaac Sim 6.1 physics validation</title><style>
body{background:#101820;color:#edf4fa;font:16px system-ui,sans-serif;margin:0;
padding:28px;max-width:1440px;margin-inline:auto}
h1{font-size:28px}h2{font-size:20px}p{color:#b6c9d8;line-height:1.6}.frames{display:flex;flex-wrap:wrap;gap:12px}
figure{margin:0;flex:1;min-width:240px;max-width:360px}img{width:100%;border-radius:8px}
figcaption{padding:8px 0;color:#b6c9d8}
section{border-top:1px solid #304250;padding-block:18px}details{margin-top:10px}
pre{overflow:auto;white-space:pre-wrap;font-size:13px}
</style><h1>RoboDojo physics validation</h1>
<p>Actual Isaac Sim 6.1.0.0 / OmniPhysX 110.3.2, NVIDIA RTX 5090.
Images are unmodified RTX camera output from genuine native solver motion, rigid ground contact, and live resets.
Reports include measured nodes, positions, velocities, image changes and actual runtime versions.</p>
<p>These runs validate native scene objects. Full Isaac Lab task/robot and multimodal validation are separate.
The new FEM surface model changes legacy particle-cloth material behavior; bundled asset migration explicitly records
adhesion removal and benchmark_equivalent=false. Fluid uses GPU dynamics with host readback;
cloth uses the GPU tensor pipeline.</p>
"""
        + "".join(sections)
        + "</html>"
    )
    (args.output / "index.html").write_text(page)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
