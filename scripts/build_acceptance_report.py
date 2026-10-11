"""Build a portable, truthful report from existing runtime evidence; never starts Kit.

Example (run from RoboDojo):
  python scripts/build_acceptance_report.py --suite-dir outputs/task-suite \
      --cpu-log outputs/cpu.log --moving-dir outputs/moving --output outputs/report

Optional evidence: --fluid-dir, --cloth-dir, --tactile-dir, --sl-dir,
--replay-dir and --physics-gallery-dir. Input JSON is retained without rewriting
its recorded status or runtime versions. The report does not evaluate policies.
"""

import argparse
import base64
from datetime import UTC, datetime
import hashlib
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
import sys

LIMITATIONS = [
    "IR uses RTX projected visible radiance as a proxy, without physical NIR material response.",
    "Surface FEM replaces removed particle cloth; material/solver behavior and benchmark physics changed.",
    "Particle-readback TacSL queries real triangle meshes. "
    "Forces cover registered rigid colliders, not fluid pressure.",
    "A noncontact recording may have zero tactile forces. "
    "Contact-response evidence comes from the separate contact test.",
    "Scene initialization, reset and sensor acceptance do not establish policy success or task completion.",
    "Evidence can come from different source revisions; original per-run versions and hashes are preserved below.",
]


def load_json(path, inputs):
    path = path.resolve(strict=True)
    raw = path.read_bytes()
    value = json.loads(raw)
    inputs.append({"file": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
    return value


def parse_cpu_log(path, inputs):
    path = path.resolve(strict=True)
    raw = path.read_bytes()
    # Use the last completed pytest summary, including failures/errors/subtests.
    summaries = re.findall(
        r"[^\r\n]*\b\d+ (?:passed|failed|errors?|skipped)[^\r\n]*\bin \d+(?:\.\d+)?s[^\r\n]*", raw.decode()
    )
    if not summaries:
        raise ValueError(f"No completed pytest summary in {path}")
    summary = summaries[-1].strip().strip("=").strip()
    counts = {}
    for count, result in re.findall(
        r"(\d+) (subtests (?:passed|failed)|passed|failed|errors?|skipped|xfailed|xpassed|warnings?)", summary
    ):
        counts[result] = counts.get(result, 0) + int(count)
    failed = any(counts.get(key, 0) for key in ("failed", "error", "errors", "subtests failed"))
    inputs.append({"file": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
    return {"summary": summary, "counts": counts, "status": "failed" if failed else "passed"}


def check_suite(suite):
    if not isinstance(suite, dict) or not all(
        key in suite for key in ("status", "complete", "context", "tasks", "selected_tasks")
    ):
        raise ValueError("Suite must contain its original status, completeness, context and task inventory")
    tasks = suite["tasks"]
    if len({task["task"] for task in tasks}) != len(tasks):
        raise ValueError("Suite contains duplicate task results")
    if not set(task["task"] for task in tasks).issubset(set(suite["selected_tasks"])):
        raise ValueError("Suite results do not match selected tasks")
    for status in ("passed", "failed"):
        actual = sum(task["status"] == status for task in tasks)
        if suite.get(status) != actual:
            raise ValueError(f"Suite {status} count differs from its individual results")
    if suite["complete"] and len(tasks) != len(suite["selected_tasks"]):
        raise ValueError("Complete suite is missing task results")
    if suite["status"] == "passed" and (
        suite["failed"] or not suite["complete"] or suite["passed"] != len(suite["selected_tasks"])
    ):
        raise ValueError("Suite claims passed despite failed or incomplete results")


def status(value):
    """Normalize only the display decision, leaving source evidence untouched."""
    result = value.get("status", value.get("passed"))
    if isinstance(result, bool):
        return "passed" if result else "failed"
    if not isinstance(result, str):
        raise ValueError("Evidence has no explicit status/passed field")
    return result


def recording(directory, inputs):
    manifest = load_json(directory / "manifest.json", inputs)
    validation = load_json(directory / "validation.json", inputs)
    if not isinstance(manifest.get("frames"), list) or not isinstance(validation.get("frames"), list):
        raise ValueError(f"Missing recording frame inventory: {directory}")
    if len(manifest["frames"]) != len(validation["frames"]):
        raise ValueError(f"Recording manifest/validation frame count differs: {directory}")
    hdf5 = None
    if manifest.get("storage") == "hdf5":
        project_root = str(Path(__file__).resolve().parents[1])
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from utils.multimodal import SCHEMA_VERSION
        from utils.multimodal_hdf5 import HDF5Reader, validate_hdf5

        filename = manifest.get("file", "episode.hdf5")
        hdf5_path = (directory / filename).resolve(strict=True)
        if not hdf5_path.is_relative_to(directory.resolve()):
            raise ValueError(f"HDF5 path escapes recording directory: {hdf5_path}")
        hdf5 = validate_hdf5(hdf5_path, require_complete=status(validation) == "passed")
        if hdf5["validated_frames"] != len(manifest["frames"]):
            raise ValueError(f"HDF5/manifest frame count differs: {directory}")
        with HDF5Reader(hdf5_path) as reader:
            if reader.metadata["observation_schema"] != SCHEMA_VERSION:
                raise ValueError("A tactile-only component cannot establish a full multimodal recording")
            for index, (frame, checked) in enumerate(zip(manifest["frames"], validation["frames"], strict=True)):
                actual = reader.sync(index)
                if frame.get("hdf5_frame", frame.get("index")) != index or checked.get("index") != index:
                    raise ValueError(f"HDF5 frame indices differ at {index}: {directory}")
                if any(frame.get(key) != actual[key] or checked.get(key) != actual[key] for key in actual):
                    raise ValueError(f"HDF5/manifest/validation clocks differ at {index}: {directory}")
        digest = hashlib.sha256()
        with hdf5_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        hdf5.update(source_hdf5=str(hdf5_path), source_hdf5_sha256=digest.hexdigest())
        inputs.append({"file": str(hdf5_path), "sha256": digest.hexdigest()})
    elif manifest.get("storage") not in (None, "npz"):
        raise ValueError(f"Unsupported recording storage: {manifest['storage']}")
    previous = -1
    for frame in manifest["frames"]:
        if hdf5 is None:
            frame_path = (directory / frame["file"]).resolve(strict=True)
            if not frame_path.is_relative_to(directory.resolve()):
                raise ValueError(f"Frame path escapes recording directory: {frame_path}")
        if frame["timestamp_ns"] <= previous:
            raise ValueError(f"Non-increasing recorded timestamps: {directory}")
        previous = frame["timestamp_ns"]
    if status(validation) == "passed" and (not manifest.get("complete") or not validation.get("real_runtime")):
        raise ValueError(f"Passed recording lacks complete real-runtime evidence: {directory}")
    count = len(manifest["frames"])
    duration = (previous - manifest["frames"][0]["timestamp_ns"]) / 1e9 if count else 0
    result = {
        "manifest": manifest,
        "validation": validation,
        "frame_count": count,
        "measured_sampling_hz": (count - 1) / duration if duration > 0 else None,
    }
    if hdf5 is not None:
        result["hdf5_validation"] = hdf5
    audit = directory / "independent-frame-audit.json"
    if audit.exists():
        result["independent_audit"] = load_json(audit, inputs)
    return result


class ResourceCheck(HTMLParser):
    """Existing inline galleries are portable; reject dangling local resources."""

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key == "src" or (tag == "link" and key == "href"):
                if value and not value.startswith(("data:", "#")):
                    raise ValueError(f"Viewer has an external resource and cannot be embedded portably: {value}")


def picture(path, label):
    raw = path.resolve(strict=True).read_bytes()
    if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"Preview is not a PNG file: {path}")
    encoded = base64.b64encode(raw).decode("ascii")
    return (
        f"<figure><figcaption>{html.escape(label)}</figcaption>"
        f'<img loading="lazy" alt="{html.escape(label)}" src="data:image/png;base64,{encoded}"></figure>'
    )


def viewer(path, label, destination, name):
    document = path.resolve(strict=True).read_text()
    ResourceCheck().feed(document)
    target = destination / "viewers" / f"{name}.html"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)
    return (
        f"<details><summary>{html.escape(label)} · interactive viewer</summary>"
        f'<p><a href="viewers/{name}.html">Open saved viewer</a></p>'
        f'<iframe title="{html.escape(label)}" sandbox="allow-scripts" '
        f'srcdoc="{html.escape(document, quote=True)}" loading="lazy"></iframe></details>'
    )


CSS = """
:root{color-scheme:dark;
--bg:#101827;
--panel:#1b2a40;
--text:#ecf2fd;
--muted:#b6c4da;
--line:#41536b;
--link:#85caff}

:root.light{color-scheme:light;
--bg:#f3f6fb;
--panel:#fff;
--text:#172238;
--muted:#40526e;
--line:#b3bfd0;
--link:#12568d}

*{box-sizing:border-box}
body{font:16px system-ui;
margin:auto;
max-width:1440px;
padding:24px;
background:var(--bg);
color:var(--text)}

a{color:var(--link)}
p,li{line-height:1.65}
.muted{color:var(--muted)}
nav{position:sticky;
top:0;
background:var(--bg);
padding:12px 0;
z-index:1;
display:flex;
gap:16px;
flex-wrap:wrap}

.cards{display:flex;
flex-wrap:wrap;
gap:12px}
.card{background:var(--panel);
padding:16px;
border-radius:10px;
flex:1;
min-width:200px}
.card b{display:block;
font-size:24px}

.passed{color:#31b879}
.failed{color:#ed6868}
.table{overflow:auto}
table{border-collapse:collapse;
width:100%}
td,th{text-align:left;
padding:10px;
border-bottom:1px solid var(--line)}

img{width:100%;
height:auto}
figure{margin:24px 0}
figcaption{margin-bottom:10px}
pre{white-space:pre-wrap;
overflow-wrap:anywhere;
background:var(--panel);
padding:16px;
border-radius:10px;
font-size:13px}

iframe{width:100%;
height:850px;
border:1px solid var(--line)}
details{margin:18px 0}
button,input{font:inherit;
padding:8px;
max-width:100%}
section{scroll-margin-top:80px}

@media(max-width:600px){body{padding:14px}
.card{min-width:160px}
iframe{height:550px}
}

@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto}
}

@media print{nav,button,input,iframe{display:none}
body{background:white;
color:black}
pre{background:#eee}
}

"""
JS = """
const root=document.documentElement;
try{if(localStorage.getItem('robodojo-theme')==='light')root.classList.add('light')}catch(e){}
document.getElementById('theme').onclick=()=>{
  root.classList.toggle('light');
  try{localStorage.setItem('robodojo-theme',root.classList.contains('light')?'light':'dark')}catch(e){}
};
document.getElementById('filter').oninput=e=>{
  for(const row of document.querySelectorAll('tbody tr')){
    row.hidden=!row.textContent.toLowerCase().includes(e.target.value.toLowerCase());
  }
};
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--cpu-log", type=Path, required=True)
    parser.add_argument("--moving-dir", type=Path)
    for name in ("fluid", "cloth", "tactile", "sl", "replay", "physics-gallery"):
        parser.add_argument(f"--{name}-dir", f"--{name}", type=Path)
    parser.add_argument(
        "--require-passed", action="store_true", help="Exit nonzero unless every supplied acceptance result passed."
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = []
    suite = load_json(args.suite_dir / "summary.json", inputs)
    check_suite(suite)
    report = {
        "schema": "robodojo.acceptance-report.v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "suite_status": suite["status"],
        "suite_complete": suite["complete"],
        "suite": suite,
        "source_context": suite["context"],
        "cpu_tests": parse_cpu_log(args.cpu_log, inputs),
        "policy_evaluation": False,
        "limitations": LIMITATIONS,
        "evidence": {},
    }
    sections = []
    args.output.mkdir(parents=True, exist_ok=True)
    for name, label in (
        ("moving", "Moving wrists: synchronized modalities"),
        ("fluid", "PBD fluid: synchronized modalities"),
        ("cloth", "Surface FEM cloth: synchronized modalities"),
    ):
        directory = getattr(args, f"{name}_dir")
        if directory is None:
            continue
        data = recording(directory, inputs)
        report["evidence"][name] = data
        if name == "cloth":
            sections.append(
                "<p>Cloth-only TacSL records actual RGB/depth. Unsupported rigid SDF force fields are explicitly null; "
                "this capture does not establish nonzero contact forces.</p>"
            )
        sections.append(
            f"<h3>{html.escape(label)}</h3><p>Status: {html.escape(status(data['validation']))}; "
            f"{data['frame_count']} saved frames; measured simulated sampling: {data['measured_sampling_hz']} Hz.</p>"
        )
        sections.append(picture(directory / "multimodal.png", label))
        sections.append(viewer(directory / "multimodal.html", label, args.output, name))
    for name, json_name, png_name, label in (
        ("tactile", "result.json", "gallery.png", "TacSL physical contact response"),
        ("sl", "report.json", "structured_light.png", "Structured light: independent geometric validation"),
        ("replay", "report.json", None, "Native model/physics variant replay and reset"),
    ):
        directory = getattr(args, f"{name}_dir")
        if directory is None:
            continue
        data = load_json(directory / json_name, inputs)
        report["evidence"][name] = data
        sections.append(f"<h3>{html.escape(label)}</h3><p>Status: {html.escape(status(data))}.</p>")
        if png_name:
            sections.append(picture(directory / png_name, label))
    if args.physics_gallery_dir:
        gallery = load_json(args.physics_gallery_dir / "manifest.json", inputs)
        if not isinstance(gallery, list) or not gallery or any("report" not in item for item in gallery):
            raise ValueError("Physics gallery must retain its original per-capture reports")
        report["evidence"]["physics_gallery"] = gallery
        sections.append(
            picture(args.physics_gallery_dir / "physics-captures.png", "Native cloth/fluid motion and reset")
        )
        sections.append(
            viewer(args.physics_gallery_dir / "index.html", "Native physics gallery", args.output, "physics")
        )
    statuses = [suite["status"], report["cpu_tests"]["status"]]
    for name, data in report["evidence"].items():
        if name == "physics_gallery":
            statuses.extend(status(item["report"]) for item in data)
        else:
            statuses.append(status(data["validation"] if "validation" in data else data))
    report["status"] = (
        "failed" if "failed" in statuses else "passed" if all(x == "passed" for x in statuses) else "incomplete"
    )
    report["inputs"] = inputs
    rows = []
    for task in suite["tasks"]:
        duration = task.get("duration_s")
        elapsed = f"{duration:.1f}s" if isinstance(duration, (int, float)) else "unknown"
        rows.append(
            f"<tr><td>{html.escape(task['task'])}</td><td>{html.escape(task['status'])}</td>"
            f"<td>{elapsed}</td><td>{html.escape(str(task.get('error', '')))}</td></tr>"
        )
    title = "RoboDojo · Isaac Sim 6.1 / Isaac Lab 3.0 EA acceptance"
    document = (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
        f'content="width=device-width,initial-scale=1"><meta name="description" content="Existing runtime evidence, '
        f'source versions and explicit limitations"><meta property="og:title" content="{title}">'
        f"<title>{title}</title><style>{CSS}</style></head><body><h1>{title}</h1>"
        "<p>Existing RTX / PhysX / TacSL evidence. CPU fixture tests are recorded separately. "
        'This report does not measure policy success.</p><nav><a href="#overview">Overview</a>'
        '<a href="#captures">Captures</a><a href="#tasks">Tasks</a><a href="#limits">Limits and provenance</a>'
        '<button id="theme">Toggle theme</button></nav><section id="overview"><h2>Recorded acceptance status</h2>'
        f'<div class="cards"><div class="card"><b>{html.escape(report["status"])}</b>Combined supplied evidence</div>'
        f'<div class="card"><b>{suite["passed"]} / {len(suite["selected_tasks"])}</b>'
        f"Task scenes: {html.escape(suite['status'])}; failed: {suite['failed']}; complete: {suite['complete']}</div>"
        f'<div class="card"><b>{html.escape(report["cpu_tests"]["status"])}</b>'
        f"{html.escape(report['cpu_tests']['summary'])}</div></div></section>"
        f'<section id="captures"><h2>Original capture previews</h2>{"".join(sections)}</section>'
        '<section id="tasks"><h2>Individual task scenes</h2><p class="muted">Fresh process per task; '
        "native scene placement, reset, physics steps and RGBD acceptance. Original status values follow.</p>"
        '<label>Filter tasks <input id="filter" type="search"></label><div class="table"><table>'
        "<thead><tr><th>Task</th><th>Status</th><th>Duration</th><th>Error</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div></section>"
        '<section id="limits"><h2>Limits and original provenance</h2><ul>'
        + "".join(f"<li>{html.escape(note)}</li>" for note in LIMITATIONS)
        + "</ul><p>Images and inline viewers are portable. Full NPZ files remain in the original recording "
        "directories listed in the input provenance. Source JSON is preserved in report.json.</p>"
        f"<details><summary>Full original reports, packages and source commit</summary>"
        f"<pre>{html.escape(json.dumps(report, ensure_ascii=False, indent=2))}</pre></details></section>"
        f"<script>{JS}</script></body></html>"
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "index.html").write_text(document)
    print(
        json.dumps(
            {
                "html": str((args.output / "index.html").resolve()),
                "report": str((args.output / "report.json").resolve()),
                "status": report["status"],
                "suite_status": suite["status"],
                "passed": suite["passed"],
                "failed": suite["failed"],
            }
        )
    )

    return 1 if args.require_passed and report["status"] != "passed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
