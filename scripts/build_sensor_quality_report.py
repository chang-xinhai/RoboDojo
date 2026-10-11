"""Build an offline sensor comparison from existing runtime and audit evidence.

No simulator, CUDA kernels or image synthesis is invoked. Original JSON and its
SHA256 remain in report.json; passed source statuses are never inferred.
"""

import argparse
import base64
from datetime import UTC, datetime
import hashlib
import html
from html.parser import HTMLParser
import io
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / "reproduction/isaac61/outputs/moving-wrists-61-v5-final"
LIMITATIONS = [
    "IR is an RTX visible-radiance proxy, not a full NIR spectral/material simulation.",
    "SL depth is sparse: invalid/confidence-rejected pixels remain NaN. There is no ground-truth filling, "
    "and passed controls do not establish dense or universal coverage.",
    "IsaacLab TacSL uses its optical/penalty contact model; it is not UniVTAC UIPC or a marker-tracking sensor.",
    "A contact sequence can establish sensor response; it does not establish policy success or task completion.",
    "Reference videos are qualitative comparisons with different scenes, cameras and sensor models.",
    "Zero-force baseline frames are noncontact observations, not failed tactile sensing.",
    "Raw force is in N; depth/penetration is in m. Displayed depth colors and amplified RGB are visualizations.",
    "Original per-run source revisions and configuration hashes are retained; no new revision is assigned to old runs.",
]


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reject_constant(value):
    raise ValueError(f"Non-finite JSON number: {value}")


def read_json(path, inputs):
    path = path.resolve(strict=True)
    raw = path.read_bytes()
    value = json.loads(raw, parse_constant=reject_constant)

    def finite_numbers(item):
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError(f"Non-finite JSON number in {path}")
        if isinstance(item, dict):
            for child in item.values():
                finite_numbers(child)
        elif isinstance(item, list):
            for child in item:
                finite_numbers(child)

    finite_numbers(value)
    inputs.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
    return value


def source_status(value):
    """Require explicit, nonconflicting acceptance fields; preserve raw input."""
    if not isinstance(value, dict):
        raise ValueError("Runtime/audit report must be a JSON object")
    statuses = []
    if "status" in value:
        if not isinstance(value["status"], str) or value["status"] not in {"passed", "failed", "incomplete", "running"}:
            raise ValueError("Unrecognized original report status")
        statuses.append(value["status"])
    if "passed" in value:
        if not isinstance(value["passed"], bool):
            raise ValueError("Original passed field must be boolean")
        statuses.append("passed" if value["passed"] else "failed")
    if not statuses:
        return "incomplete"
    if len(set(statuses)) != 1:
        raise ValueError("Contradictory original acceptance fields")
    return statuses[0]


def local_file(value, directory):
    if not isinstance(value, str) or not value:
        raise ValueError("Source file path must be a nonempty string")
    raw = Path(value)
    if ".." in raw.parts:
        raise ValueError("Relative source traversal is not allowed")
    candidates = [raw] if raw.is_absolute() else [directory / raw, ROOT / raw]
    for candidate in candidates:
        if candidate.is_file():
            resolved = candidate.resolve(strict=True)
            if not resolved.is_relative_to(directory.resolve()):
                raise ValueError(f"Source file is outside the supplied evidence directory: {resolved}")
            return resolved
    raise ValueError(f"Missing source file: {value}")


def checked_hash(path, expected):
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"Missing SHA256 provenance for {path}")
    actual = fingerprint(path)
    if actual != expected:
        raise ValueError(f"Source SHA256 mismatch: {path}")
    return actual


def reference_evidence(directory, inputs):
    """Reference frames are qualitative, never runtime acceptance results."""
    candidates = [
        directory / "extraction-manifest.json",
        directory.parent / "review-reference/extraction-manifest.json",
    ]
    manifest_path = next((path for path in candidates if path.is_file()), None)
    if manifest_path is None:
        return {"status": "incomplete", "frames": [], "reason": "No original extraction manifest"}
    manifest = read_json(manifest_path, inputs)
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("Reference extraction manifest must contain original frame records")
    records = []
    for item in manifest:
        if not isinstance(item, dict):
            raise ValueError("Malformed reference extraction record")
        image = local_file(item.get("frame_path"), manifest_path.parent)
        checked_hash(image, item.get("frame_sha256"))
        source = Path(item.get("source", ""))
        if not source.is_file():
            raise ValueError(f"Original reference video is missing: {source}")
        checked_hash(source, item.get("source_sha256"))
        timestamp = item.get("timestamp_seconds")
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
            raise ValueError("Reference timestamp must be finite seconds")
        records.append({"original": item, "image": str(image), "source_verified": True})
    extras = {}
    for name in ("depthumi-reference-source.json", "official-tacsl-reference-metrics.json"):
        path = manifest_path.parent / name
        if path.is_file():
            extras[name] = read_json(path, inputs)
    return {"status": "verified_reference", "frames": records, "original_manifest": manifest, "context": extras}


def component_preview(reader, original):
    """Verify the component viewer's optical support ranking against raw depth.

    Its selected frame is an integer and its criterion is optical support, not
    force magnitude. Keep the producer JSON intact and return separate evidence.
    """
    from env.tactile_manager.contact_metrics import contact_frame_metrics
    from utils.multimodal_hdf5 import TACTILE_COMPONENT_SCHEMA

    if reader.metadata.get("observation_schema") != TACTILE_COMPONENT_SCHEMA:
        raise ValueError("Integer preview selection requires actual tactile component HDF5")
    if original.get("observation_schema") != TACTILE_COMPONENT_SCHEMA:
        raise ValueError("Component preview must declare its original observation schema")
    criterion = "maximum sum of actual largest positive optical-depth components over four pads; first tie"
    if original.get("selection") != criterion:
        raise ValueError("Unrecognized component preview selection criterion")
    index = original.get("selected_frame")
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(reader):
        raise ValueError("Component selected-frame index is outside the recording")
    scores = []
    for frame_index in range(len(reader)):
        frame = reader.read_frame(frame_index)
        score = 0.0
        for side in ("left", "right"):
            for finger in ("finger_0", "finger_1"):
                recorded = frame["contact_metrics"][side][finger]
                threshold = recorded.get("depth_threshold_m")
                minimum = recorded.get("minimum_component_pixels")
                if (
                    isinstance(threshold, bool)
                    or not isinstance(threshold, (int, float))
                    or not math.isfinite(threshold)
                    or threshold <= 0
                    or isinstance(minimum, bool)
                    or not isinstance(minimum, int)
                    or minimum < 1
                ):
                    raise ValueError("Component preview lacks valid original optical metric thresholds")
                actual = contact_frame_metrics(
                    frame["tactile"][side][finger],
                    depth_threshold_m=threshold,
                    minimum_component_pixels=minimum,
                )
                value = recorded.get("largest_component_pixel_fraction")
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or not math.isclose(value, actual["largest_component_pixel_fraction"], abs_tol=1e-12)
                    or recorded.get("closed_contour_hole_count") != actual["closed_contour_hole_count"]
                ):
                    raise ValueError("Component optical support/holes differ from actual HDF5 depth")
                score += actual["largest_component_pixel_fraction"]
        scores.append(score)
    strongest = max(range(len(scores)), key=scores.__getitem__)
    if index != strongest:
        raise ValueError("Component preview does not show the strongest optical support frame/first tie")
    selected = reader.read_frame(index)
    if original.get("selected_physics_frame") != selected["sync"]["physics_frame"]:
        raise ValueError("Component selected-frame clock differs from actual HDF5")
    if original.get("selected_phase") != selected["contact_demo"]["last_command"]["phase"]:
        raise ValueError("Component selected-frame phase differs from actual HDF5")
    if original.get("frame_count") != len(reader):
        raise ValueError("Component preview frame count differs from actual HDF5")
    return {
        "index": index,
        **selected["sync"],
        "score": scores[index],
        "score_semantics": "sum of largest positive optical-depth components over four pads (fraction)",
        "selected_by": criterion,
        "raw_optical_support_recomputed": True,
        "policy_evaluation": False,
    }


def bundle(directory, kind, inputs):
    """Keep failed/incomplete evidence visible; require linked audit for passed."""
    directory = directory.resolve(strict=True)
    names = ("validation.json",) if kind != "tactile" else ("validation.json", "result.json", "report.json")
    validation_path = next((directory / name for name in names if (directory / name).is_file()), None)
    if validation_path is None:
        return {"status": "incomplete", "directory": str(directory), "reason": "No original runtime validation"}
    validation = read_json(validation_path, inputs)
    validation_status = source_status(validation)
    manifest_path = directory / "manifest.json"
    manifest = read_json(manifest_path, inputs) if manifest_path.is_file() else None
    if manifest is not None and not isinstance(manifest, dict):
        raise ValueError("Capture manifest must be an object")
    audits = {}
    for name in ("independent-frame-audit.json", "independent-audit.json", "audit.json"):
        path = directory / name
        if path.is_file():
            audits[name] = read_json(path, inputs)
    audit_statuses = [source_status(value) for value in audits.values()]
    reasons = []
    if validation_status != "passed":
        reasons.append(f"Original runtime status: {validation_status}")
    if not audits or not all(value == "passed" for value in audit_statuses):
        reasons.append("Missing or non-passed independent audit")
    if validation.get("real_runtime") is not True:
        reasons.append("No explicit real_runtime:true in original validation")
    if kind != "tactile" and (manifest is None or manifest.get("complete") is not True):
        reasons.append("No complete original capture manifest")
    if kind == "capture" and (manifest is None or manifest.get("storage") != "hdf5"):
        reasons.append("New multimodal capture must use its original canonical HDF5 manifest")
    if validation.get("complete") is False or (manifest and manifest.get("complete") is False):
        reasons.append("Original run is incomplete")
    frame_counts = []
    for document in [validation, manifest, *audits.values()]:
        if document is None:
            continue
        count = document.get("frame_count")
        if count is None and isinstance(document.get("frames"), list):
            count = len(document["frames"])
        if count is not None:
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError("Evidence frame counts must be positive integers")
            frame_counts.append(count)
    if not frame_counts:
        reasons.append("No original positive frame inventory")
    elif len(set(frame_counts)) != 1:
        raise ValueError("Runtime/manifest/audit frame counts disagree")
    source_files = []
    hdf5_claims = []
    original_hdf5 = None
    hdf5_report = directory / "hdf5-validation.json"
    if hdf5_report.is_file():
        original_hdf5 = read_json(hdf5_report, inputs)
        if not isinstance(original_hdf5, dict):
            raise ValueError("HDF5 validation must be an object")
    for document in [validation, manifest, original_hdf5, *audits.values()]:
        if document and document.get("source_hdf5"):
            hdf5_claims.append((document["source_hdf5"], document.get("source_hdf5_sha256")))
    hdf5_check = None
    if kind == "capture" and manifest and manifest.get("storage") == "hdf5":
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from scripts.build_acceptance_report import recording

        hdf5_check = recording(directory, inputs)["hdf5_validation"]
        source_files.append({"path": hdf5_check["source_hdf5"], "sha256": hdf5_check["source_hdf5_sha256"]})
    for claim, digest in hdf5_claims:
        path = local_file(claim, directory)
        with path.open("rb") as stream:
            if stream.read(8) != b"\x89HDF\r\n\x1a\n":
                raise ValueError(f"Recorded HDF5 source has an invalid signature: {path}")
        source_files.append({"path": str(path), "sha256": checked_hash(path, digest) if digest else fingerprint(path)})
    if kind == "tactile" and hdf5_claims:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from utils.multimodal_hdf5 import validate_hdf5

        hdf5_check = validate_hdf5(Path(source_files[0]["path"]), require_complete=validation_status == "passed")
        if frame_counts and hdf5_check["validated_frames"] != frame_counts[0]:
            raise ValueError("Contact report/HDF5 frame counts disagree")
    if hdf5_claims and len({item["sha256"] for item in source_files}) != 1:
        raise ValueError("Validation and audit refer to different HDF5 recordings")
    if kind in {"capture", "tactile"} and not hdf5_claims:
        reasons.append("No original HDF5 path and SHA256 linkage")
    if kind in {"capture", "tactile"} and not any(
        audit.get("source_hdf5") and audit.get("source_hdf5_sha256") for audit in audits.values()
    ):
        reasons.append("Independent audit is not linked to the original HDF5 path/SHA256")
    if kind == "tactile" and frame_counts and frame_counts[0] < 2:
        reasons.append("Single-frame evidence does not establish a continuous contact sequence")
    # Older NPZ baseline audits already contain original per-frame hashes.
    if kind == "baseline":
        for audit in audits.values():
            for frame in audit.get("frames", []):
                if "file" in frame and "sha256" in frame:
                    path = local_file(frame["file"], directory)
                    source_files.append({"path": str(path), "sha256": checked_hash(path, frame["sha256"])})
    status = "failed" if "failed" in [validation_status, *audit_statuses] else "incomplete" if reasons else "passed"
    result = {
        "status": status,
        "directory": str(directory),
        "validation": validation,
        "manifest": manifest,
        "audits": audits,
        "frame_count": frame_counts[0] if frame_counts else None,
        "verified_sources": source_files,
        "hdf5_validation": hdf5_check,
        "original_hdf5_validation": original_hdf5,
        "limitations": reasons,
    }
    native = directory / "native-log-audit.json"
    if native.is_file():
        original = read_json(native, inputs)
        count = original.get("known_fatal_count") if isinstance(original, dict) else None
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("Malformed native log audit")
        result["native_log_audit"] = original
        if original["known_fatal_count"] > 0:
            result["status"] = "failed"
    preview = directory / "visualization.json"
    if preview.is_file():
        original = read_json(preview, inputs)
        if not isinstance(original, dict):
            raise ValueError("Visualization provenance must be an object")
        if original.get("source_hdf5"):
            source = local_file(original["source_hdf5"], directory)
            digest = checked_hash(source, original.get("source_hdf5_sha256"))
            if source_files and digest not in {item["sha256"] for item in source_files}:
                raise ValueError("Preview is sourced from a different HDF5 episode")
        if original.get("source_hdf5"):
            selected = original.get("selected_frame")
            if kind == "tactile" and isinstance(selected, int) and not isinstance(selected, bool):
                if str(ROOT) not in sys.path:
                    sys.path.insert(0, str(ROOT))
                from utils.multimodal_hdf5 import HDF5Reader

                with HDF5Reader(source) as reader:
                    result["verified_preview_selection"] = component_preview(reader, original)
                result["preview_selection_verified"] = True
                result["preview_contact_kind"] = (
                    "positive_optical_support" if result["verified_preview_selection"]["score"] > 0 else "noncontact"
                )
                result["visualization"] = original
                return result
            if not isinstance(selected, dict):
                raise ValueError("HDF5 preview must record its selected frame")
            index = selected.get("index")
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < (result["frame_count"] or 0):
                raise ValueError("Preview selected-frame index is outside the recording")
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))
            from reproduction.visualize_multimodal import _score
            from utils.multimodal_hdf5 import HDF5Reader

            with HDF5Reader(source) as reader:
                if any(selected.get(key) != value for key, value in reader.sync(index).items()):
                    raise ValueError("Preview selected-frame clock differs from actual HDF5")
                score, semantics = _score(reader.read_frame(index))
                if not math.isclose(score, selected.get("score", -1), rel_tol=1e-7, abs_tol=1e-12):
                    raise ValueError("Preview tactile score differs from its actual HDF5 frame")
                if selected.get("score_semantics") != semantics:
                    raise ValueError("Preview tactile score units/semantics differ from the actual data")
                best_score = max(_score(reader.read_frame(i))[0] for i in range(len(reader)))
                if score + 1e-12 < best_score:
                    raise ValueError("Preview does not show the strongest committed contact frame")
            result["preview_selection_verified"] = True
            result["preview_contact_kind"] = (
                "measured_force_contact"
                if score > 0 and semantics.endswith("(N)")
                else "signed_optical_proxy_only"
                if score > 0
                else "noncontact"
            )
        result["visualization"] = original
    return result


def control_evidence(directory, inputs):
    """Retain independently audited SL controls as separate geometric evidence."""
    directory = directory.resolve(strict=True)
    validation = read_json(directory / "report.json", inputs)
    original_status = source_status(validation)
    audit_path = directory / "independent-audit.json"
    audit = read_json(audit_path, inputs) if audit_path.is_file() else None
    audit_status = source_status(audit) if audit is not None else "incomplete"
    status = (
        "failed"
        if "failed" in (original_status, audit_status)
        else ("passed" if original_status == audit_status == "passed" else "incomplete")
    )
    reasons = ["Standalone geometric control; does not replace synchronized HDF5 capture or tactile contact."]
    paths = sorted(directory.glob("*.npz"))
    if len(paths) > 32 or sum(path.stat().st_size for path in paths) > 1024 * 1024 * 1024:
        raise ValueError("Geometric control exceeds the bounded 32-file / 1 GiB input inventory")
    expected = {}
    if audit is not None:
        if audit.get("source_report_sha256"):
            checked_hash(directory / "report.json", audit["source_report_sha256"])
        else:
            reasons.append("Independent control audit lacks its original report SHA256")
            if status == "passed":
                status = "incomplete"
        records = audit.get("source_files", [])
        if not isinstance(records, list):
            raise ValueError("Control audit source_files must be a list")
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("Malformed control audit source file")
            path = local_file(record.get("file", record.get("path")), directory)
            if path in expected:
                raise ValueError("Duplicate control source file in audit")
            expected[path] = record.get("sha256")
    if audit is not None and audit.get("frame_count") is not None:
        count = audit["frame_count"]
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("Control audit frame_count must be a positive integer")
    sources = []
    for path in paths:
        path = path.resolve(strict=True)
        digest = checked_hash(path, expected[path]) if path in expected else fingerprint(path)
        sources.append({"path": str(path), "sha256": digest})
        inputs.append({"path": str(path), "sha256": digest, "kind": "raw_geometric_control"})
    if not paths or {path.resolve() for path in paths} != set(expected):
        reasons.append("Raw control inventory is not completely linked to the independent audit hashes")
        if status == "passed":
            status = "incomplete"
    return {
        "status": status,
        "directory": str(directory),
        "validation": validation,
        "audits": {"independent-audit.json": audit} if audit is not None else {},
        "verified_sources": sources,
        "frame_count": audit.get("frame_count") if audit is not None else None,
        "limitations": reasons,
    }


def capture_viewer(directory, capture, inputs):
    """Verify a separate offline export without rewriting native capture evidence."""
    import numpy as np
    from PIL import Image

    from reproduction.visualize_multimodal import MODES, _score, mosaic
    from utils.multimodal import json_value
    from utils.multimodal_hdf5 import HDF5Reader

    directory = directory.resolve(strict=True)
    native = capture.get("hdf5_validation") or {}
    source = Path(native.get("source_hdf5", "")).resolve(strict=True)
    digest = native.get("source_hdf5_sha256")
    visualization = read_json(directory / "visualization.json", inputs)
    source_claim = visualization.get("source_hdf5")
    if (
        not isinstance(source_claim, str)
        or not Path(source_claim).is_absolute()
        or Path(source_claim).resolve(strict=True) != source
        or visualization.get("source_hdf5_sha256") != digest
    ):
        raise ValueError("Offline viewer must name the exact verified native HDF5 path/SHA256")
    checked_hash(source, digest)
    receipt_path = next(
        (
            directory / name
            for name in ("offline-generation-receipt.json", "candidate-generation-receipt.json")
            if (directory / name).is_file()
        ),
        None,
    )
    if receipt_path is None:
        raise ValueError("Offline viewer requires its separate generation receipt")
    receipt = read_json(receipt_path, inputs)
    if (
        source_status(receipt) != "passed"
        or receipt.get("offline_export") is not True
        or receipt.get("simulation_rerun") is not False
        or isinstance(receipt.get("cpu_process_return_code"), bool)
        or not isinstance(receipt.get("cpu_process_return_code"), int)
        or receipt.get("cpu_process_return_code") != 0
        or receipt.get("source_hdf5") != str(source)
        or receipt.get("source_hdf5_sha256_before") != digest
        or receipt.get("source_hdf5_sha256_after") != digest
        or isinstance(receipt.get("frame_stride"), bool)
        or not isinstance(receipt.get("frame_stride"), int)
        or receipt.get("frame_stride") != 1
        or receipt.get("max_frames") is not None
    ):
        raise ValueError("Offline generation receipt is not a complete unchanged-source export")
    paths = {kind: local_file(visualization.get(kind), directory) for kind in ("png", "video", "html")}
    files = [{"kind": kind, "path": str(path), "sha256": fingerprint(path)} for kind, path in paths.items()]
    inputs.extend({**item, "kind": "offline_viewer_" + item["kind"]} for item in files)
    depth_range = visualization.get("depth_range_m")
    delta_range = visualization.get("optical_delta_range_m")
    mode = visualization.get("tactile_display_mode")
    if (
        not isinstance(depth_range, list)
        or len(depth_range) != 2
        or not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in depth_range)
        or not 0 <= depth_range[0] < depth_range[1]
        or not isinstance(delta_range, list)
        or len(delta_range) != 2
        or not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in delta_range)
        or not delta_range[0] < 0 < delta_range[1]
        or delta_range[0] != -delta_range[1]
        or mode not in MODES
    ):
        raise ValueError("Offline viewer must declare finite display ranges and a supported mode")
    selected = visualization.get("selected_frame")
    if not isinstance(selected, dict):
        raise ValueError("Offline viewer must retain its actual selected frame/score")
    index = selected.get("index")
    with HDF5Reader(source) as reader:
        if (
            visualization.get("source_format") != "hdf5"
            or isinstance(visualization.get("frames"), bool)
            or not isinstance(visualization.get("frames"), int)
            or visualization.get("frames") != len(reader)
            or len(reader) != capture["frame_count"]
            or isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < len(reader)
        ):
            raise ValueError("Offline viewer must include the complete original HDF5 frame inventory")
        if any(selected.get(key) != value for key, value in reader.sync(index).items()):
            raise ValueError("Offline viewer selected-frame clock differs from native HDF5")
        selected_score, semantics = _score(reader.read_frame(index))
        if (
            not isinstance(selected.get("score"), (int, float))
            or isinstance(selected.get("score"), bool)
            or not math.isclose(selected["score"], selected_score, rel_tol=1e-7, abs_tol=1e-12)
            or selected.get("score_semantics") != semantics
        ):
            raise ValueError("Offline viewer selected-frame score/units differ from native HDF5")

        class Inventory(HTMLParser):
            def __init__(self):
                super().__init__()
                self.in_script = False
                self.parts = []
                self.rows = []
                self.scripts = 0
                self.max_script_bytes = 0
                self.best_score = -1.0
                self.selected_png_sha256 = None
                self.bootstrap = False
                self.controller = ""

            def handle_starttag(self, tag, attrs):
                if tag == "script":
                    self.in_script, self.parts = True, []

            def handle_data(self, text):
                if self.in_script:
                    self.parts.append(text)

            def handle_endtag(self, tag):
                if tag != "script":
                    return
                self.in_script = False
                code = "".join(self.parts)
                self.parts = []
                self.scripts += 1
                self.max_script_bytes = max(self.max_script_bytes, len(code.encode()))
                if self.max_script_bytes >= 536_870_888:
                    raise ValueError("Offline viewer script exceeds the V8 string limit")
                if code.strip() == "const frames=[];":
                    self.bootstrap = True
                    return
                if not code.startswith("frames.push("):
                    self.controller = code
                    return
                if not code.endswith(");") or len(self.rows) >= len(reader):
                    raise ValueError("Malformed/excess offline viewer frame chunks")
                chunk = json.loads(code[len("frames.push(") : -2], parse_constant=reject_constant)
                frame_index = len(self.rows)
                frame = reader.read_frame(frame_index)
                if chunk.get("details", {}).get("sync") != frame["sync"]:
                    raise ValueError("Offline viewer chunk clock differs from its original HDF5 frame")
                expected_details = {
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
                if chunk["details"] != json_value(expected_details):
                    raise ValueError("Offline viewer metadata differs from original camera/pad calibration")
                views = chunk.get("views", {})
                if set(views) != set(MODES):
                    raise ValueError("Offline viewer must retain all six declared display modes")
                hashes = {}
                for view_mode, uri in views.items():
                    if not isinstance(uri, str) or not uri.startswith("data:image/png;base64,"):
                        raise ValueError("Offline viewer has a nonlocal/invalid PNG frame")
                    raw = base64.b64decode(uri.split(",", 1)[1], validate=True)
                    pixels = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
                    expected = np.asarray(
                        mosaic(frame, mode=view_mode, depth_range=depth_range, delta_limit=delta_range[1])
                    )
                    if not np.array_equal(pixels, expected):
                        raise ValueError("Offline viewer PNG differs from the original frame/display mapping")
                    hashes[view_mode] = hashlib.sha256(raw).hexdigest()
                if frame_index == index:
                    self.selected_png_sha256 = hashes[mode]
                score, _ = _score(frame)
                self.best_score = max(self.best_score, score)
                self.rows.append({"index": frame_index, "sync": frame["sync"], "png_sha256": hashes})

        inventory = Inventory()
        with paths["html"].open() as stream:
            for block in iter(lambda: stream.read(1024 * 1024), ""):
                inventory.feed(block)
        inventory.close()
        if len(inventory.rows) != len(reader) or inventory.scripts != len(reader) + 2 or not inventory.bootstrap:
            raise ValueError("Offline viewer HTML omits original frames/bounded chunks")
        if (
            f"const selectedPreviewIndex={index};" not in inventory.controller
            or "frames[i].views[mode.value]" not in inventory.controller
            or "slider.oninput=show" not in inventory.controller
            or "mode.onchange=show" not in inventory.controller
        ):
            raise ValueError("Offline viewer controls do not select the declared original preview")
        if selected_score + 1e-12 < inventory.best_score:
            raise ValueError("Offline viewer must show the strongest actual committed pad score")
        if fingerprint(paths["png"]) != inventory.selected_png_sha256:
            raise ValueError("Offline viewer PNG is not its verified selected frame")
    probe = json.loads(
        subprocess.check_output(
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
                "json",
                str(paths["video"]),
            ],
            text=True,
        )
    )
    streams = probe.get("streams", [])
    if len(streams) != 1 or int(streams[0].get("nb_read_frames", -1)) != capture["frame_count"]:
        raise ValueError("Offline viewer MP4 omits original frames")
    return {
        "status": "passed",
        "directory": str(directory),
        "visualization": visualization,
        "generation_receipt": receipt,
        "verified_files": files,
        "frame_count": len(inventory.rows),
        "script_blocks": inventory.scripts,
        "max_script_bytes": inventory.max_script_bytes,
        "frames": inventory.rows,
        "selected_frame_verified": True,
        "png_pixel_lineage_verified": True,
        "role": "Offline export of original native data; no new native capture or policy evaluation",
    }


def metrics(data):
    """Present selected finite measurements with their original field names/units."""
    rows = []
    suffixes = ("_N", "_m", "_mm", "_Hz", "_hz", "_fraction", "_seconds", "_ns", "_rad", "_deg", "_dn", "_pixels")

    def visit(value, prefix=""):
        if isinstance(value, dict):
            for key, item in value.items():
                path = f"{prefix}.{key}" if prefix else key
                if key.endswith(suffixes) and not isinstance(item, bool) and isinstance(item, (int, float)):
                    if not math.isfinite(item):
                        raise ValueError(f"Non-finite metric {path}")
                    unit = next(suffix[1:] for suffix in suffixes if key.endswith(suffix))
                    rows.append({"field": path, "value": item, "unit": unit})
                elif key not in {"original_manifest", "ffmpeg_command"}:
                    visit(item, path)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{prefix}[{index}]")

    visit(data)
    return rows


def image(path, label, inputs):
    path = path.resolve(strict=True)
    raw = path.read_bytes()
    if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"Invalid PNG preview: {path}")
    inputs.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "kind": "preview_png"})
    return (
        f'<figure><img loading="lazy" alt="{html.escape(label)}" '
        f'src="data:image/png;base64,{base64.b64encode(raw).decode()}">'
        f"<figcaption>{html.escape(label)}</figcaption></figure>"
    )


def media(data, label, output, copy_videos, inputs):
    if data.get("offline_viewer"):
        data = data["offline_viewer"]
    directory = Path(data["directory"])
    parts = []
    visualization = data.get("visualization", {})
    preferred = (
        (visualization.get("preview_png") or visualization.get("png")) if isinstance(visualization, dict) else None
    )
    candidates = (
        [preferred]
        if preferred
        else [
            "contact-rich.png",
            "contact_rich.png",
            "contact.png",
            "multimodal.png",
            "gallery.png",
            "structured_light.png",
            "rgb-before-after.png",
        ]
    )
    preview = next((directory / name for name in candidates if name and (directory / name).is_file()), None)
    if preview:
        selection = data.get("verified_preview_selection", visualization.get("selected_frame", {}))
        caption = f"{label} · original saved preview"
        if isinstance(selection, dict) and selection:
            caption += (
                f" · frame {selection.get('index')} · {selection.get('score')} {selection.get('score_semantics')}"
            )
        parts.append(image(local_file(str(preview), directory), caption, inputs))
    for path in sorted(directory.glob("*.html")):
        parts.append(
            f'<p><a href="{html.escape(path.resolve().as_uri(), quote=True)}">'
            f"Open original {html.escape(path.name)}</a></p>"
        )
    for path in sorted(directory.glob("*.mp4")):
        target = path.resolve()
        inputs.append({"path": str(target), "sha256": fingerprint(target), "kind": "video"})
        if copy_videos:
            target = output / "videos" / f"{label.lower().replace(' ', '-')}-{path.name}"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            link = "videos/" + target.name
        else:
            link = target.as_uri()
        parts.append(
            f'<video controls preload="metadata" aria-label="{html.escape(label)}" '
            f'src="{html.escape(link, quote=True)}"></video>'
        )
    if not parts:
        parts.append('<p class="muted">No original preview files supplied.</p>')
    if visualization:
        parts.append(
            "<details><summary>Original preview frame selection / scaling</summary>"
            f"<pre>{html.escape(json.dumps(visualization, indent=2, ensure_ascii=False))}</pre></details>"
        )
    return "".join(parts)


CSS = """
:root{color-scheme:light;
--bg:#f4f6fa;
--fg:#142334;
--panel:white;
--muted:#55677b;
--line:#cbd5df;
--link:#075eaa}

:root.dark{color-scheme:dark;
--bg:#111a26;
--fg:#e4eef8;
--panel:#1c2a3c;
--muted:#b2c2d3;
--line:#40546b;
--link:#95cdff}

*{box-sizing:border-box}
body{margin:0 auto;
padding:24px;
max-width:1440px;
background:var(--bg);
color:var(--fg);
font:16px system-ui}

header p,li{line-height:1.7}
h1{max-width:900px}
a{color:var(--link)}
.muted{color:var(--muted)}

nav{position:sticky;
top:0;
background:var(--bg);
z-index:2;
display:flex;
gap:16px;
flex-wrap:wrap;
padding:14px 0}

button,input{font:inherit;
padding:8px;
max-width:100%}
.cards,.grid{display:grid;
grid-template-columns:repeat(auto-fit,minmax(280px,1fr));
gap:16px}

.card,figure,details{background:var(--panel);
padding:16px;
border:1px solid var(--line);
border-radius:12px}
.card strong{display:block;
font-size:24px}

figure{margin:12px 0}
img,video{width:100%;
height:auto;
display:block}
figcaption{padding-top:10px;
line-height:1.5;
overflow-wrap:anywhere}

.table{overflow:auto}
table{border-collapse:collapse;
width:100%;
font-size:14px}
th,td{text-align:left;
padding:10px;
border-bottom:1px solid var(--line)}

pre{white-space:pre-wrap;
overflow-wrap:anywhere;
font-size:12px}
details{margin:16px 0}
summary{cursor:pointer}
section{scroll-margin-top:90px;
margin-bottom:36px}

.passed,.verified_reference{color:#138254}
.failed{color:#ce3434}
.incomplete{color:#b47709}

@media(max-width:600px){body{padding:12px}
.cards,.grid{grid-template-columns:1fr}
nav{gap:10px;
font-size:14px}
}

@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto}
}

@media print{nav,button,input,video{display:none}
body{color:black;
background:white}
.card,figure{break-inside:avoid}
}

"""
JS = """
const root=document.documentElement;
try{if(localStorage.getItem('sensor-quality-theme')==='dark')root.classList.add('dark')}catch(e){}
document.getElementById('theme').onclick=()=>{root.classList.toggle('dark');
try{localStorage.setItem('sensor-quality-theme',root.classList.contains('dark')?'dark':'light')}catch(e){}};
document.getElementById('filter').oninput=e=>{for(const row of document.querySelectorAll('tbody tr'))
row.hidden=!row.textContent.toLowerCase().includes(e.target.value.toLowerCase());};
"""


def build(args):
    inputs = []
    if args.output.resolve() in {args.reference_dir.resolve(), args.capture_dir.resolve(), args.tactile_dir.resolve()}:
        raise ValueError("Output must not overwrite an original evidence directory")
    reference = reference_evidence(args.reference_dir.resolve(strict=True), inputs)
    evidence = {
        "capture": bundle(args.capture_dir, "capture", inputs),
        "tactile": bundle(args.tactile_dir, "tactile", inputs),
    }
    viewer_directory = getattr(args, "capture_viewer_dir", None)
    if viewer_directory is not None:
        if args.output.resolve() == viewer_directory.resolve():
            raise ValueError("Output must not overwrite an original offline viewer directory")
        evidence["capture"]["offline_viewer"] = capture_viewer(viewer_directory, evidence["capture"], inputs)
    if args.baseline_dir.is_dir():
        evidence["baseline"] = bundle(args.baseline_dir, "baseline", inputs)
    for index, directory in enumerate(getattr(args, "sl_dir", []) or []):
        evidence[f"sl_control_{index}"] = control_evidence(directory, inputs)
    statuses = [data["status"] for data in evidence.values()]
    status = "failed" if "failed" in statuses else "passed" if all(x == "passed" for x in statuses) else "incomplete"
    if reference["status"] != "verified_reference" and status == "passed":
        status = "incomplete"
    report = {
        "schema": "robodojo.sensor-quality-report.v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "generator": {
            "source_file": str(Path(__file__).resolve()),
            "source_sha256": fingerprint(Path(__file__).resolve()),
            "role": "Offline report generator, distinct from original capture revisions",
        },
        "status": status,
        "complete": status == "passed",
        "policy_evaluation": False,
        "reference": reference,
        "evidence": evidence,
        "limitations": LIMITATIONS,
        "inputs": inputs,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    figures = []
    for record in reference["frames"]:
        original = record["original"]
        label = f"{Path(original['source']).name} · t={original['timestamp_seconds']} s · original reference"
        figures.append(image(Path(record["image"]), label, inputs))
    panels = []
    metric_rows = []
    for name, data in evidence.items():
        control_kind = (
            "Frozen-state RGB" if data.get("validation", {}).get("same_physics_state") is True else "Structured-light"
        )
        label = {
            "capture": "New multimodal capture",
            "tactile": "Continuous tactile contact",
            "baseline": "Old noncontact baseline",
        }.get(name, f"{control_kind} control · {Path(data['directory']).name}")
        reasons = " · ".join(data.get("limitations", [data.get("reason", "")]))
        panels.append(
            f'<section id="{name}"><h2>{html.escape(label)}</h2><p>Status: '
            f'<strong class="{data["status"]}">{data["status"]}</strong>; '
            f"{data.get('frame_count') or 'unverified'} frames.</p><p>{html.escape(reasons)}</p>"
            + media(data, label, args.output, args.copy_videos, inputs)
            + "</section>"
        )
        for row in metrics(data):
            metric_rows.append(
                f"<tr><td>{name}</td><td>{html.escape(row['field'])}</td>"
                f"<td>{row['value']:.6g}</td><td>{html.escape(row['unit'])}</td></tr>"
            )
    title = "RoboDojo sensor quality · original references and measured captures"
    document = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta name="description" '
        'content="Original video references, measured sensor captures and preserved audit provenance">'
        f'<meta property="og:title" content="{title}"><title>{title}</title><style>{CSS}</style></head><body>'
        f"<header><h1>{title}</h1><p>Visual comparison of saved real-runtime evidence. "
        "No simulator or generated substitute images are used by this report.</p></header>"
        '<nav><a href="#overview">Evidence</a><a href="#reference">Reference videos</a>'
        '<a href="#capture">New capture</a><a href="#tactile">Contact sequence</a>'
        '<a href="#measurements">Measurements</a><a href="#provenance">Limits / provenance</a>'
        '<button id="theme">Toggle theme</button></nav><section id="overview"><h2>Original evidence status</h2>'
        f'<div class="cards"><div class="card"><strong class="{status}">{status}</strong>'
        "Combined supplied evidence; no policy success claim</div>"
        + "".join(
            f'<div class="card"><strong class="{data["status"]}">{data["status"]}</strong>{name}</div>'
            for name, data in evidence.items()
        )
        + '</div></section><section id="reference"><h2>Reference video frames</h2>'
        "<p>Source video and extracted frame SHA256 values were checked. Comparisons are qualitative; "
        'reference scenes and sensor models differ.</p><div class="grid">'
        + "".join(figures)
        + "</div></section>"
        + "".join(panels)
        + '<section id="measurements"><h2>Original reported measurements</h2>'
        "<p>Field names identify the source report and units. Fraction means 0–1; "
        "ns denotes nanoseconds. No improvement percentage is inferred across different scenes.</p>"
        '<label>Filter measurements <input id="filter" type="search"></label><div class="table"><table>'
        "<thead><tr><th>Evidence</th><th>Original field</th><th>Value</th><th>Unit</th></tr></thead><tbody>"
        + "".join(metric_rows)
        + '</tbody></table></div></section><section id="provenance"><h2>Limits and preserved provenance</h2><ul>'
        + "".join(f"<li>{html.escape(note)}</li>" for note in LIMITATIONS)
        + "</ul><p>PNG previews and JSON are embedded. Full HDF5/NPZ remains at its original verified path. "
        "Videos use local links unless --copy-videos is selected; original viewers remain local links.</p>"
        "<details><summary>Original reports, source hashes and package/configuration metadata</summary>"
        f"<pre>{html.escape(json.dumps(report, ensure_ascii=False, indent=2))}</pre></details></section>"
        f"<script>{JS}</script></body></html>"
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (args.output / "index.html").write_text(document)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "capture", "tactile"):
        parser.add_argument(f"--{name}-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--capture-viewer-dir", type=Path, help="Separate verified offline export of the original capture"
    )
    parser.add_argument(
        "--sl-dir", type=Path, action="append", help="Optional independent SL geometric controls; repeatable"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Local output directory containing index.html/report.json"
    )
    parser.add_argument(
        "--copy-videos", action="store_true", help="Copy existing MP4 files; otherwise link their local paths"
    )
    parser.add_argument(
        "--require-passed", action="store_true", help="Exit 1 if supplied evidence is failed/incomplete"
    )
    args = parser.parse_args(argv)
    report = build(args)
    print(json.dumps({"status": report["status"], "html": str((args.output / "index.html").resolve())}))
    return 1 if args.require_passed and report["status"] != "passed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
