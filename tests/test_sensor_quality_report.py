"""Malformed or stale evidence cannot become a completed visual quality claim."""

import json
from types import SimpleNamespace

from PIL import Image
import pytest

from scripts.build_sensor_quality_report import build, bundle, control_evidence, fingerprint, read_json, source_status
from tests.test_acceptance_report import hdf5_recording


def write_json(path, data):
    path.write_text(json.dumps(data))


def audited_capture(directory):
    source = hdf5_recording(directory)
    audit = {
        "passed": True,
        "fixture_not_runtime": True,
        "frame_count": 2,
        "source_hdf5": str(source),
        "source_hdf5_sha256": fingerprint(source),
    }
    write_json(directory / "independent-frame-audit.json", audit)
    return source


def reference(directory):
    directory.mkdir()
    video = directory / "fixture.mp4"
    video.write_bytes(b"unit-test reference bytes; not a runtime video")
    image = directory / "fixture.png"
    Image.new("RGB", (2, 2), "red").save(image)
    write_json(
        directory / "extraction-manifest.json",
        [
            {
                "source": str(video),
                "source_sha256": fingerprint(video),
                "frame_path": str(image),
                "frame_sha256": fingerprint(image),
                "timestamp_seconds": 2,
            }
        ],
    )
    return directory


@pytest.mark.parametrize(
    "data",
    [
        {"status": ["passed"]},
        {"status": "complete"},
        {"passed": "true"},
        {"status": "passed", "passed": False},
    ],
)
def test_conflicting_or_malformed_acceptance_status_is_rejected(data):
    with pytest.raises(ValueError):
        source_status(data)
    assert source_status({"other_metadata": True}) == "incomplete"


@pytest.mark.parametrize("bad_number", ["NaN", "Infinity", "1e999"])
def test_nonfinite_metadata_is_rejected_even_outside_displayed_metrics(tmp_path, bad_number):
    source = tmp_path / "report.json"
    source.write_text('{"hidden_frame": {"position": ' + bad_number + "}}")
    with pytest.raises(ValueError, match="Non-finite"):
        read_json(source, [])


def test_passed_capture_requires_actual_hdf5_and_linked_independent_audit(tmp_path):
    source = audited_capture(tmp_path / "capture")
    report = bundle(source.parent, "capture", [])
    assert report["status"] == "passed"
    assert report["hdf5_validation"]["validated_frames"] == 2
    assert report["verified_sources"][0]["sha256"] == fingerprint(source)
    assert report["validation"]["fixture_not_runtime"] is True
    assert report["manifest"]["run"] == {"test_fixture": True}


@pytest.mark.parametrize("audit_status", [None, False])
def test_missing_or_failed_audit_never_becomes_passed(tmp_path, audit_status):
    source = hdf5_recording(tmp_path / "capture")
    if audit_status is not None:
        write_json(source.parent / "audit.json", {"passed": audit_status})
    report = bundle(source.parent, "capture", [])
    assert report["status"] == ("incomplete" if audit_status is None else "failed")


def test_source_hash_and_original_frame_count_are_not_trusted_blindly(tmp_path):
    source = audited_capture(tmp_path / "capture")
    path = source.parent / "independent-frame-audit.json"
    audit = json.loads(path.read_text())
    audit["source_hdf5_sha256"] = "0" * 64
    write_json(path, audit)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        bundle(source.parent, "capture", [])
    audit["source_hdf5_sha256"] = fingerprint(source)
    audit["frame_count"] = 3
    write_json(path, audit)
    with pytest.raises(ValueError, match="frame counts disagree"):
        bundle(source.parent, "capture", [])


@pytest.mark.parametrize("damage", ["index", "clock", "score", "units"])
def test_preview_must_point_to_its_actual_frame_and_measured_tactile_score(tmp_path, damage):
    source = audited_capture(tmp_path / "capture")
    selected = {
        "index": 1,
        "physics_frame": 2,
        "timestamp_ns": 80_000_000,
        "score": 0.0,
        "score_semantics": "maximum absolute measured normal force (N)",
    }
    if damage == "index":
        selected["index"] = 99
    elif damage == "clock":
        selected["timestamp_ns"] = 123
    elif damage == "score":
        selected["score"] = 1000.0
    else:
        selected["score_semantics"] = "fabricated pressure in Pa"
    write_json(
        source.parent / "visualization.json",
        {
            "source_hdf5": str(source),
            "source_hdf5_sha256": fingerprint(source),
            "selected_frame": selected,
        },
    )
    with pytest.raises(ValueError, match="selected-frame|score"):
        bundle(source.parent, "capture", [])


def test_missing_acceptance_is_visibly_incomplete_and_output_preserves_all_original_inputs(tmp_path):
    ref = reference(tmp_path / "reference")
    capture = audited_capture(tmp_path / "capture")
    tactile = tmp_path / "tactile"
    tactile.mkdir()
    write_json(tactile / "report.json", {"status": "running", "real_runtime": True, "note": "<script>bad()</script>"})
    output = tmp_path / "report"
    args = SimpleNamespace(
        reference_dir=ref,
        capture_dir=capture.parent,
        tactile_dir=tactile,
        baseline_dir=tmp_path / "no-baseline",
        output=output,
        copy_videos=False,
    )
    result = build(args)
    assert result["status"] == "incomplete" and result["complete"] is False
    assert result["policy_evaluation"] is False
    assert result["evidence"]["tactile"]["validation"]["status"] == "running"
    document = (output / "index.html").read_text()
    assert "data:image/png;base64," in document
    assert "&lt;script&gt;bad()&lt;/script&gt;" in document
    assert "<script>bad()</script>" not in document
    assert all(len(item["sha256"]) == 64 for item in result["inputs"])
    original_manifest = json.loads((capture.parent / "manifest.json").read_text())
    assert result["evidence"]["capture"]["manifest"] == original_manifest


def test_changed_reference_frame_and_source_overwrite_are_rejected(tmp_path):
    ref = reference(tmp_path / "reference")
    capture = audited_capture(tmp_path / "capture")
    args = SimpleNamespace(
        reference_dir=ref,
        capture_dir=capture.parent,
        tactile_dir=capture.parent,
        baseline_dir=tmp_path / "no-baseline",
        output=ref,
        copy_videos=False,
    )
    with pytest.raises(ValueError, match="overwrite"):
        build(args)
    args.output = tmp_path / "output"
    (ref / "fixture.png").write_bytes(b"modified")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        build(args)


def test_tactile_component_does_not_need_or_fabricate_three_camera_payloads(tmp_path):
    source = hdf5_recording(tmp_path / "tactile", component=True)
    write_json(
        source.parent / "audit.json",
        {
            "passed": True,
            "frame_count": 2,
            "source_hdf5": str(source),
            "source_hdf5_sha256": fingerprint(source),
        },
    )
    report = bundle(source.parent, "tactile", [])
    assert report["status"] == "passed"
    assert report["hdf5_validation"]["observation_schema"] == "robodojo.tactile_component.v1"
    assert report["hdf5_validation"]["validated_frames"] == 2


def test_idle_frame_cannot_replace_a_stronger_recorded_contact_preview(tmp_path):
    import h5py

    source = audited_capture(tmp_path / "capture")
    with h5py.File(source, "r+") as handle:
        handle["frames/00000001/tactile/left/finger_0/normal_force"][:] = 0.2
    digest = fingerprint(source)
    write_json(
        source.parent / "independent-frame-audit.json",
        {
            "passed": True,
            "frame_count": 2,
            "source_hdf5": str(source),
            "source_hdf5_sha256": digest,
        },
    )
    write_json(
        source.parent / "visualization.json",
        {
            "source_hdf5": str(source),
            "source_hdf5_sha256": digest,
            "selected_frame": {
                "index": 0,
                "physics_frame": 1,
                "timestamp_ns": 40_000_000,
                "score": 0.0,
                "score_semantics": "maximum absolute official TacSL taxel Z estimate (N)",
            },
        },
    )
    with pytest.raises(ValueError, match="strongest committed contact"):
        bundle(source.parent, "capture", [])


def test_optional_controls_keep_separate_status_and_reject_stale_raw_source_hashes(tmp_path):
    directory = tmp_path / "plane"
    directory.mkdir()
    write_json(directory / "report.json", {"passed": True, "fixture_not_runtime": True})
    raw = directory / "cam_head.npz"
    raw.write_bytes(b"hashed synthetic control source, not native runtime evidence")
    audit = {
        "status": "passed",
        "frame_count": 1,
        "source_report_sha256": fingerprint(directory / "report.json"),
        "source_files": [{"file": raw.name, "sha256": fingerprint(raw)}],
    }
    write_json(directory / "independent-audit.json", audit)
    result = control_evidence(directory, [])
    assert result["status"] == "passed"
    assert result["validation"]["fixture_not_runtime"] is True
    assert "does not replace" in result["limitations"][0]
    raw.write_bytes(b"modified after independent audit")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        control_evidence(directory, [])


def test_control_without_source_linkage_is_incomplete_and_input_inventory_is_bounded(tmp_path):
    directory = tmp_path / "control"
    directory.mkdir()
    write_json(directory / "report.json", {"passed": True})
    write_json(directory / "independent-audit.json", {"passed": True})
    (directory / "frame.npz").write_bytes(b"fixture")
    assert control_evidence(directory, [])["status"] == "incomplete"
    for index in range(33):
        (directory / f"frame{index}.npz").write_bytes(b"fixture")
    with pytest.raises(ValueError, match="bounded"):
        control_evidence(directory, [])


def audited_component_preview(directory):
    """Three real typed fixture frames: negative depth, ring contact, less support.

    The idle frame has larger model force deliberately, so force ranking cannot
    accidentally pass a test for the component's declared optical criterion.
    """
    import numpy as np

    from env.tactile_manager.contact_metrics import contact_frame_metrics
    from tests.test_multimodal import fixture
    from utils.multimodal_hdf5 import TACTILE_COMPONENT_SCHEMA, HDF5Writer

    directory.mkdir()
    writer = HDF5Writer(directory, {"fixture_not_runtime": True}, observation_schema=TACTILE_COMPONENT_SCHEMA)
    for index in range(3):
        frame = fixture(index + 1)
        frame["sync"]["schema"] = TACTILE_COMPONENT_SCHEMA
        del frame["vision"]
        frame["contact_demo"] = {"last_command": {"phase": f"fixture_phase_{index}"}}
        frame["contact_metrics"] = {}
        for side in ("left", "right"):
            frame["contact_metrics"][side] = {}
            for finger in ("finger_0", "finger_1"):
                pad = frame["tactile"][side][finger]
                pad["rgb"] = np.zeros((24, 24, 3), np.uint8)
                pad["depth"] = np.ones((24, 24, 1), np.float32)
                delta = np.full((24, 24), -0.001, np.float32)
                if index == 1:
                    delta[3:21, 3:21] = 0.001
                    delta[8:16, 8:16] = -0.001  # Actual enclosed 64-pixel hole.
                elif index == 2:
                    delta[3:11, 3:11] = 0.001
                pad["optical_depth_delta"] = delta[..., None]
                pad["normal_force"][:] = -2.0 if index == 0 else 0.1
                frame["contact_metrics"][side][finger] = contact_frame_metrics(pad)
        writer.append(frame)
    source = writer.complete()
    digest = fingerprint(source)
    report = {
        "status": "passed",
        "real_runtime": True,
        "fixture_not_runtime": True,
        "frame_count": 3,
        "source_hdf5": str(source),
        "source_hdf5_sha256": digest,
    }
    write_json(directory / "report.json", report)
    write_json(directory / "independent-audit.json", report)
    preview = {
        "observation_schema": TACTILE_COMPONENT_SCHEMA,
        "source_hdf5": str(source),
        "source_hdf5_sha256": digest,
        "frame_count": 3,
        "selected_frame": 1,
        "selected_physics_frame": 2,
        "selected_phase": "fixture_phase_1",
        "selection": "maximum sum of actual largest positive optical-depth components over four pads; first tie",
    }
    write_json(directory / "visualization.json", preview)
    return source, preview


def test_component_integer_preview_uses_raw_ring_support_not_force_and_preserves_original_json(tmp_path):
    source, original = audited_component_preview(tmp_path / "component")
    result = bundle(source.parent, "tactile", [])
    assert result["status"] == "passed"
    assert result["visualization"] == original
    verified = result["verified_preview_selection"]
    assert verified["index"] == 1
    assert verified["score"] == pytest.approx(4 * (18 * 18 - 64) / (24 * 24))
    assert verified["raw_optical_support_recomputed"] is True
    assert verified["score_semantics"].endswith("(fraction)")
    assert result["preview_contact_kind"] == "positive_optical_support"


@pytest.mark.parametrize("damage", ["idle", "bool", "clock", "phase", "schema", "criterion"])
def test_component_integer_preview_rejects_wrong_selection_or_clock(tmp_path, damage):
    source, original = audited_component_preview(tmp_path / "component")
    field, value = {
        "idle": ("selected_frame", 0),
        "bool": ("selected_frame", True),
        "clock": ("selected_physics_frame", 123),
        "phase": ("selected_phase", "wrong"),
        "schema": ("observation_schema", "invented"),
        "criterion": ("selection", "maximum force"),
    }[damage]
    original[field] = value
    write_json(source.parent / "visualization.json", original)
    with pytest.raises(ValueError, match="preview|selected-frame"):
        bundle(source.parent, "tactile", [])


@pytest.mark.parametrize("field,value", [("largest_component_pixel_fraction", 0.99), ("closed_contour_hole_count", 0)])
def test_component_preview_recomputes_support_and_holes_instead_of_trusting_saved_metrics(tmp_path, field, value):
    import h5py

    source, _ = audited_component_preview(tmp_path / "component")
    with h5py.File(source, "r+") as handle:
        node = handle[f"frames/00000001/contact_metrics/left/finger_0/{field}/value"]
        node[()] = json.dumps(value)
    digest = fingerprint(source)
    for name in ("report.json", "independent-audit.json", "visualization.json"):
        path = source.parent / name
        data = json.loads(path.read_text())
        data["source_hdf5_sha256"] = digest
        write_json(path, data)
    with pytest.raises(ValueError, match="support/holes"):
        bundle(source.parent, "tactile", [])


def test_integer_component_preview_cannot_label_full_camera_hdf5_as_component(tmp_path):
    from scripts.build_sensor_quality_report import component_preview
    from utils.multimodal_hdf5 import HDF5Reader

    full = audited_capture(tmp_path / "capture")
    _, declaration = audited_component_preview(tmp_path / "component")
    with HDF5Reader(full) as reader, pytest.raises(ValueError, match="actual tactile component HDF5"):
        component_preview(reader, declaration)
