"""Actual HDF5 evidence checks; synthetic fixtures never establish runtime acceptance."""

import json

import h5py
import pytest

from scripts.build_acceptance_report import recording
from tests.test_multimodal import fixture
from utils.multimodal import MultimodalWriter
from utils.multimodal_hdf5 import TACTILE_COMPONENT_SCHEMA, HDF5Reader, HDF5Writer


def hdf5_recording(directory, *, component=False):
    directory.mkdir(parents=True, exist_ok=True)
    kwargs = {"observation_schema": TACTILE_COMPONENT_SCHEMA} if component else {}
    writer = HDF5Writer(directory, {"test_fixture": True}, **kwargs)
    for index in (1, 2):
        observation = fixture(index)
        if component:
            observation["sync"]["schema"] = TACTILE_COMPONENT_SCHEMA
            observation["vision"] = {}
        writer.append(observation)
    frames = [{"index": i, "physics_frame": i + 1, "timestamp_ns": (i + 1) * 40_000_000} for i in range(2)]
    validation = {"status": "passed", "real_runtime": True, "fixture_not_runtime": True, "frames": frames}
    path = writer.complete(validation=validation)
    with HDF5Reader(path) as reader:
        manifest = {
            "storage": "hdf5",
            "complete": True,
            "file": path.name,
            "run": {"test_fixture": True},
            "frames": [{"file": path.name, "hdf5_frame": i, **reader.sync(i)} for i in range(len(reader))],
        }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "validation.json").write_text(json.dumps(validation))
    return path


def test_hdf5_report_checks_actual_data_without_npz_and_preserves_original_metadata(tmp_path):
    source = hdf5_recording(tmp_path / "capture")
    inputs = []
    report = recording(source.parent, inputs)
    assert report["frame_count"] == 2 and report["measured_sampling_hz"] == pytest.approx(25)
    assert report["manifest"]["run"] == {"test_fixture": True}
    assert report["validation"]["fixture_not_runtime"] is True
    assert report["hdf5_validation"]["source_hdf5"] == str(source)
    assert report["hdf5_validation"]["validated_frames"] == 2
    assert not list(source.parent.glob("*.npz"))
    assert inputs[-1]["sha256"] == report["hdf5_validation"]["source_hdf5_sha256"]


@pytest.mark.parametrize("damage", ["count", "clock", "index", "escape"])
def test_hdf5_report_rejects_manifest_claims_that_disagree_with_source(tmp_path, damage):
    source = hdf5_recording(tmp_path / "capture")
    path = source.parent / "manifest.json"
    manifest = json.loads(path.read_text())
    if damage == "count":
        manifest["frames"].pop()
        validation_path = source.parent / "validation.json"
        validation = json.loads(validation_path.read_text())
        validation["frames"].pop()
        validation_path.write_text(json.dumps(validation))
    elif damage == "clock":
        manifest["frames"][0]["timestamp_ns"] += 1
    elif damage == "index":
        manifest["frames"][1]["hdf5_frame"] = 0
    else:
        external = tmp_path / "outside.hdf5"
        source.rename(external)
        manifest["file"] = "../outside.hdf5"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="differs|differ|escapes"):
        recording(source.parent, [])


@pytest.mark.parametrize("damage", ["incomplete", "stale_sensor"])
def test_hdf5_report_rejects_invalid_recording_even_with_passed_json(tmp_path, damage):
    source = hdf5_recording(tmp_path / "capture")
    with h5py.File(source, "r+") as handle:
        if damage == "incomplete":
            handle.attrs["complete"] = False
        else:
            # Change the actual typed camera timestamp, keeping the JSON reports passed.
            node = handle["frames/00000000/vision/head/timestamp_ns/value"]
            node[()] = json.dumps(0)
    with pytest.raises(ValueError, match="Recording|timestamp|completion"):
        recording(source.parent, [])


def test_component_hdf5_cannot_be_presented_as_three_camera_capture(tmp_path):
    source = hdf5_recording(tmp_path / "component", component=True)
    with pytest.raises(ValueError, match="tactile-only component"):
        recording(source.parent, [])


def test_legacy_npz_acceptance_keeps_its_original_source_fields(tmp_path):
    directory = tmp_path / "legacy"
    writer = MultimodalWriter(directory, {"git_commit": "old-source", "fixture_not_runtime": True})
    writer.append(fixture(1))
    writer.append(fixture(2))
    writer.close()
    manifest = json.loads((directory / "manifest.json").read_text())
    validation = {
        "status": "passed",
        "real_runtime": True,
        "fixture_not_runtime": True,
        "frames": [{"index": index} for index in range(2)],
    }
    (directory / "validation.json").write_text(json.dumps(validation))
    report = recording(directory, [])
    assert report["manifest"] == manifest
    assert report["manifest"]["run"]["git_commit"] == "old-source"
    assert "hdf5_validation" not in report
    assert report["frame_count"] == 2
