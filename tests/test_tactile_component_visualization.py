"""CPU display contracts only; these fixture arrays are not runtime evidence."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from reproduction.isaac61.visualize_tactile_component import PADS, mosaic, render
from tests.test_multimodal import fixture
from utils.multimodal_hdf5 import TACTILE_COMPONENT_SCHEMA, HDF5Reader, HDF5Writer


def component_fixture(index):
    frame = fixture(index)
    del frame["vision"]
    frame["sync"]["schema"] = TACTILE_COMPONENT_SCHEMA
    frame["contact_demo"] = {"last_command": {"phase": "hold", "elapsed_time_s": index / 25}}
    frame["contact_metrics"] = {}
    for side, finger in PADS:
        data = frame["tactile"][side][finger]
        data["optical_depth_delta"] = np.full_like(data["depth"], -0.001)
        data["metadata"] = {"array_size": [2, 4], "taxel_array_axes": ["sensor_local_x", "sensor_local_z"]}
        frame["contact_metrics"].setdefault(side, {})[finger] = {
            "largest_component_pixel_fraction": 0.5 if index == 2 else 0
        }
    return frame


def test_component_hdf5_direct_video_and_selection_preserve_signed_arrays(tmp_path):
    writer = HDF5Writer(
        tmp_path,
        {"sampling_frequency_hz": 25, "synthetic_cpu_fixture": True},
        observation_schema=TACTILE_COMPONENT_SCHEMA,
    )
    for index in (1, 2, 3):
        writer.append(component_fixture(index))
    path = writer.complete()
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    report = render(path)
    assert report["selected_frame"] == 1
    assert report["selected_physics_frame"] == 2
    assert Path(report["contact_png"]).stat().st_size > 100
    assert Path(report["contact_mp4"]).stat().st_size > 100
    assert report["source_hdf5_sha256"] == before == hashlib.sha256(path.read_bytes()).hexdigest()
    assert not list(tmp_path.glob("*.npz"))
    with HDF5Reader(path) as reader:
        assert "vision" not in reader.read_frame(0)
        np.testing.assert_array_equal(
            reader.read_frame(0)["tactile"]["left"]["finger_0"]["optical_depth_delta"],
            component_fixture(1)["tactile"]["left"]["finger_0"]["optical_depth_delta"],
        )


def test_component_view_rejects_invented_taxel_axis_orientation():
    frame = component_fixture(1)
    del frame["tactile"]["left"]["finger_0"]["metadata"]["taxel_array_axes"]
    with pytest.raises(ValueError, match="undeclared taxel"):
        mosaic(frame)
