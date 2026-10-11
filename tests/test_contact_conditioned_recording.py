"""Contact-zero sentinels and lossless raw optical data on explicit CPU fixtures."""

import importlib.util
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from reproduction.visualize_multimodal import mosaic
from tests.test_multimodal import fixture
from utils.multimodal import validate_multimodal
from utils.multimodal_hdf5 import HDF5Reader, HDF5Writer


def contact_fixture(active, frame=1):
    observation = fixture(frame)
    for fingers in observation["tactile"].values():
        for data in fingers.values():
            data["provenance"].update(contact_conditioned_observations=True, force_field_available=True)
            data["metadata"] = {"contact_object_labels": ["tube0", "tube1", "tube2"]}
            data["contact_active"] = np.asarray(active, dtype=np.bool_)
            data["raw_optical_rgb"] = np.full_like(data["rgb"], 179)
            data["raw_optical_depth"] = np.full_like(data["depth"], 0.025)
            data["raw_optical_depth_delta"] = np.full_like(data["depth"], -0.001)
            data["optical_depth_delta"] = data["raw_optical_depth_delta"].copy()
            for field, raw in (
                ("rgb", "raw_optical_rgb"),
                ("depth", "raw_optical_depth"),
                ("optical_depth_delta", "raw_optical_depth_delta"),
            ):
                data[field] = data[raw].copy() if active else np.zeros_like(data[raw])
            if active:
                data["penetration_depth"][0] = 0.001
                data["normal_force"][0] = 0.02
    return observation


class ContactRecordingTests(unittest.TestCase):
    def test_idle_contact_release_hdf5_preserves_zero_outputs_and_signed_raw_optics(self):
        with tempfile.TemporaryDirectory() as directory:
            with HDF5Writer(directory, {"synthetic_cpu_fixture": True}) as writer:
                for index, active in enumerate((False, True, False), start=1):
                    writer.append(contact_fixture(active, index))
            with HDF5Reader(writer.path) as reader:
                for index, active in enumerate((False, True, False)):
                    frame = reader.read_frame(index)
                    data = frame["tactile"]["left"]["finger_0"]
                    self.assertEqual(bool(data["contact_active"]), active)
                    self.assertTrue((data["raw_optical_depth"] > 0).all())
                    self.assertTrue((data["raw_optical_depth_delta"] < 0).all())
                    for field in ("rgb", "depth", "optical_depth_delta"):
                        if not active:
                            self.assertFalse(data[field].any())
                    if active:
                        np.testing.assert_equal(data["rgb"], data["raw_optical_rgb"])
            # Raw mode displays actual gel optics while the contact RGB tile is black.
            idle = contact_fixture(False)
            primary = np.asarray(mosaic(idle, tile_width=100, tile_height=100))
            raw = np.asarray(mosaic(idle, tile_width=100, tile_height=100, mode="raw_optical"))
            y = 58 + 3 * 100 + 26
            self.assertFalse(primary[y + 1 : y + 51, :100].any())
            self.assertTrue((raw[y : y + 52, :100] == 179).any())

    def test_gate_contact_and_raw_data_corruptions_are_rejected(self):
        cases = (
            (False, "contact_active", lambda data: data.update(contact_active=1)),
            (False, "rgb", lambda data: data["rgb"].fill(1)),
            (False, "depth", lambda data: data["depth"].fill(0.01)),
            (False, "optical_depth_delta", lambda data: data["optical_depth_delta"].fill(0.01)),
            (False, "penetration_depth", lambda data: data["penetration_depth"].fill(0.001)),
            (True, "contact_active", lambda data: data.update(contact_active=False)),
            (True, "rgb", lambda data: data["rgb"].fill(0)),
            (True, "depth", lambda data: data["depth"].fill(0.03)),
            (False, "raw_optical_depth", lambda data: data["raw_optical_depth"].fill(np.nan)),
            (False, "raw_optical_depth", lambda data: data["raw_optical_depth"].fill(0)),
            (False, "raw_optical_depth_delta", lambda data: data.pop("raw_optical_depth_delta")),
        )
        for active, field, corrupt in cases:
            with self.subTest(active=active, field=field):
                observation = contact_fixture(active)
                corrupt(observation["tactile"]["left"]["finger_0"])
                with self.assertRaises(ValueError):
                    validate_multimodal(observation)

    def test_two_environments_select_raw_fields_and_contact_state_independently(self):
        description = ModuleType("env.description_manager.desc_manager")
        description.DescManager = object
        path = Path(__file__).resolve().parents[1] / "env/observation_manager/obs_manager.py"
        spec = importlib.util.spec_from_file_location("_contact_obs_manager", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", {description.__name__: description}):
            spec.loader.exec_module(module)
        idle, contact = contact_fixture(False), contact_fixture(True)
        batch = {}
        for side, fingers in idle["tactile"].items():
            batch[side] = {}
            for finger, data in fingers.items():
                batch[side][finger] = {
                    key: [value, contact["tactile"][side][finger][key]]
                    if key.startswith("raw_optical") or key in ("contact_active", "metadata", "provenance")
                    else np.stack([value, contact["tactile"][side][finger][key]])
                    if isinstance(value, np.ndarray)
                    else value
                    for key, value in data.items()
                }
        manager = module.ObsManager.__new__(module.ObsManager)
        manager.num_envs = 2
        manager.env = SimpleNamespace(
            sim=SimpleNamespace(sim=SimpleNamespace(get_physics_step_count=lambda: 1, get_physics_dt=lambda: 0.04))
        )
        manager.camera_manager = SimpleNamespace(
            capture_multimodal=lambda env_id: {"cam_" + key: value for key, value in idle["vision"].items()}
        )
        manager.tactile_manager = SimpleNamespace(get_observations=lambda: batch)
        observations = {0: {}, 1: {}}
        manager._capture_multimodal(observations, [0, 1])
        for env_id in (0, 1):
            validate_multimodal(observations[env_id])
            data = observations[env_id]["tactile"]["left"]["finger_0"]
            self.assertEqual(bool(data["contact_active"]), bool(env_id))
            self.assertEqual(data["raw_optical_depth"].shape, (2, 4, 1))
            with tempfile.TemporaryDirectory() as directory:
                with HDF5Writer(directory, {"synthetic_cpu_fixture": True}) as writer:
                    writer.append(observations[env_id])
                with HDF5Reader(writer.path) as reader:
                    self.assertEqual(
                        bool(reader.read_frame(0)["tactile"]["left"]["finger_0"]["contact_active"]), bool(env_id)
                    )


if __name__ == "__main__":
    unittest.main()
