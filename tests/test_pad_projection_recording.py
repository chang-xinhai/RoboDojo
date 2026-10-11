"""Native pose/force schema checks with analytical CPU fixtures, not GPU proof."""

import importlib.util
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from tests.test_multimodal import fixture
from utils.multimodal import validate_multimodal
from utils.multimodal_hdf5 import HDF5Reader, HDF5Writer

FIELDS = ("force_body_xyz", "pad_normal_force", "pad_shear_force_xz", "taxel_orientation_w_xyzw", "gel_pose_w_xyzw")


def projected_fixture():
    observation = fixture()
    for fingers in observation["tactile"].values():
        for data in fingers.values():
            # Analytical -90deg X: original (X,Y,Z) -> body (X,Z,-Y).
            data["normal_force"][:] = 0.25
            data["shear_force"] = np.tile(np.array([0.1, 0.2], np.float32), (8, 1))
            data["force_body_xyz"] = np.tile(np.array([0.1, 0.25, -0.2], np.float32), (8, 1))
            data["pad_normal_force"] = np.full(8, -0.25, np.float32)
            data["pad_shear_force_xz"] = np.tile(np.array([0.1, -0.2], np.float32), (8, 1))
            data["taxel_orientation_w_xyzw"] = np.tile(np.array([-(2**-0.5), 0, 0, 2**-0.5], np.float32), (8, 1))
            data["gel_pose_w_xyzw"] = np.array([1, 2, 3, 0, 0, 0, 1], np.float32)
            data["optical_depth_delta"] = np.full((2, 4, 1), -0.001, np.float32)
            data["metadata"] = {"array_size": [2, 4], "taxel_array_axes": ["sensor_local_x", "sensor_local_z"]}
    return observation


class PadRecordingTests(unittest.TestCase):
    def test_projection_and_signed_compression_roundtrip_without_changing_raw_forces(self):
        observation = projected_fixture()
        validate_multimodal(observation)
        with tempfile.TemporaryDirectory() as directory:
            with HDF5Writer(directory, {"synthetic_cpu_fixture": True}) as writer:
                writer.append(observation)
            with HDF5Reader(writer.path) as reader:
                restored = reader.read_frame(0)["tactile"]["left"]["finger_0"]
                np.testing.assert_equal(
                    restored["normal_force"], observation["tactile"]["left"]["finger_0"]["normal_force"]
                )
                np.testing.assert_equal(restored["pad_normal_force"], np.full(8, -0.25, np.float32))
                for field in FIELDS:
                    np.testing.assert_equal(restored[field], observation["tactile"]["left"]["finger_0"][field])

    def test_nonfinite_shape_and_nonunit_quaternion_fail(self):
        for field in FIELDS:
            for corruption in ("nan", "shape"):
                with self.subTest(field=field, corruption=corruption):
                    sample = projected_fixture()
                    values = sample["tactile"]["left"]["finger_0"][field]
                    if corruption == "nan":
                        values.flat[0] = np.nan
                    else:
                        sample["tactile"]["left"]["finger_0"][field] = values[:-1]
                    with self.assertRaises(ValueError):
                        validate_multimodal(sample)
        for field in ("taxel_orientation_w_xyzw", "gel_pose_w_xyzw"):
            with self.subTest(field=field):
                sample = projected_fixture()
                sample["tactile"]["left"]["finger_0"][field][...] *= 2
                with self.assertRaisesRegex(ValueError, "unit norm"):
                    validate_multimodal(sample)

    def test_wrong_rotation_sign_and_incomplete_projection_fail(self):
        for field in ("force_body_xyz", "pad_normal_force", "pad_shear_force_xz"):
            with self.subTest(field=field):
                observation = projected_fixture()
                observation["tactile"]["left"]["finger_0"][field] *= -1
                with self.assertRaisesRegex(ValueError, "disagrees"):
                    validate_multimodal(observation)
        observation = projected_fixture()
        del observation["tactile"]["left"]["finger_0"]["gel_pose_w_xyzw"]
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_multimodal(observation)

    def test_mixed_envs_select_derived_fields_and_image_only_nulls_before_hdf5(self):
        description = ModuleType("env.description_manager.desc_manager")
        description.DescManager = object
        path = Path(__file__).resolve().parents[1] / "env/observation_manager/obs_manager.py"
        spec = importlib.util.spec_from_file_location("_projection_obs_manager", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", {description.__name__: description}):
            spec.loader.exec_module(module)
        rigid = projected_fixture()
        batch = {}
        for side, fingers in rigid["tactile"].items():
            batch[side] = {}
            for finger, data in fingers.items():
                sample = {
                    key: np.stack([value, value]) if isinstance(value, np.ndarray) else value
                    for key, value in data.items()
                }
                for field in (*FIELDS, "normal_force", "shear_force", "penetration_depth", "points_pos_w"):
                    sample[field] = [None, data[field]]
                # Both image deltas exist; list-valued buffers must also be sliced.
                sample["optical_depth_delta"] = [data["optical_depth_delta"], data["optical_depth_delta"] * 2]
                sample["provenance"] = [
                    {
                        "backend": "isaaclab_tacsl",
                        "force_field_available": False,
                        "force_unavailable_reason": "fixture cloth",
                    },
                    {"backend": "isaaclab_tacsl", "force_field_available": True},
                ]
                sample["metadata"] = [{"env_id": 0}, {"env_id": 1, **data["metadata"]}]
                batch[side][finger] = sample
        manager = module.ObsManager.__new__(module.ObsManager)
        manager.num_envs = 2
        context = SimpleNamespace(get_physics_step_count=lambda: 1, get_physics_dt=lambda: 0.04)
        manager.env = SimpleNamespace(sim=SimpleNamespace(sim=context))
        manager.camera_manager = SimpleNamespace(
            capture_multimodal=lambda env_id: {"cam_" + name: data for name, data in rigid["vision"].items()}
        )
        manager.tactile_manager = SimpleNamespace(get_observations=lambda: batch)
        observations = {0: {}, 1: {}}
        manager._capture_multimodal(observations, [1, 0])
        with tempfile.TemporaryDirectory() as directory:
            for env_id, observation in observations.items():
                with HDF5Writer(Path(directory) / str(env_id)) as writer:
                    writer.append(observation)
                with HDF5Reader(writer.path) as reader:
                    data = reader.read_frame(0)["tactile"]["left"]["finger_0"]
                    self.assertEqual(data["metadata"]["env_id"], env_id)
                    self.assertEqual(data["provenance"]["force_field_available"], env_id == 1)
                    for field in FIELDS:
                        if env_id == 0:
                            self.assertIsNone(data[field])
                        else:
                            np.testing.assert_equal(data[field], rigid["tactile"]["left"]["finger_0"][field])
                    expected = rigid["tactile"]["left"]["finger_0"]["optical_depth_delta"] * (env_id + 1)
                    np.testing.assert_equal(data["optical_depth_delta"], expected)
        image_only = observations[0]
        image_only["tactile"]["left"]["finger_0"]["pad_normal_force"] = np.zeros(8, np.float32)
        with self.assertRaisesRegex(ValueError, "must be null"):
            validate_multimodal(image_only)
