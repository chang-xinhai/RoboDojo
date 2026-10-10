"""Contract/serialization tests: synthetic fixtures are never runtime evidence."""

import importlib.util
import json
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from utils.multimodal import CAMERAS, SCHEMA_VERSION, MultimodalWriter, read_frame, validate_multimodal


def fixture(frame=1):
    stamp = frame * 40_000_000
    obs = {
        "sync": {
            "schema": SCHEMA_VERSION,
            "physics_frame": frame,
            "timestamp_ns": stamp,
            "simulation_time_s": stamp / 1e9,
        },
        "vision": {},
        "tactile": {},
        "state": {},
        "action": {},
    }
    for camera in CAMERAS:
        obs["vision"][camera] = {
            "rgb": np.arange(36, dtype=np.uint8).reshape(3, 4, 3),
            "depth_gt": np.ones((3, 4), np.float32),
            "depth_sl": np.full((2, 4), np.nan, np.float32),
            "ir_left": np.zeros((2, 4), np.uint8),
            "ir_right": np.ones((2, 4), np.uint8),
            "timestamp_ns": stamp,
            "calibration": {"K": np.eye(3)},
            "provenance": {"ir": "unit_test_fixture"},
        }
    for side in ("left", "right"):
        obs["tactile"][side] = {}
        for finger in ("finger_0", "finger_1"):
            obs["tactile"][side][finger] = {
                "rgb": np.zeros((2, 4, 3), np.uint8),
                "depth": np.ones((2, 4, 1), np.float32),
                "normal_force": np.zeros(8, np.float32),
                "shear_force": np.zeros((8, 3), np.float32),
                "penetration_depth": np.zeros(8, np.float32),
                "points_pos_w": np.zeros((8, 3), np.float32),
                "provenance": {"backend": "isaaclab_tacsl"},
                "timestamp_ns": stamp,
            }
    return obs


class RecordingTests(unittest.TestCase):
    def test_round_trip_preserves_depth_nan_dtype_and_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = MultimodalWriter(directory, {"test_fixture": True})
            original = fixture()
            filename = writer.append(original)
            writer.close()
            restored = read_frame(filename)
            for name in CAMERAS:
                for field in ("rgb", "depth_gt", "depth_sl", "ir_left", "ir_right"):
                    np.testing.assert_equal(restored["vision"][name][field], original["vision"][name][field])
                    self.assertEqual(restored["vision"][name][field].dtype, original["vision"][name][field].dtype)
                np.testing.assert_equal(restored["vision"][name]["calibration"]["K"], np.eye(3))
            self.assertTrue(json.loads((Path(directory) / "manifest.json").read_text())["complete"])
            with self.assertRaises(FileExistsError):
                MultimodalWriter(directory)

    def test_missing_sensor_is_not_filled(self):
        for side in ("vision", "tactile"):
            data = fixture()
            del data[side][next(iter(data[side]))]
            with self.assertRaisesRegex(ValueError, "Missing required"):
                validate_multimodal(data)

    def test_stale_clock_is_rejected(self):
        data = fixture()
        data["vision"]["head"]["timestamp_ns"] = 0
        with self.assertRaisesRegex(ValueError, "timestamp"):
            validate_multimodal(data)
        data = fixture()
        data["tactile"]["left"]["finger_1"]["timestamp_ns"] = 0
        with self.assertRaisesRegex(ValueError, "timestamp"):
            validate_multimodal(data)

    def test_wrong_depth_units_and_shape_are_rejected(self):
        data = fixture()
        data["vision"]["head"]["depth_gt"] = np.full((3, 4), 1000, np.uint16)
        with self.assertRaisesRegex(ValueError, "float32"):
            validate_multimodal(data)
        data = fixture()
        data["vision"]["head"]["depth_sl"] = np.ones((3, 4), np.float32)
        with self.assertRaisesRegex(ValueError, "camera frame"):
            validate_multimodal(data)

    def test_bad_frame_does_not_produce_file_or_complete_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = MultimodalWriter(directory)
            writer.append(fixture(2))
            with self.assertRaisesRegex(ValueError, "strictly increase"):
                writer.append(fixture(1))
            bad = fixture(3)
            bad["vision"]["head"]["depth_gt"][0, 0] = -1
            with self.assertRaisesRegex(ValueError, "negative"):
                writer.append(bad)
            self.assertEqual(len(list(Path(directory).glob("frame_*.npz"))), 1)
            self.assertFalse((Path(directory) / "manifest.json").exists())
            self.assertEqual(read_frame(Path(directory) / "frame_000000.npz")["sync"]["physics_frame"], 2)

    def test_missing_finger_and_mismatched_taxels_are_rejected(self):
        data = fixture()
        del data["tactile"]["left"]["finger_1"]
        with self.assertRaisesRegex(ValueError, "fingers"):
            validate_multimodal(data)
        data = fixture()
        data["tactile"]["right"]["finger_0"]["points_pos_w"] = np.zeros((7, 3), np.float32)
        with self.assertRaisesRegex(ValueError, "taxel positions"):
            validate_multimodal(data)

    def test_image_only_mode_declares_force_limit_without_fabrication(self):
        data = fixture()
        sample = data["tactile"]["left"]["finger_0"]
        fields = ("normal_force", "shear_force", "penetration_depth", "points_pos_w")
        for field in fields:
            sample[field] = None
        with self.assertRaisesRegex(ValueError, "missing available"):
            validate_multimodal(data)
        sample["provenance"].update(
            force_field_available=False, force_unavailable_reason="deformable contact unsupported"
        )
        validate_multimodal(data)
        with tempfile.TemporaryDirectory() as directory:
            writer = MultimodalWriter(directory)
            restored = read_frame(writer.append(data))
            self.assertIsNone(restored["tactile"]["left"]["finger_0"]["normal_force"])
        sample["normal_force"] = np.zeros(8, np.float32)
        with self.assertRaisesRegex(ValueError, "never fabricated"):
            validate_multimodal(data)

    def test_lab3_frozen_physics_envelope_and_environment_selection(self):
        # Isolate the description subsystem so this contract test needs NumPy
        # but no Isaac, torch, Warp or renderer process.
        description_module = ModuleType("env.description_manager.desc_manager")
        description_module.DescManager = object
        path = Path(__file__).resolve().parents[1] / "env/observation_manager/obs_manager.py"
        spec = importlib.util.spec_from_file_location("_sync_contract_obs_manager", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", {description_module.__name__: description_module}):
            spec.loader.exec_module(module)

        class PhysicsContext:
            frame = 2

            def get_physics_step_count(self):
                return self.frame

            def get_physics_dt(self):
                return 0.04

        context = PhysicsContext()
        source = fixture(2)

        class Cameras:
            advance = False

            def capture_multimodal(self, env_id):
                if self.advance:
                    context.frame += 1
                return {"cam_" + name: data for name, data in source["vision"].items()}

        class Tactile:
            def get_observations(self):
                return {
                    side: {
                        finger: {
                            key: np.stack([value] * 3) if isinstance(value, np.ndarray) else value
                            for key, value in data.items()
                        }
                        for finger, data in fingers.items()
                    }
                    for side, fingers in source["tactile"].items()
                }

        manager = module.ObsManager.__new__(module.ObsManager)
        manager.env = SimpleNamespace(sim=SimpleNamespace(sim=context))
        manager.camera_manager = Cameras()
        manager.tactile_manager = Tactile()
        obs = {2: {}, 0: {}}
        manager._capture_multimodal(obs, [2, 0])
        self.assertEqual(obs[2]["sync"]["physics_frame"], 2)
        self.assertEqual(obs[2]["sync"]["env_id"], 2)
        validate_multimodal(obs[2])
        validate_multimodal(obs[0])
        self.assertFalse(
            np.shares_memory(
                obs[2]["tactile"]["left"]["finger_0"]["rgb"],
                source["tactile"]["left"]["finger_0"]["rgb"],
            )
        )
        manager.camera_manager.advance = True
        with self.assertRaisesRegex(RuntimeError, "Physics advanced"):
            manager._capture_multimodal({0: {}}, [0])

    def test_fake_tactile_backend_is_rejected(self):
        data = fixture()
        data["tactile"]["left"]["finger_0"]["provenance"]["backend"] = "contact_force_proxy"
        with self.assertRaisesRegex(ValueError, "actual Isaac Lab TacSL"):
            validate_multimodal(data)


if __name__ == "__main__":
    unittest.main()
