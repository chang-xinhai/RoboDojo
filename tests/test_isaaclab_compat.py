"""Independent public-contract tests for the Lab 3 quaternion/array boundary."""

import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from utils.isaaclab_compat import (
    add_app_launcher_args,
    as_torch,
    pose_xyzw_to_wxyz,
    simulation_snapshot,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)


def rotate_xyzw(quaternion, vector):
    """Rotate a vector using the vector/scalar quaternion definition."""
    xyz = quaternion[:3]
    return vector + 2 * np.cross(xyz, np.cross(xyz, vector) + quaternion[3] * vector)


class QuaternionContractTests(unittest.TestCase):
    def test_known_quarter_turns_preserve_physical_orientation(self):
        s = np.sqrt(0.5)
        cases = (
            ([s, s, 0, 0], [0, 1, 0], [0, 0, 1]),
            ([s, 0, s, 0], [0, 0, 1], [1, 0, 0]),
            ([s, 0, 0, s], [1, 0, 0], [0, 1, 0]),
            ([1, 0, 0, 0], [1, 2, 3], [1, 2, 3]),
            ([0, 0, 0, 1], [1, 2, 3], [-1, -2, 3]),
        )
        for wxyz, vector, expected in cases:
            with self.subTest(wxyz=wxyz):
                xyzw = wxyz_to_xyzw(wxyz)
                np.testing.assert_allclose(rotate_xyzw(xyzw, np.asarray(vector)), expected, atol=1e-12)
                np.testing.assert_allclose(xyzw_to_wxyz(xyzw), wxyz)

    def test_batch_sign_dtype_and_source_are_preserved(self):
        source = np.array([[[0.5, 0.5, 0.5, 0.5], [-0.5, -0.5, -0.5, -0.5]]], dtype=np.float32)
        original = source.copy()
        converted = wxyz_to_xyzw(source)
        np.testing.assert_array_equal(xyzw_to_wxyz(converted), original)
        self.assertEqual(converted.shape, source.shape)
        self.assertEqual(converted.dtype, np.float32)
        converted[...] = 0
        np.testing.assert_array_equal(source, original)

    def test_pose_translation_survives_rotation_boundary(self):
        source = np.array([[10, -20, 30, 0, 0, np.sqrt(0.5), np.sqrt(0.5)]])
        converted = pose_xyzw_to_wxyz(source)
        np.testing.assert_array_equal(converted[0, :3], [10, -20, 30])
        np.testing.assert_allclose(converted[0, 3:], [np.sqrt(0.5), 0, 0, np.sqrt(0.5)])
        converted[0, :3] = 0
        np.testing.assert_array_equal(source[0, :3], [10, -20, 30])

    def test_bad_component_counts_are_rejected(self):
        for shape in ((3,), (2, 5), (1, 2, 6)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                wxyz_to_xyzw(np.zeros(shape))
        with self.assertRaises(ValueError):
            pose_xyzw_to_wxyz(np.zeros((2, 8)))

    def test_empty_batch_is_supported(self):
        self.assertEqual(wxyz_to_xyzw(np.empty((0, 4))).shape, (0, 4))


class ArrayAndLaunchContractTests(unittest.TestCase):
    def test_snapshot_uses_physics_steps_without_advancing_context(self):
        class Context:
            frame = 0

            def get_physics_step_count(self):
                return self.frame

            def get_physics_dt(self):
                return 0.004

            def get_simulation_time(self):
                # Early Access PhysX currently leaves this upstream clock at 0.
                return 0.0

            def step(self):
                raise AssertionError("Reading an acquisition clock must not advance physics")

        ctx = Context()
        self.assertEqual(simulation_snapshot(ctx), (0.0, 0))
        ctx.frame = 250
        self.assertEqual(simulation_snapshot(ctx), (1.0, 250))
        ctx.frame = 260
        self.assertEqual(simulation_snapshot(ctx), (1.04, 260))
        self.assertEqual(ctx.frame, 260)

    def test_snapshot_legacy_seconds_are_not_inferred_from_frame(self):
        ctx = SimpleNamespace(current_time=7.5, current_time_step_index=19)
        self.assertEqual(simulation_snapshot(ctx), (7.5, 19))

    def test_array_passthrough_does_not_copy(self):
        source = np.arange(6).reshape(2, 3)
        self.assertIs(as_torch(source), source)

    def test_launcher_legacy_flags_are_registered_only_once(self):
        class Launcher:
            @staticmethod
            def add_app_launcher_args(parser):
                if "--headless" not in parser._option_string_actions:
                    parser.add_argument("--headless", action="store_true")

        parser = argparse.ArgumentParser()
        add_app_launcher_args(parser, Launcher)
        add_app_launcher_args(parser, Launcher)
        args = parser.parse_args(["--headless", "--enable_cameras"])
        self.assertTrue(args.headless)
        self.assertTrue(args.enable_cameras)

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("warp"), "torch/warp missing")
    def test_official_proxy_array_cpu_zero_copy(self):
        # Load upstream's actual class without importing the Kit-dependent package.
        import torch
        import warp as wp

        path = (
            Path(__file__).resolve().parents[1]
            / "third_party/IsaacLab3/source/isaaclab/isaaclab/utils/warp/proxy_array.py"
        )
        spec = importlib.util.spec_from_file_location("official_proxy_array_contract", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        source = wp.array([[0, 0, np.sqrt(0.5), np.sqrt(0.5)]], dtype=wp.quatf, device="cpu")
        proxy = module.ProxyArray(source)
        tensor = as_torch(proxy)
        self.assertIs(tensor, proxy.torch)
        self.assertIs(as_torch(tensor), tensor)
        self.assertEqual(tensor.shape, (1, 4))
        np.testing.assert_allclose(xyzw_to_wxyz(proxy).numpy(), [[np.sqrt(0.5), 0, 0, np.sqrt(0.5)]])
        tensor[0, 0] = 0.25
        self.assertAlmostEqual(float(source.numpy()[0, 0]), 0.25)
        self.assertEqual(xyzw_to_wxyz(proxy).dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
