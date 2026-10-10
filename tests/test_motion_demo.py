"""CPU actuator contract tests; stand-ins are not runtime motion evidence."""

from types import SimpleNamespace
import unittest

import numpy as np

from utils.motion_demo import WristMotionDemo


class TargetCommand:
    def __init__(self):
        self.calls = []

    def set_position_index(self, *, value, joint_ids, env_ids):
        self.calls.append({"value": np.asarray(value).copy(), "joint_ids": list(joint_ids), "env_ids": list(env_ids)})


def robot_manager():
    robots = [
        SimpleNamespace(robot_type="arm", type="target", arm_name=name, arm_joint_indices=[0, 1])
        for name in ("left_arm", "right_arm")
    ]
    entities = [
        SimpleNamespace(
            data=SimpleNamespace(
                joint_pos=np.zeros((1, 4), np.float32), soft_joint_pos_limits=np.tile([-1.0, 1.0], (1, 4, 1))
            ),
            joint_names=["shoulder", "wrist", "grip_a", "grip_b"],
            actuators=SimpleNamespace(target_command=TargetCommand()),
        )
        for _ in robots
    ]
    manager = SimpleNamespace(robot_list=robots, robot_key=entities)
    manager.get_joint = lambda robot, env_idx_list: {
        0: entities[robots.index(robot)].data.joint_pos[0, robot.arm_joint_indices].copy()
    }
    return manager


class MotionDemoTests(unittest.TestCase):
    def test_commands_real_actuator_interface_without_gripper_joints(self):
        manager = robot_manager()
        demo = WristMotionDemo(manager, amplitude_rad=0.03, frequency_hz=0.5)
        command = demo.apply(0.5, 42)
        self.assertEqual(command["issued_before_physics_frame"], 42)
        for index, entity in enumerate(manager.robot_key):
            call = entity.actuators.target_command.calls[-1]
            self.assertEqual(call["joint_ids"], [0, 1])
            self.assertEqual(call["env_ids"], [0])
            self.assertEqual(call["value"].shape, (1, 2))
            self.assertAlmostEqual(call["value"][0, 1], 0.03 if index == 0 else -0.03)
            self.assertEqual(call["value"][0, 0], 0)
        measured = demo.sample()
        # Commands do not claim a measured response before physics updates.
        np.testing.assert_equal(measured["measured_joint_positions_rad"]["left_arm"], [0, 0])
        with self.assertRaisesRegex(RuntimeError, "at least two"):
            demo.validate_motion()
        for entity in manager.robot_key:
            target = entity.actuators.target_command.calls[-1]["value"][0]
            entity.data.joint_pos[0, :2] = target * 0.5
        demo.sample()
        report = demo.validate_motion()
        self.assertAlmostEqual(report["measured_joint_ranges_rad"]["left_arm"][-1], 0.015)
        self.assertAlmostEqual(report["measured_joint_ranges_rad"]["right_arm"][-1], 0.015)

    def test_frozen_measured_positions_fail_even_when_targets_move(self):
        manager = robot_manager()
        demo = WristMotionDemo(manager)
        demo.apply(0.04, 1)
        demo.sample()
        demo.apply(0.08, 2)
        demo.sample()
        with self.assertRaisesRegex(RuntimeError, "did not measurably move"):
            demo.validate_motion()

    def test_amplitude_respects_remaining_soft_limit_clearance(self):
        manager = robot_manager()
        manager.robot_key[0].data.joint_pos[0, 1] = 0.99
        demo = WristMotionDemo(manager, amplitude_rad=0.03)
        configuration = demo.configuration()
        amplitude = configuration["arms"]["left_arm"]["amplitude_rad"]
        self.assertAlmostEqual(amplitude, 0.009, places=6)
        demo.apply(0.5, 1)
        target = manager.robot_key[0].actuators.target_command.calls[-1]["value"][0, 1]
        self.assertLess(target, 1.0)
        manager.robot_key[0].data.joint_pos[0, 1] = 1.0
        with self.assertRaisesRegex(ValueError, "no safe wrist"):
            WristMotionDemo(manager)

    def test_parameters_cannot_exceed_demo_bounds(self):
        for amplitude, frequency in ((0, 0.5), (0.06, 0.5), (0.03, 2), (float("nan"), 0.5)):
            with self.assertRaises(ValueError):
                WristMotionDemo(robot_manager(), amplitude_rad=amplitude, frequency_hz=frequency)


if __name__ == "__main__":
    unittest.main()
