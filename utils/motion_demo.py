"""Small real wrist joint motion for synchronized sensor capture validation."""

from __future__ import annotations

import math

import numpy as np

from utils.isaaclab_compat import as_torch, set_joint_target
from utils.multimodal import numpy_copy


class WristMotionDemo:
    """Command the final arm joint around its measured initial position.

    Only arm position targets change. The environment advances physics and
    subsequently reads its actual sensor buffers and measured joint positions.
    """

    def __init__(self, robot_manager, amplitude_rad=0.03, frequency_hz=0.5, env_id=0):
        if not 0 < amplitude_rad <= 0.05 or not 0 < frequency_hz <= 1.0:
            raise ValueError("Motion demo requires amplitude in (0,0.05] rad and frequency in (0,1] Hz")
        self.robot_manager = robot_manager
        self.frequency_hz = float(frequency_hz)
        self.env_id = int(env_id)
        self.arms = []
        self.last_command = None
        self.measured = {}
        for index, robot in enumerate(robot_manager.robot_list):
            if robot.robot_type != "arm" or robot.type != "target":
                continue
            entity = robot_manager.robot_key[index]
            indices = list(robot.arm_joint_indices)
            if not indices:
                raise ValueError(f"Motion demo found no arm joints for {robot.arm_name}")
            positions = numpy_copy(as_torch(entity.data.joint_pos)[env_id][indices]).astype(np.float64)
            limits = numpy_copy(as_torch(entity.data.soft_joint_pos_limits)[env_id][indices])
            clearance = min(positions[-1] - limits[-1, 0], limits[-1, 1] - positions[-1])
            amplitude = min(float(amplitude_rad), float(clearance) * 0.9)
            if not math.isfinite(amplitude) or amplitude < 0.001:
                raise ValueError(f"{robot.arm_name}: no safe wrist motion range around measured position")
            self.arms.append(
                {
                    "robot": robot,
                    "entity": entity,
                    "joint_ids": indices,
                    "baseline": positions,
                    "amplitude_rad": amplitude,
                    "sign": 1 if len(self.arms) % 2 == 0 else -1,
                }
            )
            self.measured[robot.arm_name] = []
        if len(self.arms) != 2:
            raise ValueError("Moving-wrist capture validation requires exactly two target arms")

    def configuration(self):
        return {
            "enabled": True,
            "frequency_hz": self.frequency_hz,
            "actuator_backend": "isaaclab_actuator_target_command",
            "arms": {
                arm["robot"].arm_name: {
                    "joint_name": arm["entity"].joint_names[arm["joint_ids"][-1]],
                    "joint_id": arm["joint_ids"][-1],
                    "amplitude_rad": arm["amplitude_rad"],
                    "baseline_joint_positions_rad": arm["baseline"].tolist(),
                    "phase_sign": arm["sign"],
                }
                for arm in self.arms
            },
        }

    def apply(self, elapsed_time_s, physics_frame):
        """Write real actuator targets immediately before one physics step."""
        if not math.isfinite(elapsed_time_s) or elapsed_time_s < 0:
            raise ValueError("Motion elapsed time must be nonnegative and finite")
        commands = {}
        for arm in self.arms:
            target = arm["baseline"].copy()
            target[-1] += (
                arm["sign"] * arm["amplitude_rad"] * math.sin(2 * math.pi * self.frequency_hz * elapsed_time_s)
            )
            current = as_torch(arm["entity"].data.joint_pos)
            value = current.new_tensor(target).unsqueeze(0) if hasattr(current, "new_tensor") else target[None, :]
            set_joint_target(arm["entity"], value, joint_ids=arm["joint_ids"], env_ids=[self.env_id])
            commands[arm["robot"].arm_name] = target.tolist()
        self.last_command = {
            "issued_before_physics_frame": int(physics_frame),
            "elapsed_time_s": float(elapsed_time_s),
            "joint_position_targets_rad": commands,
        }
        return self.last_command

    def sample(self):
        """Record measured articulation positions separately from commands."""
        measured = {}
        for arm in self.arms:
            name = arm["robot"].arm_name
            positions = self.robot_manager.get_joint(arm["robot"], env_idx_list=[self.env_id])[self.env_id]
            measured[name] = numpy_copy(positions)
            self.measured[name].append(measured[name])
        return {"last_actuator_command": self.last_command, "measured_joint_positions_rad": measured}

    def validate_motion(self):
        ranges = {}
        for name, samples in self.measured.items():
            if len(samples) < 2:
                raise RuntimeError("Moving-wrist validation requires at least two captured frames")
            span = np.ptp(np.stack(samples), axis=0)
            if float(span[-1]) <= 1e-5:
                raise RuntimeError(f"{name}: commanded wrist joint did not measurably move in captured frames")
            ranges[name] = span.tolist()
        return {"configuration": self.configuration(), "measured_joint_ranges_rad": ranges}
