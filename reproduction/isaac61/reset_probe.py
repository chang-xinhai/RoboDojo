"""Actual GPU motion and reset invariants shared by task smoke scripts."""

import numpy as np
import torch
import transforms3d as t3d

from utils.isaaclab_compat import as_torch, set_joint_target


def _numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def probe_motion_and_reset(env, *, steps=12):
    """Move native bodies and joints, then verify the production reset methods.

    Immediate component resets must be exact; the full task reset restores the
    layout and lets gravity/contacts settle, so its final poses have a tolerance.
    """
    inactive = env.scene_manager._inactive_scene_object_names[0]
    bodies = [
        obj
        for name, obj in env.scene_manager._rigid_and_dynamic_objects[0].items()
        if name not in inactive and hasattr(obj, "_rigid_prim_view")
    ]
    arms = list({id(arm): arm for arm in env.robot_manager.robot_key}.values())
    before = []
    for obj in bodies:
        position, orientation = map(_numpy, obj.get_world_pose())
        position = position.copy()
        position[2] += 0.05
        obj.set_world_pose(position=position, orientation=orientation)
        points = obj.sample_mesh_vertices()[0].copy()
        before.append((position.copy(), orientation.copy(), points))
        view = obj._rigid_prim_view
        velocity = view._backend_utils.convert(
            [[0.12, 0.03, 0.0, 0.0, 0.0, 0.08]], device=view._device, dtype="float32"
        )
        view.set_velocities(velocity)
    joint_before = []
    for arm in arms:
        joint_before.append(as_torch(arm.data.joint_pos).clone())
        target = joint_before[-1].clone()
        target[:, arm.joint_names.index("joint1")] += 0.05
        set_joint_target(arm, target)
    for _ in range(steps):
        env.sim_step(render=False)
    evidence = {"steps": steps, "bodies": {}, "robots": []}
    for obj, (position_before, orientation_before, points_before) in zip(bodies, before, strict=True):
        position, orientation = map(_numpy, obj.get_world_pose())
        motion = float(np.linalg.norm(position - position_before))
        assert motion > 1e-4, (obj.prim_path, motion)
        points_after = obj.sample_mesh_vertices()[0]
        expected = (points_before - position_before) @ t3d.quaternions.quat2mat(
            orientation_before
        ) @ t3d.quaternions.quat2mat(orientation).T + position
        mesh_error = float(np.max(np.abs(points_after - expected)))
        assert mesh_error < 2e-4, (obj.prim_path, "stale reward mesh", mesh_error)
        velocity_before = float(np.max(np.abs(_numpy(obj._rigid_prim_view.get_velocities()))))
        assert velocity_before > 1e-4, (obj.prim_path, velocity_before)
        obj.apply_saved_pose()
        local_position, local_orientation = map(_numpy, obj.get_local_pose())
        np.testing.assert_allclose(local_position, obj.default_pos, atol=1e-5)
        assert abs(float(np.dot(local_orientation, obj.default_ori))) > 1.0 - 1e-5
        reset_velocity = float(np.max(np.abs(_numpy(obj._rigid_prim_view.get_velocities()))))
        assert reset_velocity < 1e-7, (obj.prim_path, reset_velocity)
        evidence["bodies"][obj.instance_name] = {
            "motion_m": motion,
            "mesh_world_max_error_m": mesh_error,
            "pre_reset_max_velocity": velocity_before,
            "reset_max_velocity": reset_velocity,
            "saved_pose_max_error_m": float(np.max(np.abs(local_position - obj.default_pos))),
        }
    for arm, initial in zip(arms, joint_before, strict=True):
        motion = float(torch.max(torch.abs(as_torch(arm.data.joint_pos) - initial)).item())
        velocity = float(torch.max(torch.abs(as_torch(arm.data.joint_vel))).item())
        assert motion > 1e-4, (arm.joint_names, motion)
        evidence["robots"].append({"motion_rad": motion, "pre_reset_max_velocity_rad_s": velocity})
    env.robot_manager.reset()
    for arm, robot_evidence in zip(arms, evidence["robots"], strict=True):
        pose_error = float(
            torch.max(torch.abs(as_torch(arm.data.joint_pos) - as_torch(arm.data.default_joint_pos))).item()
        )
        velocity_error = float(torch.max(torch.abs(as_torch(arm.data.joint_vel))).item())
        assert pose_error < 1e-6 and velocity_error < 1e-7, (pose_error, velocity_error)
        robot_evidence.update(reset_pose_error_rad=pose_error, reset_max_velocity_rad_s=velocity_error)
    env.reset(seed=0)
    env.scene_manager.apply_saved_poses([0])
    for obj in bodies:
        position, _ = map(_numpy, obj.get_local_pose())
        error = float(np.linalg.norm(position - np.asarray(obj.default_pos)))
        assert error < 0.025, (obj.instance_name, "full reset layout drift", error)
        evidence["bodies"][obj.instance_name]["full_reset_settled_position_error_m"] = error
    evidence["full_task_reset_and_layout_restore"] = True
    return evidence
