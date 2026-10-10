"""Run production RoboDojo X5 FK, IK and collision-aware cuRobo planning.

This launches no Kit app. Run on CUDA after installing the pinned 6.1 runtime:
  bash reproduction/isaac61/run.sh python reproduction/isaac61/smoke_planner.py
The parent runner must serialize this GPU test with other simulation jobs.
"""

import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
import traceback
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def urdf_fk(urdf_path, base_link, tool_link, joint_values):
    """Independent NumPy FK from the URDF chain, without cuRobo math kernels."""
    import numpy as np
    import transforms3d as t3d

    joints = {joint.find("child").attrib["link"]: joint for joint in ET.parse(urdf_path).getroot().findall("joint")}
    chain = []
    child = tool_link
    while child != base_link:
        joint = joints[child]
        chain.append(joint)
        child = joint.find("parent").attrib["link"]
    transform = np.eye(4)
    for joint in reversed(chain):
        origin = joint.find("origin")
        xyz = np.fromstring(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" ")
        rpy = np.fromstring(origin.get("rpy", "0 0 0") if origin is not None else "0 0 0", sep=" ")
        local = np.eye(4)
        local[:3, :3] = t3d.euler.euler2mat(*rpy, axes="sxyz")
        local[:3, 3] = xyz
        motion = np.eye(4)
        joint_type = joint.get("type")
        if joint_type != "fixed":
            axis_element = joint.find("axis")
            axis = np.fromstring(axis_element.get("xyz", "1 0 0") if axis_element is not None else "1 0 0", sep=" ")
            value = float(joint_values[joint.get("name")])
            if joint_type in ("revolute", "continuous"):
                motion[:3, :3] = t3d.axangles.axangle2mat(axis, value)
            elif joint_type == "prismatic":
                motion[:3, 3] = axis * value
            else:
                raise ValueError(f"Unsupported URDF joint type: {joint_type}")
        transform = transform @ local @ motion
    return transform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reproduction/isaac61/planner-smoke.json")
    parser.add_argument("--fk-position-tolerance", type=float, default=1e-4)
    parser.add_argument("--ik-position-tolerance", type=float, default=0.003)
    parser.add_argument("--ik-rotation-tolerance", type=float, default=0.03)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.unlink(missing_ok=True)
    result = {"status": "failed", "checks": [], "robot": "x5", "quaternion_order": "WXYZ"}
    planner = None
    try:
        import numpy as np
        import torch
        import transforms3d as t3d

        from env.planner_manager.curobo_planner import CuroboPlanner
        from utils.isaaclab_compat import pose_xyzw_to_wxyz, wxyz_to_xyzw

        result["packages"] = {name: metadata.version(name) for name in ("torch", "warp-lang", "nvidia-curobo")}
        assert torch.__version__.startswith("2.11."), torch.__version__
        assert metadata.version("warp-lang") == "1.16.0"
        assert torch.cuda.is_available(), "CUDA device unavailable"
        torch.manual_seed(0)
        np.random.seed(0)
        active = [f"joint{index}" for index in range(1, 7)]
        # Same table-relative height as the production ARX scene: table 0.740m,
        # robot root 0.765m. Self-collision and table collision remain enabled.
        planner = CuroboPlanner(
            robot_origin_pose=[0, 0, 0, 1, 0, 0, 0],
            active_joints_name=active,
            all_joints=active,
            dt=0.004,
            yml_path=str(ROOT / "Assets/Robots/x5/curobo.yml"),
            table_height=-0.025,
        )
        result["checks"].append("production single/batch planner warmup and CUDA graph capture")
        start = planner.cspace_retract_config[planner._active_to_cspace_idx].copy()
        goal = start.copy()
        goal[0] += 0.12
        goal[-1] -= 0.08

        def fk(joints):
            state = planner._build_joint_state(joints)
            pose = planner.motion_planner.compute_kinematics(state).tool_poses.to_dict()[planner.ee_link]
            position = pose.position.detach().cpu().numpy().reshape(-1, 3)[0]
            quaternion = pose.quaternion.detach().cpu().numpy().reshape(-1, 4)[0]
            assert np.isfinite(position).all() and np.isfinite(quaternion).all()
            np.testing.assert_allclose(np.linalg.norm(quaternion), 1, atol=1e-4)
            return position, quaternion

        goal_position, goal_quaternion = fk(goal)
        kinematics = planner.robot_cfg["robot_cfg"]["kinematics"]
        reference = urdf_fk(
            kinematics["urdf_path"],
            kinematics["base_link"],
            planner.ee_link,
            dict(zip(planner.cspace_joint_names, planner._build_cspace_joint_values(goal), strict=True)),
        )
        fk_position_error = float(np.linalg.norm(goal_position - reference[:3, 3]))
        fk_rotation_error = float(np.max(np.abs(t3d.quaternions.quat2mat(goal_quaternion) - reference[:3, :3])))
        assert fk_position_error < args.fk_position_tolerance, fk_position_error
        assert fk_rotation_error < 1e-4, (
            f"FK quaternion WXYZ disagrees with independent URDF rotation: {fk_rotation_error}"
        )
        result["checks"].append("GPU FK position and WXYZ rotation agree with independent URDF FK")

        # Exercise the actual IsaacLab XYZW -> public RoboDojo WXYZ boundary,
        # then a nonidentity world/root transform as used by RobotManager.
        base_pose = pose_xyzw_to_wxyz(
            np.r_[goal_position - np.asarray(planner.frame_bias), wxyz_to_xyzw(goal_quaternion)]
        )
        root_pose = np.array([-0.3, -0.45, 0.765, np.sqrt(0.5), 0, 0, np.sqrt(0.5)])
        root_rotation = t3d.quaternions.quat2mat(root_pose[3:])
        target = np.r_[
            root_pose[:3] + root_rotation @ base_pose[:3],
            t3d.quaternions.qmult(root_pose[3:], base_pose[3:]),
        ]
        ik = planner.solve_ik_to_joint(start, target.tolist(), real_robot_pose=root_pose.tolist())
        assert ik["status"] == "Success", ik
        assert ik["joint_names"] == active and np.asarray(ik["joint_value"]).shape == (6,)
        ik_position, ik_quaternion = fk(ik["joint_value"])
        ik_position_error = float(np.linalg.norm(ik_position - goal_position))
        cosine = float(np.clip(abs(np.dot(ik_quaternion, goal_quaternion)), 0, 1))
        ik_rotation_error = float(2 * np.arccos(cosine))
        assert ik_position_error < args.ik_position_tolerance, ik_position_error
        assert ik_rotation_error < args.ik_rotation_tolerance, ik_rotation_error
        result["checks"].append("production IK roundtrip with nonidentity WXYZ root transform")

        trajectory = planner.plan_joint(start, goal)
        assert trajectory["status"] == "Success", trajectory
        positions = np.asarray(trajectory["position"])
        velocities = np.asarray(trajectory["velocity"])
        assert positions.ndim == 2 and positions.shape[0] > 1 and positions.shape[1] == 6, positions.shape
        assert velocities.shape == positions.shape
        assert np.isfinite(positions).all() and np.isfinite(velocities).all()
        np.testing.assert_allclose(positions[0], start, atol=0.015)
        np.testing.assert_allclose(positions[-1], goal, atol=0.015)
        result["checks"].append("production collision-aware nonzero joint trajectory with finite commands")
        torch.cuda.synchronize()
        np.savez_compressed(args.output.with_suffix(".npz"), joint_position=positions, joint_velocity=velocities)
        result.update(
            status="passed",
            gpu=torch.cuda.get_device_name(),
            tool_frame=planner.ee_link,
            active_joints=active,
            trajectory_shape=list(positions.shape),
            fk_position_error_m=fk_position_error,
            fk_rotation_matrix_max_error=fk_rotation_error,
            ik_position_error_m=ik_position_error,
            ik_orientation_error_rad=ik_rotation_error,
            goal_pose_base_wxyz=np.r_[goal_position, goal_quaternion].tolist(),
            cuda_graph=True,
            self_collision_check=True,
            table_height_m=-0.025,
        )
    except BaseException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        if planner is not None:
            for name in ("ik_solver", "motion_planner", "motion_planner_batch"):
                resource = getattr(planner, name, None)
                if resource is not None:
                    try:
                        resource.destroy()
                        delattr(planner, name)
                    except Exception as exc:
                        result["status"] = "failed"
                        result.setdefault("cleanup_errors", []).append(f"{name}: {type(exc).__name__}: {exc}")
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
