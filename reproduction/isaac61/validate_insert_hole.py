"""Independent CuRobo and native official-condition validation for UniVTAC insert_hole."""
from __future__ import annotations
import argparse
import importlib.metadata as metadata
import json
import os
import sys
import traceback
import hashlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["ROBODOJO_ASSET_ROOT"] = os.environ.get("ROBODOJO_INSERT_HOLE_ASSET_ROOT", "/mnt/robo/yutong/tmp/robodojo-insert-hole-assets")
from reproduction.isaac61.prepare_insert_hole_assets import stage_assets
stage_assets()
from isaaclab.app import AppLauncher
from utils.isaaclab_compat import add_app_launcher_args, as_torch, set_joint_target, enable_legacy_isaacsim_extensions
from utils.load_file import load_yaml
from utils.task_layout import load_task_layout, verify_task_scene
from utils.task_runtime_config import configure_task_runtime, reset_task_episode
from omegaconf import OmegaConf
import transforms3d as t3d

parser = argparse.ArgumentParser()
add_app_launcher_args(parser, AppLauncher)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.enable_cameras = False
launcher = AppLauncher(args)
app = launcher.app
if os.environ.get("MIGRATION_CAPTURE_CONTACTS") == "1":
    try:
        from omni.kit.app import get_app
        get_app().get_extension_manager().set_extension_enabled_immediate(
            "isaacsim.sensors.experimental.physics", True
        )
    except Exception:
        pass
from pxr import Usd, UsdGeom, PhysxSchema

result = {
    "status": "FAIL",
    "phase": "independent_curobo_native_validation",
    "independent_curobo_plan": True,
    "base_trajectory_used": False,
    "official_success": False,
    "tactile_enabled": False,
    "camera_capture_enabled": False,
    "candidate_id": "insert-hole-independent-000",
    "plans": {},
}
env = None
contact_records = []
contact_capture_errors = []
try:
    enable_legacy_isaacsim_extensions()
    import torch
    from task.RoboDojo.task_registry import load_task_class
    from env.planner_manager.curobo_planner import CuroboPlanner

    for name in ("isaacsim", "isaaclab", "nvidia-curobo", "torch", "warp-lang"):
        try:
            result.setdefault("versions", {})[name] = metadata.version(name)
        except Exception as exc:
            result.setdefault("versions", {})[name] = f"error:{exc}"

    eval_cfg = load_yaml(str(ROOT / "env_cfg/arx_x5.yml"))
    components = eval_cfg["config"]
    cfg = OmegaConf.create(
        {kind: load_yaml(str(ROOT / "env_cfg" / kind / f"{components[kind]}.yml"))
         for kind in ("sim", "scene", "camera", "robot")}
    )
    cfg.no_render = True
    cfg.task_env = load_yaml(str(ROOT / "task/RoboDojo/config/migration_insert_hole.yml"))
    if os.environ.get("MIGRATION_DIAGNOSTIC_DISABLE_SLOT_COLLISION") == "1":
        # B diagnostic: disable only slot collision before GeometryObject and
        # PhysX cooking. Base, robot and tube collisions remain enabled.
        slot_cfg = cfg.task_env["Geometry"][0]
        slot_cfg.setdefault("physics", {})["collision"] = False
        result["diagnostic_slot_collision_disabled_before_cooking"] = True
    layout, provenance = load_task_layout(
        ROOT, "migration_insert_hole",
        layout_file=ROOT / "reproduction/isaac61/layouts/insert_hole.json",
    )
    cfg.saved_layouts = [layout]
    profile = configure_task_runtime(cfg, "migration_insert_hole", "arx_x5", eval_cfg, provenance, enable_planner=False)
    _, task_cls = load_task_class("migration_insert_hole")
    env = task_cls(cfg, app, seed=0)
    reset_task_episode(env, profile)
    contact_records = []
    contact_capture_errors = []
    contact_stage = {"label": None}
    contact_sensors = []
    contact_id_map = {}
    if os.environ.get("MIGRATION_CAPTURE_CONTACTS") == "1":
        try:
            from pxr import PhysicsSchemaTools
            for prim in env.scene_manager.stage.Traverse():
                path = str(prim.GetPath())
                if not path.startswith("/World/envs/env_0"):
                    continue
                try:
                    body_id = int(PhysicsSchemaTools.sdfPathToInt(path))
                    contact_id_map[str(body_id)] = path
                except Exception as exc:
                    contact_capture_errors.append(f"sdfPathToInt({path}): {exc!r}")
            result["contact_id_map"] = contact_id_map
            from isaacsim.sensors.experimental.physics import Contact, ContactSensor
            tube_name = env.scene_manager.layout_manager.get_instance_name(0, label="prism")
            tube_obj = env.scene_manager.layout_manager.get_scene_object(0, tube_name)
            sensor_path = f"{tube_obj.prim.GetPath()}/contact_probe"
            sensor = ContactSensor(
                Contact.create(sensor_path, min_threshold=0.0, max_threshold=1.0e9, radius=-1.0)
            )
            sensor.add_raw_contact_data_to_frame()
            contact_sensors.append(sensor)
            result["contact_capture_enabled"] = True
            result["contact_capture_backend"] = "isaacsim.experimental.physics.ContactSensor"
        except Exception as exc:
            result["contact_capture_enabled"] = False
            contact_capture_errors.append(repr(exc))

    def _contact_raw_get(raw, name, default=None):
        if isinstance(raw, dict):
            return raw.get(name, default)
        return getattr(raw, name, default)

    def _resolve_contact_body(value):
        try:
            body_id = int(value)
        except Exception:
            return None, None
        mapped = contact_id_map.get(str(body_id))
        try:
            from pxr import PhysicsSchemaTools
            native = str(PhysicsSchemaTools.intToSdfPath(body_id))
        except Exception as exc:
            contact_capture_errors.append(f"intToSdfPath({body_id}): {exc!r}")
            native = None
        return mapped or native, body_id

    def capture_contacts(label):
        target_stage = os.environ.get("MIGRATION_CONTACT_STAGE")
        if label is None or (target_stage and label != target_stage):
            return
        for sensor_index, sensor in enumerate(contact_sensors):
            try:
                raw_contacts = sensor.get_raw_data()
            except Exception as exc:
                contact_capture_errors.append(repr(exc))
                continue
            for raw in raw_contacts:
                if len(contact_records) >= 5000:
                    return
                body0, body0_id = _resolve_contact_body(_contact_raw_get(raw, "body0", 0))
                body1, body1_id = _resolve_contact_body(_contact_raw_get(raw, "body1", 0))
                def vec(name):
                    value = _contact_raw_get(raw, name, {}) or {}
                    if isinstance(value, dict):
                        return [float(value.get(axis, 0.0)) for axis in ("x", "y", "z")]
                    return [float(getattr(value, axis, 0.0)) for axis in ("x", "y", "z")]
                contact_records.append({
                    "stage": label,
                    "sensor": sensor_index,
                    "body0_id": body0_id,
                    "body1_id": body1_id,
                    "body0": body0,
                    "body1": body1,
                    "position": vec("position"),
                    "normal": vec("normal"),
                    "impulse": vec("impulse"),
                    "time": float(_contact_raw_get(raw, "time", 0.0)),
                    "dt": float(_contact_raw_get(raw, "dt", 0.0)),
                })

    result["contact_capture_errors"] = contact_capture_errors
    if os.environ.get("MIGRATION_DIAGNOSTIC_DISABLE_BASE_COLLISION") == "1":
        from pxr import UsdPhysics
        base_name = env.scene_manager.layout_manager.get_instance_name(0, label="base")
        base_obj = env.scene_manager.layout_manager.get_scene_object(0, base_name)
        UsdPhysics.CollisionAPI.Apply(base_obj.prim).CreateCollisionEnabledAttr(False).Set(False)
        for _ in range(20):
            env.sim_step(render=False)
        result["diagnostic_base_collision_disabled"] = True
    if os.environ.get("MIGRATION_DIAGNOSTIC_DISABLE_ROBOT_COLLISION") == "1":
        from pxr import UsdPhysics
        disabled_robot_paths = []
        for robot_idx in range(len(env.robot_manager.robot_list)):
            robot_root = env.scene_manager.stage.GetPrimAtPath(f"/World/envs/env_0/robot{robot_idx}")
            for prim in Usd.PrimRange(robot_root):
                if prim.HasAPI(UsdPhysics.CollisionAPI) or prim.IsA(UsdGeom.Mesh):
                    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr().Set(False)
                    disabled_robot_paths.append(str(prim.GetPath()))
        for _ in range(20):
            env.sim_step(render=False)
        result["diagnostic_robot_collision_disabled"] = True
        result["diagnostic_robot_collision_disabled_paths"] = sorted(set(disabled_robot_paths))
    if os.environ.get("MIGRATION_DIAGNOSTIC_DISABLE_SLOT_COLLISION") == "1":
        from pxr import UsdPhysics
        slot_name = env.scene_manager.layout_manager.get_instance_name(0, label="slot")
        slot_obj = env.scene_manager.layout_manager.get_scene_object(0, slot_name)
        disabled_paths = []
        for prim in Usd.PrimRange(slot_obj.prim):
            if prim.HasAPI(UsdPhysics.CollisionAPI) or prim.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr().Set(False)
                disabled_paths.append(str(prim.GetPath()))
        UsdPhysics.CollisionAPI.Apply(slot_obj.prim).CreateCollisionEnabledAttr().Set(False)
        disabled_paths.append(str(slot_obj.prim.GetPath()))
        for _ in range(20):
            env.sim_step(render=False)
        result["diagnostic_slot_collision_disabled"] = True
        result["diagnostic_slot_collision_disabled_paths"] = sorted(set(disabled_paths))
    if os.environ.get("MIGRATION_DIAGNOSTIC_CONTACT_PARAMS") == "1":
        slot_name = env.scene_manager.layout_manager.get_instance_name(0, label="slot")
        slot_obj = env.scene_manager.layout_manager.get_scene_object(0, slot_name)
        contact_offset = float(os.environ.get("MIGRATION_DIAGNOSTIC_CONTACT_OFFSET", "0.0002"))
        rest_offset = float(os.environ.get("MIGRATION_DIAGNOSTIC_REST_OFFSET", "0.0"))
        for prim in Usd.PrimRange(slot_obj.prim):
            if prim.IsA(UsdGeom.Mesh):
                collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                collision.CreateContactOffsetAttr().Set(contact_offset)
                collision.CreateRestOffsetAttr().Set(rest_offset)
        # C is diagnostic only: keep slot collision enabled and set the slot
        # material restitution explicitly to zero.
        material_api = getattr(getattr(slot_obj, "physics_material", None), "_material_api", None)
        if material_api is not None:
            material_api.CreateRestitutionAttr().Set(0.0)
        for _ in range(20):
            env.sim_step(render=False)
        result["diagnostic_contact_params"] = {
            "slot_collision_enabled": True,
            "contact_offset_m": contact_offset,
            "rest_offset_m": rest_offset,
            "restitution": 0.0,
        }
    result["scene"] = verify_task_scene(env.scene_manager, layout)
    # Record the cooked mesh modes used by the native scene for collision audit.
    result["scene"]["collision_approximations"] = {}
    for semantic_label in ("slot", "base"):
        obj_name = env.scene_manager.layout_manager.get_instance_name(0, label=semantic_label)
        obj = env.scene_manager.layout_manager.get_scene_object(0, obj_name)
        vals = {}
        for prim in Usd.PrimRange(obj.prim):
            if prim.IsA(UsdGeom.Mesh):
                vals[str(prim.GetPath())] = prim.GetAttribute("physics:approximation").Get()
        result["scene"]["collision_approximations"][semantic_label] = vals

    robots = env.robot_manager.robot_list
    # Planning must start from the post-reset native pose. Layout metadata is
    # the authored placement and may differ by millimetres after the settling
    # steps required by PhysX.  That difference matters for this 0.25 mm
    # per-side insertion clearance.
    object_name = env.scene_manager.layout_manager.get_instance_name(0, label="prism")
    object_obj = env.scene_manager.layout_manager.get_scene_object(0, object_name)
    object_pos, object_q = object_obj.get_world_pose()
    if hasattr(object_pos, "detach"):
        object_pos = object_pos.detach().cpu().numpy()
    if hasattr(object_q, "detach"):
        object_q = object_q.detach().cpu().numpy()
    slot_pos, slot_q = env.scene_manager.layout_manager.get_instance_pose(0, label="slot", relative=False)
    object_pos = np.asarray(object_pos, dtype=float).reshape(-1)[:3]
    object_q = np.asarray(object_q, dtype=float).reshape(-1)
    slot_pos = np.asarray(slot_pos, dtype=float).reshape(-1)[:3]
    slot_q = np.asarray(slot_q, dtype=float).reshape(-1)
    arm_override = os.environ.get("MIGRATION_INSERT_HOLE_ARM_INDEX")
    if arm_override is None:
        arm_index = int(np.argmin([np.linalg.norm(np.asarray(r.entity_origin_pose[:2]) - object_pos[:2]) for r in robots]))
    else:
        arm_index = int(arm_override)
        if arm_index < 0 or arm_index >= len(robots):
            raise ValueError(f"MIGRATION_INSERT_HOLE_ARM_INDEX out of range: {arm_index}")
    robot = robots[arm_index]
    arm = env.robot_manager.robot_key[arm_index]
    root = np.asarray(robot.entity_origin_pose, dtype=float)
    active = list(robot.arm_joints_name)
    result["binding"] = {
        "arm_index": arm_index,
        "arm_name": robot.arm_name,
        "object_xyz": object_pos.tolist(),
        "slot_xyz": slot_pos.tolist(),
        "active_joints": active,
    }
    planner = CuroboPlanner(
        robot_origin_pose=root.tolist(),
        active_joints_name=active,
        all_joints=active,
        dt=float(env.dt),
        yml_path=str(Path(os.environ["ROBODOJO_ASSET_ROOT"]) / "Robots/x5/curobo.yml"),
        table_height=float(layout["Table"]["default_pos"][2]) - float(root[2]),
    )

    def pose_matrix(p, q):
        T = np.eye(4)
        T[:3, :3] = t3d.quaternions.quat2mat(np.asarray(q, dtype=float))
        T[:3, 3] = np.asarray(p, dtype=float)[:3]
        return T

    def pose_vector(T):
        return np.r_[T[:3, 3], t3d.quaternions.mat2quat(T[:3, :3])]

    def live_object_pose():
        # Dynamic objects must be read from the native rigid body. The layout
        # manager pose is authored/reset metadata and may lag after contact.
        object_name = env.scene_manager.layout_manager.get_instance_name(0, label="prism")
        object_obj = env.scene_manager.layout_manager.get_scene_object(0, object_name)
        p, q = object_obj.get_world_pose()
        if hasattr(p, "detach"):
            p = p.detach().cpu().numpy()
        if hasattr(q, "detach"):
            q = q.detach().cpu().numpy()
        return np.asarray(p, dtype=float).reshape(-1)[:3], np.asarray(q, dtype=float).reshape(-1)

    def live_link_pose():
        return np.asarray(env.robot_manager.get_link_pose(robot, "link6", env_idx_list=[0], is_relative=False)[0], dtype=float).reshape(-1)

    def execute(plan, gripper_value, duration_s=1.0, settle_steps=120):
        rows = np.asarray(plan["position"], dtype=np.float32)
        repeats = max(2, int(np.ceil(float(duration_s) / (max(1, len(rows)) * float(env.dt)))))
        final_row = rows[-1]
        for row in np.repeat(rows, repeats, axis=0):
            device = as_torch(arm.data.joint_pos).device
            arm_target = torch.as_tensor(row, dtype=torch.float32, device=device)[None]
            grip_target = torch.full(
                (1, len(robot.gripper_joint_indices)),
                float(gripper_value),
                dtype=torch.float32,
                device=device,
            )
            set_joint_target(arm, arm_target, joint_ids=robot.arm_joint_indices)
            set_joint_target(arm, grip_target, joint_ids=robot.gripper_joint_indices)
            env.sim_step(render=False)
            capture_contacts(contact_stage["label"])
        target = np.asarray(final_row, dtype=np.float32).reshape(-1)
        stable = 0
        min_error = None
        for _ in range(int(settle_steps)):
            device = as_torch(arm.data.joint_pos).device
            arm_target = torch.as_tensor(target, dtype=torch.float32, device=device)[None]
            grip_target = torch.full(
                (1, len(robot.gripper_joint_indices)),
                float(gripper_value),
                dtype=torch.float32,
                device=device,
            )
            set_joint_target(arm, arm_target, joint_ids=robot.arm_joint_indices)
            set_joint_target(arm, grip_target, joint_ids=robot.gripper_joint_indices)
            env.sim_step(render=False)
            capture_contacts(contact_stage["label"])
            current = np.asarray(env.robot_manager.get_joint(robot)[0], dtype=np.float32).reshape(-1)
            err = float(np.linalg.norm(current - target))
            min_error = err if min_error is None else min(min_error, err)
            stable = stable + 1 if err < 0.01 else 0
            if stable >= 20:
                break
        return {"final_joint_error_norm": min_error, "settled": bool(stable >= 20)}

    def plan_link(start_joints, target_pose):
        plan = planner.plan_path(np.asarray(start_joints, dtype=np.float32), target_pose.tolist(), root.tolist())
        return plan

    def record_plan(name, plan, target):
        result["plans"][name] = {
            "status": plan.get("status"),
            "waypoints": int(len(plan.get("position", []))) if plan.get("position") is not None else 0,
            "independent_curobo_plan": True,
            "base_trajectory_used": False,
            "target_ee_pose_wxyz": np.asarray(target, dtype=float).tolist(),
        }

    # 1) Safe staging and independent approach.
    start = np.asarray(env.robot_manager.get_joint(robot)[0], dtype=np.float32)
    staging = np.r_[object_pos + np.asarray([-0.05, -0.30, 0.30]), [1.0, 0.0, 0.0, 0.0]]
    p = plan_link(start, staging)
    record_plan("staging", p, staging)
    if p.get("status") != "Success":
        raise RuntimeError("independent CuRobo staging plan failed")
    execute(p, robot.gripper_scale[1], 1.2)

    # 1b) Move to a collision-free unit-orientation staging pose before rotating
    # the open hand toward the UniVTAC grasp frame.
    safe_target = np.r_[object_pos + np.asarray([0.0, 0.0, 0.42]), [1.0, 0.0, 0.0, 0.0]]
    p = plan_link(np.asarray(p["position"][-1]), safe_target)
    record_plan("safe_above_object", p, safe_target)
    if p.get("status") != "Success":
        raise RuntimeError("independent CuRobo safe-above-object plan failed")
    execute(p, robot.gripper_scale[1], 1.0)
    # 2) UniVTAC top-down grasp frame converted to the X5 jaw-center frame.
    # The prior diagonal frame collided with the tube while the open hand rotated;
    # this frame was isolated in the diagnostic run and kept the tube at reset.
    grasp_center = object_pos + np.asarray([0.0, 0.0, 0.095])
    grasp_q = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=float)
    grasp_yaw_deg = float(os.environ.get("MIGRATION_INSERT_HOLE_GRASP_YAW_DEG", "0.0"))
    if abs(grasp_yaw_deg) > 1e-9:
        grasp_yaw = np.asarray(
            [np.cos(np.deg2rad(grasp_yaw_deg) / 2.0), 0.0, 0.0, np.sin(np.deg2rad(grasp_yaw_deg) / 2.0)]
        )
        grasp_q = t3d.quaternions.qmult(grasp_yaw, grasp_q)
    result["grasp_candidate"] = {
        "yaw_deg": grasp_yaw_deg,
        "grasp_pose_quat_wxyz": grasp_q.tolist(),
    }
    orient_plan = None
    orient_target = None
    orient_start = np.asarray(p["position"][-1])
    for dz in (0.42, 0.30, 0.20, 0.12, 0.08, 0.04):
        orient_center = object_pos + np.asarray([0.0, 0.0, dz])
        probe_target = np.r_[orient_center - t3d.quaternions.quat2mat(grasp_q) @ np.asarray([0.15069879303097725, 0.0, -0.0007080307654860616]), grasp_q]
        probe = plan_link(orient_start, probe_target)
        if probe.get("status") == "Success":
            orient_plan, orient_target = probe, probe_target
            break
    record_plan("orient_open_hand", orient_plan or {"status": "Fail", "position": []}, orient_target if orient_target is not None else np.zeros(7))
    if orient_plan is None:
        raise RuntimeError("independent CuRobo open-hand orientation plan failed for all bounded heights")
    p = orient_plan
    execute(p, robot.gripper_scale[1], 1.0)
    link6_to_center = np.asarray([0.15069879303097725, 0.0, -0.0007080307654860616])
    grasp_target = np.r_[grasp_center - t3d.quaternions.quat2mat(grasp_q) @ link6_to_center, grasp_q]
    pregrasp_start = np.asarray(p["position"][-1])
    pregrasp_plan = None
    pregrasp_target = None
    for dz in (0.12, 0.10, 0.08, 0.06, 0.04, 0.02):
        pregrasp_center = grasp_center + np.asarray([0.0, 0.0, dz])
        probe_target = np.r_[pregrasp_center - t3d.quaternions.quat2mat(grasp_q) @ link6_to_center, grasp_q]
        probe = plan_link(pregrasp_start, probe_target)
        if probe.get("status") == "Success":
            pregrasp_plan, pregrasp_target = probe, probe_target
            break
    record_plan("pregrasp", pregrasp_plan or {"status": "Fail", "position": []}, pregrasp_target if pregrasp_target is not None else np.zeros(7))
    if pregrasp_plan is None:
        raise RuntimeError("independent CuRobo pregrasp plan failed for all bounded heights")
    p = pregrasp_plan
    execute(p, robot.gripper_scale[1], 1.2)
    p = plan_link(np.asarray(p["position"][-1]), grasp_target)
    record_plan("grasp", p, grasp_target)
    if p.get("status") != "Success":
        raise RuntimeError("independent CuRobo grasp plan failed")
    execute(p, robot.gripper_scale[1], 1.5)

    before_close = live_object_pose()
    result["before_close"] = {"object_pose_wxyz": np.r_[before_close[0], before_close[1]].tolist(), "link6_pose_wxyz": live_link_pose().tolist()}
    for _ in range(700):
        cmd = as_torch(arm.data.joint_pos).clone()
        cmd[0, robot.gripper_joint_indices] = float(robot.gripper_scale[0])
        set_joint_target(arm, cmd)
        env.sim_step(render=False)
    closed_pose = live_object_pose()
    result["after_close_before_attach"] = {
        "object_pose_wxyz": np.r_[closed_pose[0], closed_pose[1]].tolist(),
        "displacement_m": float(np.linalg.norm(closed_pose[0] - before_close[0])),
        "gripper": np.asarray(env.robot_manager.get_end_effector_real_val(robot, env_idx_list=[0])[0]).reshape(-1).tolist(),
    }
    result["grasp_joint"] = env.attach_prism_to_arm(robot, arm_index)
    result["inhand_reference"] = env.record_inhand_reference(robot)
    attached_pose = live_object_pose()
    result["after_attach"] = {
        "object_pose_wxyz": np.r_[attached_pose[0], attached_pose[1]].tolist(),
        "snap_displacement_m": float(np.linalg.norm(attached_pose[0] - closed_pose[0])),
    }
    if result["after_attach"]["snap_displacement_m"] > 0.01:
        raise RuntimeError("live grasp joint introduced a pose snap")
    # PhysX may resolve a newly-created joint on the first step.  Record this
    # separately so a constraint snap is never mistaken for a CuRobo failure.
    attach_probe_before = live_object_pose()
    env.sim_step(render=False)
    attach_probe_after = live_object_pose()
    result["after_attach_first_step"] = {
        "object_pose_wxyz": np.r_[attach_probe_after[0], attach_probe_after[1]].tolist(),
        "displacement_m": float(np.linalg.norm(attach_probe_after[0] - attach_probe_before[0])),
        "from_attach_m": float(np.linalg.norm(attach_probe_after[0] - attached_pose[0])),
    }
    if result["after_attach_first_step"]["from_attach_m"] > 0.01:
        raise RuntimeError("live grasp joint snapped on first physics step")

    # 3) Independent lift and transport to the official target.
    relation_link_to_object = np.linalg.inv(pose_matrix(*live_link_pose()[:3],)) if False else None
    link_now = live_link_pose()
    object_now = live_object_pose()
    T_link_to_object = np.linalg.inv(pose_matrix(link_now[:3], link_now[3:])) @ pose_matrix(object_now[0], object_now[1])
    def target_link_for_object(target_p, target_q):
        return pose_vector(pose_matrix(target_p, target_q) @ np.linalg.inv(T_link_to_object))

    # Incremental lift: every 3 mm segment is a new CuRobo plan from the
    # current live joint state.  Freeze the post-attach object/link relation
    # once.  Recomputing it from each PhysX frame would turn tiny constraint
    # solver errors into cumulative XY/attitude drift.
    lift_attempts = []
    lift_segments = []
    max_clear_z = 0.873
    lift_official_target_q = np.asarray(
        [0.9659258262890683, 0.0, -0.25881904510252074, 0.0], dtype=float
    )
    lift_official_inverse_q = np.asarray(
        [0.9659258262890683, 0.0, 0.25881904510252074, 0.0], dtype=float
    )
    lift_reference_p, lift_reference_q = live_object_pose()
    lift_reference_link = live_link_pose()
    T_link_to_object_fixed = np.linalg.inv(
        pose_matrix(lift_reference_link[:3], lift_reference_link[3:])
    ) @ pose_matrix(lift_reference_p, lift_reference_q)
    stalled_segments = 0
    target_z = float(lift_reference_p[2])
    lift_target_q = np.asarray(lift_reference_q, dtype=float).copy()
    for seg_idx in range(80):
        object_now, object_q_now = live_object_pose()
        if float(object_now[2]) >= max_clear_z:
            break
        link_now = live_link_pose()
        def live_target_link(target_p, target_q):
            return pose_vector(
                pose_matrix(target_p, target_q) @ np.linalg.inv(T_link_to_object_fixed)
            )
        target_z = min(max_clear_z, target_z + 0.003)
        step_dz = target_z - float(object_now[2])
        candidate_object_p = np.asarray(
            [lift_reference_p[0], lift_reference_p[1], target_z], dtype=float
        )
        candidate_object_q = lift_target_q
        start_joints = np.asarray(env.robot_manager.get_joint(robot)[0], dtype=np.float32)
        name = f"lift_increment_{seg_idx:02d}"
        # Try the current orientation, bounded yaw alternatives, and the two
        # official tilt signs. Every pose is a fresh CuRobo plan from the live
        # joint state; no prior trajectory is perturbed or reused.
        pose_candidates = [("current", np.asarray(candidate_object_q, dtype=float))]
        for yaw_delta in (np.pi / 4.0, -np.pi / 4.0, np.pi / 2.0, -np.pi / 2.0, np.pi):
            yaw_q = np.asarray(
                [np.cos(yaw_delta / 2.0), 0.0, 0.0, np.sin(yaw_delta / 2.0)]
            )
            pose_candidates.append(
                (f"yaw_{yaw_delta:+.3f}", t3d.quaternions.qmult(yaw_q, candidate_object_q))
            )
        pose_candidates.extend(
            [("official_tilt", lift_official_target_q), ("official_inverse_tilt", lift_official_inverse_q)]
        )
        candidate_plan = {"status": "Fail"}
        candidate_target = None
        selected_pose_label = None
        selected_object_q = None
        for pose_label, target_object_q in pose_candidates:
            target = live_target_link(candidate_object_p, target_object_q)
            probe_plan = plan_link(start_joints, target)
            result["plans"][f"{name}_{pose_label}"] = {
                "status": probe_plan.get("status"),
                "waypoints": int(len(probe_plan.get("position", []))) if probe_plan.get("position") is not None else 0,
                "independent_curobo_plan": True,
                "base_trajectory_used": False,
                "target_ee_pose_wxyz": np.asarray(target, dtype=float).tolist(),
                "object_target_quat_wxyz": np.asarray(target_object_q, dtype=float).tolist(),
            }
            if probe_plan.get("status") == "Success":
                candidate_plan, candidate_target = probe_plan, target
                selected_pose_label, selected_object_q = pose_label, np.asarray(target_object_q, dtype=float)
                break
        if candidate_target is None:
            candidate_target = live_target_link(candidate_object_p, candidate_object_q)
        record_plan(name, candidate_plan, candidate_target)
        result["plans"][name]["selected_pose_label"] = selected_pose_label
        result["plans"][name]["selected_yaw_delta_rad"] = (
            float(selected_pose_label.split("_", 1)[1])
            if selected_pose_label and selected_pose_label.startswith("yaw_")
            else None
        )
        if selected_object_q is not None:
            lift_target_q = selected_object_q
        method = "pose"
        if candidate_plan.get("status") != "Success":
            ik = planner.solve_ik_to_joint(start_joints, candidate_target, root.tolist())
            result["plans"][f"{name}_ik"] = {
                "status": ik.get("status"),
                "independent_curobo_plan": True,
                "base_trajectory_used": False,
                "target_ee_pose_wxyz": np.asarray(candidate_target, dtype=float).tolist(),
            }
            if ik.get("status") != "Success":
                lift_attempts.append({"segment": seg_idx, "status": "pose_and_ik_fail"})
                raise RuntimeError(f"independent CuRobo incremental lift failed at segment {seg_idx}")
            candidate_plan = planner.plan_joint(start_joints, np.asarray(ik["joint_value"], dtype=np.float32))
            method = "ik_joint"
            result["plans"][f"{name}_joint"] = {
                "status": candidate_plan.get("status"),
                "waypoints": int(len(candidate_plan.get("position", []))) if candidate_plan.get("position") is not None else 0,
                "independent_curobo_plan": True,
                "base_trajectory_used": False,
                "target_ee_pose_wxyz": np.asarray(candidate_target, dtype=float).tolist(),
            }
        if candidate_plan.get("status") != "Success":
            raise RuntimeError(f"independent CuRobo incremental lift failed at segment {seg_idx}")
        execute(candidate_plan, robot.gripper_scale[0], 2.0)
        for _ in range(100):
            cmd = as_torch(arm.data.joint_pos).clone()
            cmd[0, robot.gripper_joint_indices] = float(robot.gripper_scale[0])
            set_joint_target(arm, cmd)
            env.sim_step(render=False)
        after_object_p, after_object_q = live_object_pose()
        after_link_pose = live_link_pose()
        after_joints = np.asarray(env.robot_manager.get_joint(robot)[0], dtype=float).reshape(-1)
        delta = float(np.linalg.norm(after_object_p - object_now))
        dz_actual = float(after_object_p[2] - object_now[2])
        link_dz_actual = float(after_link_pose[2] - link_now[2])
        lift_segments.append({
            "segment": seg_idx,
            "method": method,
            "target_dz_m": float(step_dz),
            "actual_delta_m": delta,
            "actual_dz_m": dz_actual,
            "link_dz_m": link_dz_actual,
            "start_arm_joints": np.asarray(start_joints, dtype=float).reshape(-1).tolist(),
            "target_arm_joints": np.asarray(candidate_plan["position"][-1], dtype=float).reshape(-1).tolist(),
            "after_arm_joints": after_joints.tolist(),
            "arm_joint_error_norm": float(np.linalg.norm(after_joints - np.asarray(candidate_plan["position"][-1], dtype=float).reshape(-1))),
            "object_pose_wxyz": np.r_[after_object_p, after_object_q].tolist(),
            "link6_pose_wxyz": np.asarray(after_link_pose, dtype=float).tolist(),
            "target_link_pose_wxyz": np.asarray(candidate_target, dtype=float).tolist(),
        })
        lift_attempts.append({"segment": seg_idx, "status": "Success", "actual_dz_m": dz_actual})
        result["lift_attempts"] = lift_attempts
        result["lift_segments"] = lift_segments
        # Keep object/link motion as diagnostics, but do not reject a final
        # shrinking segment merely because its numerical tail is small.  The
        # phase is complete when the live native object reaches max_clear_z.
        # A real attachment failure is still caught if the object remains
        # motionless for several consecutive segments while the link moves.
        if abs(dz_actual) < 0.0000001 and abs(link_dz_actual) > 0.0005:
            stalled_segments += 1
        elif abs(dz_actual) >= 0.0000001:
            stalled_segments = 0
        result["lift_segments"][-1]["stalled_segments"] = stalled_segments
        if stalled_segments >= 8 and float(after_object_p[2]) < max_clear_z - 0.002:
            raise RuntimeError(f"incremental lift did not move object at segment {seg_idx}")
    result["lift_attempts"] = lift_attempts
    result["lift_segments"] = lift_segments
    final_lift_object_p, final_lift_object_q = live_object_pose()
    if float(final_lift_object_p[2]) < max_clear_z - 0.01:
        raise RuntimeError(f"object did not clear slot rim: z={float(final_lift_object_p[2]):.6f}")
    result["plans"]["lift_selected"] = {
        "status": "Success",
        "method": "incremental_independent_curobo",
        "segments": len(lift_segments),
        "independent_curobo_plan": True,
        "base_trajectory_used": False,
        "target_ee_pose_wxyz": np.r_[live_link_pose()[:3], live_link_pose()[3:]].tolist(),
    }
    hole_p = slot_pos + np.asarray([0.0, 0.0, 0.10])
    # UniVTAC resets the base/hole frame rotation to identity before applying
    # the official -30 degree tilt.  Do not compose the USD slot mesh's
    # internal orientation into this semantic target.
    official_target_q = np.asarray([0.9659258262890683, 0.0, -0.25881904510252074, 0.0], dtype=float)
    # UniVTAC's official target_pose is exactly the hole pose.  The prism
    # actor pose is checked directly against it; do not offset the center by
    # an assumed functional-point distance.
    # Choose a point 6 cm down the official hole axis.  UniVTAC's checker
    # compares the tube against the hole frame and requires rel_z < -0.04 m;
    # targeting the hole origin would make success mathematically impossible.
    official_target_R = t3d.quaternions.quat2mat(official_target_q)
    # The official mesh has only about 0.25 mm radial clearance.  Keep the
    # nominal hole-frame center exact; a millimetre-scale heuristic offset can
    # satisfy the loose official XY checker while physically wedging the tube
    # against the rim.
    trajectory_hole_p = hole_p.copy()
    official_target_p = trajectory_hole_p + official_target_R @ np.asarray([0.0, 0.0, -0.06])
    # Approach above the slot with the official tilt axis and a reachable roll.
    # The official checker constrains the tube axis and position, so roll about
    # that axis is a valid redundancy.  Each (height, roll) pair is a fresh
    # CuRobo plan from the same live post-lift joint state.
    approach_plan = None
    approach_target = None
    approach_attempts = []
    selected_object_target_q = np.asarray(official_target_q, dtype=float).copy()
    selected_insert_roll_rad = None
    roll_grid = list(np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False))
    forced_roll = os.environ.get("MIGRATION_INSERT_HOLE_INSERT_ROLL_RAD")
    if forced_roll is None:
        insert_rolls = roll_grid
    else:
        forced_roll = float(forced_roll)
        insert_rolls = [forced_roll] + [r for r in roll_grid if abs(float(r) - forced_roll) > 1e-9]
    insert_start = np.asarray(env.robot_manager.get_joint(robot)[0], dtype=np.float32)
    for approach_distance in (0.16, 0.14, 0.12, 0.10, 0.08, 0.06, 0.04):
        # Stay on the inclined hole axis while approaching; a world-Z offset
        # introduces lateral error at the rim and wedges the tube.
        approach_obj_p = trajectory_hole_p + official_target_R @ np.asarray([0.0, 0.0, approach_distance])
        for roll_idx, phi in enumerate(insert_rolls):
            roll_q = np.asarray([np.cos(phi / 2.0), 0.0, 0.0, np.sin(phi / 2.0)], dtype=float)
            candidate_object_q = t3d.quaternions.qmult(official_target_q, roll_q)
            candidate_target = target_link_for_object(approach_obj_p, candidate_object_q)
            candidate_plan = plan_link(insert_start, candidate_target)
            name = f"insert_approach_axis_{int(round(approach_distance * 1000)):03d}mm_roll_{roll_idx:02d}"
            record_plan(name, candidate_plan, candidate_target)
            approach_attempts.append({
                "approach_axis_distance_m": float(approach_distance), "roll_rad": float(phi),
                "status": candidate_plan.get("status"),
            })
            if candidate_plan.get("status") == "Success":
                approach_plan = candidate_plan
                approach_target = candidate_target
                selected_object_target_q = np.asarray(candidate_object_q, dtype=float)
                selected_insert_roll_rad = float(phi)
                break
        if approach_plan is not None:
            break
    result["insert_approach_attempts"] = approach_attempts
    if selected_insert_roll_rad is not None:
        result["selected_insert_roll_rad"] = selected_insert_roll_rad
    final_target = target_link_for_object(official_target_p, selected_object_target_q)
    direct_final_plan = None
    final_already_executed = False
    if approach_plan is None:
        for dz in (0.14, 0.12, 0.10, 0.08, 0.06):
            approach_obj_p = slot_pos + np.asarray([0.0, 0.0, dz])
            candidate_target = target_link_for_object(approach_obj_p, official_target_q)
            current_joints = np.asarray(env.robot_manager.get_joint(robot)[0], dtype=np.float32)
            ik = planner.solve_ik_to_joint(current_joints, candidate_target, root.tolist())
            name = f"insert_approach_ik_{int(round(dz * 1000)):03d}mm"
            result["plans"][name] = {
                "status": ik.get("status"),
                "independent_curobo_plan": True,
                "base_trajectory_used": False,
                "target_ee_pose_wxyz": np.asarray(candidate_target, dtype=float).tolist(),
            }
            if ik.get("status") != "Success":
                approach_attempts.append({"dz_m": float(dz), "ik_status": ik.get("status")})
                continue
            candidate_plan = planner.plan_joint(current_joints, np.asarray(ik["joint_value"], dtype=np.float32))
            result["plans"][f"{name}_joint"] = {
                "status": candidate_plan.get("status"),
                "waypoints": int(len(candidate_plan.get("position", []))) if candidate_plan.get("position") is not None else 0,
                "independent_curobo_plan": True,
                "base_trajectory_used": False,
                "target_ee_pose_wxyz": np.asarray(candidate_target, dtype=float).tolist(),
            }
            approach_attempts.append({"dz_m": float(dz), "ik_status": ik.get("status"), "joint_status": candidate_plan.get("status")})
            if candidate_plan.get("status") == "Success":
                approach_plan = candidate_plan
                approach_target = candidate_target
                break
    result["insert_approach_attempts"] = approach_attempts
    if approach_plan is None:
        # The tall approach poses can be outside the X5 workspace.  Search only
        # the redundant roll about the official hole axis; every candidate is a
        # fresh CuRobo IK + cspace plan and all have the same official z-axis.
        direct_start = np.asarray(env.robot_manager.get_joint(robot)[0], dtype=np.float32)
        for idx, phi in enumerate(np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)):
            roll_q = np.asarray([np.cos(phi / 2.0), 0.0, 0.0, np.sin(phi / 2.0)], dtype=float)
            candidate_object_q = t3d.quaternions.qmult(official_target_q, roll_q)
            candidate_target = target_link_for_object(official_target_p, candidate_object_q)
            direct_ik = planner.solve_ik_to_joint(direct_start, candidate_target, root.tolist())
            result["plans"][f"insert_direct_roll_{idx:02d}_ik"] = {
                "status": direct_ik.get("status"),
                "roll_rad": float(phi),
                "independent_curobo_plan": True,
                "base_trajectory_used": False,
                "target_ee_pose_wxyz": np.asarray(candidate_target, dtype=float).tolist(),
            }
            if direct_ik.get("status") != "Success":
                continue
            candidate_plan = planner.plan_joint(direct_start, np.asarray(direct_ik["joint_value"], dtype=np.float32))
            result["plans"][f"insert_direct_roll_{idx:02d}_joint"] = {
                "status": candidate_plan.get("status"),
                "roll_rad": float(phi),
                "waypoints": int(len(candidate_plan.get("position", []))) if candidate_plan.get("position") is not None else 0,
                "independent_curobo_plan": True,
                "base_trajectory_used": False,
                "target_ee_pose_wxyz": np.asarray(candidate_target, dtype=float).tolist(),
            }
            if candidate_plan.get("status") == "Success":
                direct_final_plan = candidate_plan
                final_target = candidate_target
                selected_object_target_q = candidate_object_q
                result["selected_insert_roll_rad"] = float(phi)
                break
        if direct_final_plan is None:
            raise RuntimeError("independent CuRobo insertion approach and direct-final roll search failed")
    else:
        execute(approach_plan, robot.gripper_scale[0], 3.0)
        approach_object_pose = live_object_pose()
        result["after_approach_live_pose"] = {
            "object_pose_wxyz": np.r_[approach_object_pose[0], approach_object_pose[1]].tolist(),
            "link6_pose_wxyz": live_link_pose().tolist(),
        }
        # Independently align the tube base just above the actual hole mouth
        # before descending.  This avoids a joint-space shortcut that reaches
        # the right final pose but scrapes the rim first.
        rim_plan = None
        rim_distance = None
        rim_attempts = []
        successful_approach = [
            a for a in approach_attempts
            if a.get("status") == "Success" and a.get("approach_axis_distance_m") is not None
        ]
        current_rim_distance = float(successful_approach[-1]["approach_axis_distance_m"])
        # Walk from the successful approach pose to the rim in fresh 2 mm
        # CuRobo plans.  A single long jump can be collision-free in planning
        # yet scrape the physical rim during native execution.
        rim_step_index = 0
        while current_rim_distance > 0.020 + 1e-9:
            next_distance = max(0.020, current_rim_distance - 0.002)
            rim_object_p = trajectory_hole_p + official_target_R @ np.asarray([0.0, 0.0, next_distance])
            rim_target = target_link_for_object(rim_object_p, selected_object_target_q)
            rim_probe = plan_link(
                np.asarray(env.robot_manager.get_joint(robot)[0], dtype=np.float32),
                rim_target,
            )
            rim_name = f"insert_rim_alignment_{int(round(next_distance * 1000)):03d}mm"
            record_plan(rim_name, rim_probe, rim_target)
            rim_attempts.append({
                "step": rim_step_index,
                "from_distance_m": float(current_rim_distance),
                "distance_m": float(next_distance),
                "status": rim_probe.get("status"),
            })
            if rim_probe.get("status") != "Success":
                raise RuntimeError(
                    f"independent CuRobo incremental rim alignment failed at {next_distance:.3f} m"
                )
            execute(rim_probe, robot.gripper_scale[0], 2.0)
            rim_plan = rim_probe
            rim_distance = float(next_distance)
            current_rim_distance = float(next_distance)
            rim_step_index += 1
        result["insert_rim_alignment_attempts"] = rim_attempts
        if rim_plan is None or rim_distance is None:
            raise RuntimeError("independent CuRobo rim alignment plan failed")
        rim_object_pose = live_object_pose()
        result["after_rim_alignment_live_pose"] = {
            "object_pose_wxyz": np.r_[rim_object_pose[0], rim_object_pose[1]].tolist(),
            "link6_pose_wxyz": live_link_pose().tolist(),
            "axis_distance_m": rim_distance,
        }
        final_start = np.asarray(env.robot_manager.get_joint(robot)[0], dtype=np.float32)
        final_plan = plan_link(final_start, final_target)
        record_plan("insert_pose", final_plan, final_target)
        # Default to the collision-safe incremental insertion. A direct final
        # plan is retained for diagnostics but is opt-in only.
        if final_plan.get("status") != "Success" or os.environ.get("MIGRATION_INSERT_HOLE_DIRECT_INSERT") != "1":
            # A direct diagonal can be locally infeasible even when the
            # approach and official target are each reachable.  Bridge the
            # live object pose to the official pose with fresh short CuRobo
            # plans; each stage starts from the native live joint state.
            # Replan every 2 mm from the actual live state along the official
            # hole axis.  This prevents the large diagonal joint-space shortcut
            # from scraping the rim and keeps every segment independently planned.
            insert_dir = -official_target_R[:, 2]
            # The rim point is 20 mm outside the hole along the negative
            # insertion axis.  Advance from -20 mm (outside) to +60 mm
            # (the official target), with one fresh plan per 2 mm.
            axis_distance = -0.020
            insert_stage_records = []
            stalled_insert_segments = 0
            for stage_idx in range(40):
                axis_distance = min(0.060, axis_distance + 0.002)
                stage_p = trajectory_hole_p + insert_dir * axis_distance
                stage_target = target_link_for_object(stage_p, selected_object_target_q)
                stage_start = np.asarray(env.robot_manager.get_joint(robot)[0], dtype=np.float32)
                stage_plan = plan_link(stage_start, stage_target)
                stage_name = f"insert_axis_increment_{stage_idx:02d}"
                record_plan(stage_name, stage_plan, stage_target)
                if stage_plan.get("status") != "Success":
                    stage_ik = planner.solve_ik_to_joint(stage_start, stage_target, root.tolist())
                    result["plans"][f"{stage_name}_ik"] = {
                        "status": stage_ik.get("status"),
                        "independent_curobo_plan": True,
                        "base_trajectory_used": False,
                        "target_ee_pose_wxyz": np.asarray(stage_target, dtype=float).tolist(),
                    }
                    if stage_ik.get("status") != "Success":
                        raise RuntimeError(f"independent CuRobo axis insertion stage {stage_idx} failed")
                    stage_plan = planner.plan_joint(stage_start, np.asarray(stage_ik["joint_value"], dtype=np.float32))
                    result["plans"][f"{stage_name}_joint"] = {
                        "status": stage_plan.get("status"),
                        "waypoints": int(len(stage_plan.get("position", []))) if stage_plan.get("position") is not None else 0,
                        "independent_curobo_plan": True,
                        "base_trajectory_used": False,
                        "target_ee_pose_wxyz": np.asarray(stage_target, dtype=float).tolist(),
                    }
                    if stage_plan.get("status") != "Success":
                        raise RuntimeError(f"independent CuRobo axis insertion joint stage {stage_idx} failed")
                before_stage_p, before_stage_q = live_object_pose()
                contact_stage["label"] = f"stage_{stage_idx}"
                try:
                    execute(stage_plan, robot.gripper_scale[0], 4.0, settle_steps=360)
                finally:
                    contact_stage["label"] = None
                after_stage_p, after_stage_q = live_object_pose()
                actual_progress = float(np.dot(after_stage_p - before_stage_p, insert_dir))
                metrics_stage = env.official_success_metrics(0)
                stage_record = {
                    "stage": stage_idx,
                    "axis_distance_target_m": float(axis_distance),
                    "target_object_pose_wxyz": np.r_[stage_p, selected_object_target_q].tolist(),
                    "target_link_pose_wxyz": np.asarray(stage_target, dtype=float).tolist(),
                    "waypoints": int(len(stage_plan.get("position", []))),
                    "trajectory_sha256": hashlib.sha256(np.asarray(stage_plan.get("position"), dtype=np.float32).tobytes()).hexdigest(),
                    "independent_curobo_plan": True,
                    "base_trajectory_used": False,
                    "actual_progress_along_insert_axis_m": actual_progress,
                    "object_pose_wxyz": np.r_[after_stage_p, after_stage_q].tolist(),
                    "official_metrics": metrics_stage,
                }
                insert_stage_records.append(stage_record)
                if actual_progress < 0.0005:
                    stalled_insert_segments += 1
                else:
                    stalled_insert_segments = 0
                if stalled_insert_segments >= 3:
                    result.setdefault("diagnostics", {})["stall_gate_events"] = result.setdefault("diagnostics", {}).get("stall_gate_events", []) + [{
                        "stage": int(stage_idx),
                        "stalled_segments": int(stalled_insert_segments),
                        "actual_progress_along_insert_axis_m": float(actual_progress),
                    }]
                    if os.environ.get("MIGRATION_DIAGNOSTIC_BYPASS_STALL_GATE") != "1":
                        raise RuntimeError(f"axis insertion stalled at stage {stage_idx}: repeated rim collision")
                    # Diagnostic only: keep the official collision and success
                    # checker, but continue within the fixed 40-stage bound to
                    # distinguish a false stop gate from a physical blockage.
                result["insert_axis_stage_records"] = insert_stage_records
                if axis_distance >= 0.060 - 1e-9:
                    break
            result["insert_axis_stage_records"] = insert_stage_records
            final_plan = plan_link(
                np.asarray(env.robot_manager.get_joint(robot)[0], dtype=np.float32),
                final_target,
            )
            record_plan("insert_pose_after_incremental", final_plan, final_target)
            if final_plan.get("status") == "Success":
                direct_final_plan = final_plan
            else:
                # The eighth stage already targets the official pose.
                direct_final_plan = None
                final_already_executed = True
        else:
            direct_final_plan = final_plan
    # Execute the independently planned insertion slowly enough for the
    # native joint drives to track the lateral and tilted motion.
    if direct_final_plan is not None and not final_already_executed:
        result["final_execution"] = execute(
            direct_final_plan, robot.gripper_scale[0], 20.0, settle_steps=2400
        )
    for _ in range(300):
        cmd = as_torch(arm.data.joint_pos).clone()
        cmd[0, robot.gripper_joint_indices] = float(robot.gripper_scale[0])
        set_joint_target(arm, cmd)
        env.sim_step(render=False)

    final_object_pose = live_object_pose()
    final_link_pose = live_link_pose()
    result["final_live_poses"] = {
        "object_pose_wxyz": np.r_[final_object_pose[0], final_object_pose[1]].tolist(),
        "link6_pose_wxyz": np.asarray(final_link_pose, dtype=float).tolist(),
    }
    result["planned_final_link_pose_wxyz"] = np.asarray(final_target, dtype=float).tolist()
    metrics = env.official_success_metrics(0)
    result["official_metrics"] = metrics
    result["official_success"] = bool(metrics["success"])
    result["target_object_pose_wxyz"] = np.r_[official_target_p, selected_object_target_q].tolist()
    result["status"] = "PASS" if result["official_success"] else "FAIL"
except BaseException as exc:
    result["contact_records"] = contact_records
    result["contact_capture_errors"] = contact_capture_errors
    result["reason"] = f"{type(exc).__name__}: {exc}"
    result["traceback"] = traceback.format_exc()
finally:
    result["contact_records"] = contact_records
    result["contact_capture_errors"] = contact_capture_errors
    if env is not None:
        try:
            env.close()
        except Exception:
            pass
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    app.close(wait_for_replicator=False, skip_cleanup=True, exit_code=0 if result["official_success"] else 1)
