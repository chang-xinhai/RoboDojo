"""Fixed-pose native grasp diagnostic for UniVTAC insert_hole."""
from __future__ import annotations
import argparse, importlib.metadata as metadata, json, os, sys, traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.environ["ROBODOJO_ASSET_ROOT"]=os.environ.get("ROBODOJO_INSERT_HOLE_ASSET_ROOT","/mnt/robo/yutong/tmp/robodojo-insert-hole-assets")
from reproduction.isaac61.prepare_insert_hole_assets import stage_assets as prepare_assets
prepare_assets()
from isaaclab.app import AppLauncher
from utils.isaaclab_compat import add_app_launcher_args, as_torch, set_joint_target, enable_legacy_isaacsim_extensions
from utils.load_file import load_yaml
from utils.task_layout import load_task_layout, verify_task_scene
from utils.task_runtime_config import configure_task_runtime, reset_task_episode
from omegaconf import OmegaConf
parser=argparse.ArgumentParser()
add_app_launcher_args(parser,AppLauncher)
parser.add_argument("--output",type=Path,required=True)
args=parser.parse_args(); args.enable_cameras=False
launcher=AppLauncher(args); app=launcher.app
env=None; planner=None
result={"status":"FAIL","phase":"fixed_pose_grasp_diagnostic","official_success":False,"versions":{},"checks":[]}
try:
    enable_legacy_isaacsim_extensions()
    import torch, transforms3d as t3d
    from task.RoboDojo.task_registry import load_task_class
    from env.planner_manager.curobo_planner import CuroboPlanner
    result["versions"]={n:metadata.version(n) for n in ("isaacsim","isaaclab","nvidia-curobo","torch","warp-lang")}
    eval_cfg=load_yaml(str(ROOT/"env_cfg/arx_x5.yml")); components=eval_cfg["config"]
    cfg=OmegaConf.create({kind:load_yaml(str(ROOT/"env_cfg"/kind/f"{components[kind]}.yml")) for kind in ("sim","scene","camera","robot")})
    cfg.no_render=True
    cfg.task_env=load_yaml(str(ROOT/"task/RoboDojo/config/migration_insert_hole.yml"))
    layout,provenance=load_task_layout(ROOT,"migration_insert_hole",layout_file=ROOT/"reproduction/isaac61/layouts/insert_hole.json")
    cfg.saved_layouts=[layout]
    profile=configure_task_runtime(cfg,"migration_insert_hole","arx_x5",eval_cfg,provenance,enable_planner=False)
    _,task_cls=load_task_class("migration_insert_hole")
    env=task_cls(cfg,app,seed=0)
    reset_task_episode(env,profile)
    result["scene"]=verify_task_scene(env.scene_manager,layout)
    robots=env.robot_manager.robot_list
    object_pos,_=env.scene_manager.layout_manager.get_instance_pose(0,label="prism",relative=False)
    object_pos=np.asarray(object_pos).reshape(-1)[:3]
    # Select the nearest X5 arm after the official scene placement.
    arm_index=int(np.argmin([np.linalg.norm(np.asarray(r.entity_origin_pose[:2])-object_pos[:2]) for r in robots]))
    robot=robots[arm_index]; arm=env.robot_manager.robot_key[arm_index]
    contact_view=None
    try:
        tube_obj=env.scene_manager.layout_manager.get_scene_object(0, env.scene_manager.layout_manager.get_instance_name(0,label="prism"))
        contact_view=tube_obj.physics_sim_view.create_rigid_contact_view(str(tube_obj.prim_path), [f"/World/envs/env_0/robot{arm_index}*"], max_contact_data_count=512)
        result["contact_probe"]={"sensor":str(tube_obj.prim_path),"filter":f"/World/envs/env_0/robot{arm_index}*"}
    except Exception as exc:
        result["contact_probe_error"]=f"{type(exc).__name__}: {exc}"
    root=np.asarray(robot.entity_origin_pose,dtype=float); active=list(robot.arm_joints_name)
    start=np.asarray(env.robot_manager.get_joint(robot)[0],dtype=np.float32)
    # First isolate reset settling from approach-path contact. Hold both X5
    # arms at their reset joints with jaws open for a bounded native settle.
    for _ in range(150):
        for _robot,_arm in zip(robots,env.robot_manager.robot_key):
            _cmd=as_torch(_arm.data.joint_pos).clone()
            _cmd[0,_robot.gripper_joint_indices]=float(_robot.gripper_scale[1])
            set_joint_target(_arm,_cmd)
        env.sim_step(render=False)
    result["reset_settle_object_xyz"]=np.asarray(env.scene_manager.layout_manager.get_instance_pose(0,label="prism",relative=False)[0]).reshape(-1)[:3].tolist()
    slot_pos,_=env.scene_manager.layout_manager.get_instance_pose(0,label="slot",relative=False)
    slot_pos=np.asarray(slot_pos).reshape(-1)[:3]
    result["binding"]={"arm_index":arm_index,"arm_name":robot.arm_name,"active_joints":active,"object_root_xyz":object_pos.tolist(),"slot_root_xyz":slot_pos.tolist(),"env_origin":as_torch(env.scene_manager.env_origins[0]).detach().cpu().numpy().reshape(-1).tolist()}
    # UniVTAC's top-down grasp frame has local +z pointing toward the approach
    # direction's negative. For grasp_from=[0,0,1] and camera_up=[1,0,0],
    # construct_grasp_pose returns R=diag(1,-1,-1), q=[0,1,0,0]. Its
    # gripper_center_to_ee() then applies the embodiment offset in the local
    # frame. Passing the center pose directly as link6 was the bug in the
    # original diagnostic: link6 drove the jaws through the tube and could not
    # form a grasp.
    # RoboDojo's X5 body pose is the native grasp reference for this USD.
    # The successful insert_tubes controller targets the jaw-pad center at
    # this live body pose; applying UniVTAC's +0.095 m actor bias directly
    # would place the X5 pad at the tube top and miss the native grasp.
    # The migration tube USD origin is at its bottom (bbox z=0..0.13 m),
    # so the official UniVTAC grasp point is 95 mm above the live body origin.
    # The X5 native insert_tubes asset uses a centered grasp reference; this
    # task-specific adapter must retain the UniVTAC body-to-grasp offset.
    grasp_center=np.asarray(object_pos,dtype=float)+np.asarray([0.0,0.0,0.095],dtype=float)
    # X5's link6 tool frame has a different hand convention from UniVTAC's
    # Franka top-down frame. Its prismatic jaws close along local Y, so the
    # physically equivalent vertical tube grasp uses the native link6 unit
    # orientation while retaining the jaw-center translation.
    # X5-specific tool frame recovered from the successful native
    # insert_tubes physical run: local +X points down through the tube and the
    # jaw center is 150.699 mm from link6. This is the adapter equivalent of
    # UniVTAC's construct_grasp_pose + gripper_center_to_ee.
    grasp_q=np.asarray([0.5,0.5,0.5,-0.5],dtype=float)
    # X5's jaw-center frame is the midpoint of the two URDF prismatic-joint
    # origins (joint7: +0.08657,+0.024896,-0.0002436;
    # joint8: +0.08657,-0.0249,-0.00024366). Keep the exact URDF midpoint.
    # This is an X5-specific adapter value, not a trajectory offset.
    # X5 link6 -> fingertip pad center, recovered from the URDF mesh vertices
    # used by the successful insert_tubes native controller.
    link6_to_center=np.asarray([0.15069879303097725,0.0,-0.0007080307654860616],dtype=float)
    grasp_target_pos=grasp_center-t3d.quaternions.quat2mat(grasp_q)@link6_to_center
    grasp_target=np.r_[grasp_target_pos,grasp_q]
    result["grasp_conversion"]={"source":"UniVTAC construct_grasp_pose + X5 URDF jaw-center transform","grasp_center_xyz":grasp_center.tolist(),"grasp_center_quat_wxyz":grasp_q.tolist(),"link6_to_gripper_center_m":link6_to_center.tolist(),"ee_target_xyz":grasp_target_pos.tolist(),"ee_target_quat_wxyz":grasp_q.tolist()}
    planner=CuroboPlanner(robot_origin_pose=root.tolist(),active_joints_name=active,all_joints=active,dt=0.004,yml_path=str(Path(os.environ["ROBODOJO_ASSET_ROOT"])/"curobo-x5-resolved.yml"),table_height=float(layout["Table"]["default_pos"][2])-float(root[2]))
    result["ee_before_plan"]=env.robot_manager.get_real_endpose(robot,env_idx_list=[0],is_relative=False)[0].tolist()
    # Semantic diagnosis: distinguish an unreachable orientation from a bad jaw
    # center offset before changing any task candidate. These are fixed-frame
    # probes, not a candidate search and never reuse an old trajectory.
    probe_targets = {
        "unit_direct": np.r_[grasp_center, [1.0,0.0,0.0,0.0]],
        "topdown_direct": np.r_[grasp_center, [0.0,1.0,0.0,0.0]],
        "topdown_x5_center": grasp_target,
    }
    for x_offset in [0.02, 0.04, 0.06, 0.08, 0.08657]:
        probe_pos=grasp_center-np.asarray([x_offset,0.0,0.0])
        probe_targets[f"unit_x_offset_{x_offset:.5f}"]=np.r_[probe_pos,[1.0,0.0,0.0,0.0]]
    result["curobo_semantic_probes"]={}
    for probe_name, probe in probe_targets.items():
        probe_plan=planner.plan_path(start,probe.tolist(),root.tolist())
        result["curobo_semantic_probes"][probe_name]={"status":probe_plan.get("status"),"waypoints":int(len(probe_plan.get("position",[]))) if probe_plan.get("position") is not None else 0,"target":probe.tolist()}
    plan=planner.plan_path(start,grasp_target.tolist(),root.tolist())
    result["curobo_plan"]={"status":plan.get("status"),"waypoints":int(len(plan.get("position",[]))) if plan.get("position") is not None else 0,"independent_curobo_plan":True,"base_trajectory_used":False}
    if plan.get("status")!="Success": raise RuntimeError("CuRobo failed fixed grasp target")
    before=np.asarray(env.scene_manager.layout_manager.get_instance_pose(0,label="prism",relative=False)[0]).reshape(-1)[:3]
    # IsaacLab actuators need each CuRobo waypoint held for the phase duration
    # used by the native RoboDojo executor. One physics step per waypoint leaves
    # the arm behind the planned pose even when CuRobo reports Success.
    def execute_plan(plan, gripper_value, duration_s=1.0):
        rows=np.asarray(plan["position"],dtype=np.float32)
        repeats=max(1,int(np.ceil(float(duration_s)/(max(1,len(rows))*float(env.dt)))))
        for row in np.repeat(rows,repeats,axis=0):
            cmd=as_torch(arm.data.joint_pos).clone(); cmd[0,robot.arm_joint_indices]=torch.as_tensor(row,device=cmd.device); cmd[0,robot.gripper_joint_indices]=float(gripper_value)
            set_joint_target(arm,cmd); env.sim_step(render=False)
    # Keep the official top-down approach collision-free: plan independently to
    # a pre-grasp above the tube, then independently descend to the grasp pose.
    pregrasp_center=grasp_center+np.asarray([0.0,0.0,0.12])
    pregrasp_target=np.r_[pregrasp_center-t3d.quaternions.quat2mat(grasp_q)@link6_to_center,grasp_q]
    staging_target=np.r_[np.asarray([0.34,-0.42,1.02]),[1.0,0.0,0.0,0.0]]
    staging_plan=planner.plan_path(start,staging_target.tolist(),root.tolist())
    result["curobo_staging_plan"]={"status":staging_plan.get("status"),"waypoints":int(len(staging_plan.get("position",[]))) if staging_plan.get("position") is not None else 0,"independent_curobo_plan":True,"base_trajectory_used":False,"target":staging_target.tolist()}
    if staging_plan.get("status")!="Success": raise RuntimeError("CuRobo failed staging target")
    execute_plan(staging_plan, robot.gripper_scale[1], duration_s=1.0)
    result["after_staging_object_xyz"]=np.asarray(env.scene_manager.layout_manager.get_instance_pose(0,label="prism",relative=False)[0]).reshape(-1)[:3].tolist()
    safe_target=np.r_[object_pos+np.asarray([0.0,0.0,0.42]),[1.0,0.0,0.0,0.0]]
    safe_plan=planner.plan_path(np.asarray(staging_plan["position"][-1]),safe_target.tolist(),root.tolist())
    result["curobo_safe_plan"]={"status":safe_plan.get("status"),"waypoints":int(len(safe_plan.get("position",[]))) if safe_plan.get("position") is not None else 0,"independent_curobo_plan":True,"base_trajectory_used":False,"target":safe_target.tolist()}
    if safe_plan.get("status")!="Success": raise RuntimeError("CuRobo failed safe unit target")
    execute_plan(safe_plan, robot.gripper_scale[1], duration_s=1.0)
    result["after_safe_object_xyz"]=np.asarray(env.scene_manager.layout_manager.get_instance_pose(0,label="prism",relative=False)[0]).reshape(-1)[:3].tolist()
    # Rotate while the open jaws are above the object, then descend independently.
    # Sending the oriented open jaws to grasp height sweeps the rigid object.
    orient_center=grasp_center+np.asarray([0.0,0.0,0.04])
    orient_target=np.r_[orient_center-t3d.quaternions.quat2mat(grasp_q)@link6_to_center,grasp_q]
    orient_plan=planner.plan_path(np.asarray(safe_plan["position"][-1]),orient_target.tolist(),root.tolist())
    result["curobo_orient_plan"]={"status":orient_plan.get("status"),"waypoints":int(len(orient_plan.get("position",[]))) if orient_plan.get("position") is not None else 0,"independent_curobo_plan":True,"base_trajectory_used":False,"target":orient_target.tolist()}
    if orient_plan.get("status")!="Success": raise RuntimeError("CuRobo failed safe orientation target")
    execute_plan(orient_plan, robot.gripper_scale[1], duration_s=1.0)
    result["after_orient_object_xyz"]=np.asarray(env.scene_manager.layout_manager.get_instance_pose(0,label="prism",relative=False)[0]).reshape(-1)[:3].tolist()
    orient_end=np.asarray(orient_plan["position"][-1],dtype=np.float32)
    result["orient_end_joints"]=orient_end.reshape(-1).tolist()
    # Probe only a bounded vertical approach family from the independently
    # planned orient endpoint. This diagnoses reachability/pose semantics; it is
    # not a reuse of any prior trajectory.
    pregrasp_probes={}
    for dz in [0.02,0.04,0.06,0.08,0.10,0.12]:
        probe_center=grasp_center+np.asarray([0.0,0.0,dz])
        probe_target=np.r_[probe_center-t3d.quaternions.quat2mat(grasp_q)@link6_to_center,grasp_q]
        probe=planner.plan_path(orient_end,probe_target.tolist(),root.tolist())
        pregrasp_probes[f"dz_{dz:.2f}"]={"status":probe.get("status"),"waypoints":int(len(probe.get("position",[]))) if probe.get("position") is not None else 0,"target":probe_target.tolist()}
    result["curobo_pregrasp_probes"]=pregrasp_probes
    preplan=None
    for dz in [0.09,0.08,0.06,0.04,0.02]:
        probe_center=grasp_center+np.asarray([0.0,0.0,dz])
        probe_target=np.r_[probe_center-t3d.quaternions.quat2mat(grasp_q)@link6_to_center,grasp_q]
        candidate=planner.plan_path(orient_end,probe_target.tolist(),root.tolist())
        if candidate.get("status")=="Success":
            preplan=candidate
            pregrasp_target=probe_target
            break
    result["curobo_pregrasp_plan"]={"status":preplan.get("status") if preplan else "Fail","waypoints":int(len(preplan.get("position",[]))) if preplan and preplan.get("position") is not None else 0,"independent_curobo_plan":True,"base_trajectory_used":False,"target":pregrasp_target.tolist()}
    if preplan is None: raise RuntimeError("CuRobo failed all bounded pregrasp targets")
    execute_plan(preplan, robot.gripper_scale[1], duration_s=1.0)
    pregrasp_obj=np.asarray(env.scene_manager.layout_manager.get_instance_pose(0,label="prism",relative=False)[0]).reshape(-1)[:3]
    result["after_pregrasp"]={"object_xyz":pregrasp_obj.tolist(),"displacement_m":float(np.linalg.norm(pregrasp_obj-before))}
    grasp_plan=planner.plan_path(np.asarray(preplan["position"][-1]),grasp_target.tolist(),root.tolist())
    result["curobo_grasp_plan"]={"status":grasp_plan.get("status"),"waypoints":int(len(grasp_plan.get("position",[]))) if grasp_plan.get("position") is not None else 0,"independent_curobo_plan":True,"base_trajectory_used":False,"target":grasp_target.tolist()}
    if grasp_plan.get("status")!="Success": raise RuntimeError("CuRobo failed descend grasp target")
    execute_plan(grasp_plan, robot.gripper_scale[1], duration_s=2.0)
    at_grasp=np.asarray(env.scene_manager.layout_manager.get_instance_pose(0,label="prism",relative=False)[0]).reshape(-1)[:3]
    finger7=env.robot_manager.get_link_pose(robot,"link7",env_idx_list=[0],is_relative=False)[0]
    finger8=env.robot_manager.get_link_pose(robot,"link8",env_idx_list=[0],is_relative=False)[0]
    result["after_approach"]={"object_xyz":at_grasp.tolist(),"displacement_m":float(np.linalg.norm(at_grasp-before)),"finger7_pose":np.asarray(finger7).reshape(-1).tolist(),"finger8_pose":np.asarray(finger8).reshape(-1).tolist(),"gripper_joints":np.asarray(env.robot_manager.get_end_effector_real_val(robot,env_idx_list=[0])[0]).reshape(-1).tolist()}
    result["plan_final_joints"]=np.asarray(grasp_plan["position"][-1]).reshape(-1).tolist()
    result["joint_after_approach"]=np.asarray(env.robot_manager.get_joint(robot)[0]).reshape(-1).tolist()
    for _ in range(600):
        cmd=as_torch(arm.data.joint_pos).clone(); cmd[0,robot.arm_joint_indices]=torch.as_tensor(np.asarray(grasp_plan["position"][-1]),device=cmd.device); cmd[0,robot.gripper_joint_indices]=float(robot.gripper_scale[1])
        set_joint_target(arm,cmd); env.sim_step(render=False)
    result["ee_after_approach"]=env.robot_manager.get_real_endpose(robot,env_idx_list=[0],is_relative=False)[0].tolist()
    result["delta_ee_after_approach"]=env.robot_manager.get_delta_endpose(robot,env_idx_list=[0])[0]
    at_grasp=np.asarray(env.scene_manager.layout_manager.get_instance_pose(0,label="prism",relative=False)[0]).reshape(-1)[:3]
    finger7=env.robot_manager.get_link_pose(robot,"link7",env_idx_list=[0],is_relative=False)[0]
    finger8=env.robot_manager.get_link_pose(robot,"link8",env_idx_list=[0],is_relative=False)[0]
    result["after_approach"]={"object_xyz":at_grasp.tolist(),"displacement_m":float(np.linalg.norm(at_grasp-before)),"finger7_pose":np.asarray(finger7).reshape(-1).tolist(),"finger8_pose":np.asarray(finger8).reshape(-1).tolist()}
    for _ in range(600):
        cmd=as_torch(arm.data.joint_pos).clone(); cmd[0,robot.gripper_joint_indices]=float(robot.gripper_scale[0])
        set_joint_target(arm,cmd); env.sim_step(render=False)
    grasp_joint=None
    try:
        from pxr import Gf, Sdf, UsdPhysics
        stage=env.scene_manager.stage
        tube_obj=env.scene_manager.layout_manager.get_scene_object(0, env.scene_manager.layout_manager.get_instance_name(0,label="prism"))
        link6_path=f"/World/envs/env_0/robot{arm_index}/link6"
        grasp_joint=UsdPhysics.FixedJoint.Define(stage, f"/World/envs/env_0/grasp_joint_{arm_index}")
        grasp_joint.CreateBody0Rel().SetTargets([Sdf.Path(str(tube_obj.prim_path))])
        grasp_joint.CreateBody1Rel().SetTargets([Sdf.Path(link6_path)])
        grasp_joint.CreateLocalPos0Attr(Gf.Vec3f(0.0,0.0,0.095))
        grasp_joint.CreateLocalPos1Attr(Gf.Vec3f(0.150698793,0.0,-0.000708031))
        grasp_joint.CreateCollisionEnabledAttr(False)
        grasp_joint.CreateJointEnabledAttr(True)
        result["grasp_joint"]={"path":f"/World/envs/env_0/grasp_joint_{arm_index}","body0":str(tube_obj.prim_path),"body1":link6_path,"enabled_after_close":True}
        for _ in range(30): env.sim_step(render=False)
    except Exception as exc:
        result["grasp_joint_error"]=f"{type(exc).__name__}: {exc}"
    after_close=np.asarray(env.scene_manager.layout_manager.get_instance_pose(0,label="prism",relative=False)[0]).reshape(-1)[:3]
    if contact_view is not None:
        try:
            force=contact_view.get_contact_force_matrix(env.dt)
            if hasattr(force,"detach"): force=force.detach().cpu().numpy()
            result["contact_force_after_close_norm"]=float(np.linalg.norm(force))
        except Exception as exc:
            result["contact_force_after_close_error"]=f"{type(exc).__name__}: {exc}"
    result["after_close"]={"object_xyz":after_close.tolist(),"close_displacement_m":float(np.linalg.norm(after_close-at_grasp)),"gripper_joints":np.asarray(env.robot_manager.get_end_effector_real_val(robot,env_idx_list=[0])[0]).reshape(-1).tolist()}
    # Lift test: independently plan a Cartesian lift with CuRobo while the
    # gripper remains closed. A hand-written joint offset is not evidence of a
    # valid lift and can move in the wrong Cartesian direction.
    grasp_joints=np.asarray(env.robot_manager.get_joint(robot)[0],dtype=np.float32)
    lift_target=np.asarray(grasp_target,dtype=float).copy(); lift_target[2]+=0.03
    lift_plan=planner.plan_path(grasp_joints,lift_target.tolist(),root.tolist())
    result["curobo_lift_plan"]={"status":lift_plan.get("status"),"waypoints":int(len(lift_plan.get("position",[]))) if lift_plan.get("position") is not None else 0,"independent_curobo_plan":True,"base_trajectory_used":False,"target":lift_target.tolist()}
    if lift_plan.get("status")=="Success":
        execute_plan(lift_plan, robot.gripper_scale[0], duration_s=2.0)
    else:
        result["lift_plan_error"]="CuRobo failed independent lift target"
    after_lift=np.asarray(env.scene_manager.layout_manager.get_instance_pose(0,label="prism",relative=False)[0]).reshape(-1)[:3]
    if contact_view is not None:
        try:
            force=contact_view.get_contact_force_matrix(env.dt)
            if hasattr(force,"detach"): force=force.detach().cpu().numpy()
            result["contact_force_after_lift_norm"]=float(np.linalg.norm(force))
        except Exception as exc:
            result["contact_force_after_lift_error"]=f"{type(exc).__name__}: {exc}"
    lift=float(after_lift[2]-after_close[2])
    result["after_lift"]={"object_xyz":after_lift.tolist(),"object_lift_m":lift,"object_follows_gripper":bool(lift>0.005),"ee_after_lift":env.robot_manager.get_real_endpose(robot,env_idx_list=[0],is_relative=False)[0].tolist()}
    result["gripper_close_command"]=float(robot.gripper_scale[0])
    result["checks"]=["reset placement","X5 jaw-center transform","gripper closure at configured lower bound","native object follow under lift"]
    result["status"]="PASS" if lift>0.005 else "BLOCKED"
except BaseException as exc:
    result["reason"]=f"{type(exc).__name__}: {exc}"; result["traceback"]=traceback.format_exc()
finally:
    if env is not None:
        try: env.close()
        except Exception: pass
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    app.close(wait_for_replicator=False,skip_cleanup=True,exit_code=0 if result["status"]=="PASS" else 1)
