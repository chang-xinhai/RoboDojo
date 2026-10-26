"""UniVTAC insert_hole semantic adapter for RoboDojo."""
from __future__ import annotations
import numpy as np
import transforms3d as t3d
from env.environment.task_env import TaskEnv
from env.reward_manager.reward_manager import RewardManager

class MigrationInsertHoleCommon:
    def __init__(self, config, app, **kwargs):
        super().__init__(config, app, **kwargs)
        self.reward_manager = RewardManager(self.num_envs)
        self.step_lim = 1200
        # UniVTAC uses the tube pose expressed in the hole target frame.  The
        # official target translation is the hole frame origin; success then
        # requires the tube's relative z to be below -0.04 m.
        self.official_target_quat = np.asarray([0.9659258262890683, 0.0, -0.25881904510252074, 0.0], dtype=float)
        self._inhand_reference_z = None
        self._inhand_robot = None

    def _post_setup_scene(self, sim):
        super()._post_setup_scene(sim)
        self.reward_manager.initialize(self)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.reward_manager.reset()

    def run_reward(self):
        rm = self.reward_manager
        rm.check([rm.is_A_depth_in_B(label_A="prism", label_B="slot", z_threshold=0.04),
                  rm.is_AB_xy_distance_within_threshold(label_A="prism", label_B="slot", threshold=0.01),
                  rm.is_axis_up(label="prism", axis=[0, 0, 1], threshold=1),
                  rm.is_A_in_B(label_A="prism", label_B="slot")])

    def get_score(self):
        rm = self.reward_manager
        rm.score([[rm.is_A_depth_in_B(label_A="prism", label_B="slot", z_threshold=0.04),
                   rm.is_AB_xy_distance_within_threshold(label_A="prism", label_B="slot", threshold=0.01),
                   rm.is_A_in_B(label_A="prism", label_B="slot")]], [100], score_mode="transition")

    def gen_instruction(self, env_idx):
        return ["Insert the test tube into the inclined hole and keep it upright."]

    def attach_prism_to_arm(self, robot, arm_index: int):
        """Create a no-snap fixed grasp at the live closed-gripper pose."""
        from pxr import Gf, Sdf, UsdPhysics
        manager = self.scene_manager.layout_manager
        object_name = manager.get_instance_name(0, label="prism")
        object_obj = manager.get_scene_object(0, object_name)
        # Use the live PhysX rigid-body pose for attachment. LayoutManager stores
        # the authored/reset pose and can lag behind a dynamic body after contact.
        object_p, object_q = object_obj.get_world_pose()
        if hasattr(object_p, "detach"):
            object_p = object_p.detach().cpu().numpy()
        if hasattr(object_q, "detach"):
            object_q = object_q.detach().cpu().numpy()
        object_p = np.asarray(object_p, dtype=float).reshape(-1)[:3]
        object_q = self._unit(object_q, "prism")
        link_pose = self.robot_manager.get_link_pose(robot, "link6", env_idx_list=[0], is_relative=False)[0]
        link = np.asarray(link_pose, dtype=float).reshape(-1)
        link_p, link_q = link[:3], self._unit(link[3:], "link6")
        link_rot = t3d.quaternions.quat2mat(link_q)
        local_pos1 = link_rot.T @ (object_p - link_p)
        local_rot1 = t3d.quaternions.qmult(t3d.quaternions.qconjugate(link_q), object_q)
        stage = self.scene_manager.stage
        path = f"/World/envs/env_0/grasp_joint_{int(arm_index)}"
        link6_path = f"/World/envs/env_0/robot{int(arm_index)}/link6"
        link6_prim = stage.GetPrimAtPath(link6_path)
        object_prim = stage.GetPrimAtPath(str(object_obj.prim_path))
        # Define the joint with the live native poses before PhysX sees it.
        # Calling physx_utils.createJoint first registers helper defaults; changing
        # USD anchors afterwards leaves PhysX using the stale constraint.
        joint = UsdPhysics.FixedJoint.Define(stage, path)
        joint.CreateBody0Rel().SetTargets([Sdf.Path(str(object_obj.prim_path))])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(link6_path)])
        joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot0Attr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
        joint.CreateLocalPos1Attr(Gf.Vec3f(*[float(x) for x in local_pos1]))
        joint.CreateLocalRot1Attr(Gf.Quatf(float(local_rot1[0]), Gf.Vec3f(float(local_rot1[1]), float(local_rot1[2]), float(local_rot1[3]))))
        joint.CreateCollisionEnabledAttr().Set(False)
        joint.CreateJointEnabledAttr().Set(True)
        return {"path": path, "body0": str(object_obj.prim_path), "body1": link6_path,
                "creation": "UsdPhysics.FixedJoint.Define", "created_path": str(path),
                "local_pos0_expected": [0.0, 0.0, 0.0],
                "local_rot0_expected_wxyz": [1.0, 0.0, 0.0, 0.0],
                "local_pos1_expected": local_pos1.tolist(),
                "local_rot1_expected_wxyz": local_rot1.tolist()}

    @staticmethod
    def _unit(value, label):
        q = np.asarray(value, dtype=float).reshape(-1)
        if q.size != 4:
            raise RuntimeError(f"{label} quaternion shape={q.shape}")
        n = float(np.linalg.norm(q))
        if not np.isfinite(n) or n < 1e-12:
            raise RuntimeError(f"{label} quaternion norm={n}")
        return q / n

    def _live_pose(self, label, env_idx):
        p, q = self.scene_manager.layout_manager.get_instance_pose(env_idx, label=label, relative=False)
        return np.asarray(p, dtype=float).reshape(-1)[:3], self._unit(q, label)

    def record_inhand_reference(self, robot, env_idx=0):
        """Record the tube/link6 relative axial coordinate after attachment."""
        object_name = self.scene_manager.layout_manager.get_instance_name(env_idx, label="prism")
        object_obj = self.scene_manager.layout_manager.get_scene_object(env_idx, object_name)
        object_p, _ = object_obj.get_world_pose()
        if hasattr(object_p, "detach"):
            object_p = object_p.detach().cpu().numpy()
        object_p = np.asarray(object_p, dtype=float).reshape(-1)[:3]
        link_pose = np.asarray(
            self.robot_manager.get_link_pose(robot, "link6", env_idx_list=[env_idx], is_relative=False)[0],
            dtype=float,
        ).reshape(-1)
        rel = t3d.quaternions.quat2mat(link_pose[3:]).T @ (object_p - link_pose[:3])
        self._inhand_reference_z = float(rel[2])
        self._inhand_robot = robot
        return {"reference_inhand_z_m": self._inhand_reference_z}

    def official_success_metrics(self, env_idx=0):
        object_name = self.scene_manager.layout_manager.get_instance_name(env_idx, label="prism")
        object_obj = self.scene_manager.layout_manager.get_scene_object(env_idx, object_name)
        prism_p, prism_q = object_obj.get_world_pose()
        if hasattr(prism_p, "detach"):
            prism_p = prism_p.detach().cpu().numpy()
        if hasattr(prism_q, "detach"):
            prism_q = prism_q.detach().cpu().numpy()
        prism_p = np.asarray(prism_p, dtype=float).reshape(-1)[:3]
        prism_q = self._unit(prism_q, "prism")
        slot_p, _ = self._live_pose("slot", env_idx)
        hole_p = slot_p + np.asarray([0.0, 0.0, 0.10])
        target_q = self._unit(self.official_target_quat, "target")
        target_R = t3d.quaternions.quat2mat(target_q)
        rel_p = target_R.T @ (prism_p - hole_p)
        rel_q = t3d.quaternions.qmult(t3d.quaternions.qconjugate(target_q), prism_q)
        vertical_dot = float(abs(t3d.quaternions.quat2mat(rel_q)[2, 2]))
        current_inhand_z = None
        inhand_bias = None
        if getattr(self, "_inhand_robot", None) is not None:
            link_pose = np.asarray(
                self.robot_manager.get_link_pose(
                    self._inhand_robot, "link6", env_idx_list=[env_idx], is_relative=False
                )[0],
                dtype=float,
            ).reshape(-1)
            # The relative position is a 3-vector; UniVTAC's in-hand check
            # compares the axial z component only.
            current_inhand_z = float(
                (t3d.quaternions.quat2mat(link_pose[3:]).T @ (prism_p - link_pose[:3]))[2]
            )
            if self._inhand_reference_z is not None:
                inhand_bias = float(abs(self._inhand_reference_z - current_inhand_z))
        metrics = {
            "rel_x_m": float(rel_p[0]),
            "rel_y_m": float(rel_p[1]),
            "insert_depth_m": float(max(0.0, -rel_p[2])),
            "vertical_dot": vertical_dot,
            "inhand_reference_z_m": self._inhand_reference_z,
            "inhand_current_z_m": current_inhand_z,
            "inhand_bias_m": inhand_bias,
        }
        metrics["success"] = bool(
            abs(metrics["rel_x_m"]) < 0.01
            and abs(metrics["rel_y_m"]) < 0.01
            and metrics["insert_depth_m"] > 0.04
            and metrics["vertical_dot"] > 0.99
            and inhand_bias is not None
            and inhand_bias < 0.04
        )
        return metrics

class migration_insert_hole(MigrationInsertHoleCommon, TaskEnv):
    pass
