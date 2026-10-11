from copy import deepcopy
from typing import List

import numpy as np

from env.description_manager.desc_manager import DescManager


class ObsManager:
    ANNOTATORS_TO_COLLECT = {
        "rgb": "color",
        "depth": "depth",
        "distance_to_image_plane": "distance_to_image_plane",
    }

    def __init__(
        self,
        obs_config,
        num_envs,
        dt,
        task_name,
        description_cfg,
        seeds_per_env: List[int] = None,
    ):
        self.obs_config = obs_config
        self.num_envs = num_envs
        self.dt = dt
        self.task_name = task_name
        self.description_cfg = description_cfg
        self.desc_manager = DescManager(
            num_envs=self.num_envs,
            description_cfg=self.description_cfg,
            desc_type="seen",
            seeds_per_env=seeds_per_env,
        )
        self.robot_cfg = obs_config.get("robot", {})
        self.multimodal = bool(obs_config.get("multimodal", False))
        self.collect_freq = self.obs_config.get("collect_freq", 0)
        # Frames the SDG dispatcher must deliver after the physics step before
        # images are read. One is exact with the zero-delay Kit settings that
        # the entrypoints enable; render_sync adds one more if they are absent.
        self.capture_render_passes = int(self.obs_config.get("capture_render_passes", 1))
        self.capture_render_max_passes = int(self.obs_config.get("capture_render_max_passes", 16))
        if self.collect_freq > 0:
            self.collect_interval = 1.0 / (self.dt * self.collect_freq)
        else:
            self.collect_interval = 0.0
        if not self.collect_interval.is_integer():
            raise ValueError("Collect Interval must be integer!")
        vision_cfg = self.obs_config.get("vision", {})
        self.collect_approximate_depth = vision_cfg.get("approximate_depth", True)
        self.collect_depth = vision_cfg.get("depth", False)
        self.collect_intrinsic_matrix = vision_cfg.get("intrinsic_matrix", False)
        self.collect_extrinsic_matrix = vision_cfg.get("extrinsic_matrix", False)
        self.collect_shape = vision_cfg.get("shape", True)

    def initialize(self, env):
        self.env = env
        if hasattr(env, "robot_manager"):
            self.robot_manager = env.robot_manager
        else:
            self.robot_manager = None
        if hasattr(env, "capture_manager"):
            self.capture_manager = env.capture_manager
        else:
            self.capture_manager = None
        if hasattr(env, "camera_manager"):
            self.camera_manager = env.camera_manager
        else:
            self.camera_manager = None

        self.tactile_manager = getattr(env, "tactile_manager", None)
        if self.multimodal:
            if not self.camera_manager or not getattr(self.camera_manager, "structured_light_enabled", False):
                raise RuntimeError("Multimodal capture requires enabled structured-light cameras")
            if self.tactile_manager is None or not getattr(self.tactile_manager, "enabled", False):
                raise RuntimeError("Multimodal capture requires the actual Isaac Lab TacSL tactile manager")
        if hasattr(env, "env_seed_list"):
            self.desc_manager.update_env_seeds(env.env_seed_list)
        self.initialize_instruction()

    def initialize_instruction(self):
        self.desc_manager.initialize(self.env)
        self.instruction = self.desc_manager.get_one_description()

    def reset(self):
        self.desc_manager.reset()
        self.instruction = self.desc_manager.get_one_description()

    def render_for_capture(self):
        """Render until the camera buffers hold the current physics state.

        Call this instead of a bare ``env.render()`` right before ``get_obs``.
        """
        from env.camera_manager.capture.render_sync import wait_for_latest_cameras

        wait_for_latest_cameras(
            self.env,
            capture_manager=self.capture_manager,
            min_passes=self.capture_render_passes,
            max_passes=self.capture_render_max_passes,
        )

    def get_obs(self, env_idx_list=None):  # batch
        if env_idx_list is None:
            env_idx_list = range(self.num_envs)
        obs = dict()
        for env_idx in env_idx_list:
            obs[env_idx] = dict()
            obs[env_idx]["vision"] = dict()
            obs[env_idx]["state"] = dict()
            obs[env_idx]["action"] = dict()
            obs[env_idx]["data_format_version"] = "v1.0"
            obs[env_idx]["additional_info"] = {
                "frequency": self.collect_freq,
            }
            obs[env_idx]["instruction"] = self.instruction[env_idx]
        if self.multimodal:
            self._capture_multimodal(obs, env_idx_list)
        elif self.camera_manager is not None and self.capture_manager is not None:
            try:
                from env.camera_manager.capture.render_sync import wait_for_camera_data
                from utils.isaaclab_compat import simulation_snapshot

                data, _ = wait_for_camera_data(
                    self.env,
                    lambda: self.capture_manager.step(env_ids=env_idx_list),
                    physics_state=lambda: simulation_snapshot(self.env.sim.sim),
                    min_passes=self.capture_render_passes,
                    max_passes=self.capture_render_max_passes,
                )
                for ith in range(self.camera_manager.num_cams):
                    cam_data = data[ith]
                    for annotator_name, env_list in cam_data.items():
                        collect_name = self.ANNOTATORS_TO_COLLECT[annotator_name]
                        for idx, env_idx in enumerate(env_idx_list):
                            camera_name = self.camera_manager.camera_names[env_idx][ith]
                            if camera_name not in obs[env_idx]["vision"]:
                                obs[env_idx]["vision"][camera_name] = dict()
                            obs[env_idx]["vision"][camera_name][collect_name] = env_list[idx]["data"]

                            if annotator_name == "rgb":
                                obs[env_idx]["vision"][camera_name][collect_name] = obs[env_idx]["vision"][camera_name][
                                    collect_name
                                ][:, :, :3]
                                if self.collect_shape:
                                    obs[env_idx]["vision"][camera_name]["shape"] = obs[env_idx]["vision"][camera_name][
                                        collect_name
                                    ].shape

            except Exception as e:
                import traceback

                stack_trace = traceback.format_exc()
                print(stack_trace)
                print("[get_obs] Camera observation capture failed.")
                raise e

            for env_idx in env_idx_list:
                for cam_data in obs[env_idx]["vision"].values():
                    if "distance_to_image_plane" in cam_data:
                        raw = cam_data.pop("distance_to_image_plane")
                        if hasattr(raw, "ndim") and raw.ndim >= 1 and raw.shape[-1] == 1:
                            raw = raw.squeeze(-1)
                        if self.collect_approximate_depth:
                            cam_data["approximate_depth"] = np.clip(raw * 1000, 0, 65535).astype(np.uint16)
                        if self.collect_depth:
                            cam_data["depth"] = raw.astype(np.float32)

            if self.collect_intrinsic_matrix or self.collect_extrinsic_matrix:
                for ith in range(self.camera_manager.num_cams):
                    for idx, env_idx in enumerate(env_idx_list):
                        camera_name = self.camera_manager.camera_names[env_idx][ith]
                        if self.collect_intrinsic_matrix:
                            obs[env_idx]["vision"][camera_name]["intrinsic_matrix"] = (
                                self.camera_manager.get_camera_intrinsics(ith, env_idx)
                            )
                        if self.collect_extrinsic_matrix:
                            obs[env_idx]["vision"][camera_name]["extrinsic_matrix"] = (
                                self.camera_manager.get_camera_extrinsics(ith, env_idx)
                            )

        if self.robot_manager is not None:
            try:
                for robot in self.robot_manager.robot_list:
                    if robot.type != "target":
                        continue
                    if self.robot_cfg["joint_states"]:
                        if robot.robot_type == "arm":
                            joints = self.robot_manager.get_joint(robot=robot, env_idx_list=env_idx_list)
                        for env_idx in env_idx_list:
                            if robot.robot_type == "arm":
                                obs[env_idx]["state"][f"{robot.arm_name}_joint_state"] = joints[env_idx]
                                obs[env_idx]["action"][f"{robot.arm_name}_joint_state"] = joints[env_idx]

                    if self.robot_cfg["world_ee_state"]:
                        endpose = self.robot_manager.get_real_endpose(robot=robot, env_idx_list=env_idx_list)
                        for env_idx in env_idx_list:
                            if self.robot_manager.target_arm_nums == 1:
                                obs[env_idx]["state"]["ee_pose"] = endpose[env_idx]
                            else:
                                name = robot.arm_name.split("_")[0]
                                obs[env_idx]["state"][f"{name}_ee_pose"] = endpose[env_idx]

                if hasattr(self.robot_manager, "control_manager"):
                    for env_idx in env_idx_list:
                        for (
                            key,
                            data,
                        ) in self.robot_manager.control_manager.prev_control[env_idx].items():
                            position = data["position"]
                            if key.endswith("ee_joint_state"):
                                end_effector_name = self.robot_manager.restore_name(deepcopy(key))
                                robot = self.robot_manager.get_robot_by_gripper_name(end_effector_name)
                                if robot.ee_type == "gripper":
                                    val = position[0]
                                    if robot.gripper_move["sign"] == 1:
                                        val = (val - robot.gripper_scale[0]) / (
                                            robot.gripper_scale[1] - robot.gripper_scale[0]
                                        )
                                    else:
                                        val = (robot.gripper_scale[1] - val) / (
                                            robot.gripper_scale[1] - robot.gripper_scale[0]
                                        )
                                    obs[env_idx]["action"][key] = [val]
                                    if self.robot_cfg["joint_states"]:
                                        obs[env_idx]["state"][key] = [val]
                                else:
                                    obs[env_idx]["action"][key] = position
                                    if self.robot_cfg["joint_states"]:
                                        obs[env_idx]["state"][key] = position

            except Exception as e:
                import traceback

                stack_trace = traceback.format_exc()
                print(stack_trace)
                print("[get_obs] Robot observation capture failed.")
                raise e
        if self.multimodal:
            from utils.multimodal import validate_multimodal

            for item in obs.values():
                validate_multimodal(item)
        return obs

    def _capture_multimodal(self, obs, env_idx_list):
        """Hold physics fixed while sequential render phases read every sensor.

        The shared envelope clock is simulation time, never wall-clock time.
        CameraManager and TacSL also provide independent acquisition clocks.
        """
        from utils.isaaclab_compat import simulation_snapshot
        from utils.multimodal import SCHEMA_VERSION, numpy_copy

        sim_context = self.env.sim.sim
        start_time, start_frame = simulation_snapshot(sim_context)
        timestamp_ns = round(start_time * 1e9)
        for env_idx in env_idx_list:
            cameras = self.camera_manager.capture_multimodal(env_id=env_idx)
            obs[env_idx]["vision"] = {name.removeprefix("cam_"): data for name, data in cameras.items()}
        tactile = self.tactile_manager.get_observations()
        for env_idx in env_idx_list:
            obs[env_idx]["tactile"] = {}
            for side, fingers in tactile.items():
                obs[env_idx]["tactile"][side] = {}
                for finger, data in fingers.items():
                    sample = {}
                    for field, value in data.items():
                        if hasattr(value, "shape"):
                            selected = value[env_idx]
                        elif isinstance(value, (list, tuple)) and field in {
                            "normal_force",
                            "shear_force",
                            "penetration_depth",
                            "points_pos_w",
                            "provenance",
                            "metadata",
                        }:
                            if len(value) != self.num_envs:
                                raise RuntimeError(
                                    f"TacSL {side}.{finger}.{field}: per-environment values must contain "
                                    f"{self.num_envs} samples, received {len(value)}"
                                )
                            selected = value[env_idx]
                        else:
                            selected = value
                        sample[field] = numpy_copy(selected) if hasattr(selected, "shape") else deepcopy(selected)
                    obs[env_idx]["tactile"][side][finger] = sample
            obs[env_idx]["sync"] = {
                "schema": SCHEMA_VERSION,
                "physics_frame": start_frame,
                "timestamp_ns": timestamp_ns,
                "simulation_time_s": start_time,
                "env_id": int(env_idx),
                "physics_held_fixed": True,
            }
            obs[env_idx]["data_format_version"] = SCHEMA_VERSION
        if simulation_snapshot(sim_context) != (start_time, start_frame):
            raise RuntimeError("Physics advanced during multimodal capture; refusing an unsynchronized observation")
