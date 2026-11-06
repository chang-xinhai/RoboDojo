from typing import Any, List

from env.camera_manager.camera_manager import CameraManager
from env.environment.base_env import BaseEnv
from env.robot_manager.control_manager import ControlSeq
from env.robot_manager.robot_manager import RobotManager
from env.scene_manager.scene_manager import SceneManager


class TaskEnv(BaseEnv):
    def __init__(self, config, app, **kwargs):
        self.sim_config = config.sim
        self.scene_config = config.scene
        self.num_envs = self.sim_config.scene.num_envs
        self.env_spacing = self.sim_config.scene.env_spacing
        self.device = self.sim_config.get("device", "cuda:0")
        self.dt = self.sim_config.dt

        # RobotManager must be created before BaseEnv.__init__ because it
        # provides the InteractiveSceneCfg used to launch the simulation.
        self.robot_config = config.robot
        self.robot_manager = RobotManager(
            num_envs=self.num_envs,
            env_spacing=self.env_spacing,
            config=self.robot_config,
            dt=self.dt,
            device=self.device,
        )
        self._interactiveSceneCfg = self.robot_manager._get_SceneCfg(
            num_envs=self.num_envs, env_spacing=self.env_spacing
        )
        super().__init__(self.sim_config, app=app, interactiveSceneCfg=self._interactiveSceneCfg, **kwargs)
        self.env_seed_list = self.env_seeds

        self.scene_manager = SceneManager(
            self.num_envs,
            self.device,
            self.sim_config.scene.env_spacing,
            seeds_per_env=self.env_seed_list,
            scene_config=self.scene_config,
            task_config=config.task_env,
        )

        self.robot_manager._load_camera_cfg(config.camera)
        self.capture_config = config.camera
        self.camera_config = config.camera
        self.camera_manager = CameraManager(
            self.num_envs,
            self.camera_config,
            self.device,
            seeds_per_env=self.env_seed_list,
        )
        from env.camera_manager.capture.capture_manager import CaptureManager

        self.capture_manager = CaptureManager(
            self.num_envs,
            self.capture_config,
            self.camera_manager,
            self.device,
        )

    def launch_sim(self, config):
        # Finalize robot mounts and camera declarations before the scene clone plan is built.
        self.interactiveSceneCfg = self.robot_manager._get_SceneCfg(
            num_envs=self.num_envs, env_spacing=self.env_spacing, replicate_physics=False)
        camera_configs = self.camera_manager.scene_configs()
        camera_configs.update(self.capture_manager.scene_configs(camera_configs))
        for key, camera_cfg in camera_configs.items():
            if hasattr(self.interactiveSceneCfg, key):
                raise ValueError(f"Duplicate scene entity: {key}")
            setattr(self.interactiveSceneCfg, key, camera_cfg)
        super().launch_sim(config)

    def _setup_scene(self, sim):
        self.scene_manager.initialize(sim)

    def _post_setup_scene(self, sim):
        self.robot_manager.initialize(sim)
        self.robot_manager.bind()
        self.scene_manager.post_init()
        self.camera_manager.bind(sim.scene)
        self.capture_manager.initialize(sim)

    def _write_data_to_sim(self):
        self.scene_manager.write_data_to_sim()

    def _update_scene(self, dt):
        self.scene_manager.update(dt)

    def _reset_idx(self, sim, env_ids):
        self.robot_manager.reset_state(env_ids)
        self.scene_manager.reset(env_ids)

    def update_seed(self, seed: Any | None = None):
        super().update_seed(seed)
        if hasattr(self, "scene_manager") and self.scene_manager is not None:
            self.scene_manager.update_env_seeds(self.env_seeds)
        if hasattr(self, "robot_manager") and self.robot_manager is not None:
            self.robot_manager.update_env_seeds(self.env_seeds)
        if hasattr(self, "camera_manager") and self.camera_manager is not None:
            self.camera_manager.update_env_seeds(self.env_seeds)

    def reset(self, seed: Any | None = None, options: dict[str, Any] | None = None):
        # Layout membership may change each episode; native contexts own immutable asset topology.
        # Preserve the manager's selected layouts while releasing all previous stage handles.
        self.close()
        super().reset(seed=seed, options=options)
        self.scene_manager.reload_scene()
        self.robot_manager.reset()
        for _ in range(300):
            self.sim_step(render=False)
        self.camera_manager.reset()
        self.capture_manager.reset()

    def step(self, meta_control_list: List[ControlSeq]):
        self.robot_manager.control_robot(meta_control_list=meta_control_list)

    def close(self):
        self.capture_manager.destroy()
        self.camera_manager.destroy()
        self.robot_manager.close()
        self.scene_manager.close()
        super().close()
