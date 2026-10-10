from typing import Any, List

from env.camera_manager.camera_manager import CameraManager
from env.environment.base_env import BaseEnv
from env.robot_manager.control_manager import ControlSeq
from env.robot_manager.robot_manager import RobotManager
from env.scene_manager.scene_manager import SceneManager
from env.tactile_manager import TactileManager


class TaskEnv(BaseEnv):
    def __init__(self, config, app, **kwargs):
        self.sim_config = config.sim
        self.scene_config = config.scene
        self.num_envs = self.sim_config.scene.num_envs
        self.env_spacing = self.sim_config.scene.env_spacing
        self.device = self.sim_config.get("device", "cpu")
        self.dt = self.sim_config.dt
        # PBD fluid rewards and resets require current particle positions in
        # USD. Select GPU dynamics with host readback before PhysX warmup.
        self._requires_particle_readback = bool(config.task_env.get("Fluid")) or bool(
            self.sim_config.get("particle_readback", False)
        )
        if self._requires_particle_readback and config.task_env.get("Garment"):
            raise NotImplementedError(
                "PBD fluid host readback and surface deformable garment tensors require different PhysX pipelines; "
                "run fluid and garment tasks in separate simulation contexts."
            )

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
            use_fabric=self.use_fabric,
            seeds_per_env=self.env_seed_list,
            scene_config=self.scene_config,
            task_config=config.task_env,
        )
        self.scene_manager.retained_layout_inventory = config.get("retained_layout_inventory")
        saved_layouts = config.get("saved_layouts")
        if saved_layouts is not None:
            if len(saved_layouts) != self.num_envs:
                raise ValueError(f"saved_layouts has {len(saved_layouts)} entries, expected {self.num_envs}")
            for env_idx, layout in enumerate(saved_layouts):
                self.scene_manager.layout_manager.set_saved_layout(env_idx, layout)
        self.tactile_manager = TactileManager(
            self.num_envs,
            config=config.get("tactile", {}),
            robot_manager=self.robot_manager,
            device=self.device,
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
        from env.camera_manager.capture.tiled_capture_manager import TiledCaptureManager

        self.capture_manager = TiledCaptureManager(
            self.num_envs,
            self.capture_config,
            self.camera_manager,
            self.device,
        )

    def _setup_scene(self, sim):
        self.robot_manager.initialize(sim)
        super()._setup_scene(sim)

    def _post_setup_scene(self, sim):
        self.scene_manager.initialize(sim)
        # TacSL mounts and SDF schemas must exist before Isaac Lab activates
        # the PhysX articulation and contact views during its first reset.
        self.tactile_manager.initialize(sim, self.scene_manager)
        super()._post_setup_scene(sim)
        self.camera_manager.initialize(sim)
        self.capture_manager.initialize(sim)
        self.camera_manager.post_init()
        # These legacy Replicator cameras are owned outside InteractiveScene,
        # so publish their rendering requirement to Isaac Lab explicitly.
        sim.sim.set_setting("/isaaclab/render/rtx_sensors", True)
        from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update

        # With --viz none, SimulationContext.render() has no Kit visualizer to
        # pump app.update(). Use the backend's deduplicated camera pump: it
        # synchronizes Fabric and pauses physics while producing the image.
        sim.sim.add_render_callback(
            "robodojo_legacy_camera_capture",
            lambda _: ensure_isaac_rtx_render_update(force=True),
        )

    def update_seed(self, seed: Any | None = None):
        super().update_seed(seed)
        if hasattr(self, "scene_manager") and self.scene_manager is not None:
            self.scene_manager.update_env_seeds(self.env_seeds)
        if hasattr(self, "robot_manager") and self.robot_manager is not None:
            self.robot_manager.update_env_seeds(self.env_seeds)
        if hasattr(self, "camera_manager") and self.camera_manager is not None:
            self.camera_manager.update_env_seeds(self.env_seeds)

    def reset(self, seed: Any | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed, options=options)
        self.scene_manager.reload_scene()
        self.robot_manager.reset()
        self.tactile_manager.reset()
        for _ in range(300):
            self.sim_step(render=False)
        self.camera_manager.reset()
        self.capture_manager.reset()
        self.tactile_manager.calibrate()

    def step(self, meta_control_list: List[ControlSeq]):
        self.robot_manager.control_robot(meta_control_list=meta_control_list)

    def close(self):
        if self.sim is not None:
            self.sim.sim.stop()
        self.tactile_manager.destroy()
        self.capture_manager.destroy()
        self.camera_manager.destroy()
        self.robot_manager.close()
        self.scene_manager.close(clear_prims=False)
        super().close()
