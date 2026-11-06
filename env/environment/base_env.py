from collections.abc import Sequence, Sequence as SequenceABC
from typing import Any
from copy import deepcopy

import gymnasium as gym
from omegaconf import DictConfig, OmegaConf

from env.environment.isaac.isaac_rl_env import IsaacRLEnv, IsaacRLEnvCfg
from env.seeding import seed_everywhere

DEFAULT_SIM_DEVICE = "cuda:0"


class BaseEnv(gym.Env):
    """Task-facing owner of one native Lab environment inside the application launcher."""

    def __init__(self, config: DictConfig, app=None, **kwargs):
        self.app = app
        self.config = config
        self.device = self.config.get("device", DEFAULT_SIM_DEVICE)
        self._seed_list: list[int] | None = None
        self._global_seed: int | None = None
        self.env_seeds: list[int] | None = None
        self._configure_seed_config(self.config.seed)
        self.sim = None
        self.stage = None
        self.interactiveSceneCfg = kwargs.get("interactiveSceneCfg")

    def launch_sim(self, config):
        self.sim_cfg = IsaacRLEnvCfg()
        self.sim_cfg.sim.device = self.device
        self.sim_cfg.decimation = int(config.decimation)
        self.sim_cfg.sim.dt = float(config.get("dt", 1 / 60))
        self.sim_cfg.sim.render_interval = int(config.get("render_interval", 1))
        self.sim_cfg.seed = self._global_seed
        # Scene entries are selected dynamically; dataclass clone drops undeclared attributes.
        self.sim_cfg.scene = deepcopy(self.interactiveSceneCfg)
        self.setup_sim_cfg(config)
        self.sim = IsaacRLEnv(cfg=self.sim_cfg, owner=self)
        self.stage = self.sim.sim.stage
        self.device = self.sim.device
        self.env_spacing = config.scene.env_spacing
        self.sim.env_spacing = self.env_spacing
        self.env_origins = self.sim.scene.env_origins
        self._post_setup_scene(self.sim)

    def setup_sim_cfg(self, config):
        """Apply explicit task physics overrides; native backend owns its defaults."""
        if "physx" in config:
            raise ValueError("Use sim.physics for native PhysxCfg overrides; sim.physx is no longer supported.")
        physics = config.get("physics", {})
        if isinstance(physics, DictConfig):
            physics = OmegaConf.to_container(physics, resolve=True)
        for key, value in physics.items():
            if key not in self.sim_cfg.sim.physics.__dataclass_fields__:
                raise ValueError(f"Unknown PhysxCfg field: {key}")
            setattr(self.sim_cfg.sim.physics, key, value)
        if "gravity" in config:
            self.sim_cfg.sim.gravity = tuple(config.gravity)

    def render(self):
        if self.sim is not None:
            self.sim.sim.render()

    def sim_step(self, render: bool = True):
        return self.sim.sim_step(render=render)

    def close(self):
        if self.sim is not None:
            self.sim.close()
            self.sim = None
        self.stage = None

    def update_seed(self, seed: Any | None = None):
        global_seed, _ = self._process_seed_input(seed)
        if global_seed is not None:
            seed_everywhere(global_seed)

    def reset(self, seed: Any | None = None, options: dict[str, Any] | None = None):
        self.update_seed(seed)
        if self.sim is None:
            self.launch_sim(self.config)
        return self.sim.reset(options=options)

    def _setup_scene(self, sim):
        """Author custom native assets before Lab emits PHYSICS_READY."""

    def _post_setup_scene(self, sim):
        """Bind task managers after the native environment is ready."""

    def _reset_idx(self, sim, env_ids):
        """Reset task-owned assets with explicit environment-to-instance indexing."""

    def _write_data_to_sim(self):
        pass

    def _update_scene(self, dt):
        pass

    def _configure_seed_config(self, seed_cfg: Any):
        if seed_cfg is None:
            self._global_seed = None
            self._seed_list = None
            self.env_seeds = None
            return
        if self._is_sequence_seed(seed_cfg):
            seeds = [int(s) for s in seed_cfg]
            self._set_env_seeds_full(seeds)
        else:
            self._set_global_seed(int(seed_cfg))

    def get_env_seed(self, env_id: int) -> int | None:
        if self._seed_list is not None:
            if env_id >= len(self._seed_list):
                raise IndexError(f"Requested env_id {env_id} exceeds configured seeds (len={len(self._seed_list)}).")
            return self._seed_list[env_id]
        return self._global_seed

    def _set_env_seed(self, env_id: int):
        seed = self.get_env_seed(env_id)
        if seed is not None:
            seed_everywhere(seed)

    # ------------------------------------------------------------------
    # Seed management helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _is_sequence_seed(seed: Any) -> bool:
        return isinstance(seed, SequenceABC) and not isinstance(seed, (str, bytes))

    def _set_global_seed(self, seed_value: int):
        self._global_seed = int(seed_value)
        if self.env_seeds is not None:
            self.env_seeds = [self._global_seed for _ in self.env_seeds]
            self._seed_list = list(self.env_seeds)
        else:
            self._seed_list = None

    def _set_env_seeds_full(self, seeds: Sequence[int]):
        num_envs = int(self.config.scene.num_envs)
        seed_list = [int(s) for s in seeds]
        if len(seed_list) != num_envs:
            raise ValueError(f"Seed list length {len(seed_list)} does not match num_envs {num_envs}.")
        self.env_seeds = list(seed_list)
        self._seed_list = list(seed_list)
        self._global_seed = seed_list[0] if seed_list else None

    def _process_seed_input(self, seed: Any) -> tuple[int | None, list[int] | None]:
        if seed is None:
            return None, self.env_seeds if self.env_seeds is not None else None

        if self._is_sequence_seed(seed):
            self._set_env_seeds_full(seed)
            return (
                self._global_seed,
                list(self.env_seeds) if self.env_seeds is not None else None,
            )

        self._set_global_seed(int(seed))
        return self._global_seed, None
