import torch
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab_physx.physics import PhysxCfg

from env.environment.isaac.direct_rl_env import CustomDirectRLEnv


@configclass
class IsaacRLEnvCfg(DirectRLEnvCfg):
    decimation = 1
    episode_length_s = 2000.0
    observation_space = 0
    action_space = 0
    state_space = 0
    num_rerenders_on_reset = 1
    scene = InteractiveSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=False)
    sim = SimulationCfg(device="cuda:0", gravity=(0.0, 0.0, -9.81), physics=PhysxCfg())


class IsaacRLEnv(CustomDirectRLEnv):
    """Native environment hooks for task-owned singleton assets and staged controls."""

    def __init__(self, cfg, owner, **kwargs):
        self.owner = owner
        super().__init__(cfg, **kwargs)

    def _setup_scene(self):
        self.owner._setup_scene(self)

    def _pre_physics_step(self, actions):
        pass  # RobotManager has already staged native actuator commands.

    def _apply_action(self):
        self.owner._write_data_to_sim()

    def _get_observations(self):
        return {"policy": torch.empty((self.num_envs, 0), device=self.device)}

    def _get_dones(self):
        # Refresh singleton asset caches before reward/observation consumers read them.
        self.owner._update_scene(self.step_dt)
        done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return done, done

    def _get_rewards(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        self.owner._reset_idx(self, env_ids)

    def reset_selected(self, env_ids):
        """Reset selected task instances without advancing physics or other environments."""
        indices = torch.as_tensor(env_ids, device=self.device, dtype=torch.int32)
        self._reset_idx(indices)
        self.scene.write_data_to_sim()
        self.owner._write_data_to_sim()
        self.sim.forward()
        if self.has_rtx_sensors:
            for _ in range(self.cfg.num_rerenders_on_reset):
                self.sim.render()
        return self._get_observations(), self.extras
