import torch
from isaaclab.envs import DirectRLEnv


class CustomDirectRLEnv(DirectRLEnv):
    """Expose RoboDojo's staged-command step through the native Lab step lifecycle."""

    def sim_step(self, render: bool = True):
        previous = self.render_enabled
        self.render_enabled = render
        try:
            self.step(torch.empty((self.num_envs, 0), device=self.device))
        finally:
            self.render_enabled = previous
        return self.extras
