from isaaclab.envs.direct_rl_env import DirectRLEnv


class CustomDirectRLEnv(DirectRLEnv):
    def sim_step(self, render: bool = True):
        # Cameras are owned by RoboDojo, so the app must pump even though
        # they are not registered as Isaac Lab Camera assets.
        is_rendering = self.sim.is_rendering or bool(self.sim.get_setting("/isaaclab/render/rtx_sensors"))

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            if (
                self.cfg.sim.render_interval > 0
                and self._sim_step_counter % self.cfg.sim.render_interval == 0
                and is_rendering
                and render
            ):
                self.sim.render()
            self.scene.update(dt=self.physics_dt)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        return self.extras
