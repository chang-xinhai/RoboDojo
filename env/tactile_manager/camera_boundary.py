"""Isaac Lab 3 camera boundary for the unchanged official TacSL model.

EA builds expose camera buffers as ProxyArray and may canonicalize depth keys.
Only array access/key names are adapted here; force physics and Taxim rendering
remain the official implementation.
"""

from utils.isaaclab_compat import as_torch


def cleanup_camera_once(camera, cleanup):
    """Release native render resources once, including TacSL's explicit destructor.

    Official TacSL calls its camera destructor explicitly and Python later calls
    it again. The pinned RTX renderer clears the render product after cleanup,
    so invoking it twice raises. Successful cleanup is remembered; failures are
    neither swallowed nor marked complete.
    """
    if getattr(camera, "_robodojo_camera_cleanup_complete", False):
        return
    cleanup()
    camera._robodojo_camera_cleanup_complete = True


def tactile_camera_depth(output):
    """Resolve either the EA canonical key or the legacy depth key explicitly."""
    for key in ("depth/distance_to_image_plane", "distance_to_image_plane", "depth"):
        if key in output:
            return as_torch(output[key])
    raise RuntimeError(f"TacSL camera did not produce image-plane depth; received {list(output)}")


class TacSLCameraBoundary:
    """Mixin adapting actual camera buffers to official TacSL's tensor inputs."""

    def get_initial_render(self):
        if not self.cfg.enable_camera_tactile:
            return None
        self._camera_sensor.update(dt=0.0, force_recompute=True)
        output = self._camera_sensor.data.output
        self._nominal_tactile = {"distance_to_image_plane": tactile_camera_depth(output).clone()}
        return self._nominal_tactile

    def _update_camera_tactile(self, env_ids):
        if self._nominal_tactile is None:
            raise RuntimeError("TacSL requires a no-contact baseline; call get_initial_render() first.")
        self._camera_sensor.update(self._sim_physics_dt, force_recompute=True)
        depth = tactile_camera_depth(self._camera_sensor.data.output)
        self._data.tactile_depth_image[env_ids] = depth[env_ids].clone()
        indentation = self._nominal_tactile["distance_to_image_plane"][env_ids] - depth[env_ids]
        self._data.tactile_rgb_image[env_ids] = self._tactile_rgb_render.render(indentation.squeeze(-1))
