"""Real host-readback PhysX tensors with CUDA processing for PBD tasks.

GPU dynamics with readback enabled exposes native CPU tensor buffers. Lab 3
allocates its processing kernels on the configured CUDA device. This adapter
copies the real native buffers across that boundary and keeps stable CUDA
shadows; it does not change the simulation device or synthesize physics state.
"""

import torch
import warp as wp


class ReadbackTensorView:
    def __init__(self, native, device):
        self._native = native
        self._target_device = str(device)
        self._shadows = {}

    def _host(self, value):
        if isinstance(value, wp.array):
            if value.device.is_cuda:
                result = wp.empty_like(value, device="cpu")
                wp.copy(result, value)
                wp.synchronize_device(value.device)
                return result
            return value
        if isinstance(value, torch.Tensor):
            return value.to("cpu")
        if isinstance(value, tuple):
            return tuple(self._host(item) for item in value)
        if isinstance(value, list):
            return [self._host(item) for item in value]
        return value

    def _cuda(self, value, key):
        if isinstance(value, wp.array):
            if value.device.is_cuda:
                return value
            shadow = self._shadows.get(key)
            if shadow is None or shadow.shape != value.shape or shadow.dtype != value.dtype:
                shadow = self._shadows[key] = wp.empty_like(value, device=self._target_device)
            wp.copy(shadow, value)
            return shadow
        if isinstance(value, torch.Tensor):
            if value.is_cuda:
                return value
            shadow = self._shadows.get(key)
            if shadow is None or shadow.shape != value.shape or shadow.dtype != value.dtype:
                shadow = self._shadows[key] = torch.empty_like(value, device=self._target_device)
            shadow.copy_(value)
            return shadow
        if isinstance(value, tuple):
            return tuple(self._cuda(item, (key, index)) for index, item in enumerate(value))
        if isinstance(value, list):
            return [self._cuda(item, (key, index)) for index, item in enumerate(value)]
        return value

    def __getattr__(self, name):
        method = getattr(self._native, name)
        if not callable(method):
            return method
        if name.startswith("create_") and name.endswith("_view"):

            def factory(*args, **kwargs):
                value = method(*args, **kwargs)
                return None if value is None else ReadbackTensorView(value, self._target_device)

            return factory
        if name.startswith("get_"):

            def getter(*args, **kwargs):
                value = method(*self._host(args), **{key: self._host(v) for key, v in kwargs.items()})
                return self._cuda(value, name)

            return getter
        if name.startswith("set_"):
            return lambda *args, **kwargs: method(
                *self._host(args), **{key: self._host(v) for key, v in kwargs.items()}
            )
        return method


def install_particle_readback_bridge(physics_manager):
    """Install before assets' PHYSICS_READY callbacks, only for PBD contexts."""
    from isaaclab.physics import PhysicsEvent
    from isaaclab.utils.string import string_to_callable

    if isinstance(physics_manager, str):
        physics_manager = string_to_callable(str(physics_manager))

    def prepare(_payload):
        native = physics_manager._view
        if native is None or isinstance(native, ReadbackTensorView):
            return
        wrapped = ReadbackTensorView(native, physics_manager.get_physics_sim_device())
        physics_manager._view = wrapped
        # Official asset data use this same simulation view via the manager.
        physics_manager._scene_data_backend.simulation_view = wrapped

    return physics_manager.register_callback(prepare, PhysicsEvent.PHYSICS_READY, order=-100)
