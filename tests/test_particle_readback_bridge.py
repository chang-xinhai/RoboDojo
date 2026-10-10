"""CPU tensor boundary contracts; GPU allocation is validated separately in Kit."""

from enum import Enum
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import warp as wp

from utils.particle_readback_bridge import ReadbackTensorView, install_particle_readback_bridge


class NativeBodyView:
    def __init__(self):
        self.count = 2
        self.positions = wp.array([[1, 2, 3], [4, 5, 6]], dtype=wp.vec3, device="cpu")
        self.velocities = torch.tensor([[0.0, 1, 2], [3, 4, 5]])
        self.received = None

    def get_transforms(self):
        return self.positions

    def get_velocities(self):
        return self.velocities

    def get_motion(self):
        return self.positions, self.velocities

    def set_transforms(self, positions, *, indices):
        self.received = positions, indices
        positions_np = positions.numpy()
        indices_np = indices.numpy()
        native = self.positions.numpy()
        native[indices_np] = positions_np
        self.positions.assign(native)


class TensorBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()

    def test_real_warp_readback_updates_stable_independent_buffer(self):
        native = NativeBodyView()
        bridged = ReadbackTensorView(native, "cpu")
        first = bridged.get_transforms()
        self.assertNotEqual(first.ptr, native.positions.ptr)
        np.testing.assert_array_equal(first.numpy(), [[1, 2, 3], [4, 5, 6]])
        native.positions.assign(np.asarray([[7, 8, 9], [10, 11, 12]], dtype=np.float32))
        second = bridged.get_transforms()
        self.assertEqual(first.ptr, second.ptr)
        np.testing.assert_array_equal(first.numpy(), [[7, 8, 9], [10, 11, 12]])
        self.assertEqual(bridged.count, 2)

    def test_real_torch_shadow_preserves_owner_and_refreshes_in_place(self):
        native = NativeBodyView()
        bridged = ReadbackTensorView(native, "cpu")
        first = bridged.get_velocities()
        self.assertNotEqual(first.data_ptr(), native.velocities.data_ptr())
        native.velocities.add_(10)
        second = bridged.get_velocities()
        self.assertEqual(first.data_ptr(), second.data_ptr())
        torch.testing.assert_close(first, torch.tensor([[10.0, 11, 12], [13, 14, 15]]))

    def test_real_native_setter_receives_host_buffers_and_subset_indices(self):
        native = NativeBodyView()
        bridged = ReadbackTensorView(native, "cpu")
        value = wp.array([[90, 91, 92]], dtype=wp.vec3, device="cpu")
        indices = torch.tensor([1], dtype=torch.int32)
        bridged.set_transforms(value, indices=indices)
        self.assertIs(native.received[0], value)
        self.assertEqual(native.received[1].device.type, "cpu")
        np.testing.assert_array_equal(bridged.get_transforms().numpy(), [[1, 2, 3], [90, 91, 92]])

    def test_mixed_result_tuple_has_separate_stable_shadows(self):
        native = NativeBodyView()
        bridged = ReadbackTensorView(native, "cpu")
        warp_result, torch_result = bridged.get_motion()
        self.assertNotEqual(warp_result.ptr, native.positions.ptr)
        self.assertNotEqual(torch_result.data_ptr(), native.velocities.data_ptr())
        native.velocities.mul_(2)
        updated_warp, updated_torch = bridged.get_motion()
        self.assertEqual(updated_warp.ptr, warp_result.ptr)
        self.assertEqual(updated_torch.data_ptr(), torch_result.data_ptr())
        torch.testing.assert_close(updated_torch, native.velocities)

    def test_shape_and_dtype_changes_reallocate_shadow(self):
        native = NativeBodyView()
        bridged = ReadbackTensorView(native, "cpu")
        original = bridged.get_velocities()
        native.velocities = torch.tensor([[42]], dtype=torch.float64)
        changed = bridged.get_velocities()
        self.assertEqual(changed.shape, (1, 1))
        self.assertEqual(changed.dtype, torch.float64)
        self.assertNotEqual(original.data_ptr(), changed.data_ptr())

    def test_child_factory_keeps_native_factory_arguments(self):
        child = NativeBodyView()
        calls = []
        native = SimpleNamespace(create_rigid_body_view=lambda *args, **kwargs: calls.append((args, kwargs)) or child)
        bridged_child = ReadbackTensorView(native, "cpu").create_rigid_body_view("/World/body", flag=True)
        self.assertEqual(calls, [(("/World/body",), {"flag": True})])
        self.assertIs(bridged_child._native, child)
        np.testing.assert_array_equal(bridged_child.get_transforms().numpy(), [[1, 2, 3], [4, 5, 6]])

    def test_native_unready_child_factory_returns_none(self):
        native = SimpleNamespace(create_rigid_body_view=lambda path: None)
        self.assertIsNone(ReadbackTensorView(native, "cpu").create_rigid_body_view("/World/body"))

    def test_nested_result_list_uses_real_tensor_shadows(self):
        native = NativeBodyView()
        native.get_nested_state = lambda: [native.positions, (native.velocities, "metadata")]
        wrapped = ReadbackTensorView(native, "cpu")
        result = wrapped.get_nested_state()
        self.assertNotEqual(result[0].ptr, native.positions.ptr)
        self.assertNotEqual(result[1][0].data_ptr(), native.velocities.data_ptr())
        self.assertEqual(result[1][1], "metadata")
        native.positions.assign(np.full((2, 3), 42, dtype=np.float32))
        refreshed = wrapped.get_nested_state()
        self.assertEqual(refreshed[0].ptr, result[0].ptr)
        np.testing.assert_array_equal(result[0].numpy(), np.full((2, 3), 42))

    def test_install_uses_one_manager_view_before_asset_callbacks(self):
        class PhysicsEvent(Enum):
            PHYSICS_READY = "physics_ready"

        modules = {name: ModuleType(name) for name in ("isaaclab.physics", "isaaclab.utils.string")}
        modules["isaaclab.physics"].PhysicsEvent = PhysicsEvent
        registered = []
        native = NativeBodyView()
        manager = SimpleNamespace(
            _view=native,
            _scene_data_backend=SimpleNamespace(simulation_view=native),
            get_physics_sim_device=lambda: "cpu",
            register_callback=lambda callback, event, order: registered.append((callback, event, order)) or 123,
        )
        modules["isaaclab.utils.string"].string_to_callable = lambda value: manager
        with patch.dict(sys.modules, modules):
            handle = install_particle_readback_bridge("fake:manager")
        self.assertEqual(handle, 123)
        self.assertEqual(registered[0][1:], (PhysicsEvent.PHYSICS_READY, -100))
        registered[0][0](None)
        self.assertIsInstance(manager._view, ReadbackTensorView)
        self.assertIs(manager._view._native, native)
        self.assertIs(manager._scene_data_backend.simulation_view, manager._view)
        first = manager._view
        registered[0][0](None)
        self.assertIs(manager._view, first)
        manager._view = None
        registered[0][0](None)
        self.assertIsNone(manager._view)


if __name__ == "__main__":
    unittest.main()
