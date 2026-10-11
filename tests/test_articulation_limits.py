"""Run the production initialization method without launching Kit or a GPU."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


class NoImplicitNumpyTensor(torch.Tensor):
    """CPU tensor with the same implicit-array prohibition as a CUDA tensor."""

    def __array__(self, *args, **kwargs):
        raise AssertionError("Implicit tensor-to-NumPy conversion is prohibited")


@pytest.fixture
def initialized_object_factory():
    source = ROOT / "env/scene_manager/objects/articulation.py"
    tree = ast.parse(source.read_text())
    production = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ArticulationObject"
    )
    initialize = next(
        node for node in production.body if isinstance(node, ast.FunctionDef) and node.name == "initialize"
    )

    class Parent:
        def initialize(self, **kwargs):
            self.parent_initialize(**kwargs)

    probe = ast.ClassDef(
        name="Probe",
        bases=[ast.Name(id="Parent", ctx=ast.Load())],
        keywords=[],
        body=[initialize],
        decorator_list=[],
    )
    physics_view = object()
    app = SimpleNamespace(update=Mock())
    namespace = {
        "Parent": Parent,
        "np": np,
        "torch": torch,
        "SimulationManager": SimpleNamespace(get_physics_sim_view=lambda: physics_view),
        "omni": SimpleNamespace(kit=SimpleNamespace(app=SimpleNamespace(get_app=lambda: app))),
    }
    module = ast.fix_missing_locations(ast.Module(body=[probe], type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    probe_type = namespace["Probe"]

    def forbidden_aggregate(self):
        raise AssertionError("Legacy aggregate DOF properties must not be queried")

    probe_type.dof_properties = property(forbidden_aggregate)

    def make(limits, num_dof=3):
        obj = probe_type()
        obj._prim_path = "/World/articulation"
        obj.num_dof = num_dof
        obj.parent_initialize = Mock()
        obj._articulation_view = SimpleNamespace(get_dof_limits=Mock(return_value=limits))
        obj.get_current_joint_positions = Mock(return_value=torch.zeros(num_dof))
        obj._capture_drive_targets = Mock(return_value={"joint": 0.2})
        return obj, physics_view, app

    return make


@pytest.mark.parametrize("frontend", ["numpy", "torch", "explicit-torch"])
def test_native_limits_are_explicit_host_snapshots(initialized_object_factory, frontend):
    limits = np.asarray([[[-2.0, 1.0], [-3.0, 4.0], [0.0, 0.0]]], dtype=np.float32)
    if frontend != "numpy":
        limits = torch.from_numpy(limits)
    if frontend == "explicit-torch":
        limits = limits.as_subclass(NoImplicitNumpyTensor)
    obj, physics_view, app = initialized_object_factory(limits)
    obj.initialize()
    np.testing.assert_array_equal(obj.lower_joint_positions, [-2, -3, 0])
    np.testing.assert_array_equal(obj.upper_joint_positions, [1, 4, 0])
    limits[...] = 99
    np.testing.assert_array_equal(obj.lower_joint_positions, [-2, -3, 0])
    np.testing.assert_array_equal(obj.upper_joint_positions, [1, 4, 0])
    obj.lower_joint_positions[0] = -10
    assert obj.upper_joint_positions[0] == 1
    obj.parent_initialize.assert_called_once_with(physics_sim_view=physics_view)
    obj._articulation_view.get_dof_limits.assert_called_once_with()
    torch.testing.assert_close(obj.initial_joint_positions, torch.zeros(3))
    assert obj.initial_drive_targets == {"joint": 0.2}
    app.update.assert_called_once_with()


@pytest.mark.parametrize(
    "limits,error",
    [(None, RuntimeError), (np.zeros((3, 2)), ValueError), (np.zeros((2, 3, 2)), ValueError)],
)
def test_missing_or_mismatched_native_limits_fail_explicitly(initialized_object_factory, limits, error):
    obj, _, app = initialized_object_factory(limits)
    with pytest.raises(error, match="articulation"):
        obj.initialize()
    obj.get_current_joint_positions.assert_not_called()
    obj._capture_drive_targets.assert_not_called()
    app.update.assert_not_called()


def test_zero_dof_native_articulation_limits(initialized_object_factory):
    obj, _, _ = initialized_object_factory(torch.empty((1, 0, 2)), num_dof=0)
    obj.initialize()
    assert obj.lower_joint_positions.shape == obj.upper_joint_positions.shape == (0,)
