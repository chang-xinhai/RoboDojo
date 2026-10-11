"""CPU reward snapshots and vectorized geometry match native/Gf conventions."""

import ast
from copy import deepcopy
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
from pxr import Gf
import pytest
import torch

from utils.live_geometry import transform_points_world

ROOT = Path(__file__).resolve().parents[1]


class NoImplicitNumpyTensor(torch.Tensor):
    def __array__(self, *args, **kwargs):
        raise AssertionError("Native CUDA-style poses require explicit host conversion")


@pytest.mark.parametrize("kind", ["rigid", "articulation"])
@pytest.mark.parametrize("relative", [True, False])
def test_actual_layout_pose_returns_independent_host_snapshots(kind, relative):
    source = ROOT / "env/scene_manager/layout_manager.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LayoutManager")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "get_instance_pose")
    namespace = {"np": np, "torch": torch, "deepcopy": deepcopy}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    position = torch.tensor([1.0, 2.0, 3.0], requires_grad=True).as_subclass(NoImplicitNumpyTensor)
    orientation = torch.tensor([0.5, 0.5, 0.5, 0.5], requires_grad=True).as_subclass(NoImplicitNumpyTensor)
    origin = torch.tensor([10.0, 20.0, 30.0]).as_subclass(NoImplicitNumpyTensor)
    native = SimpleNamespace(_get_object_transform=lambda: (position, orientation, "cpu"))
    layout = SimpleNamespace(
        instance_type_by_env=[{"native": kind}],
        get_scene_object=lambda env_idx, inst_name: native,
        get_instance_name=lambda env_idx, label: "native",
        scene_manager=SimpleNamespace(env_origins=[origin]),
    )
    layout.get_instance_pose = MethodType(namespace["get_instance_pose"], layout)
    pos, quat = layout.get_instance_pose(0, label="target", relative=relative)
    assert isinstance(pos, np.ndarray) and isinstance(quat, np.ndarray)
    np.testing.assert_array_equal(pos, [1, 2, 3] if relative else [11, 22, 33])
    np.testing.assert_array_equal(quat, [0.5] * 4)
    np.testing.assert_array_equal(np.concatenate([pos, quat])[3:], [0.5] * 4)
    pos[:] = 99
    quat[:] = 99
    np.testing.assert_array_equal(position.detach().cpu().numpy(), [1, 2, 3])
    np.testing.assert_array_equal(orientation.detach().cpu().numpy(), [0.5] * 4)


@pytest.mark.parametrize("kind", ["geometry", "garment"])
@pytest.mark.parametrize("env_id", [0, 1])
@pytest.mark.parametrize("relative", [True, False])
def test_non_rigid_state_preserves_relative_mode_and_independent_snapshots(kind, env_id, relative):
    source = ROOT / "env/scene_manager/layout_manager.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LayoutManager")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "get_instance_pose")
    namespace = {"np": np, "torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    origins = [np.array([10, 20, 30]), np.array([-40, 50, -60])]
    pose = torch.tensor([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]).as_subclass(NoImplicitNumpyTensor)
    calls = []

    def get_state(*, is_relative):
        calls.append(is_relative)
        state = pose.clone()
        if not is_relative:
            state[:3] += torch.from_numpy(origins[env_id])
        return {"root_pose": state}

    layout = SimpleNamespace(
        instance_type_by_env=[{"native": kind}, {"native": kind}],
        get_scene_object=lambda env_idx, inst_name: SimpleNamespace(get_state=get_state),
    )
    result = namespace["get_instance_pose"](layout, env_id, inst_name="native", relative=relative)
    np.testing.assert_array_equal(result[0], [1, 2, 3] if relative else np.array([1, 2, 3]) + origins[env_id])
    assert calls == [relative]
    result[0][:] = 99
    result[1][:] = 99
    np.testing.assert_array_equal(pose.detach().cpu().numpy(), [1, 2, 3, 1, 0, 0, 0])


@pytest.mark.parametrize("angle", [0, 90, 137])
def test_vectorized_geometry_matches_gf_with_nested_nonuniform_affines(angle):
    parent = Gf.Transform()
    parent.SetScale(Gf.Vec3d(2, 3, 4))
    parent.SetRotation(Gf.Rotation(Gf.Vec3d(0, 0, 1), angle))
    parent.SetTranslation(Gf.Vec3d(11, 22, 33))
    child = Gf.Transform()
    child.SetScale(Gf.Vec3d(0.5, 1.2, 1.7))
    child.SetRotation(Gf.Rotation(Gf.Vec3d(1, 1, 0).GetNormalized(), 31))
    child.SetTranslation(Gf.Vec3d(0.5, -0.2, 0.3))
    matrix = child.GetMatrix() * parent.GetMatrix()
    points = np.random.default_rng(7).normal(size=(128, 3)).astype(np.float32)
    expected = np.array([list(matrix.Transform(Gf.Vec3d(*map(float, point)))) for point in points], dtype=np.float32)
    actual = transform_points_world(points, matrix)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32


def test_vectorized_geometry_empty_and_invalid_shapes():
    assert transform_points_world(np.empty((0, 3)), Gf.Matrix4d(1)).shape == (0, 3)
    with pytest.raises(ValueError, match="points"):
        transform_points_world(np.zeros((3,)), Gf.Matrix4d(1))
    matrix = np.eye(4)
    matrix[0, 3] = 0.1
    with pytest.raises(ValueError, match="affine"):
        transform_points_world(np.zeros((2, 3)), matrix)
