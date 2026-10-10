"""Independent frame/reset checks using actual USD math without starting Kit."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def actual_method(name):
    Gf = pytest.importorskip("pxr.Gf")
    torch = pytest.importorskip("torch")
    path = Path(__file__).resolve().parents[1] / "env/scene_manager/objects/garment.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GarmentObject")
    function = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    scope = {"Gf": Gf, "np": np, "torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), scope)
    return scope[name], Gf, torch


def garment_fixture():
    method, Gf, torch = actual_method("_world_reset_points")
    child = Gf.Matrix4d(1).SetTranslate(Gf.Vec3d(0.5, 0, 0))
    sample = SimpleNamespace(
        init_pos=np.array([0.3, -0.2, 0.1]),
        env_origin=np.array([10.0, 20.0, 30.0]),
        init_ori=np.array([np.sqrt(0.5), 0, 0, np.sqrt(0.5)]),
        init_scale=np.array([2.0, 3.0, 4.0]),
        _mesh_to_root=child,
        _rest_points=np.eye(3, dtype=np.float32),
    )
    sample._world_reset_points = lambda: method(sample)
    return sample, torch


def test_reset_world_points_apply_child_scale_rotation_and_origin_once():
    sample, _ = garment_fixture()
    before = sample._rest_points.copy()
    # Child translation precedes root nonuniform scale and a +90 degree Z turn.
    # The world root is at (10.3, 19.8, 30.1), independently of USD parent pose.
    expected = [[10.3, 22.8, 30.1], [7.3, 20.8, 30.1], [10.3, 20.8, 34.1]]
    np.testing.assert_allclose(sample._world_reset_points(), expected, atol=2e-6)
    np.testing.assert_array_equal(sample._rest_points, before)


def test_live_reset_writes_world_nodes_and_zero_velocity_without_usd_teleport():
    sample, torch = garment_fixture()
    method, _, _ = actual_method("reset")
    calls = []
    sample._surface_view = SimpleNamespace(
        get_simulation_nodal_positions=lambda: torch.full((1, 3, 3), 999.0),
        set_simulation_nodal_positions=lambda values, ids: calls.append(("positions", values.clone(), ids.clone())),
        set_simulation_nodal_velocities=lambda values, ids: calls.append(("velocities", values.clone(), ids.clone())),
    )
    sample.set_world_pose = lambda *_args, **_kwargs: pytest.fail("Live reset must preserve the cooked USD body")
    method(sample)
    assert [name for name, _, _ in calls] == ["positions", "velocities"]
    np.testing.assert_allclose(calls[0][1][0].numpy(), sample._world_reset_points(), atol=2e-6)
    assert calls[0][1].dtype == torch.float32
    assert calls[0][2].dtype == torch.int32
    assert calls[0][2].tolist() == [0]
    assert torch.count_nonzero(calls[1][1]) == 0
