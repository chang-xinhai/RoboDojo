"""Actual static geometry snapshots preserve host and coordinate contracts."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from pxr import Gf, Usd, UsdGeom
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


class ExplicitHostTensor(torch.Tensor):
    host_requests = []

    def __array__(self, *args, **kwargs):
        raise AssertionError("CUDA-style origin requires explicit host conversion")

    def to(self, *args, **kwargs):
        self.host_requests.append(kwargs.get("device"))
        return super().to(*args, **kwargs)


def geometry_snapshot(origin, local_position=(1, 2, 3)):
    source = ROOT / "env/scene_manager/objects/geometry.py"
    cls = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.ClassDef))
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "get_state")
    stage = Usd.Stage.CreateInMemory()
    parent = UsdGeom.Xform.Define(stage, "/World/env")
    parent.AddTranslateOp().Set(Gf.Vec3d(*map(float, origin)))
    child = UsdGeom.Xform.Define(stage, "/World/env/geometry")
    transform = Gf.Transform()
    transform.SetScale(Gf.Vec3d(2, 3, 4))
    transform.SetRotation(Gf.Rotation(Gf.Vec3d(0, 0, 1), 90))
    transform.SetTranslation(Gf.Vec3d(*map(float, local_position)))
    child.AddTransformOp().Set(transform.GetMatrix())
    namespace = {
        "np": np,
        "torch": torch,
        "Gf": Gf,
        "omni": SimpleNamespace(
            usd=SimpleNamespace(
                get_world_transform_matrix=lambda prim: UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                    Usd.TimeCode.Default()
                )
            )
        ),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    owner = SimpleNamespace(prim=child.GetPrim(), _prim_path=str(child.GetPath()), stage=stage)
    return owner, namespace["get_state"], namespace


@pytest.mark.parametrize("origin", [(10, 20, 30), (-40, 50, -60)])
@pytest.mark.parametrize("kind", ["torch", "numpy", "list"])
@pytest.mark.parametrize("relative", [False, True])
def test_actual_geometry_state_host_snapshot_origin_and_scaled_rotation(origin, kind, relative):
    owner, get_state, _ = geometry_snapshot(origin)
    if kind == "torch":
        owner.env_origin = torch.tensor(origin, dtype=torch.float64, requires_grad=True).as_subclass(ExplicitHostTensor)
        ExplicitHostTensor.host_requests.clear()
    elif kind == "numpy":
        owner.env_origin = np.array(origin, dtype=np.float64)
    else:
        owner.env_origin = list(origin)
    pose = get_state(owner, is_relative=relative)["root_pose"]
    assert pose.device.type == "cpu" and pose.dtype == torch.float32 and not pose.requires_grad
    assert tuple(pose.shape) == (7,)
    np.testing.assert_array_equal(pose[:3].numpy(), [1, 2, 3] if relative else np.array(origin) + [1, 2, 3])
    np.testing.assert_allclose(pose[3:].numpy(), [np.sqrt(0.5), 0, 0, np.sqrt(0.5)], atol=1e-7)
    assert float(torch.linalg.vector_norm(pose[3:])) == pytest.approx(1, abs=1e-7)
    if kind == "torch" and relative:
        assert ExplicitHostTensor.host_requests == ["cpu"]
    pose[:] = 99
    fresh = get_state(owner, is_relative=relative)["root_pose"]
    assert float(fresh[0]) == (1 if relative else origin[0] + 1)
    if kind == "torch":
        np.testing.assert_array_equal(owner.env_origin.detach().cpu().numpy(), origin)
    else:
        np.testing.assert_array_equal(owner.env_origin, origin)


@pytest.mark.parametrize("origin", [0, [1, 2], [1, 2, 3, 4], [1, np.nan, 3], torch.tensor([1, float("inf"), 3])])
def test_actual_geometry_rejects_invalid_relative_origins(origin):
    owner, get_state, _ = geometry_snapshot((10, 20, 30))
    owner.env_origin = origin
    with pytest.raises(ValueError, match="three finite coordinates"):
        get_state(owner, is_relative=True)


def test_actual_geometry_world_read_failure_is_not_fabricated_zero_pose():
    owner, get_state, namespace = geometry_snapshot((10, 20, 30))
    owner.env_origin = [10, 20, 30]

    def broken_pose(prim):
        raise RuntimeError("invalid static prim")

    namespace["omni"].usd.get_world_transform_matrix = broken_pose
    with pytest.raises(RuntimeError, match="Cannot read.*World/env/geometry") as error:
        get_state(owner)
    assert str(error.value.__cause__) == "invalid static prim"


@pytest.mark.parametrize("translation", [(float("nan"), 2, 3), (1, float("inf"), 3)])
def test_actual_geometry_rejects_nonfinite_static_world_pose(translation):
    owner, get_state, _ = geometry_snapshot((10, 20, 30), translation)
    owner.env_origin = [10, 20, 30]
    with pytest.raises(ValueError, match="Non-finite static geometry pose"):
        get_state(owner)
