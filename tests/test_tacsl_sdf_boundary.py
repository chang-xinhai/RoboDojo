"""Check real Warp buffers and USD transforms at the native SDF boundary."""

import math

from pxr import Gf, Usd, UsdGeom
import pytest
import torch
import warp as wp

from env.tactile_manager.sdf_boundary import BodyFrameSdfView, shape_pose_in_body


@pytest.mark.parametrize("geometry_scale", [1.0, 1.25, 6.0])
def test_body_points_and_native_gradients_roundtrip_rigid_shape_pose(geometry_scale):
    wp.init()
    rotation = torch.tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    translation = torch.tensor([0.02, 0.03, 0.04])
    shape_points = torch.tensor([[[0.003, 0.004, 0.0], [0.005, 0.0, 0.0]]])
    body_points = (shape_points * geometry_scale) @ rotation + translation

    class NativeSphereQuery:
        count = 1

        def get_sdf_and_gradients(self, query):
            self.points = wp.to_torch(query).clone()
            distance = self.points.norm(dim=-1, keepdim=True)
            self.result = torch.cat((self.points / distance, distance - 0.006), dim=-1)
            return wp.from_torch(self.result)

    native = NativeSphereQuery()
    view = BodyFrameSdfView(native, rotation.tolist(), translation.tolist(), "cpu", geometry_scale=geometry_scale)
    actual = wp.to_torch(view.get_sdf_and_gradients(wp.from_torch(body_points)))
    assert torch.allclose(native.points, shape_points, atol=1e-8)
    assert torch.allclose(actual[..., :3], (shape_points / 0.005) @ rotation, atol=1e-6)
    assert torch.allclose(actual[..., 3], torch.full((1, 2), -0.001 * geometry_scale), atol=1e-8)
    assert torch.allclose(native.result[..., :3], shape_points / 0.005, atol=1e-6)
    assert view.count == 1


def make_shape_stage(mesh_scale):
    stage = Usd.Stage.CreateInMemory()
    body = UsdGeom.Xform.Define(stage, "/body")
    body.AddTranslateOp().Set(Gf.Vec3d(1, 2, 3))
    body.AddRotateZOp().Set(90)
    body.AddScaleOp().Set(Gf.Vec3d(2))
    mesh = UsdGeom.Mesh.Define(stage, "/body/mesh")
    mesh.AddTranslateOp().Set(Gf.Vec3d(0.02, 0, 0))
    mesh.AddRotateZOp().Set(45)
    mesh.AddScaleOp().Set(Gf.Vec3d(*mesh_scale))
    return stage, mesh.GetPrim(), body.GetPrim()


def test_uniform_world_scale_scales_shape_offset_without_scaling_rigid_axes():
    stage, mesh, body = make_shape_stage((3, 3, 3))
    rotation, translation, scale = shape_pose_in_body(mesh, body)
    c = math.sqrt(0.5)
    assert torch.allclose(torch.tensor(rotation), torch.tensor([[c, c, 0], [-c, c, 0], [0, 0, 1]]), atol=1e-6)
    assert translation == pytest.approx((0.04, 0, 0), abs=1e-6)
    assert scale == pytest.approx(6)
    assert stage.GetPrimAtPath("/body/mesh").IsValid()


def test_nonuniform_mesh_scale_fails_explicitly():
    stage, mesh, body = make_shape_stage((2, 3, 2))
    with pytest.raises(ValueError, match="positive uniform scale"):
        shape_pose_in_body(mesh, body)
    assert stage.GetPrimAtPath("/body/mesh").IsValid()
