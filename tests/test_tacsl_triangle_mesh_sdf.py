"""Real Warp geometry queries at the TacSL particle-readback boundary."""

from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics
import pytest
import torch
import trimesh
import warp as wp

from env.tactile_manager.sdf_boundary import BodyFrameSdfView, shape_pose_in_body
from env.tactile_manager.triangle_mesh_sdf import WarpTriangleMeshSdfView


def sphere_view():
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.0055)
    return WarpTriangleMeshSdfView(sphere.vertices, sphere.faces, "cpu")


def test_real_triangle_query_has_signed_metric_distance_and_outward_gradient():
    view = sphere_view()
    points = torch.tensor([[[0.0065, 0, 0], [0.0045, 0, 0], [0, 0.0065, 0]]], dtype=torch.float32)
    actual = wp.to_torch(view.get_sdf_and_gradients(wp.from_torch(points)))
    assert actual[0, :, 3].tolist() == pytest.approx([0.001, -0.001, 0.001], abs=3e-5)
    assert torch.allclose(actual[0, :, :3], torch.tensor([[1.0, 0, 0], [1.0, 0, 0], [0, 1.0, 0]]), atol=0.1)
    assert view.query_backend == "warp_triangle_mesh_signed_distance"


def test_real_mesh_query_retains_body_frame_and_uniform_scale():
    view = sphere_view()
    rotation = torch.tensor([[0.0, 1.0, 0], [-1.0, 0, 0], [0, 0, 1.0]])
    translation = torch.tensor([0.025, 0.01, 0.02])
    wrapped = BodyFrameSdfView(view, rotation.tolist(), translation.tolist(), "cpu", geometry_scale=1.25)
    mesh_points = torch.tensor([[[0.0063, 0, 0], [0.0047, 0, 0]]])
    body_points = (mesh_points * 1.25) @ rotation + translation
    actual = wp.to_torch(wrapped.get_sdf_and_gradients(wp.from_torch(body_points)))
    assert actual[0, :, 3].tolist() == pytest.approx([0.001, -0.001], abs=4e-5)
    assert torch.allclose(actual[0, :, :3], torch.tensor([[0, 1.0, 0], [0, 1.0, 0]]), atol=0.1)


def test_actual_usd_mesh_points_and_left_handed_winding():
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.006)
    stage = Usd.Stage.CreateInMemory()
    body = UsdGeom.Xform.Define(stage, "/body")
    body.AddScaleOp().Set(Gf.Vec3d(1.25))
    mesh = UsdGeom.Mesh.Define(stage, "/body/mesh")
    mesh.AddTranslateOp().Set(Gf.Vec3d(0.02, 0, 0))
    mesh.CreatePointsAttr(sphere.vertices.tolist())
    mesh.CreateFaceVertexCountsAttr([3] * len(sphere.faces))
    mesh.CreateFaceVertexIndicesAttr(sphere.faces[:, ::-1].flatten().tolist())
    mesh.CreateOrientationAttr("leftHanded")
    view = WarpTriangleMeshSdfView.from_usd(mesh.GetPrim(), "cpu")
    rotation, translation, scale = shape_pose_in_body(mesh.GetPrim(), body.GetPrim())
    wrapped = BodyFrameSdfView(view, rotation, translation, "cpu", scale)
    actual = wp.to_torch(wrapped.get_sdf_and_gradients(wp.from_torch(torch.tensor([[[0.032, 0, 0]]]))))
    assert actual[0, 0, 3] < 0
    assert translation == pytest.approx((0.025, 0, 0))


def test_invalid_indices_fail_instead_of_invalid_geometry():
    with pytest.raises(ValueError, match="invalid vertex indices"):
        WarpTriangleMeshSdfView(np.zeros((3, 3)), [[0, 1, 3]], "cpu")


def test_wholly_degenerate_mesh_fails_instead_of_false_valid_surface():
    with pytest.raises(ValueError, match="no nondegenerate triangle surface"):
        WarpTriangleMeshSdfView(np.zeros((3, 3)), [[0, 1, 2]], "cpu")
    with pytest.raises(ValueError, match="no nondegenerate triangle surface"):
        WarpTriangleMeshSdfView([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[0, 1, 2]], "cpu")


def test_zero_area_faces_do_not_change_actual_solid_query():
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.006)
    faces = np.concatenate((sphere.faces, [[0, 0, 0]]), axis=0)
    view = WarpTriangleMeshSdfView(sphere.vertices, faces, "cpu")
    assert view.discarded_degenerate_triangles == 1
    assert view.triangle_count == len(sphere.faces)
    points = torch.tensor([[[0.007, 0, 0], [0.005, 0, 0]]])
    actual = wp.to_torch(view.get_sdf_and_gradients(wp.from_torch(points)))
    assert actual[0, :, 3].tolist() == pytest.approx([0.001, -0.001], abs=3e-5)
    assert torch.allclose(actual[0, :, :3].norm(dim=-1), torch.ones(2), atol=1e-6)


def test_actual_fluid_task_bottle_has_correct_surface_distance_sign():
    asset = Path(__file__).resolve().parents[1] / "Assets/Object/RoboDojo/Rigid/wuliangye/00000/object.usdz"
    if not asset.is_file():
        pytest.skip("RoboDojo task assets are not installed")
    stage = Usd.Stage.Open(str(asset))
    prim = next(prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI))
    view = WarpTriangleMeshSdfView.from_usd(prim, "cpu")
    mesh = UsdGeom.Mesh(prim)
    vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
    faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    triangles = vertices[faces[[100, 500, 1500]]]
    normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    centre = triangles.mean(axis=1)
    points = np.stack((centre + normal * 1e-5, centre - normal * 1e-5), axis=1).reshape(1, -1, 3)
    actual = wp.to_torch(view.get_sdf_and_gradients(wp.from_torch(torch.tensor(points))))
    assert actual[0, 0::2, 3].tolist() == pytest.approx([1e-5] * 3, abs=1e-8)
    assert actual[0, 1::2, 3].tolist() == pytest.approx([-1e-5] * 3, abs=1e-8)


def test_original_precision_cup_quads_initialize_without_changing_asset():
    asset = Path(__file__).resolve().parents[1] / "Assets/Object/RoboDojo/Rigid/cup/00006/object.usdz"
    if not asset.is_file():
        pytest.skip("RoboDojo precision cup asset is not installed")
    stage = Usd.Stage.Open(str(asset))
    prim = next(p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh) and p.HasAPI(UsdPhysics.CollisionAPI))
    mesh = UsdGeom.Mesh(prim)
    original_points = np.asarray(mesh.GetPointsAttr().Get()).copy()
    view = WarpTriangleMeshSdfView.from_usd(prim, "cpu")
    assert view.triangle_count > 0
    np.testing.assert_array_equal(view.points.numpy(), original_points)
    np.testing.assert_array_equal(np.asarray(mesh.GetPointsAttr().Get()), original_points)


def test_nonplanar_quad_retains_authored_vertices():
    stage = Usd.Stage.CreateInMemory()
    mesh = UsdGeom.Mesh.Define(stage, "/mesh")
    points = [[0, 0, 0], [1, 0, 0], [1, 1, 0.2], [0, 1, 0]]
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    view = WarpTriangleMeshSdfView.from_usd(mesh.GetPrim(), "cpu")
    np.testing.assert_array_equal(view.points.numpy(), np.asarray(points, dtype=np.float32))
    np.testing.assert_array_equal(view.indices.numpy().reshape(-1, 3), [[0, 1, 2], [0, 2, 3]])


def test_concave_quad_uses_interior_diagonal_and_rejects_crossing_edges():
    stage = Usd.Stage.CreateInMemory()
    mesh = UsdGeom.Mesh.Define(stage, "/mesh")
    mesh.CreatePointsAttr([[0, 0, 0], [1, 0.25, 0], [2, 0, 0], [1, 1, 0]])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    view = WarpTriangleMeshSdfView.from_usd(mesh.GetPrim(), "cpu")
    np.testing.assert_array_equal(view.indices.numpy().reshape(-1, 3), [[0, 1, 3], [1, 2, 3]])
    mesh.GetPointsAttr().Set([[0, 0, 0], [1, 1, 0], [0, 1, 0], [1, 0, 0]])
    with pytest.raises(ValueError, match="query quad"):
        WarpTriangleMeshSdfView.from_usd(mesh.GetPrim(), "cpu")
