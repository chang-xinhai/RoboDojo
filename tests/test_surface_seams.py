"""Real topology and semantic-ID checks for welded FEM garment seams."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, Vt
import pytest
import torch

from utils.surface_deformable import weld_exact_surface_seams


def test_exact_seams_keep_source_order_and_join_panels():
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    compact, triangles, mapping, representatives = weld_exact_surface_seams(points, [0, 1, 2, 3, 4, 5])
    np.testing.assert_array_equal(mapping, [0, 1, 2, 1, 3, 2])
    np.testing.assert_array_equal(representatives, [0, 1, 2, 4])
    np.testing.assert_array_equal(compact[mapping], points)
    np.testing.assert_array_equal(triangles, [0, 1, 2, 1, 3, 2])


def test_welding_does_not_merge_nearby_points():
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 1e-7], [1, 1, 0]])
    compact, _, mapping, _ = weld_exact_surface_seams(points, [0, 1, 2, 3, 4, 2])
    assert len(compact) == 5
    np.testing.assert_array_equal(mapping, np.arange(5))


@pytest.mark.parametrize("triangles", [[0, 1, 3], [0, 1, 2, 0, 3, 2]])
def test_welding_rejects_collapsed_or_duplicate_triangles(triangles):
    with pytest.raises(ValueError, match="degenerate|duplicate"):
        weld_exact_surface_seams([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0]], triangles)


@pytest.mark.parametrize("asset_id,duplicates", [(1, 0), (4, 32), (6, 0), (7, 40), (9, 40), (11, 0)])
def test_actual_bundled_mesh_seams_and_semantic_vertices(asset_id, duplicates):
    path = Path(__file__).resolve().parents[1] / f"Assets/Object/RoboDojo/Garment/Top_Long/{asset_id:05d}/object.usdz"
    if not path.exists():
        pytest.skip("Large bundled garment assets have not been downloaded")
    stage = Usd.Stage.Open(str(path))
    mesh = next(UsdGeom.Mesh(p) for p in stage.Traverse() if p.IsA(UsdGeom.Mesh))
    points = np.asarray(mesh.GetPointsAttr().Get())
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
    compact, triangles, mapping, _ = weld_exact_surface_seams(points, indices)
    assert len(points) - len(compact) == duplicates
    np.testing.assert_array_equal(compact[mapping], points)
    np.testing.assert_array_equal(compact[triangles], points[indices])


def test_actual_authoring_preserves_corner_uv_normals_and_records_vertex_map():
    tree = ast.parse((Path(__file__).resolve().parents[1] / "env/scene_manager/objects/garment.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GarmentObject")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_weld_known_seams")
    namespace = dict(np=np, UsdGeom=UsdGeom, Vt=Vt, Sdf=Sdf, weld_exact_surface_seams=weld_exact_surface_seams)
    exec(compile(ast.Module(body=[method], type_ignores=[]), "garment.py", "exec"), namespace)
    stage = Usd.Stage.CreateInMemory()
    mesh = UsdGeom.Mesh.Define(stage, "/Cloth")
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float32)
    mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
    mesh.GetFaceVertexCountsAttr().Set([3, 3])
    mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3, 4, 5])
    uv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying")
    uv.Set([Gf.Vec2f(i / 6, 0) for i in range(6)])
    mesh.SetNormalsInterpolation("faceVarying")
    mesh.GetNormalsAttr().Set([Gf.Vec3f(0, 0, 1)] * 6)
    old_uv, old_normals = uv.Get(), mesh.GetNormalsAttr().Get()
    garment = SimpleNamespace(_mesh=mesh, _rest_points=points, original_vertex_count=6, physics_migration={})
    namespace[method.name](garment)
    assert uv.Get() == old_uv
    assert mesh.GetNormalsAttr().Get() == old_normals
    np.testing.assert_array_equal(garment._rest_points[garment._original_to_simulation], points)
    assert list(mesh.GetPrim().GetAttribute("robodojo:originalVertexToSimulationNode").Get()) == [0, 1, 2, 1, 3, 2]
    assert garment.physics_migration["exact_seam_welding"]["welded_vertices"] == 2


def test_rewards_read_original_vertex_ids_from_live_welded_nodes():
    tree = ast.parse((Path(__file__).resolve().parents[1] / "env/scene_manager/objects/garment.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GarmentObject")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "sample_mesh_vertices")
    namespace = dict(np=np, UsdGeom=UsdGeom, Gf=Gf, Usd=Usd)
    exec(compile(ast.Module(body=[method], type_ignores=[]), "garment.py", "exec"), namespace)
    stage = Usd.Stage.CreateInMemory()
    mesh = UsdGeom.Mesh.Define(stage, "/Cloth")
    mapping = np.array([0, 1, 2, 1, 3, 2])
    # Values represent live state after deformation, rather than the rest shape.
    live_nodes = torch.tensor([[[0.2, 0.1, 0.4], [1.0, 0.1, 0.5], [0.3, 0.8, 0.1], [0.9, 0.7, 0.6]]])
    garment = SimpleNamespace(
        _mesh=mesh,
        _original_to_simulation=mapping,
        _surface_view=SimpleNamespace(get_simulation_nodal_positions=lambda: live_nodes),
        get_world_pose=lambda: (np.zeros(3), np.array([1, 0, 0, 0])),
    )
    world, local, _, _ = namespace[method.name](garment)
    assert len(world) == 6
    np.testing.assert_array_equal(world, live_nodes[0].numpy()[mapping])
    np.testing.assert_array_equal(local, world)
    np.testing.assert_array_equal(world[1], world[3])
    np.testing.assert_array_equal(world[2], world[5])
