"""Actual mesh queries retain physical convex cooking without a GPU launch."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics
import pytest
import torch
import trimesh

from env.tactile_manager import TactileManager
from env.tactile_manager.contact_query import create_contact_query_view, find_warp_contact_components
from utils.particle_readback_bridge import ReadbackTensorView


def collider():
    stage = Usd.Stage.CreateInMemory()
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.02)
    mesh = UsdGeom.Mesh.Define(stage, "/World/tube/collision")
    mesh.CreatePointsAttr(sphere.vertices.tolist())
    mesh.CreateFaceVertexCountsAttr([3] * len(sphere.faces))
    mesh.CreateFaceVertexIndicesAttr(sphere.faces.reshape(-1).tolist())
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexDecomposition")
    return stage, mesh.GetPrim()


def forbidden_native_sdf(*args):
    raise AssertionError("Explicit real-triangle query must not ask native physics to replace its convex collider")


def test_explicit_warp_queries_real_surface_and_does_not_modify_original_physical_collider():
    stage, mesh = collider()
    before = stage.GetRootLayer().ExportToString()
    native = SimpleNamespace(create_sdf_shape_view=forbidden_native_sdf)
    view, backend, metadata = create_contact_query_view(native, mesh, 2, "cpu", "warp_triangle_mesh")
    result = view.get_sdf_and_gradients(torch.tensor([[[0.021, 0.0, 0.0], [0.019, 0.0, 0.0]]])).numpy()
    assert backend == "warp_triangle_mesh_signed_distance"
    assert metadata["physical_collision_preserved"] is True
    assert metadata["physical_collision_approximation"] == "convexDecomposition"
    assert metadata["query_geometry"].startswith("original USD")
    assert result[0, 0, 3] == pytest.approx(0.001, abs=1e-7)
    assert result[0, 1, 3] < 0
    assert np.all(np.isfinite(result))
    assert stage.GetRootLayer().ExportToString() == before


def test_default_native_and_existing_host_readback_choices_remain_explicit():
    _, mesh = collider()
    UsdPhysics.MeshCollisionAPI(mesh).CreateApproximationAttr("sdf")
    calls = []
    actual_native_handle = object()
    native = SimpleNamespace(create_sdf_shape_view=lambda *args: calls.append(args) or actual_native_handle)
    view, backend, metadata = create_contact_query_view(native, mesh, 500, "cpu")
    assert view is actual_native_handle
    assert backend == "physx_native_sdf"
    assert calls == [(str(mesh.GetPath()), 500)]
    assert metadata["requested_query_backend"] == "auto"
    host = ReadbackTensorView(SimpleNamespace(create_sdf_shape_view=forbidden_native_sdf), "cpu")
    view, backend, _ = create_contact_query_view(host, mesh, 500, "cpu")
    assert backend == "warp_triangle_mesh_signed_distance"
    assert view.triangle_count > 0
    with pytest.raises(ValueError, match="do not implement native SDF"):
        create_contact_query_view(host, mesh, 500, "cpu", "physx_native_sdf")


def test_manager_explicit_warp_skips_all_sdf_schema_and_cooking_changes():
    manager = TactileManager(1, {"contact_query_backend": "warp_triangle_mesh"}, None, "cpu")
    manager.contact_paths = ["/World/tube0/collision", "/World/tube1/collision"]
    manager._prepare_sdf = forbidden_native_sdf
    manager._prepare_contact_collisions()
    legacy = TactileManager(1, {}, None, "cpu")
    calls = []
    legacy.contact_paths = manager.contact_paths
    legacy._prepare_sdf = calls.append
    legacy._prepare_contact_collisions()
    assert calls == []
    explicit = TactileManager(1, {"contact_query_backend": "physx_native_sdf"}, None, "cpu")
    explicit.contact_paths = manager.contact_paths
    explicit._prepare_sdf = calls.append
    explicit._prepare_contact_collisions()
    assert calls == manager.contact_paths


def test_actual_standard_tube_asset_convex_approximation_is_retained_by_real_warp_query():
    path = Path(__file__).resolve().parents[1] / "Assets/Object/RoboDojo/Rigid/test_tube/00002/object.usdz"
    if not path.exists():
        pytest.skip("Downloaded original standard-task tube unavailable")
    stage = Usd.Stage.Open(str(path))
    mesh = next(p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh) and p.HasAPI(UsdPhysics.CollisionAPI))
    before = stage.GetRootLayer().ExportToString()
    view, _, metadata = create_contact_query_view(
        SimpleNamespace(create_sdf_shape_view=forbidden_native_sdf), mesh, 500, "cpu", "warp_triangle_mesh"
    )
    assert view.triangle_count > 0
    assert metadata["physical_collision_approximation"] == "convexDecomposition"
    assert stage.GetRootLayer().ExportToString() == before


def test_unknown_query_backend_fails_instead_of_silently_modifying_physics():
    with pytest.raises(ValueError, match="contact_query_backend"):
        TactileManager(1, {"contact_query_backend": "fake"}, None, "cpu")


def test_explicit_mesh_lookup_uses_enabled_original_collider_and_nearest_rigid_body():
    stage = Usd.Stage.CreateInMemory()
    for path in ("/World/object", "/World/object/link"):
        UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, path).GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/World/object/link/collider")
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexDecomposition")
    before = stage.GetRootLayer().ExportToString()
    collider_prim, body = find_warp_contact_components(stage, str(mesh.GetPath()))
    assert collider_prim.GetPath() == mesh.GetPath()
    assert str(body.GetPath()) == "/World/object/link"
    assert stage.GetRootLayer().ExportToString() == before
    assert UsdPhysics.MeshCollisionAPI(collider_prim).GetApproximationAttr().Get() == "convexDecomposition"
    UsdPhysics.CollisionAPI(mesh.GetPrim()).CreateCollisionEnabledAttr(False)
    with pytest.raises(ValueError, match="enabled collision mesh.*found 0"):
        find_warp_contact_components(stage, str(mesh.GetPath()))


def test_explicit_lookup_rejects_ambiguous_missing_and_nonrigid_meshes():
    stage = Usd.Stage.CreateInMemory()
    for suffix in ("mesh0", "mesh1"):
        mesh = UsdGeom.Mesh.Define(stage, "/World/object/" + suffix)
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    with pytest.raises(ValueError, match="found 2"):
        find_warp_contact_components(stage, "/World/object")
    with pytest.raises(ValueError, match="rigid-body ancestor"):
        find_warp_contact_components(stage, "/World/object/mesh0")
    with pytest.raises(ValueError, match="No actual"):
        find_warp_contact_components(stage, "/World/missing")


def test_actual_convex_tube_reference_resolves_without_any_sdf_api_or_cooking_changes():
    asset = Path(__file__).resolve().parents[1] / "Assets/Object/RoboDojo/Rigid/test_tube/00002/object.usdz"
    if not asset.exists():
        pytest.skip("Downloaded original standard-task tube unavailable")
    stage = Usd.Stage.CreateInMemory()
    body = UsdGeom.Xform.Define(stage, "/World/envs/env_0/test_tube_2_1")
    body.GetPrim().GetReferences().AddReference(str(asset))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    before = stage.GetRootLayer().ExportToString()
    collider_prim, found_body = find_warp_contact_components(stage, str(body.GetPath()))
    assert found_body == body.GetPrim()
    assert collider_prim.IsA(UsdGeom.Mesh)
    assert UsdPhysics.MeshCollisionAPI(collider_prim).GetApproximationAttr().Get() == "convexDecomposition"
    assert stage.GetRootLayer().ExportToString() == before
