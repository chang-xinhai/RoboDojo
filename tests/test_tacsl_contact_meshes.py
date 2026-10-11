"""Validate contact discovery against real USD rigid/collision APIs."""

from types import SimpleNamespace

import pytest

from env.tactile_manager import TactileManager


def test_every_mesh_of_articulated_and_rigid_objects_is_collected():
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    for path in ("/World/rigid", "/World/hinged/base", "/World/hinged/door"):
        body = UsdGeom.Xform.Define(stage, path)
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        for index in range(2):
            mesh = UsdGeom.Mesh.Define(stage, path + f"/collider_{index}")
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdGeom.Mesh.Define(stage, path + "/visual_only")
    cloth = UsdGeom.Mesh.Define(stage, "/World/cloth")
    UsdPhysics.CollisionAPI.Apply(cloth.GetPrim())
    manager = TactileManager(1, {"enabled": True}, None, "cuda:0")
    manager.stage = stage
    objects = SimpleNamespace(
        _rigid_and_dynamic_objects=[
            {"rigid": SimpleNamespace(prim_path="/World/rigid"), "cloth": SimpleNamespace(prim_path="/World/cloth")}
        ],
        _articulation_objects=[{"hinged": SimpleNamespace(prim_path="/World/hinged")}],
    )
    found = manager._discover_contact_bodies(objects)
    assert set(found) == {
        f"{body}/collider_{index}"
        for body in ("/World/rigid", "/World/hinged/base", "/World/hinged/door")
        for index in range(2)
    }
    assert len(found) == 6
    manager.scene_manager = objects
    manager.contact_paths_by_env = [found]
    manager.synchronize_contact_objects()
    stage.RemovePrim("/World/hinged/door/collider_1")
    with pytest.raises(RuntimeError, match="topology changed"):
        manager.synchronize_contact_objects()


def test_disabled_collision_mesh_is_excluded_without_losing_enabled_link_siblings():
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    base = UsdGeom.Xform.Define(stage, "/World/hinged/base")
    UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
    link = UsdGeom.Xform.Define(stage, "/World/hinged/base/door")
    UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
    for name, enabled in (("disabled", False), ("enabled", True), ("default", None)):
        mesh = UsdGeom.Mesh.Define(stage, str(link.GetPath()) + f"/colliders/{name}")
        mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
        mesh.CreateFaceVertexCountsAttr([3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
        collision = UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        if enabled is not None:
            collision.CreateCollisionEnabledAttr(enabled)
        else:
            assert not collision.GetCollisionEnabledAttr().HasAuthoredValueOpinion()
            assert collision.GetCollisionEnabledAttr().Get() is True
    manager = TactileManager(1, {"enabled": True}, None, "cuda:0")
    manager.stage = stage
    objects = SimpleNamespace(
        _rigid_and_dynamic_objects=[{}],
        _articulation_objects=[{"hinged": SimpleNamespace(prim_path="/World/hinged")}],
    )
    found = manager._discover_contact_bodies(objects)
    assert set(found) == {str(link.GetPath()) + f"/colliders/{name}" for name in ("enabled", "default")}
    assert not any(path.endswith("/disabled") for path in found)
    # Both selected siblings belong to the moving door, not its rigid base.
    for path in found:
        nearest = stage.GetPrimAtPath(path)
        while not nearest.HasAPI(UsdPhysics.RigidBodyAPI):
            nearest = nearest.GetParent()
        assert nearest == link.GetPrim()
    manager.scene_manager = objects
    manager.contact_paths_by_env = [found]
    manager.synchronize_contact_objects()
    collision = UsdPhysics.CollisionAPI(stage.GetPrimAtPath(found[0]))
    collision.CreateCollisionEnabledAttr(False)
    with pytest.raises(RuntimeError, match="topology changed"):
        manager.synchronize_contact_objects()
