"""Validate contact discovery against real USD rigid/collision APIs."""

import ast
from pathlib import Path
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


def test_calibration_fixture_collection_preserves_task_objects_and_environment_isolation():
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    task = []
    fixtures = []
    for index in range(2):
        prefix = f"/World/envs/env_{index}"
        for suffix in ("task_object", "calibration_probe"):
            body = UsdGeom.Xform.Define(stage, prefix + "/" + suffix)
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            collision = UsdGeom.Mesh.Define(stage, str(body.GetPath()) + "/collision")
            UsdPhysics.CollisionAPI.Apply(collision.GetPrim())
        # Same dictionary label must not let calibration erase a task object.
        task.append({"object": SimpleNamespace(prim_path=prefix + "/task_object")})
        fixtures.append({"object": SimpleNamespace(prim_path=prefix + "/calibration_probe")})
    manager = TactileManager(2, {"enabled": True}, None, "cuda:0")
    manager.stage = stage
    objects = SimpleNamespace(
        _rigid_and_dynamic_objects=task,
        _articulation_objects=[{}, {}],
        _tactile_calibration_objects=fixtures,
    )
    for index in range(2):
        assert set(manager._discover_contact_bodies(objects, index)) == {
            f"/World/envs/env_{index}/{name}/collision" for name in ("task_object", "calibration_probe")
        }
    assert all(len(group) == 1 for group in task)


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


def test_native_tube_labels_exclude_other_objects_and_forbid_added_probes():
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    groups = []
    records_by_env = []
    mapping_by_env = []
    for env_id in range(2):
        objects = {}
        mapping = {f"tube{i}": f"test_tube_2_{i + 1 if env_id == 0 else 3 - i}" for i in range(3)}
        mapping.update(slot="test_tube_slot_0_1", robot_jaw="robot_link7")
        records_by_env.append([{"label": label, "inst_name": name} for label, name in mapping.items()])
        mapping_by_env.append(mapping)
        for instance in mapping.values():
            path = f"/World/envs/env_{env_id}/{instance}"
            body = UsdGeom.Xform.Define(stage, path)
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            mesh = UsdGeom.Mesh.Define(stage, path + "/collision")
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            objects[instance] = SimpleNamespace(prim_path=path)
        groups.append(objects)
    # Execute the actual production lookup method without importing Kit APIs.
    source = Path(__file__).resolve().parents[1] / "env/scene_manager/layout_manager.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LayoutManager")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "get_instance_name")
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    layout = SimpleNamespace(
        object_records_by_type={
            kind: SimpleNamespace(layout_records_by_env=records_by_env if kind == "Rigid" else [[], []])
            for kind in ("Rigid", "Dynamic", "Geometry", "Articulation", "Garment", "Fluid")
        }
    )
    layout.get_instance_name = namespace["get_instance_name"].__get__(layout)
    labels = ["tube0", "tube1", "tube2"]
    manager = TactileManager(
        2, {"enabled": True, "contact_object_labels": labels, "contact_conditioned_observations": True}, None, "cuda:0"
    )
    manager.stage = stage
    scene = SimpleNamespace(_rigid_and_dynamic_objects=groups, _articulation_objects=[{}, {}], layout_manager=layout)
    for env_id in range(2):
        expected = [f"/World/envs/env_{env_id}/{mapping_by_env[env_id][label]}/collision" for label in labels]
        assert manager._discover_contact_bodies(scene, env_id) == expected
    scene._tactile_calibration_objects = [{"nut": groups[0]["test_tube_slot_0_1"]}, {}]
    with pytest.raises(ValueError, match="forbid"):
        manager._discover_contact_bodies(scene)
    scene._tactile_calibration_objects = [{}, {}]
    del groups[1][mapping_by_env[1]["tube2"]]
    with pytest.raises(ValueError, match="Missing native.*tube2"):
        manager._discover_contact_bodies(scene, 1)
    records_by_env[1][2]["label"] = "unregistered_label"
    with pytest.raises(ValueError, match="Missing native.*tube2.*instance=None"):
        manager._discover_contact_bodies(scene, 1)
    UsdPhysics.CollisionAPI(
        stage.GetPrimAtPath("/World/envs/env_0/test_tube_2_1/collision")
    ).CreateCollisionEnabledAttr(False)
    with pytest.raises(ValueError, match="tube0.*no actual rigid mesh"):
        manager._discover_contact_bodies(scene, 0)


@pytest.mark.parametrize(
    "config",
    [
        {"contact_conditioned_observations": True},
        {"contact_object_labels": "tube0"},
        {"contact_object_labels": ["tube0", "tube0"]},
        {"contact_object_labels": ["tube0"], "contact_object_paths": ["/World/probe"]},
    ],
)
def test_native_contact_configuration_rejects_ambiguous_or_unspecified_scope(config):
    with pytest.raises(ValueError):
        TactileManager(1, config, None, "cuda:0")


def test_actual_collector_omegaconf_native_label_list_is_accepted():
    from omegaconf import OmegaConf

    config = OmegaConf.create(
        {
            "enabled": True,
            "contact_object_labels": ["tube0", "tube1", "tube2"],
            "contact_conditioned_observations": True,
        }
    )
    manager = TactileManager(1, config, None, "cuda:0")
    assert list(manager.config.contact_object_labels) == ["tube0", "tube1", "tube2"]


def test_mechanical_default_and_explicit_task_mount_keep_native_jaw_frame_contract():
    from omegaconf import OmegaConf
    from pxr import Gf, Usd, UsdGeom

    root = Path(__file__).resolve().parents[1]
    asset = root / "Assets/Robots/x5/ARX.usd"
    if not asset.exists():
        pytest.skip("Downloaded actual X5 asset unavailable")
    stage = Usd.Stage.Open(str(asset))
    cache = UsdGeom.XformCache()
    link6 = stage.GetPrimAtPath("/X5A/link6")
    legacy = TactileManager(1, {}, None, "cpu")
    configured = TactileManager(1, OmegaConf.create({"mount_position_link": [0.05843, 0.024, -0.06776]}), None, "cpu")
    # Native asset taxel geometry is centred at z=.06776 in the gel body.
    # Two opposing jaw rotations must keep the contact point at the same X.
    for link_name, sign in (("link7", -1), ("link8", 1)):
        link = stage.GetPrimAtPath("/X5A/" + link_name)
        link_in_6 = cache.GetLocalToWorldTransform(link) * cache.GetLocalToWorldTransform(link6).GetInverse()
        for manager, expected_x in ((legacy, 0.150070002), (configured, 0.145)):
            rotation = Gf.Rotation(Gf.Vec3d(0, 0, 1), 180 if sign > 0 else 0)
            mount = Gf.Matrix4d(1).SetRotate(rotation)
            mount.SetTranslateOnly(Gf.Vec3d(manager._mount_translation(sign)))
            center = (mount * link_in_6).Transform(Gf.Vec3d(0, 0, 0.06776))
            assert center[0] == pytest.approx(expected_x, abs=1e-8)
        assert legacy._mount_translation(sign) == (0.063500002, sign * 0.024, -0.068956808)
        assert configured._mount_translation(sign) == (0.05843, sign * 0.024, -0.06776)


@pytest.mark.parametrize("mount", [[0.035, -0.024, -0.06776], [float("nan"), 0.024, -0.06776], [0.035, 0.024], "bad"])
def test_invalid_physical_mount_configuration_fails_early(mount):
    with pytest.raises(ValueError, match="mount_position_link"):
        TactileManager(1, {"mount_position_link": mount}, None, "cpu")
