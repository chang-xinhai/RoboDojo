"""Real USD geometry, deliberately stale USD poses and moving CPU native buffers.

These tests execute the actual public wrapper methods without loading Kit. They
verify state-source contracts, not a claim that the substitute buffers simulate
rigid dynamics; a separate GPU smoke must establish actual PhysX motion.
"""

import ast
import math
from pathlib import Path
import sys
from types import MethodType, ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics
import pytest
import torch

from utils.isaaclab_compat import as_torch, set_joint_target
from utils.live_geometry import get_live_world_transform

ROOT = Path(__file__).resolve().parents[1]
Q90 = [math.sqrt(0.5), 0, 0, math.sqrt(0.5)]


class NativeFactory:
    def __init__(self, pose, count=1):
        self.pose = torch.as_tensor([pose], dtype=torch.float32)
        self.count = count
        self.paths = []

    def create_rigid_body_view(self, path):
        self.paths.append(path)
        return SimpleNamespace(count=self.count, get_transforms=lambda: self.pose)


def bind_public_methods(owner, filename, classname, names):
    path = ROOT / "env/scene_manager/objects" / filename
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == classname)
    namespace = {
        "np": np,
        "torch": torch,
        "Gf": Gf,
        "Usd": Usd,
        "UsdGeom": UsdGeom,
        "get_live_world_transform": get_live_world_transform,
        "omni": SimpleNamespace(
            usd=SimpleNamespace(
                get_world_transform_matrix=lambda prim: UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                    Usd.TimeCode.Default()
                )
            )
        ),
    }
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
            setattr(owner, node.name, MethodType(namespace[node.name], owner))


@pytest.fixture
def scene():
    stage = Usd.Stage.CreateInMemory()
    env = UsdGeom.Xform.Define(stage, "/World/envs/env_0")
    env.AddTranslateOp().Set(Gf.Vec3d(10, 20, 30))
    body = UsdGeom.Xform.Define(stage, "/World/envs/env_0/body")
    body.AddTranslateOp().Set(Gf.Vec3d(91, 72, 63))
    body.AddScaleOp().Set(Gf.Vec3d(2, 3, 4))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/World/envs/env_0/body/mesh")
    mesh.AddTranslateOp().Set(Gf.Vec3d(0.5, 0, 0))
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)])
    factory = NativeFactory([11, 22, 33, 0, 0, Q90[3], Q90[0]])
    owner = SimpleNamespace(
        stage=stage,
        _prim_path=str(body.GetPath()),
        env_origin=np.array([10, 20, 30]),
        physics_sim_view=factory,
        _find_first_mesh_in_hierarchy=lambda _: str(mesh.GetPath()),
        _find_link_prim_by_name=lambda _: mesh.GetPrim(),
    )
    return owner, mesh, factory


@pytest.mark.parametrize("filename,classname", [("rigid.py", "RigidObject"), ("dynamic.py", "DynamicObject")])
def test_public_mesh_and_bbox_follow_native_motion_while_usd_remains_stale(scene, filename, classname):
    owner, mesh, factory = scene
    bind_public_methods(owner, filename, classname, {"sample_mesh_vertices", "get_bbox"})
    authored = UsdGeom.Xformable(mesh).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    points, local, position, orientation = owner.sample_mesh_vertices()
    # Independently known scale + Z quarter-turn + mesh offset + live translation.
    expected = [[11, 23, 33], [11, 25, 33], [8, 23, 33], [11, 23, 37]]
    np.testing.assert_allclose(points, expected, atol=1e-5)
    np.testing.assert_allclose(position, [11, 23, 33], atol=1e-5)
    np.testing.assert_allclose(orientation, Q90, atol=1e-6)
    np.testing.assert_array_equal(local, mesh.GetPointsAttr().Get())
    bbox_position, bbox_orientation, bounds = owner.get_bbox(is_relative=True)
    np.testing.assert_allclose(bbox_position, [1, 3, 3], atol=1e-5)
    np.testing.assert_allclose(bbox_orientation, Q90, atol=1e-6)
    np.testing.assert_array_equal(bounds, [0, 0, 0, 1, 1, 1])
    factory.pose[:, :3] += torch.tensor([2, -1, 0.5])
    moved = owner.sample_mesh_vertices()[0]
    np.testing.assert_allclose(moved - points, np.tile([2, -1, 0.5], (4, 1)), atol=1e-5)
    assert UsdGeom.Xformable(mesh).ComputeLocalToWorldTransform(Usd.TimeCode.Default()) == authored
    assert factory.paths == [owner._prim_path]


def test_articulated_functional_frame_follows_native_link_and_reports_unit_wxyz(scene):
    owner, mesh, factory = scene
    bind_public_methods(owner, "articulation.py", "ArticulationObject", {"get_link_pose"})
    pose = owner.get_link_pose("functional_mesh")
    np.testing.assert_allclose(pose[:3], [11, 23, 33], atol=1e-5)
    np.testing.assert_allclose(pose[3:], Q90, atol=1e-6)
    factory.pose[:, :3] += torch.tensor([1, 2, 3])
    np.testing.assert_allclose(owner.get_link_pose("functional_mesh")[:3], [12, 25, 36], atol=1e-5)
    assert factory.paths == [owner._prim_path]


def test_recreated_native_scene_does_not_reuse_destroyed_body_view(scene):
    owner, mesh, first_factory = scene
    first = get_live_world_transform(owner, mesh.GetPrim())
    replacement = NativeFactory([21, 32, 43, 0, 0, Q90[3], Q90[0]])
    owner.physics_sim_view = replacement
    second = get_live_world_transform(owner, mesh.GetPrim())
    np.testing.assert_allclose(second.ExtractTranslation(), [21, 33, 43], atol=1e-5)
    assert second != first
    assert first_factory.paths == replacement.paths == [owner._prim_path]


def test_nonrigid_dynamic_asset_explicitly_reads_fabric_pose(scene):
    owner, mesh, _factory = scene
    owner.stage.GetPrimAtPath(owner._prim_path).RemoveAPI(UsdPhysics.RigidBodyAPI)
    live = Mock(return_value=(torch.tensor([[11, 22, 33]]), torch.tensor([Q90])))
    owner._xform_prim_view = SimpleNamespace(get_world_poses=live)
    transform = get_live_world_transform(owner, mesh.GetPrim())
    live.assert_called_once_with(usd=False)
    np.testing.assert_allclose(transform.ExtractTranslation(), [11, 23, 33], atol=1e-5)


def set_global_native_factory(monkeypatch, factory):
    manager = ModuleType("isaacsim.core.simulation_manager")
    manager.SimulationManager = SimpleNamespace(get_physics_sim_view=lambda: factory)
    monkeypatch.setitem(sys.modules, manager.__name__, manager)


@pytest.mark.parametrize("body_type", ["rigid", "xform"])
@pytest.mark.parametrize("owner_state", ["missing", "none"])
def test_explicit_prephysics_geometry_queries_use_authoritative_usd(scene, monkeypatch, body_type, owner_state):
    owner, mesh, factory = scene
    if owner_state == "missing":
        del owner.physics_sim_view
    else:
        owner.physics_sim_view = None
    if body_type == "xform":
        owner.stage.GetPrimAtPath(owner._prim_path).RemoveAPI(UsdPhysics.RigidBodyAPI)
    set_global_native_factory(monkeypatch, None)
    authored = UsdGeom.Xformable(mesh).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    assert get_live_world_transform(owner, mesh.GetPrim()) == authored
    assert factory.paths == []


@pytest.mark.parametrize("count", [0, 1])
def test_active_global_physics_never_uses_usd_when_owner_is_not_yet_initialized(scene, monkeypatch, count):
    owner, mesh, factory = scene
    del owner.physics_sim_view
    factory.count = count
    set_global_native_factory(monkeypatch, factory)
    if count == 0:
        with pytest.raises(RuntimeError, match="exactly one native"):
            get_live_world_transform(owner, mesh.GetPrim())
    else:
        transform = get_live_world_transform(owner, mesh.GetPrim())
        np.testing.assert_allclose(transform.ExtractTranslation(), [11, 23, 33], atol=1e-5)
    assert factory.paths == [owner._prim_path]


@pytest.mark.parametrize("failure", ["absent", "nonfinite"])
def test_invalid_native_pose_never_falls_back_to_stale_usd(scene, failure):
    owner, mesh, factory = scene
    if failure == "absent":
        factory.count = 0
    else:
        factory.pose[0, 0] = float("nan")
    with pytest.raises(RuntimeError):
        get_live_world_transform(owner, mesh.GetPrim())


def test_robot_episode_reset_restores_actual_joint_state_and_removes_previous_momentum():
    path = ROOT / "env/robot_manager/robot_manager.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RobotManager")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "set_robot_init_pose")
    namespace = {"as_torch": as_torch, "set_joint_target": set_joint_target}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)

    class Arm:
        def __init__(self, offset):
            self.data = SimpleNamespace(
                default_joint_pos=torch.tensor([[0.1, 0.2], [0.3, 0.4]]) + offset,
                default_joint_vel=torch.zeros((2, 2)),
            )
            self.position = torch.full((2, 2), 9.0)
            self.velocity = torch.full((2, 2), 2.0)
            self.position_target = self.velocity_target = None
            self.events = []
            self.actuators = SimpleNamespace(
                target_command=SimpleNamespace(
                    set_position_index=self.store_position_target,
                    set_velocity_index=self.store_velocity_target,
                )
            )

        def write_joint_state_to_sim_index(self, *, position, velocity):
            self.position, self.velocity = position.clone(), velocity.clone()
            self.events.append("actual_state_restored")

        def reset(self):
            assert self.events == ["actual_state_restored"]
            self.events.append("episode_buffers_reset")

        def store_position_target(self, *, value, joint_ids=None, env_ids=None):
            self.position_target = value.clone()

        def store_velocity_target(self, *, value, joint_ids=None, env_ids=None):
            self.velocity_target = value.clone()

    left, right = Arm(0), Arm(1)
    owner = SimpleNamespace(
        robot_key=[left, right],
        robot_list=[SimpleNamespace(robot_type="arm"), SimpleNamespace(robot_type="arm")],
    )
    namespace[method.name](owner)
    for arm in (left, right):
        assert torch.equal(arm.position, arm.data.default_joint_pos)
        assert torch.count_nonzero(arm.velocity) == 0
        assert torch.equal(arm.position_target, arm.data.default_joint_pos)
        assert torch.count_nonzero(arm.velocity_target) == 0
        assert arm.position.data_ptr() != arm.data.default_joint_pos.data_ptr()
