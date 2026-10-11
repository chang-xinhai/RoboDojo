"""CPU geometry/trajectory contracts; these do not certify native contact."""

import ast
from dataclasses import replace
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics
import pytest
from scipy.spatial.transform import Rotation
import torch
import trimesh

from utils.contact_demo import (
    PADS,
    PHASES,
    ContactDemoConfig,
    ContactFixtures,
    ContactMotionDemo,
    annulus_mesh,
    close_contact_physics,
    configure_contact_physics,
    contact_demo_task_class,
    load_contact_geometry,
    phase_command,
    pose_at_pad,
)


@pytest.mark.parametrize(
    "values",
    [
        {"indentation_m": 0.0009},
        {"indentation_m": math.nan},
        {"slide_m": 0.004},
        {"twist_deg": 31},
        {"phase_duration_s": 0},
        {"asset_scale": 1.1},
        {"annulus_inner_radius_m": 0.01},
        {"annulus_sections": 3},
    ],
)
def test_invalid_physical_commands_rejected(values):
    with pytest.raises(ValueError):
        ContactDemoConfig(**values)


def test_contact_path_continuity_and_physical_bounds():
    config = ContactDemoConfig()
    samples = [phase_command(config, time) for time in np.linspace(0, config.duration_s, 10001)]
    assert {sample["phase"] for sample in samples} == set(PHASES)
    values = np.asarray(
        [[sample[field] for field in ("normal_separation_m", "tangent_offset_m", "twist_rad")] for sample in samples]
    )
    assert values[:, 0].min() == pytest.approx(-config.indentation_m)
    assert values[:, 0].max() == pytest.approx(config.clearance_m)
    assert np.max(np.abs(values[:, 1])) <= config.slide_m
    assert np.max(np.abs(values[:, 2])) <= math.radians(config.twist_deg)
    # Every phase joins with zero endpoint velocity, including unload.
    for index in range(1, len(PHASES)):
        boundary = index * config.phase_duration_s
        left, right = phase_command(config, boundary - 1e-8), phase_command(config, boundary + 1e-8)
        for field in ("normal_separation_m", "tangent_offset_m", "twist_rad"):
            assert left[field] == pytest.approx(right[field], abs=1e-12)
    assert phase_command(config, 100)["normal_separation_m"] == config.clearance_m
    for invalid in (-1, math.inf, math.nan):
        with pytest.raises(ValueError):
            phase_command(config, invalid)


def test_annulus_is_closed_ring_with_true_inner_hole():
    config = ContactDemoConfig(shape="annulus")
    vertices, faces = annulus_mesh(config)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    assert mesh.is_watertight and mesh.is_winding_consistent
    assert mesh.volume > 0
    assert np.linalg.norm(vertices[:, :2], axis=1).min() == pytest.approx(config.annulus_inner_radius_m)
    # The center has no triangle surface: planar bearing faces remain outside
    # the physical aperture, so RTX and signed-distance queries see a hole.
    distance = trimesh.proximity.closest_point_naive(mesh, np.asarray([[0, 0, -config.annulus_height_m / 2]]))[1]
    assert distance[0] > config.annulus_inner_radius_m * 0.99


@pytest.mark.parametrize(
    "gel_rotation",
    [
        Rotation.identity(),
        Rotation.from_euler("xyz", [30, -42, 110], degrees=True),
    ],
)
def test_pose_places_actual_bearing_face_at_measured_gel(gel_rotation):
    config = ContactDemoConfig()
    command = phase_command(config, 2.5 * config.phase_duration_s)
    command.update(tangent_offset_m=0.002, twist_rad=0.3)
    center, support = np.asarray([1.5, -2.3, 0.4]), 0.0039
    pose = pose_at_pad(center, gel_rotation.as_quat(), support, command)
    world_rotation = Rotation.from_quat(pose[3:])
    face = pose[:3] + world_rotation.apply([0, 0, -support])
    outward = gel_rotation.apply([0, -1, 0])
    tangent = gel_rotation.apply([1, 0, 0])
    np.testing.assert_allclose(face, center + outward * command["normal_separation_m"] + tangent * 0.002, atol=2e-7)
    np.testing.assert_allclose(world_rotation.apply([0, 0, 1]), outward, atol=1e-7)


def test_raw_asset_transforms_scale_and_zero_face_diagnosis(tmp_path):
    path = tmp_path / "rigid_fixture.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Nut")
    stage.SetDefaultPrim(root.GetPrim())
    root.AddTranslateOp().Set(Gf.Vec3d(1, 2, 3))
    mesh = UsdGeom.Mesh.Define(stage, "/Nut/collisions")
    mesh.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.01))
    mesh.CreatePointsAttr([(0, 0, 0), (0.02, 0, 0), (0, 0.02, 0.01)])
    mesh.CreateFaceVertexCountsAttr([3, 3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 0, 0])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    stage.GetRootLayer().Save()
    vertices, faces, provenance = load_contact_geometry(ContactDemoConfig(asset_path=str(path), asset_scale=0.5))
    np.testing.assert_allclose(vertices, [[-0.005, -0.005, -0.0025], [0.005, -0.005, -0.0025], [-0.005, 0.005, 0.0025]])
    np.testing.assert_array_equal(faces, [[0, 1, 2], [0, 0, 0]])
    np.testing.assert_allclose(provenance["source_bbox_center_m"], [1.01, 2.01, 3.015])
    assert provenance["zero_area_source_triangle_ids"] == [1]
    assert provenance["zero_area_source_triangle_count"] == 1
    assert len(provenance["asset_sha256"]) == 64


def test_missing_official_geometry_is_not_replaced_by_generated_ring(tmp_path):
    with pytest.raises(FileNotFoundError, match="Official TacSL nut"):
        load_contact_geometry(ContactDemoConfig(asset_path=str(tmp_path / "missing.usd")))


def test_task_extension_has_separate_inventory_and_default_hook(monkeypatch):
    sentinel = [{"pad": SimpleNamespace(prim_path="/World/calibration/pad")}]
    called = []

    class OriginalTask:
        def __init__(self, config):
            assert config.sim["particle_readback"] is True
            assert config.sim["host_readback_reason"] == "kinematic_contact_calibration"

        def _setup_contact_fixtures(self, sim):
            called.append(sim)

    def spawn(self, stage):
        assert stage is stage_token
        return sentinel

    stage_token = object()
    monkeypatch.setattr(ContactFixtures, "spawn", spawn)
    config = SimpleNamespace(sim={}, task_env={})
    task = contact_demo_task_class(OriginalTask, replace(ContactDemoConfig(), shape="annulus"))(config)
    task.num_envs = 1
    canonical = [{"bowl": object()}]
    task.scene_manager = SimpleNamespace(_rigid_and_dynamic_objects=canonical)
    sim = SimpleNamespace(sim=SimpleNamespace(stage=stage_token))
    task._setup_contact_fixtures(sim)
    assert called == [sim]
    assert task.scene_manager._rigid_and_dynamic_objects is canonical
    assert task.scene_manager._tactile_calibration_objects is sentinel


def test_actual_task_env_hook_authors_geometry_before_contact_cooking():
    source = Path(__file__).resolve().parents[1] / "env/environment/task_env.py"
    tree = ast.parse(source.read_text())
    task = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TaskEnv")
    method = next(node for node in task.body if isinstance(node, ast.FunctionDef) and node.name == "_post_setup_scene")
    events = []

    class NativeWarmupBoundary(Exception):
        pass

    class Parent:
        def _post_setup_scene(self, sim):
            events.append("native_warmup")
            raise NativeWarmupBoundary

    probe = ast.ClassDef(
        name="Probe", bases=[ast.Name(id="Parent", ctx=ast.Load())], keywords=[], body=[method], decorator_list=[]
    )
    namespace = {"Parent": Parent}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[probe], type_ignores=[])), str(source), "exec"), namespace)
    env = namespace["Probe"]()
    env.scene_manager = SimpleNamespace(initialize=lambda sim: events.append("scene"))
    env._setup_contact_fixtures = lambda sim: events.append("fixture")
    env.tactile_manager = SimpleNamespace(initialize=lambda sim, scene: events.append("contact_schemas"))
    with pytest.raises(NativeWarmupBoundary):
        env._post_setup_scene(object())
    assert events == ["scene", "fixture", "contact_schemas", "native_warmup"]


def test_runtime_commands_only_native_kinematic_targets_and_records_measured_state():
    config = ContactDemoConfig(shape="annulus")
    command_pose = np.asarray([1, 2, 3, 0, 0, 0, 1], dtype=np.float32)
    measured_pose = command_pose.copy()
    measured_pose[0] = -5

    class NativeView:
        def __init__(self):
            self.pose = measured_pose[None].copy()
            self.target = None

        def set_kinematic_targets(self, targets, indices):
            np.testing.assert_array_equal(indices, [0])
            self.target = targets.numpy().copy()

        def get_transforms(self):
            return torch.from_numpy(self.pose)

        def get_velocities(self):
            return torch.zeros((1, 6))

        def set_transforms(self, *args):
            raise AssertionError("Runtime motion must not teleport a body")

        def set_velocities(self, *args):
            raise AssertionError("Kinematic bodies do not support velocity writes")

    # Execute the production controller methods with a native-view boundary
    # double. The double deliberately does not simulate physics or gel force.
    demo = ContactMotionDemo.__new__(ContactMotionDemo)
    demo.config = config
    demo.fixtures = SimpleNamespace(paths=[[f"/World/probe{i}" for i in range(4)]])
    demo.env_id, demo.device = 0, torch.device("cpu")
    demo.sim = SimpleNamespace(current_time=0.2, current_time_step_index=20)
    demo.views = [NativeView() for _ in PADS]
    demo.indices = np.asarray([0], dtype=np.int32)
    demo.torch = torch
    demo.wp = SimpleNamespace(from_torch=lambda value: value, to_torch=lambda value: value)
    demo._poses = lambda command: [command_pose.copy() for _ in PADS]
    demo.history = []
    demo.apply(config.phase_duration_s, 20)
    for view in demo.views:
        np.testing.assert_array_equal(view.target, command_pose[None])
    with pytest.raises(RuntimeError, match="has not advanced"):
        demo.sample()
    demo.sim.current_time_step_index = 21
    demo.sim.current_time = 0.21
    sample = demo.sample()
    np.testing.assert_array_equal(sample["measured_probe_poses_w_xyzw"], np.stack([measured_pose] * 4))
    assert sample["last_command"]["issued_before_physics_frame"] == 20
    assert sample["physics_frame"] == 21 and sample["timestamp_ns"] == 210000000
    # Mutating backend buffers/commands cannot rewrite the archived sample.
    demo.views[0].pose[:] = 99
    demo.last_command["commanded_probe_poses_w_xyzw"][:] = 99
    assert sample["measured_probe_poses_w_xyzw"][0, 0] == -5
    assert sample["last_command"]["commanded_probe_poses_w_xyzw"][0, 0] == 1


def test_contact_readback_must_precede_native_warmup():
    writes = {}
    context = SimpleNamespace(
        physics_manager=SimpleNamespace(_view=None), set_setting=writes.__setitem__, get_setting=lambda name: False
    )
    configure_contact_physics(context, install_bridge=False)
    assert writes == {
        "/physics/suppressReadback": False,
        "/physics/updateToUsd": True,
        "/isaaclab/robodojo/particle_readback": True,
        "/isaaclab/robodojo/contact_calibration_readback": True,
    }
    writes.clear()
    context.physics_manager._view = object()
    with pytest.raises(RuntimeError, match="before native physics"):
        configure_contact_physics(context, install_bridge=False)
    assert not writes
    close_contact_physics(context)
    assert writes == {"/isaaclab/robodojo/contact_calibration_readback": False}


def test_standalone_bridge_callback_lifetime_and_idempotent_configuration(monkeypatch):
    handle = SimpleNamespace(deregister=Mock())
    installer = Mock(return_value=handle)
    monkeypatch.setitem(
        sys.modules, "utils.particle_readback_bridge", SimpleNamespace(install_particle_readback_bridge=installer)
    )
    writes = {}
    context = SimpleNamespace(
        physics_manager=SimpleNamespace(_view=None), set_setting=writes.__setitem__, get_setting=lambda name: True
    )
    assert configure_contact_physics(context) is handle
    assert configure_contact_physics(context) is handle
    installer.assert_called_once_with(context.physics_manager)
    close_contact_physics(context)
    close_contact_physics(context)
    handle.deregister.assert_called_once_with()
    assert writes["/isaaclab/robodojo/contact_calibration_readback"] is True
    assert context._robodojo_contact_readback_handle is None


def test_kinematic_reset_and_park_never_write_unsupported_velocities():
    class NativeView:
        def __init__(self):
            self.transforms, self.targets = [], []

        def set_transforms(self, poses, indices):
            self.transforms.append(poses.clone())

        def set_kinematic_targets(self, poses, indices):
            self.targets.append(poses.clone())

        def set_velocities(self, *args):
            raise AssertionError("PhysX rejects velocity writes for kinematic actors")

    demo = ContactMotionDemo.__new__(ContactMotionDemo)
    demo.config, demo.env_id, demo.device = ContactDemoConfig(), 0, torch.device("cpu")
    demo.wp, demo.torch = SimpleNamespace(from_torch=lambda value: value), torch
    demo._poses = lambda command: [np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.float32) for _ in PADS]
    demo.views, demo.indices, demo.history = [NativeView() for _ in PADS], [0], ["old"]
    demo.reset()
    assert demo.last_command is None and not demo.history
    demo.park()
    for index, view in enumerate(demo.views):
        torch.testing.assert_close(view.transforms[-1], torch.tensor([[10, index, 1, 0, 0, 0, 1]], dtype=torch.float32))
        torch.testing.assert_close(view.transforms[-1], view.targets[-1])


def test_contact_profile_rejects_fem_before_setting_readback():
    class OriginalTask:
        def __init__(self, *args):
            raise AssertionError("Incompatible pipelines must fail before initialization")

    config = SimpleNamespace(sim={}, task_env={"Garment": [object()]})
    with pytest.raises(NotImplementedError, match="surface FEM"):
        contact_demo_task_class(OriginalTask)(config)
    assert not config.sim


def test_validate_rejects_stationary_native_state_despite_a_complete_command_sequence():
    demo = ContactMotionDemo.__new__(ContactMotionDemo)
    demo.config = ContactDemoConfig()
    pose = np.asarray([[0, 0, 0, 0, 0, 0, 1]] * 4, dtype=np.float32)
    demo.history = [
        {
            "last_command": {"phase": phase, "commanded_probe_poses_w_xyzw": pose},
            "measured_probe_poses_w_xyzw": pose,
            "measured_probe_velocities_w": np.zeros((4, 6)),
        }
        for phase in PHASES
    ]
    with pytest.raises(RuntimeError, match="did not measurably move"):
        demo.validate()


def validation_probe():
    """Validator input contract only; this does not simulate native physics."""
    demo = ContactMotionDemo.__new__(ContactMotionDemo)
    demo.config = ContactDemoConfig()
    demo.configuration = lambda: {"test": "CPU validator fixture"}
    demo.history = []
    for index, phase in enumerate(PHASES):
        command = phase_command(demo.config, (index + 0.5) * demo.config.phase_duration_s)
        pose = pose_at_pad(np.zeros(3), [0, 0, 0, 1], 0.0039, command)
        poses = np.stack([pose] * 4)
        demo.history.append(
            {
                "last_command": {"phase": phase, "commanded_probe_poses_w_xyzw": poses.copy()},
                "measured_probe_poses_w_xyzw": poses.copy(),
                "measured_probe_velocities_w": np.full((4, 6), 0.01),
            }
        )
    return demo


def test_quaternion_validation_accepts_sign_equivalent_native_orientation():
    demo = validation_probe()
    for record in demo.history:
        record["measured_probe_poses_w_xyzw"][:, 3:] *= -1
    result = demo.validate()
    assert result["native_orientation_max_error_rad"] < 1e-7
    assert result["native_quaternion_norm_max_error"] < 1e-6
    assert all(angle > 0.1 for angle in result["native_rotation_from_start_max_rad"])


def test_quaternion_validation_rejects_wrong_twist_despite_nonzero_angular_velocity():
    demo = validation_probe()
    actual = demo.history[6]["measured_probe_poses_w_xyzw"]
    actual[:, 3:] = (Rotation.from_quat(actual[:, 3:]) * Rotation.from_rotvec([0, 0, 0.01])).as_quat()
    with pytest.raises(RuntimeError, match="prescribed twist"):
        demo.validate()


def test_position_error_larger_than_two_tenths_mm_is_not_accepted():
    demo = validation_probe()
    demo.history[2]["measured_probe_poses_w_xyzw"][:, 0] += 0.0003
    with pytest.raises(RuntimeError, match="within 0.2mm"):
        demo.validate()
