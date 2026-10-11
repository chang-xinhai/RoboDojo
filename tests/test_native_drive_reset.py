import ast
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def candidate_class(events):
    source = Path(__file__).resolve().parents[1] / "env/scene_manager/objects/articulation.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    names = {
        "initialize",
        "_clone_native_drive_targets",
        "_clone_native_joint_buffer",
        "_capture_drive_targets",
        "reset_drive_targets",
        "apply_saved_pose",
        "relocate_offscreen",
        "set_current_joint_positions",
    }
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]

    class Parent:
        def initialize(self, physics_sim_view):
            events.append("native_view_initialized")

    new_cls = ast.ClassDef(
        name="Candidate", bases=[ast.Name(id="Parent", ctx=ast.Load())], keywords=[], body=methods, decorator_list=[]
    )
    ns = dict(
        Parent=Parent,
        np=np,
        torch=torch,
        random=random,
        SimulationManager=SimpleNamespace(get_physics_sim_view=lambda: object()),
        omni=SimpleNamespace(
            kit=SimpleNamespace(
                app=SimpleNamespace(get_app=lambda: SimpleNamespace(update=lambda: events.append("app_update")))
            )
        ),
        UsdGeom=SimpleNamespace(Imageable=lambda prim: None),
    )
    exec(compile(ast.fix_missing_locations(ast.Module(body=[new_cls], type_ignores=[])), str(source), "exec"), ns)
    return ns["Candidate"]


def owner(kind="torch", reset=True):
    events = []
    obj = candidate_class(events)()
    initial = np.array([[0.006, np.pi / 2]], np.float32)
    state = SimpleNamespace(
        targets=torch.tensor(initial) if kind == "torch" else initial.copy(),
        reads=0,
        positions=torch.tensor([0.003, 0.2]),
        position_reads=0,
    )

    def read(clone):
        assert clone is True
        events.append("capture_targets")
        state.reads += 1
        return SimpleNamespace(joint_positions=state.targets)

    def write(value):
        events.append("restore_targets")
        state.targets = value.clone() if isinstance(value, torch.Tensor) else value.copy()

    obj._articulation_view = SimpleNamespace(
        get_applied_actions=read,
        set_joint_position_targets=write,
        get_dof_limits=lambda: np.array([[[0, 0.01], [-3, 3]]], np.float32),
    )
    obj.physics_cfg = {"reset_drive": reset}
    obj._drive_targets_captured = False
    obj._joint_positions_captured = False
    obj.initial_drive_targets = None
    obj._prim_path = "/World/Button"
    obj.num_dof = 2

    def read_positions():
        state.position_reads += 1
        return state.positions

    obj.get_current_joint_positions = read_positions

    def teleport_joint_positions(value):
        # The native state setter also changes PD targets, independently of the
        # cached reset baseline. Both production reset entrypoints call it.
        state.targets = value.reshape(1, 2).clone()
        state.positions = value.clone()

    obj.set_joint_positions = teleport_joint_positions
    obj.set_local_pose = lambda *args: events.append("pose_reset")
    obj.set_local_scale = lambda *args: events.append("scale_reset")
    obj._apply_default_velocities = lambda: events.append("velocity_reset")
    obj.default_pos = [0, 0, 0]
    obj.default_ori = [1, 0, 0, 0]
    obj.init_scale = [1, 1, 1]
    obj.visual_cfg = {}
    obj.prim = object()
    return obj, state, events, initial


@pytest.mark.parametrize("kind", ["torch", "numpy"])
def test_repeat_initialization_never_overwrites_owned_native_baseline(kind):
    obj, state, events, initial = owner(kind)
    obj.initialize()
    assert events.index("capture_targets") < events.index("app_update")
    state.targets[:] = 0.001
    state.positions[:] = 0.009
    obj.initialize()
    assert state.reads == 1 and state.position_reads == 1
    np.testing.assert_array_equal(obj.initial_joint_positions.numpy(), np.array([0.003, 0.2], np.float32))
    obj.reset_drive_targets()
    np.testing.assert_array_equal(np.asarray(state.targets), initial)
    state.targets[:] = 0.007
    np.testing.assert_array_equal(np.asarray(obj.initial_drive_targets), initial)


@pytest.mark.parametrize("entrypoint", ["apply_saved_pose", "relocate_offscreen"])
def test_actual_reset_entrypoints_restore_motor_targets_not_current_positions(entrypoint):
    obj, state, events, initial = owner()
    obj.initialize()
    getattr(obj, entrypoint)()
    np.testing.assert_array_equal(np.asarray(state.targets), initial)
    np.testing.assert_array_equal(state.positions.numpy(), np.array([0.003, 0.2], np.float32))
    assert events.count("restore_targets") == 1
    assert events.index("restore_targets") < events.index("pose_reset")


def test_reset_disabled_does_not_capture_or_restore_native_targets():
    obj, state, _, _ = owner(reset=False)
    obj.initialize()
    obj.reset_drive_targets()
    assert state.reads == 0 and obj.initial_drive_targets is None


@pytest.mark.parametrize(
    "value,error",
    [
        (None, RuntimeError),
        (np.array([0.006], np.float32), ValueError),
        (np.array([[1, 2]], np.int32), ValueError),
        (torch.tensor([[float("nan"), 0.2]]), ValueError),
        ([0.006, 0.2], TypeError),
    ],
)
def test_invalid_native_snapshot_fails_explicitly(value, error):
    obj, _, _, _ = owner()
    with pytest.raises(error):
        obj._clone_native_drive_targets(value)


def test_reset_before_native_initialization_has_no_usd_fallback():
    obj, _, _, _ = owner()
    with pytest.raises(RuntimeError, match="initialized snapshot"):
        obj.reset_drive_targets()


@pytest.mark.parametrize("positions", [torch.tensor([float("nan"), 0.2]), torch.tensor([[0.003, 0.2]])])
def test_state_setter_rejects_invalid_position_buffer_before_native_write(positions):
    obj, state, _, _ = owner()
    original = state.positions.clone()
    with pytest.raises(ValueError):
        obj.set_current_joint_positions(positions)
    torch.testing.assert_close(state.positions, original)


def test_missing_native_actions_does_not_mark_drive_snapshot_captured():
    obj, _, events, _ = owner()
    obj._articulation_view.get_applied_actions = lambda clone: None
    with pytest.raises(RuntimeError, match="actions are unavailable"):
        obj.initialize()
    assert obj._drive_targets_captured is False
    assert "app_update" not in events


def test_snapshot_preserves_native_dtype_device_and_detaches_ownership():
    obj, _, _, _ = owner()
    native = torch.tensor([[0.006, np.pi / 2]], dtype=torch.float64, requires_grad=True)
    snapshot = obj._clone_native_drive_targets(native)
    assert snapshot.device == native.device and snapshot.dtype == native.dtype
    assert snapshot.requires_grad is False and snapshot.data_ptr() != native.data_ptr()
    torch.testing.assert_close(snapshot, native.detach())
