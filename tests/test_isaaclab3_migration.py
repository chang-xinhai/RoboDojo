"""Test RoboDojo's actual manager methods without starting Kit or importing cuRobo."""

import ast
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import List, Literal

import numpy as np
import pytest
import torch

from utils.isaaclab_compat import as_torch, pose_xyzw_to_wxyz, set_joint_target, wxyz_to_xyzw

ROOT = Path(__file__).resolve().parents[1]


class ProxyArray:
    def __init__(self, tensor):
        self.torch = tensor


def load_class(path, name, **namespace):
    """Execute original class bodies while excluding Kit-dependent imports."""
    tree = ast.parse((ROOT / path).read_text())
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    scope = dict(
        Sequence=Sequence,
        List=List,
        Literal=Literal,
        DictConfig=dict,
        deepcopy=deepcopy,
        np=np,
        torch=torch,
        as_torch=as_torch,
        pose_xyzw_to_wxyz=pose_xyzw_to_wxyz,
        set_joint_target=set_joint_target,
        wxyz_to_xyzw=wxyz_to_xyzw,
    )
    scope.update(namespace)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / path), "exec"), scope)
    return scope[name]


@pytest.fixture
def manager():
    cls = load_class("env/robot_manager/robot_manager.py", "RobotManager", IsaacRLEnv=object)
    instance = cls.__new__(cls)
    robot = SimpleNamespace(arm_joint_indices=[2, 0], gripper_joint_indices=[3, 1], robot_name="x5")
    instance.robot_list = [robot]
    instance.num_envs = 3
    instance.scene = SimpleNamespace(env_origins=torch.tensor([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [8.0, 0.0, 0.0]]))
    # Nonidentity rotations detect incorrect quaternion order even for symmetric roots.
    poses = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 0.2, 0.3, 0.4, 0.8]],
            [[5.0, 2.0, 3.0, 0.1, 0.2, 0.7, 0.6]],
            [[9.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]],
        ]
    )
    instance.robot_key = [
        SimpleNamespace(
            body_names=["tool"],
            data=SimpleNamespace(
                body_link_pose_w=ProxyArray(poses), joint_pos=ProxyArray(torch.arange(12.0).reshape(3, 4))
            ),
        )
    ]
    return instance, robot, poses


def test_public_link_pose_wxyz_relative_subset_preserves_simulator(manager):
    instance, robot, raw = manager
    before = raw.clone()
    result = instance.get_link_pose(robot, "tool", env_idx_list=[1], is_relative=True)
    np.testing.assert_allclose(result[1], [1.0, 2.0, 3.0, 0.6, 0.1, 0.2, 0.7])
    assert result[0] is None and result[2] is None
    assert torch.equal(raw, before)


def test_public_link_pose_world_and_missing_link(manager):
    instance, robot, _ = manager
    result = instance.get_link_pose(robot, "tool", is_relative=False)
    np.testing.assert_allclose(result[0], [1.0, 2.0, 3.0, 0.8, 0.2, 0.3, 0.4])
    with pytest.raises(ValueError, match="Link name unknown"):
        instance.get_link_pose(robot, "unknown")


def test_joint_order_and_selected_environments(manager):
    instance, robot, _ = manager
    joints = instance.get_joint(robot, env_idx_list=[2])
    grip = instance.get_end_effector_real_val(robot, env_idx_list=[2])
    np.testing.assert_array_equal(joints[2], [10.0, 8.0])
    np.testing.assert_array_equal(grip[2], [11.0, 9.0])
    assert joints[0] is None and grip[1] is None


def test_sim_step_renders_only_requested_interval():
    cls = load_class("env/environment/isaac/direct_rl_env.py", "CustomDirectRLEnv", DirectRLEnv=object)
    instance = cls()
    calls = []
    instance.cfg = SimpleNamespace(decimation=3, sim=SimpleNamespace(render_interval=2), events=None)
    instance.scene = SimpleNamespace(
        write_data_to_sim=lambda: calls.append("write"), update=lambda dt: calls.append(("update", dt))
    )
    instance.sim = SimpleNamespace(
        is_rendering=True,
        get_setting=lambda _: True,
        step=lambda render: calls.append("step"),
        render=lambda: calls.append("render"),
    )
    instance.physics_dt = 0.004
    instance._sim_step_counter = instance.common_step_counter = 0
    instance.episode_length_buf = torch.zeros(3, dtype=torch.int64)
    instance.extras = {}
    instance.sim_step(render=False)
    assert calls == ["write", "step", ("update", 0.004)] * 3
    calls.clear()
    instance.sim_step(render=True)
    assert calls.count("render") == 2  # global physics steps 4 and 6
    assert instance._sim_step_counter == 6 and instance.common_step_counter == 2
    assert instance.episode_length_buf.tolist() == [2, 2, 2]


def test_launcher_preserves_existing_options():
    import argparse

    from utils.isaaclab_compat import add_app_launcher_args

    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    launcher = SimpleNamespace(add_app_launcher_args=lambda parser: parser.add_argument("--viz"))
    add_app_launcher_args(parser, launcher)
    args = parser.parse_args(["--headless", "--enable_cameras", "--viz", "none"])
    assert args.headless and args.enable_cameras and args.viz == "none"


@pytest.mark.parametrize("value", [None, 1, np.array(1)])
def test_scalar_pose_inputs_raise_clear_value_error(value):
    with pytest.raises(ValueError, match="Expected quaternion"):
        wxyz_to_xyzw(value)
    with pytest.raises(ValueError, match="Expected pose"):
        pose_xyzw_to_wxyz(value)


@pytest.mark.parametrize("requires_readback", [False, True])
def test_particle_pipeline_selector_preserves_cuda_manager(requires_readback, monkeypatch):
    monkeypatch.setattr("utils.legacy_prim_adapter.install_legacy_prim_adapter", lambda: (object(), ()))
    readback_managers = []
    monkeypatch.setattr("utils.particle_readback_bridge.install_particle_readback_bridge", readback_managers.append)
    physics_manager = SimpleNamespace(device="cuda:0")
    settings = {}
    cls = load_class(
        "env/environment/base_env.py",
        "BaseEnv",
        gym=SimpleNamespace(Env=object),
        Any=object,
        IsaacRLEnv=object,
        get_physx_interface=lambda: object(),
    )
    instance = cls.__new__(cls)
    instance._requires_particle_readback = requires_readback
    instance.config = {}
    context = SimpleNamespace(physics_manager=physics_manager, set_setting=settings.__setitem__)
    instance.setup_physics(SimpleNamespace(sim=context))
    assert readback_managers == ([physics_manager] if requires_readback else [])
    assert instance.physics_context is physics_manager
    assert instance.physics_context.device == "cuda:0"
    if requires_readback:
        assert settings["/physics/suppressReadback"] is False
        assert settings["/physics/updateParticlesToUsd"] is True
        assert settings["/isaaclab/robodojo/particle_readback"] is True
    else:
        assert settings == {}


@pytest.mark.parametrize("original_interop", [False, True])
@pytest.mark.parametrize("close_fails", [False, True])
def test_surface_cpu_fabric_transport_restores_original_setting_after_close(monkeypatch, original_interop, close_fails):
    monkeypatch.setattr("utils.legacy_prim_adapter.install_legacy_prim_adapter", lambda: (object(), ()))
    cls = load_class(
        "env/environment/base_env.py",
        "BaseEnv",
        gym=SimpleNamespace(Env=object),
        Any=object,
        IsaacRLEnv=object,
        get_physx_interface=lambda: object(),
    )
    calls = []
    settings = {"/physics/fabricUseGPUInterop": original_interop}

    def set_setting(name, value):
        settings[name] = value
        calls.append((name, value))

    context = SimpleNamespace(
        physics_manager=SimpleNamespace(device="cuda:0"),
        get_setting=settings.get,
        set_setting=set_setting,
    )
    instance = cls.__new__(cls)
    instance._requires_surface_fabric_copy = True
    instance.setup_physics(SimpleNamespace(sim=context))
    assert instance.physics_context is context.physics_manager
    assert settings["/physics/fabricUseGPUInterop"] is False
    assert settings["/isaaclab/robodojo/surface_fabric_cpu_copy"] is True

    def close():
        calls.append(("close", None))
        if close_fails:
            raise RuntimeError("native close error")

    instance.sim = SimpleNamespace(close=close)
    for name in ("omni", "omni.timeline", "isaaclab", "isaaclab.sim", "isaaclab.sim.utils", "isaaclab.sim.utils.stage"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules["omni"].timeline = sys.modules["omni.timeline"]
    sys.modules["omni.timeline"].get_timeline_interface = lambda: SimpleNamespace(
        stop=lambda: None, set_current_time=lambda value: None
    )
    sys.modules["isaaclab.sim.utils.stage"].create_new_stage = lambda: calls.append(("new_stage", None))
    if close_fails:
        with pytest.raises(RuntimeError, match="native close error"):
            instance.close()
    else:
        instance.close()
    assert instance.sim is None
    assert settings["/physics/fabricUseGPUInterop"] is original_interop
    assert settings["/isaaclab/robodojo/surface_fabric_cpu_copy"] is False
    assert calls.index(("close", None)) < len(calls) - 1
    assert not hasattr(instance, "_surface_fabric_restore")
    # A later ordinary context neither overrides the restored user value nor
    # inherits the garment-specific provenance marker as active.
    next_env = cls.__new__(cls)
    next_env.setup_physics(SimpleNamespace(sim=context))
    assert settings["/physics/fabricUseGPUInterop"] is original_interop
    assert not hasattr(next_env, "_surface_fabric_restore")
