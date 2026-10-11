from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from env.reward_manager.reward_manager import RewardManager
from reproduction.isaac61.task_runtime_probe import (
    REWARD_COLLECTIONS,
    finite_scalar,
    registered_predicates,
    validate_task_runtime,
)


def test_predicate_extraction_handles_boolean_groups_aliases_and_cycles():
    first = ("first", {"label": "A"})
    second = ("second", {"label": "B"})
    nested = [first, [second, first], False, ("rising_edge", False)]
    nested.append(nested)
    manager = SimpleNamespace(**{name: [[]] for name in REWARD_COLLECTIONS})
    manager.check_list[0] = nested
    manager.trigger_query_list[0] = [([first], [second], "level", False, 1, 0)]
    assert list(registered_predicates(manager, 0)) == [first, second]


@pytest.mark.parametrize("value", [0, 1.5, False, True, np.float32(0.3), torch.tensor(0.7)])
def test_finite_scalar_accepts_real_numeric_and_boolean_results(value):
    assert np.isfinite(finite_scalar(value, "probe"))


@pytest.mark.parametrize("value", [None, "1", [1], np.nan, np.inf, torch.tensor(float("nan"))])
def test_finite_scalar_rejects_nonfinite_or_non_scalar_rewards(value):
    with pytest.raises(ValueError, match="finite numeric scalar"):
        finite_scalar(value, "probe")


class NativeParserProbe:
    def __init__(self, env):
        self.env = env
        self.pre_state = [{"state": 7}]
        self.robot_origin_endpose = [{"robot": [1, 2, 3]}]
        self.joint_ratio_transition_state = [{}]
        self.calls = []

    def init_state(self):
        self.calls.append("init_state")

    def _check_env_success(self, env_idx):
        return self.env.success[env_idx]

    def native_false(self, args):
        self.calls.append("false")
        return False

    def native_joint(self, args):
        self.calls.append("joint")
        assert self.pre_state == [{"state": 7}]
        assert self.env.success == [True]
        if self.env.nonfinite_predicate:
            return np.nan
        self.pre_state[0]["state"] = 99
        self.joint_ratio_transition_state[0]["edge"] = True
        self.env.success[0] = False
        if self.env.advance_physics:
            self.env.frame += 1
        return torch.tensor(0.5)


def make_task():
    env = SimpleNamespace(num_envs=1, frame=30, nonfinite_predicate=False, advance_physics=False)
    env.sim = SimpleNamespace(
        sim=SimpleNamespace(get_physics_step_count=lambda: env.frame, get_physics_dt=lambda: 0.01)
    )
    manager = RewardManager(1)
    parser = NativeParserProbe(env)
    manager.func_parser = parser
    env.reward_manager = manager
    env.gen_instruction = lambda env_id: ["Move the real object."]

    def register():
        manager.check([("native_false", {}), ("native_joint", {})])

    env.run_reward = register
    return env, manager, parser


def test_real_reward_step_and_extra_predicates_restore_episode_histories():
    env, manager, parser = make_task()
    result = validate_task_runtime(env)
    reward = result["reward"]
    assert reward["reward_values"] == [0.0]
    assert reward["score_values"] == [0.0]
    assert reward["predicate_evaluations"] == [
        {"env_id": 0, "name": "native_false", "value": 0.0},
        {"env_id": 0, "name": "native_joint", "value": 0.5},
    ]
    # Normal step short circuits after the false condition; isolated extra
    # calls actually covered the following joint API while preserving state.
    assert parser.calls == ["init_state", "false", "joint", "false"]
    assert parser.pre_state == [{"state": 7}]
    assert parser.joint_ratio_transition_state == [{}]
    assert env.success == [True]
    assert "env_idx" not in manager.check_list[0][0][1][1]
    assert result["physics_step_before"] == result["physics_step_after"] == 30


def test_nonfinite_later_predicate_cannot_hide_behind_a_false_first_check():
    env, manager, parser = make_task()
    env.nonfinite_predicate = True
    histories = deepcopy(parser.pre_state)
    with pytest.raises(ValueError, match="native_joint"):
        validate_task_runtime(env)
    assert parser.pre_state == histories
    assert env.success == [True]
    assert manager.check_list[0]


def test_bound_label_callback_retains_native_owner_without_deepcopy():
    env, manager, parser = make_task()

    class NativeOwner:
        def __deepcopy__(self, memo):
            raise AssertionError("Native stage owners cannot be deep-copied")

        def select(self, args):
            return 0.25

    owner = NativeOwner()
    callback = owner.select
    predicate = ("native_selector", {"label": callback})
    parser.native_selector = lambda args: args["label"](args)
    env.run_reward = lambda: manager.check([predicate])
    result = validate_task_runtime(env)
    assert result["reward"]["predicate_evaluations"] == [{"env_id": 0, "name": "native_selector", "value": 0.25}]
    assert predicate[1]["label"] is callback


def test_runtime_probe_requires_physics_to_stay_frozen():
    env, _, _ = make_task()
    env.advance_physics = True
    with pytest.raises(RuntimeError, match="advanced physics"):
        validate_task_runtime(env)


@pytest.mark.parametrize("templates", [[], [""], ["   "], [None], "single string"])
def test_empty_or_invalid_instructions_fail_before_reward_registration(templates):
    env, manager, parser = make_task()
    env.gen_instruction = lambda env_id: templates
    with pytest.raises(ValueError, match="instruction"):
        validate_task_runtime(env)
    assert not manager.check_list[0]
    assert not parser.calls
