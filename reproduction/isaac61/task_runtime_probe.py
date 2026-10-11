"""Exercise real task language and reward APIs at frozen physics state."""

from collections.abc import Mapping
from copy import deepcopy
import json
import sys
from time import perf_counter

import numpy as np

from utils.isaaclab_compat import simulation_snapshot
from utils.transformer import safe_deepcopy_keep_callable

REWARD_COLLECTIONS = (
    "check_list",
    "final_check_list",
    "query_list",
    "trigger_check_list",
    "trigger_query_list",
    "score_list",
    "final_score_list",
    "score_trigger_meta",
)
PARSER_STATE = ("pre_state", "robot_origin_endpose", "joint_ratio_transition_state")


def _trace(phase, **details):
    print("[task-runtime] " + json.dumps({"phase": phase, **details}), file=sys.stderr, flush=True)


def registered_predicates(manager, env_id):
    """Find real (parser name, arguments) leaves without repeating aliases."""
    visited = set()

    def walk(value):
        if not isinstance(value, (tuple, list, Mapping)) or id(value) in visited:
            return
        visited.add(id(value))
        if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], str) and isinstance(value[1], Mapping):
            yield value
            return
        for child in value.values() if isinstance(value, Mapping) else value:
            yield from walk(child)

    for name in REWARD_COLLECTIONS:
        yield from walk(getattr(manager, name)[env_id])


def finite_scalar(value, name):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "biuf" or not np.isfinite(array).item():
        raise ValueError(f"Task reward {name} must be a finite numeric scalar: {value!r}")
    return float(array.item())


def validate_task_runtime(env):
    """Follow evaluation reward setup and separately probe all registered APIs.

    Bare TaskEnv omits EvalEnv's episode-validity flag. Enabling it here allows
    the reward parser to run; this flag makes no claim about policy success.
    Extra predicate calls restore parser histories and validity after each call,
    so the ordinary reward step retains its production transition semantics.
    """
    before = simulation_snapshot(env.sim.sim)
    started = perf_counter()
    _trace("instruction_begin")
    templates = env.gen_instruction(0)
    if (
        not isinstance(templates, list)
        or not templates
        or any(not isinstance(template, str) or not template.strip() for template in templates)
    ):
        raise ValueError("Task instruction must contain nonempty string templates")
    # Each invocation starts a fresh diagnostic episode, including after a
    # previous replay's legitimate reward checks marked it invalid.
    env.success = [True] * env.num_envs
    manager = env.reward_manager
    _trace("state_init_begin")
    manager.init_state()
    _trace("registration_begin")
    registration_result = env.run_reward()
    if registration_result is not None:
        finite_scalar(registration_result, "registration return")
    score_registered = hasattr(env, "get_score")
    if score_registered:
        env.get_score()
    predicates = list(registered_predicates(manager, 0))
    if not predicates:
        raise ValueError("Task did not register any native reward predicates")
    parser = manager.func_parser
    histories = {name: deepcopy(getattr(parser, name)) for name in PARSER_STATE}
    validity = deepcopy(env.success)
    extra = []
    for predicate in predicates:
        predicate_started = perf_counter()
        _trace("predicate_begin", name=predicate[0], env_id=0)
        try:
            result = manager.call_func_parser(safe_deepcopy_keep_callable(predicate), 0)
            extra.append({"env_id": 0, "name": predicate[0], "value": finite_scalar(result, predicate[0])})
            _trace("predicate_end", name=predicate[0], elapsed_seconds=perf_counter() - predicate_started)
        finally:
            for name, snapshot in histories.items():
                setattr(parser, name, deepcopy(snapshot))
            env.success = deepcopy(validity)
    _trace("normal_reward_step_begin")
    manager.step(env_idx_list=[0])
    rewards = manager.get_reward()
    scores = manager.get_score()
    if len(rewards) != env.num_envs or len(scores) != env.num_envs:
        raise ValueError("Task reward/score result must contain one value per environment")
    reward_values = [finite_scalar(value, "reward") for value in rewards]
    score_values = [finite_scalar(value, "score") for value in scores]
    after = simulation_snapshot(env.sim.sim)
    if after != before:
        raise RuntimeError(f"Task runtime validation advanced physics: {before} -> {after}")
    _trace("completed", elapsed_seconds=perf_counter() - started)
    return {
        "instruction": {"env_id": 0, "templates": templates},
        "reward": {
            "executed": True,
            "registered": True,
            "step_executed": True,
            "score_registered": score_registered,
            "reward_values": reward_values,
            "score_values": score_values,
            "predicate_count": len(extra),
            "predicate_evaluations": extra,
            "predicate_evaluation_mode": "isolated_extra_calls_before_normal_reward_step",
            "parser_state_restored_between_extra_calls": True,
            "harness_success_is_episode_validity": True,
        },
        "physics_step_before": before[1],
        "physics_step_after": after[1],
    }
