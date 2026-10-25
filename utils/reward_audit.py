"""Observe original reward calls exactly once; never replay stateful predicates."""

from pathlib import Path
from copy import deepcopy
import json

from utils.isaaclab_compat import simulation_snapshot


def value(item):
    if isinstance(item, dict):
        return {str(k): value(v) for k, v in item.items()}
    if isinstance(item, (tuple, list)):
        return [value(v) for v in item]
    if hasattr(item, "tolist"):
        return value(item.tolist())
    if item is None or isinstance(item, (str, bool, int, float)):
        return item
    return repr(item)


class OfficialRewardAudit:
    def __init__(self, env, output=None):
        self.env = env
        self.manager = env.reward_manager
        self.original = self.manager.call_func_parser
        self.registered_groups = deepcopy(self.manager.check_list[0])
        self.consumed = []
        self.calls = []
        self.last_failure = None
        self.phase = "normal_reward"
        self.step_number = 0
        self.path = Path(output) / "official-reward-events.jsonl" if output else None
        self.manager.call_func_parser = self.observe
        self.write({"event": "registered", "groups": self.registered_groups})

    def clock(self):
        t, f = simulation_snapshot(self.env.sim.sim)
        return dict(timestamp_ns=round(t * 1_000_000_000.0), physics_frame=f)

    def write(self, event):
        event = value({**self.clock(), **event})
        if self.path:
            try:
                with self.path.open("a") as stream:
                    stream.write(json.dumps(event, allow_nan=False) + "\n")
            except OSError:
                pass

    def observe(self, predicate, env_index):
        description = deepcopy(predicate)
        raw = self.original(predicate, env_index)
        event = dict(
            event="original_predicate_return",
            phase=self.phase,
            env_index=env_index,
            predicate=description,
            raw_result=value(raw),
            step=self.step_number,
        )
        if self.phase == "normal_reward_step" and env_index == 0:
            self.calls.append(value({**self.clock(), **event}))
            if not bool(raw):
                self.last_failure = self.calls[-1]
        self.write(event)
        return raw

    def step(self):
        before = len(self.manager.check_list[0])
        self.calls = []
        self.step_number += 1
        self.phase = "normal_reward_step"
        try:
            self.manager.step(env_idx_list=[0])
            self.phase = "normal_reward"
            after = len(self.manager.check_list[0])
            if after not in (before, before - 1):
                raise RuntimeError("Unexpected official queue transition")
            if after < before:
                index = len(self.consumed)
                event = dict(
                    event="original_group_consumed",
                    registered_group_index=index,
                    group=self.registered_groups[index],
                    original_returns=self.calls.copy(),
                    remaining_groups=after,
                    **self.clock(),
                )
                self.consumed.append(value(event))
                self.last_failure = None
                self.write(event)
        except Exception:
            self.phase = "normal_reward"
            raise

    def summary(self):
        return dict(
            kind="observed_original_ordered_reward_execution",
            registered_group_count=len(self.registered_groups),
            consumed_group_count=len(self.consumed),
            remaining_group_count=len(self.manager.check_list[0]),
            consumed_groups=deepcopy(self.consumed),
            first_remaining_group=value(self.manager.check_list[0][0]) if self.manager.check_list[0] else None,
            latest_original_failure=deepcopy(self.last_failure),
            event_path=str(self.path) if self.path else None,
            predicates_reexecuted_for_history=False,
        )
