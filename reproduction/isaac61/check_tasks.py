#!/usr/bin/env python3
"""Run fresh Isaac Sim 6.1 processes for each task's scene and runtime APIs.

Sequential execution avoids competing Kit processes on one GPU. Reports are
accepted only with zero process exit plus a fresh smoke JSON containing all three
RGBD cameras, generated instructions and executed reward predicates. Resume
requires the same commit, source contents, runtime packages,
Lab source pin and configuration. This does not evaluate a robot policy or prove
TacSL / structured-light operation (their collectors have separate acceptance).
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "robodojo.task-scene-smoke.v4"
NATIVE_LOG_POLICY = "physx-known-fatal.v1"
PHYSICS_LOG_LINE = re.compile(
    r"\[(?P<level>Error|Fatal|Critical|Warning)\]\s+"
    r"\[(?P<module>omni\.(?:physx|physics)(?:\.[^\]]+)?)\]\s+(?P<message>.*)",
    re.IGNORECASE,
)
CAMERAS = {"head", "left_wrist", "right_wrist"}
REQUIRED_CHECKS = {
    "task scene initialization",
    "robot reset",
    "RGBD observation capture",
    "task instruction generation",
    "task reward evaluation",
}
OBJECT_STORAGES = {
    "Rigid": "rigid_and_dynamic_objects",
    "Dynamic": "rigid_and_dynamic_objects",
    "Articulation": "articulation_objects",
    "Garment": "garment_objects",
    "Geometry": "geometry_objects",
    "Fluid": "fluid_objects",
}
PLACEMENT_SOURCES = {
    "Rigid": "native_body_pose",
    "Articulation": "native_body_pose",
    "Dynamic": "live_mesh_vertices",
    "Garment": "native_surface_nodes",
    "Geometry": "static_geometry_pose",
    "Fluid": "native_pbd_particle_readback",
}
MAX_LAYOUT_DISTANCE_M = 5.0
GENERATED_REPORTS = {
    "reproduction/doctor.json",
    "reproduction/runtime-smoke.json",
    "reproduction/task-smoke.json",
    "reproduction/task-inventory.json",
}
CHILD_LAUNCH = (
    "source reproduction/isaac61/environment.sh; "
    "exec bash reproduction/isaac61/run.sh python reproduction/smoke_task.py "
    '--viz none --kit_args "$ROBODOJO_KIT_ARGS" "$@"'
)


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", *args], cwd=cwd).decode().strip()


def source_identity(root=ROOT):
    """Hash runtime sources, excluding generated logs and reproduction evidence."""
    digest = hashlib.sha256()
    paths = (
        subprocess.check_output(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=root)
        .decode()
        .split("\0")
    )
    for relative in sorted(set(paths) - {""}):
        path = root / relative
        if path.suffix == ".log" or relative in GENERATED_REPORTS:
            continue
        digest.update(relative.encode())
        if path.is_file():
            digest.update(path.read_bytes())
        elif path.is_dir() and (path / ".git").exists():
            digest.update(git("rev-parse", "HEAD", cwd=path).encode())
            digest.update(subprocess.check_output(["git", "diff", "HEAD", "--binary"], cwd=path))
        else:
            digest.update(b"<missing>")
    return {
        "git_commit": git("rev-parse", "HEAD", cwd=root),
        "source_sha256": digest.hexdigest(),
        "isaaclab_source_commit": git("rev-parse", "HEAD", cwd=root / "third_party/IsaacLab3"),
    }


def runtime_packages(environment):
    program = (
        "import importlib.metadata as m,json; "
        "print(json.dumps({n:m.version(n) for n in "
        "('isaacsim','isaaclab','torch','numpy','warp-lang')}))"
    )
    result = subprocess.run(
        [str(environment / "bin/python"), "-c", program], capture_output=True, text=True, timeout=30, check=True
    )
    return json.loads(result.stdout)


def is_sim61(version):
    return str(version).split(".")[:3] == ["6", "1", "0"]


def inventory_tasks():
    path = ROOT / "scripts/internal/task_inventory.py"
    spec = importlib.util.spec_from_file_location("task_smoke_inventory", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return [task["name"] for task in module.build_inventory()["tasks"] if task["runnable"]]


def finite_vector(value, field):
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(
            not isinstance(item, (float, int)) or isinstance(item, bool) or not math.isfinite(item) for item in value
        )
    ):
        raise ValueError(f"Missing finite three-dimensional placement {field}")
    return value


def vector_distance(left, right):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def check_placement_evidence(scene, source_layout, wrappers):
    """Check real-state geometry and saved positions instead of trusting counts."""
    if scene.get("placement_verified") is not True:
        raise ValueError("Task objects require verified live placement after reset")
    origin = finite_vector(scene.get("env_origin"), "environment origin")
    placements = scene.get("live_placements", {})
    if set(placements) != set(wrappers):
        raise ValueError("Every actual task object must have a live placement")
    expected = []
    for physical_type in OBJECT_STORAGES:
        for category, instances in source_layout.get(physical_type, {}).items():
            for instance in instances:
                position = finite_vector(instance.get("default_pos"), "saved layout position")
                expected.append((physical_type, f"{category}_{instance.get('category_idx')}_", position))
    unmatched = list(expected)
    for path, wrapper in wrappers.items():
        placement = placements[path]
        physical_type = wrapper["physical_type"]
        if placement.get("state_source") != PLACEMENT_SOURCES[physical_type]:
            raise ValueError(f"Incorrect live placement state source at {path}")
        world = finite_vector(placement.get("world_center"), "world center")
        relative = finite_vector(placement.get("relative_center"), "relative center")
        saved = finite_vector(placement.get("saved_relative_position"), "saved position")
        if vector_distance([value - offset for value, offset in zip(world, origin)], relative) > 1e-4:
            raise ValueError(f"Live world/environment placement frames disagree at {path}")
        maximum = placement.get("max_distance_from_env_origin_m")
        if not isinstance(maximum, (float, int)) or not math.isfinite(maximum) or not 0 <= maximum <= 100:
            raise ValueError(f"Task object is offscreen/unplaced at {path}")
        if maximum + 1e-4 < vector_distance(relative, [0, 0, 0]):
            raise ValueError(f"Live placement maximum is smaller than its center distance at {path}")
        measured = vector_distance(relative, saved)
        reported = placement.get("center_distance_from_saved_position_m")
        if not isinstance(reported, (float, int)) or not math.isfinite(reported) or abs(reported - measured) > 1e-4:
            raise ValueError(f"Saved layout displacement evidence disagrees at {path}")
        if measured > MAX_LAYOUT_DISTANCE_M:
            raise ValueError(f"Task object is too far from its saved layout position at {path}")
        match = next(
            (
                index
                for index, (kind, prefix, position) in enumerate(unmatched)
                if kind == physical_type
                and wrapper["instance_name"].startswith(prefix)
                and vector_distance(position, saved) < 1e-4
            ),
            None,
        )
        if match is not None:
            unmatched.pop(match)
        elif any(
            kind == physical_type and wrapper["instance_name"].startswith(prefix) for kind, prefix, _ in expected
        ) and not any(
            kind == physical_type
            and wrapper["instance_name"].startswith(prefix)
            and vector_distance(position, saved) < 1e-4
            for kind, prefix, position in expected
        ):
            raise ValueError(f"Live object's saved position does not belong to the canonical layout at {path}")
    if unmatched:
        raise ValueError("Live placements do not cover all objects in the saved task layout")


def check_scene_evidence(evidence, task):
    """Require the recorded layout bytes and live objects, independent of status."""
    layout, scene = evidence.get("layout", {}), evidence.get("scene", {})
    if layout.get("task") != task or not layout.get("file") or scene.get("env_id") != 0:
        raise ValueError("Missing task layout provenance or live scene environment")
    path = Path(layout["file"])
    if not path.is_absolute():
        raise ValueError("Task layout provenance must contain an absolute file path")
    raw = path.read_bytes()
    if layout.get("sha256") != hashlib.sha256(raw).hexdigest():
        raise ValueError("Saved task layout SHA256 differs from current layout bytes")
    source_layout = json.loads(raw)
    expected = {
        category: sum(len(instances) for instances in source_layout.get(category, {}).values())
        for category in OBJECT_STORAGES
    }
    if not sum(expected.values()):
        raise ValueError("Task layout is empty; fixtures cannot certify task scene loading")
    if layout.get("expected_object_counts") != expected or scene.get("expected_object_counts") != expected:
        raise ValueError("Reported expected object counts differ from the actual saved layout")
    storage_counts = dict.fromkeys(OBJECT_STORAGES.values(), 0)
    observed_types = dict.fromkeys(OBJECT_STORAGES, 0)
    observed_paths = []
    wrappers = scene.get("wrapper_types", {})
    for category, storage in OBJECT_STORAGES.items():
        storage_counts[storage] += expected[category]
    for storage, required in storage_counts.items():
        actual = scene.get("actual_object_counts", {}).get(storage)
        paths = scene.get("object_prim_paths", {}).get(storage)
        if not isinstance(actual, int) or isinstance(actual, bool) or actual < required:
            raise ValueError(f"Missing live {storage}: expected at least {required}, received {actual}")
        if not isinstance(paths, list) or len(paths) != actual or len(set(paths)) != actual:
            raise ValueError(f"Live {storage} counts do not match unique USD prim paths")
        if any(not isinstance(prim, str) or not prim.startswith("/World/envs/env_0/") for prim in paths):
            raise ValueError(f"Task {storage} prim path is outside the accepted environment")
        for prim in paths:
            wrapper = wrappers.get(prim, {})
            physical_type = wrapper.get("physical_type")
            if physical_type not in OBJECT_STORAGES or OBJECT_STORAGES[physical_type] != storage:
                raise ValueError(f"Unrecognized or wrongly registered physical wrapper at {prim}")
            if not wrapper.get("class") or not wrapper.get("instance_name"):
                raise ValueError(f"Missing actual physics wrapper class or instance at {prim}")
            if wrapper.get("layout_type") != physical_type.lower():
                raise ValueError(f"Saved physical type differs from live wrapper at {prim}")
            observed_types[physical_type] += 1
        observed_paths.extend(paths)
    if set(wrappers) != set(observed_paths) or len(set(observed_paths)) != len(observed_paths):
        raise ValueError("Reported physics wrappers must match unique live object prim paths")
    if scene.get("actual_physical_type_counts") != observed_types:
        raise ValueError("Reported physical object counts disagree with actual wrapper inventory")
    if any(observed_types[category] < required for category, required in expected.items()):
        raise ValueError("Shared storage counts cannot certify the requested physical object types")
    check_placement_evidence(scene, source_layout, wrappers)


def check_task_runtime(evidence):
    """Require task API execution without requiring a successful robot policy."""
    runtime = evidence.get("task_runtime", {})
    instruction = runtime.get("instruction", {})
    templates = instruction.get("templates")
    if (
        instruction.get("env_id") != 0
        or isinstance(instruction.get("env_id"), bool)
        or not isinstance(templates, list)
        or not templates
        or any(not isinstance(template, str) or not template.strip() for template in templates)
    ):
        raise ValueError("Missing actual task instruction generation for environment zero")
    reward = runtime.get("reward", {})
    if any(reward.get(flag) is not True for flag in ("executed", "registered", "step_executed")):
        raise ValueError("Task rewards must be registered and actually executed through a reward step")
    if (
        reward.get("harness_success_is_episode_validity") is not True
        or reward.get("parser_state_restored_between_extra_calls") is not True
        or reward.get("predicate_evaluation_mode") != "isolated_extra_calls_before_normal_reward_step"
    ):
        raise ValueError("Task reward diagnostics must preserve parser state and distinguish episode validity")
    before, after = runtime.get("physics_step_before"), runtime.get("physics_step_after")
    if (
        any(not isinstance(step, int) or isinstance(step, bool) or step < 0 for step in (before, after))
        or before != after
    ):
        raise ValueError("Task API evaluation requires the same frozen physics step before and after")

    def finite_number(value):
        return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)

    for field in ("reward_values", "score_values"):
        values = reward.get(field)
        if not isinstance(values, list) or len(values) != 1 or not finite_number(values[0]):
            raise ValueError(f"Missing finite one-environment task {field}")
    predicates = reward.get("predicate_evaluations")
    count = reward.get("predicate_count")
    if (
        not isinstance(count, int)
        or isinstance(count, bool)
        or count < 1
        or not isinstance(predicates, list)
        or len(predicates) != count
    ):
        raise ValueError("Every registered task predicate requires actual evaluation evidence")
    for predicate in predicates:
        if (
            not isinstance(predicate, dict)
            or predicate.get("env_id") != 0
            or isinstance(predicate.get("env_id"), bool)
            or not isinstance(predicate.get("name"), str)
            or not predicate["name"].strip()
            or not finite_number(predicate.get("value"))
        ):
            raise ValueError("Task predicate evaluation must identify its environment, name and finite result")


def check_evidence(evidence, task, env_cfg):
    """Validate smoke output structure independently of its status string."""
    if evidence.get("status") != "passed" or evidence.get("task") != task or evidence.get("env_cfg") != env_cfg:
        raise ValueError("Smoke status/task/config does not match this invocation")
    if evidence.get("num_envs") != 1 or evidence.get("policy_evaluation") is not False:
        raise ValueError("Expected one-environment scene smoke without policy evaluation")
    if evidence.get("real_runtime") is not True or evidence.get("physics_steps", 0) < 10:
        raise ValueError("Expected actual runtime evidence with at least ten physics steps")
    if not is_sim61(evidence.get("packages", {}).get("isaacsim")):
        raise ValueError("Smoke must identify actual Isaac Sim 6.1.0 runtime")
    if not evidence.get("isaaclab_source_commit"):
        raise ValueError("Smoke must identify the Isaac Lab source commit")
    if not REQUIRED_CHECKS.issubset(evidence.get("checks", [])):
        raise ValueError("Missing scene/reset/RGBD/instruction/reward acceptance checks")
    check_task_runtime(evidence)
    check_scene_evidence(evidence, task)
    cameras = {name.removeprefix("cam_"): shape for name, shape in evidence.get("camera_shapes", {}).items()}
    if set(cameras) != CAMERAS:
        raise ValueError(f"Expected head and both wrist RGBD cameras, received {sorted(cameras)}")
    for name, fields in cameras.items():
        rgb, depth = fields.get("rgb", fields.get("color", [])), fields.get("depth", [])
        if len(rgb) != 3 or rgb[2] != 3 or min(rgb[:2]) < 1:
            raise ValueError(f"Invalid {name} RGB shape: {rgb}")
        if depth != rgb[:2]:
            raise ValueError(f"Invalid {name} depth shape: {depth}")
    return evidence


def known_native_error(line):
    """Identify physics data loss or invalid native bodies, including warnings.

    Renderer normal-adjacency errors and supported CPU collision fallback
    warnings belong to separate diagnostics, not this physics deny policy.
    """
    match = PHYSICS_LOG_LINE.search(line)
    if match is None:
        return None
    level, message = match["level"].lower(), match["message"].lower()
    if "buffer overflow" in message or "capacity exceeded" in message or "contacts dropped" in message:
        return "physics_capacity_or_data_loss"
    if "pxvalidatetrianglemesh for pxdeformablesurface failed" in message:
        return "invalid_surface_mesh"
    if level in {"error", "fatal", "critical"}:
        if "physx error:" in message:
            return "physx_error"
        if "assertion failed" in message or "assertion failure" in message or "assert failed" in message:
            return "native_assertion"
        if "did not match any" in message and any(
            kind in message for kind in ("deformable bod", "rigid bod", "articulation", "particle")
        ):
            return "missing_native_body"
    return None


def audit_native_log(path):
    """Bind acceptance to the real process log and retain fatal line evidence."""
    raw = Path(path).read_bytes()
    errors = []
    for number, line in enumerate(raw.decode("utf-8", errors="replace").splitlines(), 1):
        kind = known_native_error(line)
        if kind:
            errors.append({"line_number": number, "kind": kind, "message": line})
    return {
        "policy": NATIVE_LOG_POLICY,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "known_fatal_count": len(errors),
        "errors": errors,
    }


def can_resume(record, context, task, env_cfg):
    if not record or record.get("status") != "passed" or record.get("context") != context:
        return False
    if not is_sim61(context.get("packages", {}).get("isaacsim")):
        return False
    try:
        check_evidence(record.get("evidence", {}), task, env_cfg)
        audit = audit_native_log(record.get("log", ""))
        if audit["known_fatal_count"] or record.get("native_log_validation") != audit:
            return False
    except (OSError, ValueError, TypeError, AttributeError):
        return False
    return (
        record.get("process_exit_code") == 0
        and record.get("real_runtime") is True
        and record["evidence"]["isaaclab_source_commit"] == context.get("isaaclab_source_commit")
        and all(
            context.get("packages", {}).get(name) == version for name, version in record["evidence"]["packages"].items()
        )
    )


def execute_process(command, log_path, timeout, *, environment=None, cwd=ROOT):
    """Kill the entire Kit process group on timeout, then reap the parent."""
    started = time.monotonic()
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, env=environment, start_new_session=True
        )
        try:
            returncode = process.wait(timeout=timeout)
            timed_out = False
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            returncode = process.wait()
            timed_out = True
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    return returncode, timed_out, round(time.monotonic() - started, 3)


def run_task(task, args, context):
    output = args.output_dir / f"{task}.json"
    log = args.output_dir / f"{task}.log"
    # A previous report cannot satisfy a fresh attempt after Kit exits early.
    output.unlink(missing_ok=True)
    command = [
        "bash",
        "-c",
        CHILD_LAUNCH,
        "check_tasks",
        "--task",
        task,
        "--env-cfg",
        args.env_cfg,
        "--output",
        str(output),
    ]
    child_environment = dict(os.environ, ROBODOJO_ISAAC61_ENV=str(args.environment))
    exit_code, timed_out, duration = execute_process(command, log, args.timeout, environment=child_environment)
    record = {
        "task": task,
        "status": "failed",
        "context": context,
        "real_runtime": True,
        "process_exit_code": exit_code,
        "timed_out": timed_out,
        "duration_s": duration,
        "log": str(log),
        "output": str(output),
        "finished_at_utc": datetime.now(UTC).isoformat(),
    }
    if timed_out:
        record["error"] = f"Fresh task process exceeded {args.timeout:g}s timeout"
    elif exit_code:
        record["error"] = f"Fresh task process exited {exit_code}; see log"
    else:
        try:
            record["evidence"] = check_evidence(json.loads(output.read_text()), task, args.env_cfg)
            if record["evidence"]["isaaclab_source_commit"] != context["isaaclab_source_commit"]:
                raise ValueError("Task used a different Isaac Lab source commit")
            if any(
                context["packages"].get(name) != version for name, version in record["evidence"]["packages"].items()
            ):
                raise ValueError("Task used different runtime package versions")
            record["native_log_validation"] = audit_native_log(log)
            if record["native_log_validation"]["known_fatal_count"]:
                first = record["native_log_validation"]["errors"][0]
                raise ValueError(
                    f"Native physics failure in process log at line {first['line_number']}: {first['message']}"
                )
            record["status"] = "passed"
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            record["error"] = f"Missing or invalid fresh runtime evidence: {exc}"
    return record


def save_summary(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", help="Optional subset; accepts whitespace or comma-separated names")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout", type=float, default=180, help="Maximum seconds per fresh task process")
    parser.add_argument("--env-cfg", default="arx_x5_rgbd")
    parser.add_argument(
        "--environment", type=Path, default=Path(os.environ.get("ROBODOJO_ISAAC61_ENV", ROOT / ".venv-isaac61"))
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reproduction/isaac61/outputs/task-suite")
    parser.add_argument("--list", action="store_true", help="Print selected task inventory without starting Kit")
    args = parser.parse_args()
    all_tasks = inventory_tasks()
    selected = (
        list(dict.fromkeys(name for value in args.tasks for name in value.split(","))) if args.tasks else all_tasks
    )
    if set(selected) - set(all_tasks):
        parser.error(f"Unknown or non-runnable tasks: {sorted(set(selected) - set(all_tasks))}")
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    if args.list:
        print("\n".join(selected))
        return 0
    args.environment, args.output_dir = args.environment.expanduser().resolve(), args.output_dir.expanduser().resolve()
    packages = runtime_packages(args.environment)
    if not is_sim61(packages["isaacsim"]):
        parser.error(f"Expected installed Isaac Sim 6.1.0, received {packages['isaacsim']}")
    context = {
        "schema": SCHEMA,
        **source_identity(),
        "packages": packages,
        "environment": str(args.environment),
        "env_cfg": args.env_cfg,
        "timeout_s": args.timeout,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    previous = json.loads(summary_path.read_text()) if args.resume and summary_path.exists() else {}
    records = {record["task"]: record for record in previous.get("tasks", [])}
    report = {
        "schema": SCHEMA,
        "context": context,
        "selected_tasks": selected,
        "tasks": [],
        "acceptance": (
            "real scene initialization, reset, physics steps, all three RGBD camera captures, "
            "task instructions, reward predicates and no known fatal native physics log signals"
        ),
        "policy_evaluation": False,
        "status": "running",
    }
    for index, task in enumerate(selected, 1):
        identity = source_identity()
        if any(context[key] != value for key, value in identity.items()):
            report["status"] = "source_changed"
            report["complete"] = False
            report["error"] = "Source changed during the suite; restart against the current revision"
            save_summary(summary_path, report)
            print(json.dumps({"status": report["status"], "error": report["error"]}), flush=True)
            return 2
        if args.resume and can_resume(records.get(task), context, task, args.env_cfg):
            record = dict(records[task], resumed=True)
        else:
            record = run_task(task, args, context)
            identity = source_identity()
            if any(context[key] != value for key, value in identity.items()):
                record["status"] = "failed"
                record["error"] = (
                    "Source changed while the task ran; runtime evidence cannot certify the original revision"
                )
        report["tasks"].append(record)
        report["passed"] = sum(entry["status"] == "passed" for entry in report["tasks"])
        report["failed"] = len(report["tasks"]) - report["passed"]
        save_summary(summary_path, report)
        print(
            json.dumps(
                {
                    "index": index,
                    "total": len(selected),
                    "task": task,
                    "status": record["status"],
                    "resumed": record.get("resumed", False),
                    "log": record["log"],
                }
            ),
            flush=True,
        )
    report["status"] = "passed" if not report["failed"] else "failed"
    report["complete"] = True
    save_summary(summary_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "passed": report["passed"],
                "failed": report["failed"],
                "summary": str(summary_path),
            }
        ),
        flush=True,
    )
    return int(bool(report["failed"]))


if __name__ == "__main__":
    raise SystemExit(main())
