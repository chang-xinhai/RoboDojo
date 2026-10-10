#!/usr/bin/env python3
"""Run fresh Isaac Sim 6.1 processes for every task's scene/reset/step/RGBD smoke.

Sequential execution avoids competing Kit processes on one GPU. Reports are
accepted only with zero process exit plus a fresh smoke JSON containing all three
RGBD cameras. Resume requires the same commit, source contents, runtime packages,
Lab source pin and configuration. This does not evaluate a robot policy or prove
TacSL / structured-light operation (their collectors have separate acceptance).
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "robodojo.task-scene-smoke.v1"
CAMERAS = {"head", "left_wrist", "right_wrist"}
REQUIRED_CHECKS = {"task scene initialization", "robot reset", "RGBD observation capture"}
CHILD_LAUNCH = (
    "source reproduction/isaac61/environment.sh; "
    "exec bash reproduction/isaac61/run.sh python reproduction/smoke_task.py "
    '--viz none --kit_args "$ROBODOJO_KIT_ARGS" "$@"'
)


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", *args], cwd=cwd).decode().strip()


def source_identity(root=ROOT):
    """Hash tracked/untracked owned files and actual submodule source changes."""
    digest = hashlib.sha256()
    paths = (
        subprocess.check_output(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=root)
        .decode()
        .split("\0")
    )
    for relative in sorted(set(paths) - {""}):
        path = root / relative
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
        raise ValueError("Missing scene/reset/RGBD acceptance checks")
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


def can_resume(record, context, task, env_cfg):
    if not record or record.get("status") != "passed" or record.get("context") != context:
        return False
    if not is_sim61(context.get("packages", {}).get("isaacsim")):
        return False
    try:
        check_evidence(record.get("evidence", {}), task, env_cfg)
    except (ValueError, TypeError, AttributeError):
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
        "acceptance": "real scene initialization, reset, physics steps and all three RGBD camera captures",
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
