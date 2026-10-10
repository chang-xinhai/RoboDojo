#!/usr/bin/env python3
"""Audit all first-party syntax/configs and optionally import every task in Kit.

Static checks do not claim physics or sensor validation. --runtime-import launches
the installed Isaac Lab app and exercises actual imports (no stubs); it does not
instantiate all tasks. Use smoke_task.py for task scene/step acceptance.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def load_inventory():
    path = ROOT / "scripts/internal/task_inventory.py"
    spec = importlib.util.spec_from_file_location("migration_task_inventory", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_inventory()


def validate_static():
    import yaml

    issues = []
    python_files = []
    config_files = []
    for directory in ("env", "env_cfg", "task", "utils", "src", "scripts", "reproduction", "tests"):
        python_files.extend(sorted((ROOT / directory).rglob("*.py")))
    for path in python_files:
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except (SyntaxError, UnicodeError) as exc:
            issues.append({"path": str(path.relative_to(ROOT)), "check": "python_syntax", "error": str(exc)})
    for directory in ("env_cfg", "task/RoboDojo/config"):
        for suffix in ("*.yml", "*.yaml"):
            config_files.extend(sorted((ROOT / directory).rglob(suffix)))
    for path in config_files:
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(document, dict):
                raise ValueError("Expected YAML mapping")
        except (yaml.YAMLError, UnicodeError, ValueError) as exc:
            issues.append({"path": str(path.relative_to(ROOT)), "check": "yaml_parse", "error": str(exc)})

    inventory = load_inventory()
    for task in inventory["tasks"]:
        if not task["runnable"] or not task["dimension"]:
            issues.append(
                {"path": task["module"], "check": "task_inventory", "error": "Missing class/config/dimension"}
            )
        task["syntax"] = "passed" if not any(issue["path"] == task["module"] for issue in issues) else "failed"
        task["runtime_import"] = "not_run"
        task["runtime_scene_step"] = "not_run"
    return {
        "schema_version": 1,
        "scope": "first-party files; dependencies/assets excluded",
        "python_files_checked": len(python_files),
        "yaml_files_checked": len(config_files),
        "inventory": inventory,
        "issues": issues,
        "runtime_import_requested": False,
    }


def validate_runtime_imports(report, args):
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(args)
    app = launcher.app
    report["runtime_import_requested"] = True
    try:
        from task.RoboDojo.task_registry import load_task_class

        for task in report["inventory"]["tasks"]:
            try:
                name, cls = load_task_class(task["name"])
                if name != task["name"] or not isinstance(cls, type):
                    raise TypeError("Registry did not return the exported task class")
                task["runtime_import"] = "passed"
            except Exception:
                task["runtime_import"] = "failed"
                report["issues"].append(
                    {"path": task["module"], "check": "runtime_import", "error": traceback.format_exc()}
                )
    finally:
        app.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-import", action="store_true", help="Launch Kit and import every registered task")
    parser.add_argument("--output", type=Path, help="Write JSON audit report to this file")
    # Register Lab CLI only for runtime mode so static validation works without Sim.
    if "--runtime-import" in sys.argv:
        from isaaclab.app import AppLauncher

        from utils.isaaclab_compat import add_app_launcher_args

        add_app_launcher_args(parser, AppLauncher)
    args = parser.parse_args()
    report = validate_static()
    if args.runtime_import:
        validate_runtime_imports(report, args)
    report["status"] = "passed" if not report["issues"] else "failed"
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "python_files": report["python_files_checked"],
                "yaml_files": report["yaml_files_checked"],
                "tasks": report["inventory"]["counts"],
                "runtime_imports": sum(task["runtime_import"] == "passed" for task in report["inventory"]["tasks"]),
                "issues": report["issues"],
            },
            indent=2,
        )
    )
    return int(bool(report["issues"]))


if __name__ == "__main__":
    raise SystemExit(main())
