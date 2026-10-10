"""Import audit must preserve failures and request a nonzero Kit exit."""

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from reproduction.isaac61.validate_migration import validate_runtime_imports


@pytest.fixture
def app(monkeypatch):
    closes = []
    instance = SimpleNamespace(close=lambda **kwargs: closes.append(kwargs))
    module = ModuleType("isaaclab.app")
    module.AppLauncher = lambda args: SimpleNamespace(app=instance)
    monkeypatch.setitem(sys.modules, "isaaclab.app", module)
    return closes


def report():
    return {
        "issues": [],
        "inventory": {"tasks": [{"name": "real_task", "module": "task.py", "runtime_import": "not_run"}]},
    }


def test_failed_extension_bootstrap_is_reported_before_task_import(monkeypatch, app):
    def bootstrap():
        raise RuntimeError("Required camera extension failed")

    monkeypatch.setattr("utils.isaaclab_compat.enable_legacy_isaacsim_extensions", bootstrap)
    result = report()
    validate_runtime_imports(result, SimpleNamespace())
    assert result["runtime_import_requested"] is True
    assert result["inventory"]["tasks"][0]["runtime_import"] == "not_run"
    assert len(result["issues"]) == 1
    assert result["issues"][0]["check"] == "runtime_bootstrap"
    assert "Required camera extension failed" in result["issues"][0]["error"]
    assert app[0]["exit_code"] == 1


@pytest.mark.parametrize("valid", [True, False])
def test_registry_class_contract_controls_task_status_and_exit(monkeypatch, app, valid):
    monkeypatch.setattr("utils.isaaclab_compat.enable_legacy_isaacsim_extensions", lambda: None)
    module = ModuleType("task.RoboDojo.task_registry")
    module.load_task_class = lambda name: (name, type("Task", (), {}) if valid else object())
    monkeypatch.setitem(sys.modules, "task.RoboDojo.task_registry", module)
    result = report()
    validate_runtime_imports(result, SimpleNamespace())
    assert result["inventory"]["tasks"][0]["runtime_import"] == ("passed" if valid else "failed")
    assert app[0]["exit_code"] == int(not valid)
    if not valid:
        assert result["issues"][0]["check"] == "runtime_import"


def test_failure_artifact_exists_before_kit_fast_exit(monkeypatch, tmp_path):
    output = tmp_path / "audit.json"
    output.write_text('{"status":"passed","stale":true}')

    def bootstrap():
        raise RuntimeError("Real extension init failed")

    def close(**kwargs):
        saved = json.loads(output.read_text())
        assert saved["status"] == "failed"
        assert "stale" not in saved
        assert saved["issues"][0]["check"] == "runtime_bootstrap"
        raise SystemExit(kwargs["exit_code"])

    module = ModuleType("isaaclab.app")
    module.AppLauncher = lambda args: SimpleNamespace(app=SimpleNamespace(close=close))
    monkeypatch.setitem(sys.modules, "isaaclab.app", module)
    monkeypatch.setattr("utils.isaaclab_compat.enable_legacy_isaacsim_extensions", bootstrap)
    with pytest.raises(SystemExit) as exc:
        validate_runtime_imports(report(), SimpleNamespace(output=output))
    assert exc.value.code == 1
