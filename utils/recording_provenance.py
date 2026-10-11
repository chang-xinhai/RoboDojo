"""Read actual capture settings and source versions without mutating runtime."""

import hashlib
from importlib import metadata
from pathlib import Path
import subprocess

SETTINGS = (
    "/isaaclab/render/rtx_sensors",
    "/app/renderer/resolution/width",
    "/app/renderer/resolution/height",
    "/rtx/post/aa/op",
    "/rtx/rendermode",
    "/rtx-transient/resourcemanager/enableTextureStreaming",
    "/physics/fabricUseGPUInterop",
    "/isaaclab/robodojo/surface_fabric_cpu_copy",
    "/physics/fabricUpdateTransformations",
    "/physics/fabricEnableCloth",
    "/physics/fabricEnableClothVisualization",
    "/physics/suppressReadback",
    "/physics/disableFabric",
)


def capture_provenance(root):
    root = Path(root)
    result = {"packages": {}}
    try:
        result["source_commit"] = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        listed = (
            subprocess.check_output(
                ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"]
            )
            .decode()
            .split("\0")
        )
        result["source_file_sha256"] = {}
        for name in sorted(set(listed)):
            path = root / name
            if path.suffix in (".py", ".yml", ".yaml", ".toml") and path.is_file():
                result["source_file_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        tree = "\n".join(f"{name}:{digest}" for name, digest in result["source_file_sha256"].items())
        result["source_tree_sha256"] = hashlib.sha256(tree.encode()).hexdigest()
        result["source_dirty"] = bool(
            subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip()
        )
    except (OSError, subprocess.CalledProcessError) as error:
        result["source_unavailable_reason"] = str(error)
    for package in ("isaacsim", "isaaclab", "h5py", "numpy"):
        try:
            result["packages"][package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result["packages"][package] = None
    try:
        import carb

        settings = carb.settings.get_settings()
        result["runtime_render_settings"] = {key: settings.get(key) for key in SETTINGS}
        result["runtime_render_settings_source"] = "actual carb.settings queried after scene initialization"
    except (ImportError, AttributeError) as error:
        result["runtime_render_settings"] = None
        result["runtime_render_settings_unavailable_reason"] = str(error)
    lock = root / "reproduction/isaac61/source-lock.json"
    if lock.exists():
        import json

        result["source_lock"] = json.loads(lock.read_text())
    return result
