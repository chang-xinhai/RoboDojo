"""Stage the official UniVTAC insert_hole assets under an isolated RoboDojo asset root."""
from __future__ import annotations
import json, os, shutil
from pathlib import Path

OFFICIAL = Path(os.environ.get("UNIVTAC_ROOT", "/mnt/robo/yutong/vendor/UniVTAC-official"))
SHARED = Path(os.environ.get("ROBODOJO_SHARED_ASSETS", "/mnt/robo/yutong/assets/Assets"))
STAGING = Path(os.environ.get("ROBODOJO_INSERT_HOLE_ASSET_ROOT", "/mnt/robo/yutong/tmp/robodojo-insert-hole-assets"))

def _link(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src, target_is_directory=src.is_dir())

def _stage_robot_assets() -> None:
    robots = STAGING / "Robots"
    if robots.is_symlink():
        robots.unlink()
    robots.mkdir(parents=True, exist_ok=True)
    _link(SHARED / "Robots" / "franka", robots / "franka")
    shared_x5 = SHARED / "Robots" / "x5"
    staged_x5 = robots / "x5"
    if staged_x5.is_symlink():
        staged_x5.unlink()
    staged_x5.mkdir(parents=True, exist_ok=True)
    for entry in shared_x5.iterdir():
        if entry.name == "curobo.yml":
            continue
        _link(entry, staged_x5 / entry.name)
    source = shared_x5 / "curobo_tmp.yml"
    if not source.exists():
        raise FileNotFoundError(source)
    text = source.read_text()
    text = text.replace("${ASSETS_PATH}/Assets/Robots/x5", str(staged_x5))
    text = text.replace("./Assets/Robots/x5", str(staged_x5))
    (staged_x5 / "curobo.yml").write_text(text)

def stage_assets() -> Path:
    STAGING.mkdir(parents=True, exist_ok=True)
    for name in ("Background", "Eval_Layout", "Material", "Room", "Sensor"):
        _link(SHARED / name, STAGING / name)
    _stage_robot_assets()
    objects = STAGING / "Object" / "RoboDojo"
    objects.mkdir(parents=True, exist_ok=True)
    for name in ("Articulation", "Clutter", "Dynamic", "Fluid", "Garment"):
        _link(SHARED / "Object" / "RoboDojo" / name, objects / name)
    entries = (
        ("Rigid", "migration_insert_hole_tube", "TestTube.usd", [[-0.01, -0.01, 0.0], [0.01, 0.01, 0.13]], 0.05),
        ("Geometry", "migration_insert_hole_slot", "TestTubeHoleSlot.usd", [[-0.05, -0.05, 0.0], [0.05, 0.05, 0.10]], 0.0),
        ("Geometry", "migration_insert_hole_base", "TestTubeBase.usd", [[-0.025, -0.025, 0.0], [0.025, 0.025, 0.05]], 0.0),
    )
    for typ, category, src, bbox, mass in entries:
        dst = objects / typ / category / "00000"
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OFFICIAL / "assets" / "objects" / src, dst / "object.usd")
        lo, hi = bbox
        ext = [hi[i] - lo[i] for i in range(3)]
        meta = {
            "uuid": f"univtac-{category}",
            "geometry": {"aligned_bbox": {"vertices": [lo, hi], "extents": ext}, "oriented_bbox": {"extents": ext}},
            "physics": {"mass": mass, "friction": 0.45 if mass else 0.0},
        }
        (dst / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    return STAGING

if __name__ == "__main__":
    print(stage_assets())
