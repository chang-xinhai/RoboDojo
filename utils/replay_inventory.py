"""Finite native asset pools for evaluation layouts before PhysX warmup.

A pool slot is specific to its collider asset, scale, physical properties and
visual configuration. Changing mass or materials must select another native
body rather than silently reusing a body authored for an earlier episode.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
from pathlib import Path

RETAINED_TYPES = ("Rigid", "Dynamic", "Articulation")
ROOT = Path(__file__).resolve().parents[1]
_LAYOUT_ALIASES = {"arx_x5_rgbd": "arx_x5", "arx_x5_multimodal": "arx_x5"}


def _canonical(value):
    if isinstance(value, Mapping):
        return {str(key): _canonical(child) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_canonical(child) for child in value]
    if isinstance(value, str):
        placeholders = {
            "$RoboDojo_ASSETS": ROOT / "Assets",
            "$Robodojo_ASSETS": ROOT / "Assets",
            "$RoboDojo_CONF": ROOT / "env_cfg",
            "$RoboDojo_HOME": ROOT,
        }
        for placeholder, path in placeholders.items():
            value = value.replace(placeholder, str(path))
        return str(Path(value).resolve()) if value.startswith("/") else value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def retained_asset_key(physical_type, category, instance):
    """Stable key shared by pool construction, requested layouts and live wrappers."""
    physical_type = physical_type.capitalize()
    if physical_type not in RETAINED_TYPES:
        raise ValueError(f"Unsupported retained physics type: {physical_type}")
    model = instance.get("category_idx")
    if not isinstance(model, int):
        raise ValueError(f"Retained {physical_type}.{category} requires integer category_idx")
    source = "Clutter" if instance.get("type") == "cluttered" else physical_type
    asset_directory = ROOT / "Assets" / "Object" / "RoboDojo" / source / category / f"{model:05d}"
    asset_path = asset_directory / "object.usdz"
    if not asset_path.exists():
        asset_path = asset_directory / "object.usd"
    physics = _canonical(instance.get("physics", {}))
    physics.setdefault("type", physical_type.lower())
    properties = {
        "collider_asset": str(asset_path.resolve()),
        "asset_source": source,
        "physics": physics,
        "scale": _canonical(instance.get("scale", [1, 1, 1])),
        "visual": _canonical(instance.get("visual", {})),
    }
    signature = hashlib.sha256(
        json.dumps(properties, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return physical_type.lower(), str(category), model, signature


def build_retained_inventory(layouts):
    """Preload maximum simultaneous quantity per immutable asset variant.

    Pool prototypes are parked apart from one another before the native scene
    initializes. First-reset active layouts restore only their selected slots.
    """
    prototypes = {}
    layout_count = 0
    for layout in layouts:
        layout_count += 1
        current = {}
        for physical_type in RETAINED_TYPES:
            for category, instances in layout.get(physical_type, {}).items():
                for instance in instances:
                    key = retained_asset_key(physical_type, category, instance)
                    current.setdefault(key, []).append(instance)
        for key, instances in current.items():
            if len(instances) > len(prototypes.get(key, [])):
                prototypes[key] = deepcopy(instances)
    inventory = {physical_type: {} for physical_type in RETAINED_TYPES}
    counts = {physical_type: 0 for physical_type in RETAINED_TYPES}
    variants = []
    slot = 0
    for key in sorted(prototypes):
        kind, category, model, signature = key
        physical_type = kind.capitalize()
        for instance in prototypes[key]:
            parked = deepcopy(instance)
            parked["default_pos"] = [100000.0 + slot * 1000.0, 100000.0, 100000.0]
            parked["need_check_stable"] = False
            inventory[physical_type].setdefault(category, []).append(parked)
            counts[physical_type] += 1
            slot += 1
        variants.append(
            {
                "physical_type": physical_type,
                "category": category,
                "category_idx": model,
                "signature": signature,
                "capacity": len(prototypes[key]),
            }
        )
    audit = {
        "layout_count": layout_count,
        "variant_count": len(prototypes),
        "instance_count": slot,
        "physical_type_counts": counts,
        "variants": variants,
    }
    return inventory, audit


def replay_seed_config(eval_config):
    """Resolve observation profiles to their unchanged embodiment layout directory."""
    config = dict(eval_config)
    name = str(config["config_name"])
    config["config_name"] = config.get("layout_config") or _LAYOUT_ALIASES.get(name, name)
    return config


def preload_replay_inventory(seed_manager, scene_manager):
    """Load the remaining finite replay cohort without consuming the seed cursor."""
    if scene_manager.sim is not None:
        raise RuntimeError("Replay asset inventory must be supplied before the first physics initialization")
    layout_ids = list(seed_manager.seed_list)
    files = []
    layouts = []
    for layout_id in layout_ids:
        path = Path(seed_manager.seed_info[layout_id]["scene_layout"]).resolve()
        raw = path.read_bytes()
        layouts.append(json.loads(raw))
        files.append({"layout_id": int(layout_id), "file": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
    inventory, audit = build_retained_inventory(layouts)
    scene_manager.retained_layout_inventory = inventory
    return {
        **audit,
        "layout_config": seed_manager.config_name,
        "layouts": files,
        "physics_initialized": False,
        "seed_cursor": int(seed_manager.idx),
    }
