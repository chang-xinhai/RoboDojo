"""Finite replay pool and immutable native variant contracts without Kit."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from utils.replay_inventory import (
    ROOT,
    build_retained_inventory,
    preload_replay_inventory,
    replay_seed_config,
    retained_asset_key,
)


def instance(model=2, mass=0.1, color="red", x=0.0):
    return {
        "category_idx": model,
        "physics": {"type": "rigid", "mass": mass, "friction": 0.5},
        "visual": {"color": color},
        "scale": [1.0, 1.0, 1.0],
        "default_pos": [x, 0.0, 0.8],
        "default_ori": [1.0, 0.0, 0.0, 0.0],
        "label": f"object_at_{x}",
    }


class ReplayInventoryTests(unittest.TestCase):
    def test_maximum_simultaneous_count_per_physics_and_visual_variant(self):
        layouts = [
            {"Rigid": {"toy": [instance(x=0)]}, "Garment": {"cloth": [{}]}},
            {"Rigid": {"toy": [instance(x=1), instance(x=2)]}},
            {"Rigid": {"toy": [instance(mass=0.2), instance(color="blue"), instance(color="blue", x=3)]}},
        ]
        original = deepcopy(layouts)
        inventory, audit = build_retained_inventory(layouts)
        self.assertEqual(layouts, original)
        self.assertEqual(audit["layout_count"], 3)
        self.assertEqual(audit["variant_count"], 3)
        self.assertEqual(audit["instance_count"], 5)
        self.assertEqual(len(inventory["Rigid"]["toy"]), 5)
        self.assertNotIn("Garment", inventory)
        positions = [item["default_pos"] for item in inventory["Rigid"]["toy"]]
        self.assertEqual(len({tuple(position) for position in positions}), 5)
        self.assertTrue(all(position[0] >= 100000 for position in positions))
        for layout in layouts:
            requested = {}
            for item in layout["Rigid"]["toy"]:
                key = retained_asset_key("rigid", "toy", item)
                requested[key] = requested.get(key, 0) + 1
            available = {}
            for item in inventory["Rigid"]["toy"]:
                key = retained_asset_key("rigid", "toy", item)
                available[key] = available.get(key, 0) + 1
            self.assertTrue(all(available[key] >= count for key, count in requested.items()))

    def test_static_variant_key_excludes_episode_pose_labels_and_normalizes_paths(self):
        raw = instance()
        raw["visual"] = {"visual_mdl_path": "$RoboDojo_ASSETS/material/../surface.mdl"}
        resolved = deepcopy(raw)
        resolved["visual"]["visual_mdl_path"] = str(ROOT / "Assets" / "surface.mdl")
        resolved["default_pos"] = [0.4, -0.3, 0.9]
        resolved["default_ori"] = [0.0, 0.0, 0.0, 1.0]
        resolved["label"] = "next_episode"
        resolved["prim_path"] = "/World/envs/env_4/rigid/toy/toy_2_3"
        self.assertEqual(retained_asset_key("Rigid", "toy", raw), retained_asset_key("rigid", "toy", resolved))
        for field, value in (
            ("physics", {"type": "rigid", "mass": 0.2}),
            ("scale", [2, 1, 1]),
            ("visual", {"color": "blue"}),
            ("type", "cluttered"),
        ):
            variant = deepcopy(raw)
            variant[field] = value
            self.assertNotEqual(retained_asset_key("Rigid", "toy", raw), retained_asset_key("Rigid", "toy", variant))

    def test_omegaconf_live_wrapper_signature_matches_raw_inventory(self):
        from omegaconf import OmegaConf

        from utils.path import deep_resolve_paths

        raw = instance()
        raw["visual"] = {"visual_usd_path": "$RoboDojo_ASSETS/Material/../surface.usd", "color": [0.1, 0.2, 0.3]}
        live = OmegaConf.create(raw)
        deep_resolve_paths(live)
        live["prim_path"] = "/World/envs/env_0/rigid/toy/toy_2_0"
        live["usd_path"] = str(ROOT / "Assets/Object/RoboDojo/Rigid/toy/00002/object.usd")
        live["inst_name"] = "toy_2_0"
        self.assertEqual(retained_asset_key("Rigid", "toy", raw), retained_asset_key("rigid", "toy", live))

    def test_replay_preload_preserves_filtered_seed_cursor_and_file_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for model in (1, 2, 3):
                path = Path(directory) / f"task_{model}.json"
                path.write_text(json.dumps({"Rigid": {"toy": [instance(model=model)]}}))
                paths.append(path)
            seeds = SimpleNamespace(
                seed_list=[1, 2],
                seed_info={i: {"scene_layout": p} for i, p in enumerate(paths)},
                config_name="arx_x5",
                idx=0,
            )
            scene = SimpleNamespace(sim=None)
            audit = preload_replay_inventory(seeds, scene)
            self.assertEqual(seeds.idx, 0)
            self.assertEqual([item["layout_id"] for item in audit["layouts"]], [1, 2])
            self.assertEqual({item["category_idx"] for item in scene.retained_layout_inventory["Rigid"]["toy"]}, {2, 3})
            self.assertTrue(all(len(item["sha256"]) == 64 for item in audit["layouts"]))
            scene.sim = object()
            with self.assertRaisesRegex(RuntimeError, "before the first physics"):
                preload_replay_inventory(seeds, scene)

    def test_observation_profile_alias_does_not_mutate_report_config(self):
        for name in ("arx_x5_rgbd", "arx_x5_multimodal"):
            config = {"config_name": name, "task_name": "stack_bowls", "num_envs": 2}
            result = replay_seed_config(config)
            self.assertEqual(result["config_name"], "arx_x5")
            self.assertEqual(config["config_name"], name)
        self.assertEqual(
            replay_seed_config({"config_name": "custom", "layout_config": "other"})["config_name"], "other"
        )


if __name__ == "__main__":
    unittest.main()
