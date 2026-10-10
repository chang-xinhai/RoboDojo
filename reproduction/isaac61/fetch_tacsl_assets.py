#!/usr/bin/env python3
"""Download the unmodified official NVIDIA TacSL R1.5 asset and calibration.

Run in the configured Isaac Sim Python environment (provides pxr). Assets are
kept outside Git under Assets/TacSL. A SHA256 manifest records exact input bytes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2] / "Assets" / "TacSL"
SOURCE = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/TacSL/"
# NVIDIA currently publishes these sensor assets under its 5.1 asset root.
# They are USD/calibration data, not an Isaac Sim runtime installation.
FILES = [
    "gelsight_r15_finger/gelsight_r15_finger.usd",
    "gelsight_r15_finger/configuration/gelsight_r15_finger_base.usd",
    "gelsight_r15_finger/configuration/gelsight_r15_finger_physics.usd",
    "gelsight_r15_finger/configuration/gelsight_r15_finger_robot.usd",
    "gelsight_r15_finger/configuration/gelsight_r15_finger_sensor.usd",
    "gelsight_r15_data/bg.jpg",
    "gelsight_r15_data/polycalib.npz",
]

EXPECTED_SHA256 = dict(
    zip(
        FILES,
        (
            "92139f53c8cff8d70ee7668dddc4912b6b15b549cf2eaacf0f85e635ae93aa43",
            "7f751764e9653020606973d78992465951425bf333b853b4e61379cfc4beb9bb",
            "5d8b0b54169d5b5133da6fb015b6e88f1d85002af90e01bfdbb79de537ee0efd",
            "10b78cc44126d2944b32fe4dab329f5b316bedaccdd2405a0dae5761420df149",
            "3e63aa6b44df45cf624f66aaa07db98caf29469328b7ed810106c7ef3bef7271",
            "a7e4284d76f605b88a18f0a5ba50018707e82099854a2b8eefa6fa2d5bf25f85",
            "baef19e074c9414ead3a7599d54ed8254db19428f163a05bef70341964b27ef9",
        ),
        strict=True,
    )
)


def main():
    manifest = {"source": SOURCE, "files": {}}
    for relative in FILES:
        target = ROOT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file():
            payload = urlopen(SOURCE + relative, timeout=60).read()
            temporary = target.with_suffix(target.suffix + ".download")
            temporary.write_bytes(payload)
            temporary.replace(target)
        payload = target.read_bytes()
        if hashlib.sha256(payload).hexdigest() != EXPECTED_SHA256[relative]:
            raise RuntimeError(f"Official TacSL asset SHA256 mismatch: {target}; remove the invalid cache and rerun.")
        manifest["files"][relative] = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        print(f"{relative}: {len(payload)} bytes", flush=True)
    try:
        from pxr import Usd
    except ImportError:
        # Isaac Sim exposes its USD extension paths after AppLauncher starts.
        # Installation runs before that launch; real mount initialization also
        # validates these prims, so raw asset installation can complete first.
        print("USD prim validation will run at simulator startup (pxr not loaded).")
    else:
        stage = Usd.Stage.Open(str(ROOT / FILES[0]))
        for relative in ("elastomer/visuals", "elastomer/collisions", "elastomer_tip/cam"):
            if not stage.GetPrimAtPath("/gelsight_r15_finger/" + relative).IsValid():
                raise RuntimeError(f"Downloaded TacSL stage is incomplete: {relative}")
    (ROOT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Official TacSL assets and calibration verified.")


if __name__ == "__main__":
    main()
