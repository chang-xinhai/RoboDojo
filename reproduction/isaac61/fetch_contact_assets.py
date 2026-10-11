#!/usr/bin/env python3
"""Fetch the unchanged NVIDIA factory nut used by the official TacSL demo."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2] / "Assets" / "TacSL" / "contact_objects"
URL = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/Factory/factory_nut_m16.usd"
SHA256 = "f212a7f5664843abab3ec7d556ebf68ca99d4ad0880121978a45e445ad59431a"


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    target = ROOT / "factory_nut_m16.usd"
    if not target.exists():
        payload = urlopen(URL, timeout=45).read()
        if hashlib.sha256(payload).hexdigest() != SHA256:
            raise RuntimeError("Official TacSL nut asset hash mismatch")
        temporary = target.with_suffix(".download")
        temporary.write_bytes(payload)
        temporary.replace(target)
    payload = target.read_bytes()
    if hashlib.sha256(payload).hexdigest() != SHA256:
        raise RuntimeError(f"Invalid official TacSL nut asset cache: {target}")
    manifest = {
        "source": URL,
        "sha256": SHA256,
        "bytes": len(payload),
        "official_demo": "IsaacLab/scripts/demos/sensors/tacsl_sensor.py NutTactileSceneCfg",
        "asset_version_note": "NVIDIA 5.1 asset bucket; runtime independently verified as Isaac Sim 6.1",
    }
    (ROOT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
