"""Convert existing actual NPZ episodes losslessly to canonical streaming HDF5.

This is a format conversion, not a new simulator run. Original evidence and
source metadata are preserved; no new runtime or contact claim is added.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.multimodal import read_frame
from utils.multimodal_hdf5 import HDF5Writer, validate_hdf5


def convert(directory, output=None):
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("complete") or manifest.get("storage") == "hdf5":
        raise ValueError("Expected a complete legacy NPZ recording")
    evidence = {"legacy_manifest": manifest, "conversion_only": True, "original_files": {}}
    for name in (
        "validation.json",
        "runtime-render-settings.json",
        "native-log-audit.json",
        "independent-frame-audit.json",
    ):
        path = directory / name
        if path.exists():
            evidence[name] = json.loads(path.read_text())
    with HDF5Writer(
        output or directory, {**manifest.get("run", {}), "format_conversion": evidence}, allow_legacy=True
    ) as writer:
        for entry in manifest["frames"]:
            source = (directory / entry["file"]).resolve()
            if not source.is_relative_to(directory.resolve()) or source.suffix != ".npz":
                raise ValueError(f"Invalid legacy frame path: {source}")
            with source.open("rb") as stream:
                evidence["original_files"][entry["file"]] = hashlib.file_digest(stream, "sha256").hexdigest()
            writer.append(read_frame(source))
    return {"file": str(writer.path.resolve()), **validate_hdf5(writer.path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    print(json.dumps(convert(**vars(parser.parse_args())), indent=2))
