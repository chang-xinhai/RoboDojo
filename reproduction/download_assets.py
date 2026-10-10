"""Download versioned RoboDojo runtime assets; --full includes every asset."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--full", action="store_true")
parser.add_argument("--task-smoke", action="store_true", help="Only stack_bowls and the default scene")
parser.add_argument("--workers", type=int, default=32)
args = parser.parse_args()
lock = json.loads((ROOT / "reproduction/asset-lock.json").read_text())
manifest = json.loads((ROOT / "reproduction/asset-manifest.json").read_text())
prefixes = (
    "Assets/Robots/",
    "Assets/Object/RoboDojo/",
    "Assets/Eval_Layout/RoboDojo/",
    "Assets/Background/",
    "Assets/Material/Base/",
    "Assets/Material/Fluid/",
    "Assets/Material/Garment/",
    "Assets/Material/Templates/",
    "Assets/Material/material_0122/",
    "Assets/Material/material_0564/",
    "Assets/Room/Simple_Room/",
    "Assets/Room/Simple_Room_nolight/",
)
if args.task_smoke:
    prefixes = (
        "Assets/Robots/x5/",
        "Assets/Object/RoboDojo/Rigid/bowl/",
        "Assets/Object/RoboDojo/Geometry/camera_stand/",
        "Assets/Background/",
        "Assets/Material/Base/",
        "Assets/Material/material_0122/",
        "Assets/Material/material_0564/",
        "Assets/Room/Simple_Room_nolight/",
        "Assets/Eval_Layout/RoboDojo/stack_bowls/",
    )
files = [
    x
    for x in manifest
    if x["type"] == "file" and (args.full or x["path"].startswith(prefixes)) and "/.thumbs/" not in x["path"]
]
# Prioritize scene and robot files before the thousands of evaluation layouts.
files.sort(
    key=lambda x: (
        not x["path"].startswith(prefixes),
        x["path"].startswith("Assets/Eval_Layout/"),
        x["path"].startswith("Assets/Object/"),
        x["size"],
    )
)
local = threading.local()


def digest_ok(path, entry):
    if not path.is_file() or path.stat().st_size != entry["size"]:
        return False
    if "lfs" in entry:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(2**20), b""):
                h.update(chunk)
        return h.hexdigest() == entry["lfs"]["oid"]
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == entry["oid"]


def download(entry):
    path = ROOT / entry["path"]
    if digest_ok(path, entry):
        return
    if not hasattr(local, "session"):
        local.session = requests.Session()
    url = (
        f"https://huggingface.co/datasets/{lock['repo_id']}/resolve/{lock['revision']}/{quote(entry['path'], safe='/')}"
    )
    for attempt in range(6):
        try:
            temporary = path.with_name(path.name + ".download-partial")
            offset = temporary.stat().st_size if temporary.exists() else 0
            if offset == entry["size"] and digest_ok(temporary, entry):
                temporary.replace(path)
                break
            if offset >= entry["size"]:
                offset = 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with local.session.get(url, headers=headers, timeout=(30, 300), stream=True) as response:
                response.raise_for_status()
                path.parent.mkdir(parents=True, exist_ok=True)
                mode = "ab" if offset and response.status_code == 206 else "wb"
                with temporary.open(mode) as f:
                    for chunk in response.iter_content(2**16):
                        f.write(chunk)
            if not digest_ok(temporary, entry):
                raise RuntimeError(f"Hash mismatch: {entry['path']}")
            temporary.replace(path)
            break
        except requests.RequestException:
            if attempt == 5:
                raise
            time.sleep(min(2**attempt, 30))
    else:
        raise RuntimeError(f"Download retries exhausted: {entry['path']}")
    if not digest_ok(path, entry):
        raise RuntimeError(f"Hash mismatch: {entry['path']}")


print(f"Downloading {len(files)} files, {sum(x['size'] for x in files) / 2**30:.2f} GiB", flush=True)
with ThreadPoolExecutor(max_workers=args.workers) as pool:
    futures = [pool.submit(download, x) for x in files]
    for i, future in enumerate(as_completed(futures), 1):
        future.result()
        if i % 250 == 0:
            print(f"{i}/{len(files)} verified", flush=True)
profile = "full" if args.full else ("stack_bowls" if args.task_smoke else "runtime")
result = {
    "status": "passed",
    "profile": profile,
    "revision": lock["revision"],
    "files": len(files),
    "bytes": sum(x["size"] for x in files),
    "verification": "SHA256 LFS / Git blob SHA1",
}
(ROOT / f"reproduction/asset-verification-{profile}.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
subprocess.run([sys.executable, str(ROOT / "utils/update_embodiment_config_path.py")], cwd=ROOT, check=True)
