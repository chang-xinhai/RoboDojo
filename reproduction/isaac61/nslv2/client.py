"""Shared, isolated NSLv2 model worker; no GT or classical depth fallback."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from reproduction.isaac61.nslv2.runtime import Transform, array_sha, canonical_json, decode
from reproduction.isaac61.nslv2.wire import read_packet, write_packet

_SESSIONS = {}
_REGISTRY_LOCK = threading.Lock()


@dataclass
class NSLv2Prediction:
    depth_left_m: np.ndarray
    disparity_left_native: np.ndarray
    covisibility_left_native: np.ndarray
    valid_mask: np.ndarray
    disparity_left_model: np.ndarray
    covisibility_left_model: np.ndarray
    provenance: dict


def resolve_config(config):
    config = dict(config)
    default_root = Path("/home/zktitan/projects/DepthUMI/third_party/nslv2")
    defaults = {
        "python_executable": "/home/zktitan/miniconda3/envs/nslv2/bin/python",
        "model_root": str(default_root),
        "weights_path": str(default_root / "ufm_1.8b_psv_step_00250000.pt"),
        "config_path": str(default_root / "config.yaml"),
        "device": "cuda:0",
        "precision": "mixed",
        "preprocess_mode": "resize",
        "confidence_threshold": None,
        "timeout_s": 600.0,
        "temporary_root": str(ROOT / ".cache/isaac61/nslv2"),
    }
    unknown = set(config) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown NSLv2 configuration keys: {sorted(unknown)}")
    defaults.update(config)
    for key in ("python_executable", "model_root", "weights_path", "config_path", "temporary_root"):
        defaults[key] = str(Path(defaults[key]).expanduser().resolve())
    if defaults["precision"] not in ("mixed", "fp32") or defaults["preprocess_mode"] not in ("resize", "letterbox"):
        raise ValueError("Invalid NSLv2 precision or preprocessing mode")
    threshold = defaults["confidence_threshold"]
    if threshold is not None and (not np.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("NSLv2 confidence threshold must be None or in [0,1]")
    if not np.isfinite(defaults["timeout_s"]) or defaults["timeout_s"] <= 0:
        raise ValueError("NSLv2 worker timeout must be positive and finite")
    return defaults


class _Session:
    def __init__(self, config):
        self.config = config
        self.lock = threading.Lock()
        self.owners = 0
        self.sequence = 0
        temporary = Path(config["temporary_root"])
        temporary.mkdir(parents=True, exist_ok=True)
        self.log_path = temporary / f"worker-{os.getpid()}-{time.time_ns()}.log"
        self.log = self.log_path.open("wb")
        self.process = subprocess.Popen(
            [config["python_executable"], "-u", str(HERE / "worker.py")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            cwd=ROOT,
        )
        try:
            write_packet(self.process.stdin, config)
            message, arrays = read_packet(self.process.stdout, config["timeout_s"])
            if message.get("event") != "ready" or arrays:
                raise RuntimeError(f"NSLv2 failed to load: {message}; log={self.log_path}")
            self.identity = message["identity"]
            self.identity["worker_log"] = str(self.log_path)
            self.identity["adapter_source_sha256"] = {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (HERE / "client.py", HERE / "runtime.py", HERE / "worker.py", HERE / "wire.py")
            }
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.process.poll() is None:
            try:
                write_packet(self.process.stdin, {"operation": "shutdown"})
                self.process.wait(timeout=15)
            except Exception:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        self.process.stdin.close()
        self.process.stdout.close()
        self.log.close()


class NSLv2Client:
    """One owner of a shared, sequential model worker for the resolved config."""

    def __init__(self, config=None):
        self.config = resolve_config({} if config is None else config)
        self._key = canonical_json(self.config)
        self._closed = False
        with _REGISTRY_LOCK:
            if self._key not in _SESSIONS:
                _SESSIONS[self._key] = _Session(self.config)
            self._session = _SESSIONS[self._key]
            self._session.owners += 1

    @classmethod
    def from_config(cls, config):
        return cls(config)

    @property
    def identity(self):
        return json.loads(canonical_json(self._session.identity))

    def infer(self, left, right, *, fx_px, baseline_m, stamp_ns, frame_index=None, camera=None):
        if self._closed:
            raise RuntimeError("NSLv2 client is closed")
        if (
            left.dtype != np.uint8
            or right.dtype != np.uint8
            or left.ndim != 2
            or left.shape != right.shape
            or not left.size
        ):
            raise ValueError("NSLv2 requires equally shaped, nonempty 2D uint8 IR")
        if not np.isfinite(fx_px) or fx_px <= 0 or not np.isfinite(baseline_m) or baseline_m <= 0:
            raise ValueError("NSLv2 calibration must be positive and finite")
        if isinstance(stamp_ns, bool) or not isinstance(stamp_ns, (int, np.integer)) or stamp_ns < 0:
            raise ValueError("NSLv2 acquisition timestamp must be a nonnegative integer")
        session = self._session
        with session.lock:
            sequence = session.sequence
            session.sequence += 1
            write_packet(
                session.process.stdin,
                {"operation": "infer", "request_id": sequence},
                {"ir_left": left, "ir_right": right},
            )
            response, arrays = read_packet(session.process.stdout, self.config["timeout_s"])
        if response.get("event") != "prediction" or response.get("request_id") != sequence:
            raise RuntimeError(f"NSLv2 model inference failed without fallback: {response}; log={session.log_path}")
        provenance = response["provenance"]
        if provenance["source_ir_left_sha256"] != array_sha(left) or provenance["source_ir_right_sha256"] != array_sha(
            right
        ):
            raise ValueError("NSLv2 worker returned different IR input identities")
        transform = Transform(**provenance["transform"])
        depth, disparity, confidence, mask = decode(
            arrays["disparity_left_model"],
            arrays["covisibility_left_model"],
            transform,
            fx_px,
            baseline_m,
            self.config["confidence_threshold"],
        )
        identity = self.identity
        identity_reference = {
            key: identity[key]
            for key in ("method", "model", "backend", "precision", "head_precision", "allow_tf32", "paths", "sha256")
        }
        provenance.update(
            method="nsl_v2",
            model_identity=identity_reference,
            model_identity_sha256=hashlib.sha256(
                canonical_json({key: value for key, value in identity.items() if key != "worker_log"}).encode()
            ).hexdigest(),
            fx_native_px=float(fx_px),
            baseline_m=float(baseline_m),
            fx_model_px=float(fx_px) * transform.scale_x,
            acquisition_timestamp_ns=int(stamp_ns),
            source_frame_index=frame_index,
            camera=camera,
            worker_request_id=sequence,
            inference_wall_seconds=response["inference_wall_seconds"],
            invalid_encoding="NaN",
            depth_reference="native left IR optical Z metres",
            disparity_reference="raw signed model pixels",
            depth_conversion="fx_native * horizontal_scale * baseline_m / abs(raw_signed_disparity)",
            confidence_threshold=self.config["confidence_threshold"],
            gt_input="none",
            fallback="none",
            output_sha256={
                "depth_left_m": array_sha(depth),
                "disparity_left_native": array_sha(disparity),
                "covisibility_left_native": array_sha(confidence),
                "valid_mask": array_sha(mask),
                **{name: array_sha(value) for name, value in arrays.items()},
            },
        )
        return NSLv2Prediction(
            depth,
            disparity,
            confidence,
            mask,
            arrays["disparity_left_model"],
            arrays["covisibility_left_model"],
            provenance,
        )

    def close(self):
        if self._closed:
            return
        self._closed = True
        with _REGISTRY_LOCK:
            self._session.owners -= 1
            if self._session.owners == 0:
                self._session.close()
                del _SESSIONS[self._key]

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.close()
        return False
