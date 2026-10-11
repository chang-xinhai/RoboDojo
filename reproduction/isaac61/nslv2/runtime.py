"""Geometry and read-only upstream NSLv2 model loading, without GT inputs.

The explicit model and precision profile follow DepthUMI's nsl_v2/runtime.py.
Upstream model code and checkpoints are reused without modifying or copying them.
"""

from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class Transform:
    source_width: int
    source_height: int
    model_width: int
    model_height: int
    content_width: int
    content_height: int
    pad_left: int
    pad_top: int
    scale_x: float
    scale_y: float
    mode: str


def prepare(left, right, width, height, mode):
    if left.dtype != np.uint8 or right.dtype != np.uint8 or left.ndim != 2 or left.shape != right.shape:
        raise ValueError("NSLv2 requires equally shaped 2D uint8 stereo IR")
    if not left.size or width <= 0 or height <= 0 or mode not in ("resize", "letterbox"):
        raise ValueError("Invalid NSLv2 input dimensions or preprocessing mode")
    sh, sw = left.shape
    cw, ch = width, height
    if mode == "letterbox":
        scale = min(width / sw, height / sh)
        cw, ch = max(1, min(width, round(sw * scale))), max(1, min(height, round(sh * scale)))
    px, py = (width - cw) // 2, (height - ch) // 2
    interpolation = cv2.INTER_AREA if cw <= sw and ch <= sh else cv2.INTER_LINEAR

    def fit(image):
        resized = cv2.resize(image, (cw, ch), interpolation=interpolation)
        return cv2.copyMakeBorder(resized, py, height - ch - py, px, width - cw - px, cv2.BORDER_REPLICATE)

    return fit(left), fit(right), Transform(sw, sh, width, height, cw, ch, px, py, cw / sw, ch / sh, mode)


def decode(disparity, covisibility, transform, fx_px, baseline_m, confidence_threshold):
    if not np.isfinite(fx_px) or fx_px <= 0 or not np.isfinite(baseline_m) or baseline_m <= 0:
        raise ValueError("NSLv2 focal length and baseline must be positive and finite")
    if confidence_threshold is not None and (
        not np.isfinite(confidence_threshold) or not 0 <= confidence_threshold <= 1
    ):
        raise ValueError("NSLv2 confidence threshold must be None or in [0,1]")
    expected = (transform.model_height, transform.model_width)
    if disparity.shape != expected or covisibility.shape != expected:
        raise ValueError("NSLv2 prediction shape differs from its model canvas")
    slices = (
        slice(transform.pad_top, transform.pad_top + transform.content_height),
        slice(transform.pad_left, transform.pad_left + transform.content_width),
    )
    disparity = np.asarray(disparity[slices], np.float32)
    confidence = np.asarray(covisibility[slices], np.float32)
    valid = np.isfinite(disparity) & (np.abs(disparity) > 1e-6) & np.isfinite(confidence)
    if confidence_threshold is not None:
        valid &= confidence >= confidence_threshold
    depth = np.full(disparity.shape, np.nan, np.float32)
    depth[valid] = float(fx_px) * transform.scale_x * float(baseline_m) / np.abs(disparity[valid])
    depth[~np.isfinite(depth) | (depth <= 0)] = np.nan
    size = (transform.source_width, transform.source_height)
    native_depth = cv2.resize(depth, size, interpolation=cv2.INTER_NEAREST).astype(np.float32)
    native_disparity = cv2.resize(disparity / transform.scale_x, size, interpolation=cv2.INTER_NEAREST)
    native_confidence = cv2.resize(confidence, size, interpolation=cv2.INTER_NEAREST)
    return native_depth, native_disparity, native_confidence, np.isfinite(native_depth)


def load_model(config):
    from omegaconf import OmegaConf
    import torch

    root = Path(config["model_root"]).expanduser().resolve()
    weights = Path(config["weights_path"]).expanduser().resolve()
    source_config = Path(config["config_path"]).expanduser().resolve()
    for path in (root / "model.py", weights, source_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    if config["precision"] not in ("mixed", "fp32"):
        raise ValueError("NSLv2 precision must be mixed or fp32")
    files = [root / "model.py"]
    for folder in (root / "ufm_stereo", root / "third_party"):
        files.extend(path for path in folder.rglob("*") if path.suffix in (".py", ".yaml", ".cpp", ".cu", ".h"))
    source_hashes = {path.relative_to(root).as_posix(): file_sha(path) for path in sorted(files)}
    weights_sha = file_sha(weights)
    mixed = config["precision"] == "mixed"
    effective = OmegaConf.merge(
        OmegaConf.load(source_config),
        {
            "inference": {
                "mixed_precision": mixed,
                "amp_dtype": "bfloat16" if mixed else "float32",
                "allow_tf32": False,
            },
            "model": {"head_amp_dtype": "float32", "head": {"kwargs": {"disparity_projection_amp_dtype": "float32"}}},
        },
    )
    sys.path.insert(0, str(root))
    from model import StereoInference

    with tempfile.TemporaryDirectory(prefix="robodojo-nslv2-", dir=config["temporary_root"]) as temporary:
        path = Path(temporary) / "effective.yaml"
        OmegaConf.save(effective, path)
        runner = StereoInference(weights_path=weights, config_path=path, device=config["device"])
    runner.model.float()
    model_config = OmegaConf.to_container(runner.cfg, resolve=True)
    identity = {
        "method": "nsl_v2",
        "model": "ufm_1.8b_psv",
        "backend": "pytorch_bfloat16" if mixed else "pytorch_float32",
        "precision": config["precision"],
        "head_precision": "float32",
        "allow_tf32": False,
        "paths": {"model_root": str(root), "weights": str(weights), "config": str(source_config)},
        "sha256": {
            "weights": weights_sha,
            "config": file_sha(source_config),
            "model_source": hashlib.sha256(canonical_json(source_hashes).encode()).hexdigest(),
            "resolved_model_config": hashlib.sha256(canonical_json(model_config).encode()).hexdigest(),
        },
        "model_source_files_sha256": source_hashes,
        "model_config": model_config,
        "packages": {
            name: importlib.metadata.version(name) for name in ("torch", "numpy", "omegaconf", "opencv-python")
        },
        "python_executable": sys.executable,
        "device": str(runner.device),
        "gpu_name": torch.cuda.get_device_name(runner.device),
        "cuda_runtime_version": torch.version.cuda,
        "cuda_compute_capability": list(torch.cuda.get_device_capability(runner.device)),
        "loaded_curope_extensions_sha256": {
            str(Path(module.__file__).resolve()): file_sha(module.__file__)
            for name, module in list(sys.modules.items())
            if "curope" in name and getattr(module, "__file__", "").endswith(".so")
        },
        "available_curope_extensions_sha256": {
            path.relative_to(root).as_posix(): file_sha(path)
            for path in sorted((root / "third_party").rglob("*.so"))
            if "curope" in str(path)
        },
        "gt_input": "none",
        "fallback": "none",
        "confidence_threshold": config["confidence_threshold"],
        "preprocess_mode": config["preprocess_mode"],
    }
    return runner, identity


def infer(runner, left, right, mode):
    import torch

    left_fit, right_fit, transform = prepare(
        left, right, int(runner.cfg.inference.image_width), int(runner.cfg.inference.image_height), mode
    )
    prediction = runner.predict(
        torch.from_numpy(np.ascontiguousarray(left_fit))[None, None],
        torch.from_numpy(np.ascontiguousarray(right_fit))[None, None],
    )
    arrays = {
        "disparity_left_model": prediction.disparity_left[0, 0].numpy().astype(np.float32),
        "covisibility_left_model": prediction.covisibility_left[0, 0].numpy().astype(np.float32),
    }
    return arrays, {
        "transform": asdict(transform),
        "source_ir_left_sha256": array_sha(left),
        "source_ir_right_sha256": array_sha(right),
        "model_ir_left_sha256": array_sha(left_fit),
        "model_ir_right_sha256": array_sha(right_fit),
        "model_input_dtype": "uint8 0..255; upstream normalizes internally",
    }
