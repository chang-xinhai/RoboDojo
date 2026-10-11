"""Streaming, typed RoboDojo multimodal HDF5 with committed-frame transactions.

Only committed frames are readable. In-progress and failed files retain the
.partial suffix; the complete episode appears through an atomic file rename.
Numeric arrays retain their dtype and NaN values. No pickle or lossy codecs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import json
import os
from pathlib import Path
from urllib.parse import quote

import h5py
import numpy as np

from utils.multimodal import SCHEMA_VERSION, numpy_copy, validate_multimodal

HDF5_SCHEMA = "robodojo.multimodal.hdf5.v1"
TACTILE_COMPONENT_SCHEMA = "robodojo.tactile_component.v1"


def _write(parent, name, value):
    if hasattr(value, "detach"):
        value = numpy_copy(value)
    if isinstance(value, (np.ndarray, np.generic)):
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ValueError(f"Object arrays cannot be recorded: {parent.name}/{name}")
        options = {"compression": "gzip", "compression_opts": 4, "shuffle": True, "fletcher32": True}
        if not array.ndim or not array.size:
            options = {}
        if array.dtype.kind == "U":
            dataset = parent.create_dataset(name, data=array.astype(h5py.string_dtype("utf-8")))
            dataset.attrs["numpy_dtype"] = array.dtype.str
        else:
            dataset = parent.create_dataset(name, data=array, **options)
        dataset.attrs["node_type"] = "ndarray" if isinstance(value, np.ndarray) else "numpy_scalar"
        return dataset
    node = parent.create_group(name, track_order=True)
    if isinstance(value, Mapping):
        node.attrs["node_type"] = "mapping"
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"Metadata keys must be strings: {node.name}")
            # Absolute USD paths are common dictionary keys, not HDF5 paths.
            encoded = quote(key, safe="") or "%EMPTY%"
            child_node = _write(node, encoded, child)
            child_node.attrs["original_key"] = key
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        node.attrs["node_type"] = "tuple" if isinstance(value, tuple) else "list"
        for index, child in enumerate(value):
            _write(node, f"{index:08d}", child)
    else:
        node.attrs["node_type"] = "null" if value is None else "scalar"
        if value is not None:
            node.create_dataset("value", data=json.dumps(value, allow_nan=False), dtype=h5py.string_dtype("utf-8"))
    return node


def _read(node):
    kind = node.attrs["node_type"]
    if kind in ("ndarray", "numpy_scalar"):
        value = node[()]
        if "numpy_dtype" in node.attrs:
            value = np.asarray(node.asstr()[()], dtype=node.attrs["numpy_dtype"])
        return np.asarray(value) if kind == "ndarray" else np.asarray(value)[()]
    if kind == "mapping":
        return {child.attrs["original_key"]: _read(child) for child in node.values()}
    if kind in ("list", "tuple"):
        values = [_read(node[key]) for key in sorted(node)]
        return tuple(values) if kind == "tuple" else values
    if kind == "null":
        return None
    if kind == "scalar":
        return json.loads(node["value"].asstr()[()])
    raise ValueError(f"Unknown HDF5 node type: {kind}")


def _flush(handle):
    handle.flush()
    descriptor = handle.id.get_vfd_handle()
    if isinstance(descriptor, int):
        os.fsync(descriptor)


class HDF5Writer:
    """One validated frame in memory; publish a completed episode atomically.

    append() failures automatically persist a failed partial episode. seal()
    closes the file for preview/cleanup, complete() publishes it, and fail()
    can withdraw that completion if a later pipeline check fails.
    """

    def __init__(self, directory, run_metadata=None, *, allow_legacy=False, observation_schema=SCHEMA_VERSION):
        if observation_schema not in (SCHEMA_VERSION, TACTILE_COMPONENT_SCHEMA):
            raise ValueError("Unsupported observation schema")
        self.observation_schema = observation_schema
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "episode.hdf5"
        self.partial_path = self.directory / "episode.hdf5.partial"
        if self.path.exists() or self.partial_path.exists():
            raise FileExistsError(f"HDF5 recording already exists: {self.directory}")
        if not allow_legacy and (
            (self.directory / "manifest.json").exists() or any(self.directory.glob("frame_*.npz"))
        ):
            raise FileExistsError(f"Recording directory already contains data: {self.directory}")
        self.run_metadata = dict(run_metadata or {})
        self.count = 0
        self.last_frame = self.last_timestamp = -1
        self.handle = h5py.File(self.partial_path, "x")
        self.handle.attrs.update(
            schema=HDF5_SCHEMA,
            observation_schema=self.observation_schema,
            status="recording",
            complete=False,
            committed_frame_count=0,
            created_at_utc=datetime.now(UTC).isoformat(),
        )
        self.handle.create_group("frames", track_order=True)
        self.handle.create_group("pending", track_order=True)
        timeline = self.handle.create_group("time")
        for name in ("timestamp_ns", "physics_frame"):
            timeline.create_dataset(name, shape=(0,), maxshape=(None,), dtype=np.int64, chunks=(1024,))
        try:
            _write(self.handle, "run", self.run_metadata)
            _flush(self.handle)
        except BaseException as error:
            self.handle.attrs.update(status="failed", failure_reason=str(error))
            if "run" in self.handle:
                del self.handle["run"]
            _write(self.handle, "run", {})
            self.handle.close()
            self.handle = None
            raise

    def append(self, observation):
        if self.handle is None:
            raise RuntimeError("Cannot append after HDF5 writer was sealed or failed")
        name = f"{self.count:08d}"
        try:
            validate_multimodal(observation, tactile_component=self.observation_schema == TACTILE_COMPONENT_SCHEMA)
            sync = observation["sync"]
            if sync["physics_frame"] <= self.last_frame or sync["timestamp_ns"] <= self.last_timestamp:
                raise ValueError("Recording physics frames and timestamps must strictly increase")
            staged = _write(self.handle["pending"], name, observation)
            staged.attrs["committed"] = False
            _flush(self.handle)
            self.handle.move(f"pending/{name}", f"frames/{name}")
            for key in ("physics_frame", "timestamp_ns"):
                dataset = self.handle[f"time/{key}"]
                dataset.resize((self.count + 1,))
                dataset[self.count] = sync[key]
            self.handle[f"frames/{name}"].attrs["committed"] = True
            # Count is the transaction's visibility marker, written last.
            self.handle.attrs["committed_frame_count"] = self.count + 1
            _flush(self.handle)
        except BaseException as error:
            for group in ("frames", "pending"):
                if name in self.handle[group]:
                    del self.handle[group][name]
            for key in ("physics_frame", "timestamp_ns"):
                self.handle[f"time/{key}"].resize((self.count,))
            self.handle.attrs["committed_frame_count"] = self.count
            self.fail(error)
            raise
        self.count += 1
        self.last_frame, self.last_timestamp = sync["physics_frame"], sync["timestamp_ns"]
        return self.partial_path

    def _metadata(self, handle):
        pending = "_run_pending"
        if pending in handle:
            del handle[pending]
        try:
            _write(handle, pending, self.run_metadata)
        except BaseException:
            if pending in handle:
                del handle[pending]
            raise
        if "run" in handle:
            del handle["run"]
        handle.move(pending, "run")

    def seal(self):
        if self.handle is None:
            return
        if not self.count:
            self.fail("Cannot finalize an empty recording")
            raise ValueError("Cannot finalize an empty recording")
        self._metadata(self.handle)
        self.handle.attrs["status"] = "captured"
        _flush(self.handle)
        self.handle.close()
        self.handle = None

    def complete(self, validation=None):
        try:
            self.seal()
            with h5py.File(self.partial_path, "r+") as handle:
                if handle.attrs["status"] == "failed":
                    raise RuntimeError("Cannot complete a failed HDF5 recording")
                self._metadata(handle)
                if validation is not None:
                    try:
                        _write(handle, "validation", validation)
                    except BaseException:
                        if "validation" in handle:
                            del handle["validation"]
                        raise
                handle.attrs.update(status="complete", complete=True, completed_at_utc=datetime.now(UTC).isoformat())
                _flush(handle)
            os.replace(self.partial_path, self.path)
            return self.path
        except BaseException as error:
            self.fail(error)
            raise

    def fail(self, error):
        if self.handle is not None:
            self.handle.attrs.update(status="failed", complete=False, failure_reason=str(error))
            try:
                self._metadata(self.handle)
            except Exception as metadata_error:
                self.handle.attrs["metadata_failure_reason"] = str(metadata_error)
            _flush(self.handle)
            self.handle.close()
            self.handle = None
        else:
            existing = self.path if self.path.exists() else self.partial_path
            with h5py.File(existing, "r+") as handle:
                handle.attrs.update(status="failed", complete=False, failure_reason=str(error))
                _flush(handle)
            if existing == self.path:
                os.replace(self.path, self.partial_path)

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        if kind is not None:
            self.fail(value)
        elif not self.path.exists():
            try:
                self.complete()
            except BaseException as error:
                self.fail(error)
                raise


class HDF5Reader:
    """Random-access committed frames without loading the complete episode."""

    def __init__(self, path):
        self.path = Path(path)
        self.handle = h5py.File(self.path, "r")
        try:
            if self.handle.attrs.get("schema") != HDF5_SCHEMA:
                raise ValueError(f"Expected HDF5 schema {HDF5_SCHEMA}")
            if self.handle.attrs.get("observation_schema") not in (SCHEMA_VERSION, TACTILE_COMPONENT_SCHEMA):
                raise ValueError("Unsupported HDF5 observation schema")
            count = self.handle.attrs["committed_frame_count"]
            if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count < 0:
                raise ValueError("Invalid committed frame count")
            self.count = int(count)
            complete = self.handle.attrs["complete"]
            status = self.handle.attrs["status"]
            if not isinstance(complete, (bool, np.bool_)) or status not in (
                "recording",
                "captured",
                "failed",
                "complete",
            ):
                raise ValueError("Invalid HDF5 completion status")
            if bool(complete) != (status == "complete"):
                raise ValueError("HDF5 status and completion flag disagree")
            for key in ("physics_frame", "timestamp_ns"):
                dataset = self.handle[f"time/{key}"]
                if dataset.ndim != 1 or dataset.dtype != np.int64:
                    raise ValueError("HDF5 timeline must be one-dimensional int64")
            if any(len(self.handle[f"time/{key}"]) < self.count for key in ("physics_frame", "timestamp_ns")):
                raise ValueError("HDF5 timeline is shorter than committed frame count")
        except BaseException:
            self.handle.close()
            raise

    def __len__(self):
        return self.count

    @property
    def metadata(self):
        return {
            "schema": HDF5_SCHEMA,
            "observation_schema": self.handle.attrs["observation_schema"],
            "status": self.handle.attrs["status"],
            "complete": bool(self.handle.attrs["complete"]),
            "frames": self.count,
            "run": _read(self.handle["run"]),
            "failure_reason": self.handle.attrs.get("failure_reason"),
            "validation": _read(self.handle["validation"]) if "validation" in self.handle else None,
        }

    def sync(self, index):
        if not 0 <= index < self.count:
            raise IndexError(index)
        return {key: int(self.handle[f"time/{key}"][index]) for key in ("physics_frame", "timestamp_ns")}

    def read_frame(self, index, *, validate=True):
        sync = self.sync(index)
        group = self.handle[f"frames/{index:08d}"]
        if not group.attrs.get("committed"):
            raise ValueError(f"Frame {index} is not committed")
        observation = _read(group)
        if any(observation["sync"][key] != value for key, value in sync.items()):
            raise ValueError(f"Frame {index} does not match its HDF5 timeline")
        if validate:
            return validate_multimodal(
                observation, tactile_component=self.handle.attrs["observation_schema"] == TACTILE_COMPONENT_SCHEMA
            )
        return observation

    def close(self):
        self.handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def validate_hdf5(path, *, require_complete=True):
    with HDF5Reader(path) as reader:
        metadata = reader.metadata
        if require_complete and (metadata["status"] != "complete" or not metadata["complete"]):
            raise ValueError(f"Recording is {metadata['status']}: {metadata['failure_reason']}")
        previous = {"physics_frame": -1, "timestamp_ns": -1}
        for index in range(len(reader)):
            reader.read_frame(index)
            current = reader.sync(index)
            if any(current[key] <= previous[key] for key in previous):
                raise ValueError(f"Non-increasing HDF5 timeline at frame {index}")
            previous = current
        if require_complete and not len(reader):
            raise ValueError("Completed recording is empty")
        return {
            "schema": HDF5_SCHEMA,
            "observation_schema": metadata["observation_schema"],
            "status": metadata["status"],
            "complete": metadata["complete"],
            "validated_frames": len(reader),
            "last_sync": previous,
            "failure_reason": metadata["failure_reason"],
        }
