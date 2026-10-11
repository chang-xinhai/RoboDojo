"""Lossless JSON/raw-array packets for a private local model subprocess."""

import json
import os
import select
import struct
import time

import numpy as np

MAX_PACKET = 64 * 1024 * 1024


def read_exact(stream, size, timeout=None):
    result = bytearray()
    deadline = None if timeout is None else time.monotonic() + timeout
    while len(result) < size:
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([stream], [], [], remaining)[0]:
                raise TimeoutError("NSLv2 worker did not deliver a complete packet")
        block = os.read(stream.fileno(), size - len(result))
        if not block:
            raise EOFError("NSLv2 worker closed its private pipe")
        result.extend(block)
    return bytes(result)


def write_packet(stream, metadata, arrays=None):
    arrays = {} if arrays is None else arrays
    descriptors = []
    payloads = []
    for name, value in arrays.items():
        value = np.ascontiguousarray(value)
        if value.dtype.hasobject:
            raise ValueError("Object arrays are not permitted in NSLv2 transport")
        descriptors.append({"name": name, "shape": list(value.shape), "dtype": value.dtype.str, "bytes": value.nbytes})
        payloads.append(value.tobytes())
    header = json.dumps({"metadata": metadata, "arrays": descriptors}, allow_nan=False).encode()
    if len(header) + sum(map(len, payloads)) > MAX_PACKET:
        raise ValueError("NSLv2 packet exceeds its private transport bound")
    stream.write(struct.pack("!Q", len(header)))
    stream.write(header)
    for payload in payloads:
        stream.write(payload)
    stream.flush()


def read_packet(stream, timeout=None):
    size = struct.unpack("!Q", read_exact(stream, 8, timeout))[0]
    if not 0 < size <= MAX_PACKET:
        raise ValueError("Invalid NSLv2 packet header size")
    document = json.loads(read_exact(stream, size, timeout))
    arrays = {}
    total = size
    for descriptor in document["arrays"]:
        dtype = np.dtype(descriptor["dtype"])
        shape = descriptor["shape"]
        size = descriptor["bytes"]
        if dtype.hasobject or size < 0 or any(not isinstance(n, int) or n < 0 for n in shape):
            raise ValueError("Invalid NSLv2 array descriptor")
        if int(np.prod(shape)) * dtype.itemsize != size or total + size > MAX_PACKET:
            raise ValueError("NSLv2 array payload size mismatch")
        total += size
        name = descriptor["name"]
        if name in arrays:
            raise ValueError("Duplicate NSLv2 array name")
        arrays[name] = np.frombuffer(read_exact(stream, size, timeout), dtype=dtype).reshape(shape).copy()
    return document["metadata"], arrays
