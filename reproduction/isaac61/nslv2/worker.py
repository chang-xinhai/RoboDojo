"""One isolated CUDA model, private pipes, no access to a source HDF5 or GT."""

import contextlib
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from reproduction.isaac61.nslv2 import runtime
from reproduction.isaac61.nslv2.wire import read_packet, write_packet


def main():
    destination = sys.stdout.buffer
    try:
        config, arrays = read_packet(sys.stdin.buffer)
        if arrays:
            raise ValueError("Model configuration must not contain image or GT inputs")
        with contextlib.redirect_stdout(sys.stderr):
            runner, identity = runtime.load_model(config)
        write_packet(destination, {"event": "ready", "identity": identity})
        while True:
            request, arrays = read_packet(sys.stdin.buffer)
            if request["operation"] == "shutdown":
                break
            if request["operation"] != "infer" or set(arrays) != {"ir_left", "ir_right"}:
                raise ValueError("NSLv2 model accepts only the two real IR arrays")
            started = time.perf_counter()
            with contextlib.redirect_stdout(sys.stderr):
                prediction, provenance = runtime.infer(
                    runner, arrays["ir_left"], arrays["ir_right"], config["preprocess_mode"]
                )
            write_packet(
                destination,
                {
                    "event": "prediction",
                    "request_id": request["request_id"],
                    "provenance": provenance,
                    "inference_wall_seconds": time.perf_counter() - started,
                },
                prediction,
            )
    except BaseException as error:
        traceback.print_exc(file=sys.stderr)
        try:
            write_packet(destination, {"event": "error", "type": type(error).__name__, "message": str(error)})
        except Exception:
            pass
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
