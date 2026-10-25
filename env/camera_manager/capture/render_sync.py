"""Make ``annotator.get_data()`` return the frame for the current physics state.

``get_data()`` hands back whatever the SDG graph last wrote; it never waits.
How many ``env.render()`` calls it takes for a physics change to reach that
buffer depends on Kit's update order:

* With ``/app/updateOrder/checkForHydraRenderComplete=1000`` plus
  ``/app/renderer/waitIdle`` and ``/app/hydraEngine/waitIdle`` (the settings
  Isaac Lab ships in its ``*.rendering.kit`` experiences under "Avoids frame
  offset issue"), an app update blocks until its own frame is rendered and
  dispatched. One render after the physics step is then exactly current,
  regardless of how many render products exist.
* Without them the frame produced by an update reflects the state of the
  previous update, and under load frames are dropped or queued, so the buffer
  can be several frames old. A fixed render count cannot fix that reliably.

This project launches with ``isaaclab.python.kit``, which lacks those settings,
so the entrypoints append them through :func:`add_zero_delay_kit_args`.

:func:`wait_for_latest_cameras` uses the provided native render-product
completion clock when available. Structured-light capture subscribes to each
product's Hydra drawable-change event: scene reference time can stay constant
while independent samples of a frozen pose arrive. The legacy fallback watches
SDG reference time, the clock polled by Replicator's orchestrator. Neither
completion clock is compared with the PhysX timestamp. ``env.render()`` does
not advance physics.
"""

from __future__ import annotations

from contextlib import contextmanager
import sys

import numpy as np

_DISPATCHER_PATH = "/Render/PostProcess/SDGPipeline/PostProcessDispatcher"

# Kit settings that make each app update wait for its own rendered frame.
ZERO_DELAY_KIT_SETTINGS = {
    "/app/updateOrder/checkForHydraRenderComplete": "1000",
    "/app/renderer/waitIdle": "true",
    "/app/hydraEngine/waitIdle": "true",
}

_warned_missing_settings = False
_reported_active = False


class CameraFrameNotReady(RuntimeError):
    """A render product has not delivered its requested image buffer yet."""


class RenderProductDeliveryFence:
    """Count actual native Hydra completions for each requested render product.

    SDG reference time describes scene time and can stay constant across
    independent samples of a frozen pose. Drawable-change events identify
    real render completion even when that clock is intentionally frozen.
    Zero-delay Kit settings fence downstream annotator dispatch before reads.
    """

    def __init__(self, sensors=(), *, render_products=()):
        import carb.eventdispatcher
        import omni.hydratexture

        products = [sensor._hydra_texture for sensor in sensors] + list(render_products)
        self.counts = [0] * len(products)
        self.subscriptions = []
        self.paths = []
        for index, product in enumerate(products):
            texture = product.hydra_texture
            self.paths.append(str(product.path))

            def on_event(event, camera_index=index):
                del event
                self.counts[camera_index] += 1

            subscription = carb.eventdispatcher.get_eventdispatcher().observe_event(
                observer_name=f"RoboDojoRenderFence:{id(self)}:{index}",
                event_name=omni.hydratexture.GLOBAL_EVENT_DRAWABLE_CHANGED,
                filter=texture.get_event_key(),
                on_event=on_event,
            )
            self.subscriptions.append(subscription)

    def snapshot(self):
        return tuple(self.counts)

    def close(self):
        self.subscriptions.clear()


@contextmanager
def _requested_product_streaming_scope():
    """Let the bounded native-product fence own readiness during acquisition.

    Lab 3's global streaming-idle wait can remain busy across sensor/material
    edits, even after requested products deliver. Keep its synchronous update,
    then require real product completions, ready payloads and finished USD file
    loads in wait_for_camera_data. Never change the global policy outside it.
    """
    native = sys.modules.get("isaaclab_physx.renderers.isaac_rtx_renderer_utils")
    if native is None:
        yield
        return
    from omni.kit.app import get_app

    original = native._wait_for_streaming_complete
    native._wait_for_streaming_complete = lambda: get_app().update()
    try:
        yield
    finally:
        native._wait_for_streaming_complete = original


def _pending_stage_files():
    if "isaaclab_physx.renderers.isaac_rtx_renderer_utils" not in sys.modules:
        return []
    from omni.usd import get_context

    message, loaded, total = get_context().get_stage_loading_status()
    return [f"USD files loading {loaded}/{total}: {message}"] if loaded < total else []


def ensure_camera_color_render_enabled(settings=None):
    """Restore color rendering after Isaac Lab creates headless depth-only cameras.

    Sim 6's IsaacRtxRenderer disables color process-wide for depth-only sensors.
    TacSL uses these sensors alongside RGB and projected HDR camera products.
    Restore their required color AOV before each render-only acquisition, so
    late depth-camera creation cannot leave RGB/HDR permanently black.
    """
    if settings is None:
        import carb.settings

        settings = carb.settings.get_settings()
    settings.set_bool("/rtx/sdg/force/disableColorRender", False)


def pending_camera_payloads(data):
    """Identify uninitialized requested RGB/depth buffers, without filling them."""
    pending = []
    for camera_id, camera in enumerate(data):
        for annotator, samples in camera.items():
            if annotator not in {"rgb", "rgba", "depth", "depth_gt", "distance_to_image_plane", "distance_to_camera"}:
                continue
            for env_id, sample in enumerate(samples):
                value = sample.get("data") if isinstance(sample, dict) else sample
                if hasattr(value, "detach"):
                    value = value.detach()
                if hasattr(value, "cpu"):
                    value = value.cpu()
                if hasattr(value, "numpy"):
                    value = value.numpy()
                image = np.asarray(value) if value is not None else np.array([])
                label = f"camera[{camera_id}]/env[{env_id}]/{annotator}"
                if image.size == 0:
                    pending.append(label + ": empty")
                elif annotator in {"rgb", "rgba"}:
                    color = image[..., :3]
                    if not np.isfinite(color).all() or not np.any(color > 0):
                        pending.append(label + ": no finite nonzero RGB signal")
                elif not np.any(np.isfinite(image) & (image > 0)):
                    pending.append(label + ": no finite positive depth")
    return pending


def wait_for_camera_data(env, read_data, *, physics_state, min_passes=1, max_passes=32, delivery_clock=None):
    """Acquire ready payloads behind a bounded, independently observed fence."""
    if delivery_clock is not None:
        if not zero_delay_active():
            raise RuntimeError("Native-product acquisition requires synchronous zero-delay rendering")
        with _requested_product_streaming_scope():
            return _wait_for_camera_data(
                env, read_data, physics_state=physics_state, min_passes=min_passes,
                max_passes=max_passes, delivery_clock=delivery_clock,
            )
    return _wait_for_camera_data(
        env, read_data, physics_state=physics_state, min_passes=min_passes,
        max_passes=max_passes, delivery_clock=delivery_clock,
    )


def _wait_for_camera_data(env, read_data, *, physics_state, min_passes=1, max_passes=32, delivery_clock=None):
    """Bounded render-only wait for actual requested image buffers at one physics state.

    A dispatcher delivery alone does not mean a newly created render product
    has compiled its shaders and populated every annotator. Return only real
    buffers; reject physics advancement and name pending channels on timeout.
    """
    before = physics_state()
    ensure_camera_color_render_enabled()
    passes = wait_for_latest_cameras(env, min_passes=min_passes, max_passes=max_passes, delivery_clock=delivery_clock)
    pending = []
    while True:
        if physics_state() != before:
            raise RuntimeError("Physics advanced while waiting for camera buffers")
        try:
            data = read_data()
            pending = pending_camera_payloads(data)
            if delivery_clock is not None:
                pending += _pending_stage_files()
        except CameraFrameNotReady as exc:
            pending = [str(exc)]
        if not pending:
            if physics_state() != before:
                raise RuntimeError("Physics advanced while reading camera buffers")
            return data, passes
        if passes >= max_passes:
            raise CameraFrameNotReady(
                f"Camera buffers not ready after {passes} frozen-physics render passes: " + "; ".join(pending)
            )
        env.render()
        passes += 1


def zero_delay_kit_args() -> str:
    """Kit command-line overrides for the zero-delay render settings."""
    return " ".join(f"--{key}={value}" for key, value in ZERO_DELAY_KIT_SETTINGS.items())


def add_zero_delay_kit_args(args) -> None:
    """Append the zero-delay overrides to an ``AppLauncher`` argparse namespace.

    Must run before ``AppLauncher(args)``; the update-order setting is read at
    startup and cannot be changed afterwards.
    """
    current = getattr(args, "kit_args", None) or ""
    missing = [f"--{key}={value}" for key, value in ZERO_DELAY_KIT_SETTINGS.items() if f"--{key}=" not in current]
    if missing:
        args.kit_args = (current + " " + " ".join(missing)).strip()


def zero_delay_active() -> bool:
    """True when the running Kit app has the zero-delay settings applied."""
    try:
        import carb.settings

        settings = carb.settings.get_settings()
    except Exception:
        return False
    order = settings.get("/app/updateOrder/checkForHydraRenderComplete")
    try:
        if order is None or int(order) < 1000:
            return False
    except (TypeError, ValueError):
        return False
    return _truthy(settings.get("/app/renderer/waitIdle")) and _truthy(settings.get("/app/hydraEngine/waitIdle"))


def _truthy(value) -> bool:
    """Command-line overrides may arrive as strings; ``bool("false")`` is True."""
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def dispatcher_time() -> float | None:
    """Reference time of the last frame the SDG pipeline dispatched."""
    try:
        import omni.graph.core as og

        node = og.get_node_by_path(_DISPATCHER_PATH)
        if node is None:
            return None
        den = node.get_attribute("outputs:referenceTimeDenominator").get()
        if not den:
            return None
        return float(node.get_attribute("outputs:referenceTimeNumerator").get()) / float(den)
    except Exception:
        return None


def _warn_missing_settings() -> None:
    global _warned_missing_settings
    if _warned_missing_settings:
        return
    _warned_missing_settings = True
    print(
        "[render_sync] zero-delay Kit settings are not active; camera frames "
        "lag physics by at least one render. Launch with: " + zero_delay_kit_args()
    )


def _report_active() -> None:
    global _reported_active
    if _reported_active:
        return
    _reported_active = True
    print("[render_sync] zero-delay Kit settings active; one delivered frame per capture.")


def wait_for_latest_cameras(
    env,
    capture_manager=None,
    min_passes: int = 1,
    max_passes: int = 16,
    delivery_clock=None,
) -> int:
    """Render until the dispatcher has delivered ``min_passes`` new frames.

    With the zero-delay settings one delivered frame carries the current
    physics state. Without them the pipeline is one frame deep, so one more
    delivery is required. Returns the number of ``env.render()`` calls made.
    """
    del capture_manager  # kept for call compatibility; the fence is global
    required = max(1, int(min_passes))
    if not zero_delay_active():
        _warn_missing_settings()
        required += 1
    else:
        _report_active()
    max_passes = max(required, int(max_passes))

    last_stamp = dispatcher_time() if delivery_clock is None else delivery_clock()
    initial_stamp = last_stamp
    delivered = 0
    passes = 0
    while passes < max_passes:
        env.render()
        passes += 1
        stamp = dispatcher_time() if delivery_clock is None else delivery_clock()
        if delivery_clock is not None:
            # Every requested render product must have actually completed.
            # A busy viewport or one fast camera cannot satisfy another eye.
            if not stamp or len(stamp) != len(initial_stamp):
                raise CameraFrameNotReady("Native render delivery clock has no stable camera set")
            delivered = min(current - initial for current, initial in zip(stamp, initial_stamp, strict=True))
        elif stamp is None or last_stamp is None:
            # No dispatcher clock to watch: fall back to counting renders.
            delivered = passes
        elif stamp != last_stamp:
            delivered += 1
        last_stamp = stamp
        if delivered >= required:
            break
    if delivered < required:
        raise CameraFrameNotReady(
            f"Camera dispatcher delivered {delivered}/{required} fresh frames after {passes} render passes"
        )
    return passes
