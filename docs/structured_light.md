# Mounted active stereo cameras

Select `camera_config_multimodal` to add a projected-pattern stereo rig to every
RoboDojo RGB camera. The robot manager supplies wrist mount links and poses;
the head uses the existing calibrated scene pose. Each rig has a centered RGB
camera, parallel IR views separated by 50 mm, and an RTX projector. The optional
`structured_light.baseline_m` sets the virtual baseline, not a factory-calibrated
Gemini baseline. Intrinsics are read from the actual camera prim after Isaac Sim
enforces square pixels.
Saved world poses, view matrices, projections, and per-view intrinsics come
from `CameraParams` on each exact render product. This follows rendered Fabric
transforms when wrist cameras move. Capture rejects a stereo pair whose actual
view transforms do not preserve the configured parallel horizontal baseline.

`CameraManager.capture_multimodal(env_id)` returns `cam_head`, `cam_left_wrist`,
and `cam_right_wrist` packets. RGB is `uint8 HWC`; IR left/right are `uint8 HW`;
both depth outputs are `float32 HW` in meters. `depth_gt` is RTX
`distance_to_image_plane` in the RGB optical frame. `depth_sl` is reconstructed
with StereoSGBM from the rectified IR pair in the left IR optical frame. Invalid
SL pixels are NaN and are never filled from simulator ground truth.

The runtime authors real RTX `RectLight` projectors with a deterministic dot
texture, an octahedral direction-to-pattern UV EXR lookup, `ShapingAPI`,
`isProjector=True`, and `visibleInPrimaryRay=False`, following NVIDIA's
[Isaac Sim 6.1 structured-light documentation](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/sensors/isaacsim_sensors_camera_structured_light.html).
Dedicated left/right cameras capture linear HDR images with each projector
off and on. Ambient subtraction isolates the spatially projected light,
including geometry, occlusion, and surface response. A shared fixed sensor gain
preserves stereo photometric consistency. The default projection coverage
(112 by 80 degrees) follows DepthUMI's Gemini335 proxy LUT implementation.
Generated assets are cached in ignored `.cache/isaac61/structured_light/assets`.

RTX does not model near-infrared wavelengths or NIR material reflectance here.
The saved provenance explicitly identifies IR as an active projector radiance
proxy. The rig is a configurable virtual sensor and has no factory calibration
claim. It does not substitute a grayscale RGB image for either stereo view.
The dot layout and radiance scale should be validated for each task scene.
Use `render.antialiasing_mode: Off` in the multimodal environment profile:
temporal AA and DLSS blend the projector phases and suppress the dot pattern.
Exact Isaac Sim `6.1.0.0` plane and rotated-rig validation passed with projector
intensity 1500 and saturation radiance 5500. All three rigs reconstructed the
known one-meter plane with median depth 1.0 m and 79.98–80.00% valid pixels.
At 1.25 m with the entire rig rolled 30 degrees, median reconstructed depth
was 1.28 m (2.4% median error), with 79.98–80.00% valid pixels. Both runs had
zero saturated IR pixels; rendered optical-pose errors were zero for the
unrotated rig and below `2.85e-8` for the rotated rig. These checks cover the
native projector, stereo geometry, frame conventions, exposure, and delivered
render-product calibration. Moving robot capture and task-specific occlusions
require the integrated task validation.

The local evidence includes complete calibration matrices, numeric reports,
raw image/depth NPZ files, and labeled galleries:

- `reproduction/isaac61/outputs/sl-plane-61/{report.json,structured_light.png}`
- `reproduction/isaac61/outputs/sl-rotated-61/{report.json,structured_light.png}`

Reproduce both checks using the isolated Kit cache configured by the wrapper:

```sh
bash reproduction/isaac61/run.sh python scripts/check_structured_light_live.py \
  --output reproduction/isaac61/outputs/sl-plane-61
bash reproduction/isaac61/run.sh python scripts/check_structured_light_live.py \
  --distance 1.25 --roll-deg 30 --output reproduction/isaac61/outputs/sl-rotated-61
```

Capture freezes physics while rendering the ambient and active phases. Each
projector is enabled separately to prevent interference between head and wrist
emitters. All returned packets carry the same simulation `timestamp_ns`;
phase-specific host acquisition timestamps are retained as provenance.
The function checks that simulation time did not advance and switches every
projector off in `finally`, including renderer failures. Launch with the existing
`add_zero_delay_kit_args` helper so each phase waits for its current render.

CPU contracts are checked with:

```sh
python -m unittest discover -s tests -p test_structured_light.py -v
```

These tests verify stereo metric geometry, illumination isolation, invalid-depth
handling, deterministic PNG/EXR assets, and cleanup. They do not certify RTX
rendering, projection coverage, moving-wrist extrinsics, or native scene
appearance; those require the live Isaac Sim validation run.
