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
including geometry, occlusion, and surface response. Each eye's own passive
exposure is combined with this response using the declared electron/ADC model,
shot/read noise and one bounded shared pair gain. The default projection coverage
(112 by 80 degrees) follows DepthUMI's Gemini335 proxy LUT implementation.
Generated assets are cached in ignored `.cache/isaac61/structured_light/assets`.

RTX does not model near-infrared wavelengths or NIR material reflectance here.
The saved provenance explicitly identifies IR as an active projector radiance
proxy. The rig is a configurable virtual sensor and has no factory calibration
claim. It does not substitute a grayscale RGB image for either stereo view.
The dot layout and radiance scale should be validated for each task scene.
The environment leaves AA off between acquisitions. Capture scopes RGB DLAA
and IR FXAA separately, restoring prior global AA, denoiser and exposure in
`finally`. RGB and IR each average eight actual render deliveries at one frozen
pose; the two projector phases settle independently with the IR denoiser off.
This avoids temporal blending of ambient and projected patterns. No spatial
image filter or ground-truth depth filling is applied.
Isaac Lab's headless depth-only renderer sets the global
`/rtx/sdg/force/disableColorRender` flag. TacSL depth cameras trigger this
optimization even when RGB/HDR products also exist. The shared capture entry
restores color rendering before every acquisition. It then waits for actual
requested RGB and finite positive depth buffers within a bounded render count,
without advancing physics or substituting an image. Missing startup buffers
are rejected before launching the tiled reshape kernel; timeout errors identify
the pending camera and channel.
Exact Isaac Sim `6.1.0.0` plane and rotated-rig validation passed with projector
intensity 1500 and saturation radiance 5500. All three rigs reconstructed the
known one-meter plane with median depth 1.0 m and 79.98–80.00% valid pixels.
At 1.25 m with the entire rig rolled 30 degrees, median reconstructed depth
was 1.28 m (2.4% offset of the median depth), with 79.98–80.00% valid pixels.
The pixelwise median relative error was 4.30%, as recorded separately in each
run's `strict-audit.json`. Both runs had
zero saturated IR pixels; rendered optical-pose errors were zero for the
unrotated rig and below `2.85e-8` for the rotated rig. These checks cover the
native projector, stereo geometry, frame conventions, exposure, and delivered
render-product calibration. Moving robot capture and task-specific occlusions
require the integrated task validation.

A further exact `6.1.0.0` run rotates the actual target plane 30 degrees about
world Y while its visible surface center remains at the world origin. An
analytic ray-plane reference uses each delivered optical pose and intrinsic
matrix, including raster pixel centers. Across head, left wrist, and right
wrist, median relative reconstruction errors were 2.30%, 1.91%, and 2.64%;
center-ROI absolute errors were 1.44 cm, 0.065 cm, and 3.17 cm. Valid stereo
coverage was 79.88–79.93%. Ground-truth depth agreed with the analytic reference
within `9.5e-8` m median absolute error. The reconstructed depth's 10th-to-90th
percentile ranges were 1.03, 0.95, and 1.12 m, confirming reconstruction of a
surface with varying depth. The live script enforces median relative error at
most 5%, center-ROI absolute error at most 8 cm, ground-truth error below 1 mm,
and both ground-truth and reconstructed depth variation above 2 cm.

The local evidence includes complete calibration matrices, numeric reports,
raw image/depth NPZ files, and labeled galleries:

- `reproduction/isaac61/outputs/sl-plane-61/{report.json,structured_light.png}`
- `reproduction/isaac61/outputs/sl-rotated-61/{report.json,structured_light.png}`
- `reproduction/isaac61/outputs/sl-tilted-plane-61/{report.json,structured_light.png}`

The complete canonical `stack_bowls` collector also passed on exact Isaac Sim
6.1 with three mounted RGB/GT-depth/IR-pair/SL-depth rigs and four TacSL finger
sensors enabled together. The one-frame production integration check reports
76.09%, 61.24%, and 61.93% valid SL depth for head, left wrist, and right wrist.
All cameras and tactile samples share physics frame 363 and simulation timestamp
`1452000000` ns. Its raw recording, numeric validation, and gallery are in
`reproduction/isaac61/outputs/sl-integration-color-guard-v5/`.

Reproduce the checks using the isolated Kit cache configured by the wrapper:

```sh
bash reproduction/isaac61/run.sh python scripts/check_structured_light_live.py \
  --output reproduction/isaac61/outputs/sl-plane-61
bash reproduction/isaac61/run.sh python scripts/check_structured_light_live.py \
  --distance 1.25 --roll-deg 30 --output reproduction/isaac61/outputs/sl-rotated-61
bash reproduction/isaac61/run.sh python scripts/check_structured_light_live.py \
  --plane-tilt-deg 30 --output reproduction/isaac61/outputs/sl-tilted-plane-61
```

Capture freezes physics while rendering the ambient and active phases. Each
projector is enabled separately to prevent interference between head and wrist
emitters. All returned packets carry the same simulation `timestamp_ns`;
phase-specific host acquisition timestamps are retained as provenance.
The function checks that simulation time did not advance and switches every
projector off in `finally`, including renderer failures. Launch with the existing
`add_zero_delay_kit_args` helper so each phase waits for its current render.

Each native camera product has its own Hydra drawable-change completion fence.
Capture waits for every requested product to complete, including both stereo
eyes, with zero-delay Kit settings fencing annotator dispatch. The scene's SDG
reference time can stay constant while new render samples arrive at one frozen
pose; it is not used as the sample counter for these products.

The current multimodal profile follows a persisted DepthUMI full-modal recording
configuration from 2026-08-03: ISO 400, 20 ms, f/10, responsivity 1.10267, RGB
DLAA, IR FXAA, eight render samples, head projector 3500 and wrist projectors
600. It vendors the original 848x480 Gemini335L PNG with SHA-256
`29e00bb08305fd575486e9726fd3d733141d0a00bca4e0a7d5754224f49dc2fc`,
using a 2048-square direction LUT. This adjacent recording's persisted metadata
is evidence for the recipe; the user's matching August 2 video lacks its original
effective-renderer configuration snapshot, so its exact exposure is not claimed.
The reference video's smooth depth column is renderer GT, not stereo depth.

On exact Isaac Sim 6.1, the 640x480 profile passes both a one-meter front-facing
plane and a true 30-degree tilted target. The tilted target's head/left/right
median relative stereo errors are 0.9504%, 0.7915%, and 1.2185%; center errors
are 0.668, 0.963, and 1.629 cm. Valid stereo coverage is 79.54%, 61.98%, and
66.83%, with invalid pixels retained as NaN. GT agrees with the analytic plane
within 1e-7 m median. Reports preserve actual live phase settings and raw
projector-only radiance in `quality-iteration/sl-reference-{plane-v3,tilted-v1}/`.

The complete three-frame StackBowls recording with these settings passed exact
6.1 acquisition and lossless HDF5 validation. Its PNG/video/HTML preview reads
`quality-iteration/native-three-v1/episode.hdf5` directly. A separate controlled
probe holds physics frame 353 fixed and repeats each mode three times, changing
only RGB AA off with one actual render sample versus DLAA with eight samples.
Mean absolute temporal RGB differences fall from 38.160 to 2.146 DN for head,
41.238 to 1.527 for left wrist and 38.647 to 1.487 for right wrist, reductions
of 94.38%, 96.30% and 96.15%. Exposure, scene geometry and projector settings
remain fixed. These are temporal-noise results, not spectral sensor accuracy
or a claim that missing stereo depth has been reconstructed. Raw samples,
actual phase settings, metrics and an unchanged-RGB comparison gallery are in
`quality-iteration/frozen-rgb-v2/`.

```sh
bash reproduction/isaac61/run.sh python scripts/check_structured_light_live.py \
  --sensor-profile env_cfg/camera/camera_config_multimodal.yml --resolution 640 480 \
  --plane-tilt-deg 30 --output reproduction/isaac61/outputs/quality-iteration/sl-reference-tilted
```

CPU contracts are checked with:

```sh
python -m unittest discover -s tests -p 'test_structured_light*.py' -v
```

These tests verify stereo metric geometry, illumination isolation, invalid-depth
handling, deterministic PNG/EXR assets, and cleanup. They do not certify RTX
rendering, projection coverage, moving-wrist extrinsics, or native scene
appearance; those require the live Isaac Sim validation run.
