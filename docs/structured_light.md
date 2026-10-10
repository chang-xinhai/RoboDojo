# Mounted active stereo cameras

Select `camera_config_multimodal` to add a projected-pattern stereo rig to every
RoboDojo RGB camera. The robot manager supplies wrist mount links and poses;
the head uses the existing calibrated scene pose. Each rig has a centered RGB
camera, parallel IR views separated by 50 mm, and an RTX projector. The optional
`structured_light.baseline_m` sets the virtual baseline, not a factory-calibrated
Gemini baseline. Intrinsics are read from the actual camera prim after Isaac Sim
enforces square pixels.

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
Generated assets are cached in ignored `outputs/structured_light/assets`.

RTX does not model near-infrared wavelengths or NIR material reflectance here.
The saved provenance explicitly identifies IR as an active projector radiance
proxy. The rig is a configurable virtual sensor and has no factory calibration
claim. It does not substitute a grayscale RGB image for either stereo view.
The dot layout and radiance scale should be validated for each task scene.
Use `render.antialiasing_mode: Off` in the multimodal environment profile:
temporal AA and DLSS blend the projector phases and suppress the dot pattern.
The initial plane validation on Isaac Sim 6.0.1 used projector intensity 1500
and saturation radiance 5500, yielding approximately 80% valid stereo depth
at the known one-meter range for all three rigs. This is a geometry/projection
debugging result; Isaac Sim 6.1 and moving robot capture require their own runs.

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
