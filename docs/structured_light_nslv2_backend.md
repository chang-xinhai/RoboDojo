# NSLv2 runtime structured-light depth

`camera_config_multimodal.yml` selects `structured_light.backend: nsl_v2`.
Each frozen capture still records the original RGB, renderer ground-truth depth,
and independently rendered left/right IR. NSLv2 receives those same uint8 IR
arrays, the focal length delivered by the actual left-eye render product, the
measured horizontal stereo baseline, and the native physics frame/timestamp.
RGB, ground-truth depth and task state are never inference inputs.

`depth_sl` is float32 depth in metres along the **left IR optical Z axis**, at
the original IR resolution. It is not registered to RGB. Invalid predictions
remain NaN. The adapter converts the model's raw signed disparity using the
actual focal length, preprocessing horizontal scale and baseline; no renderer
depth filling or runtime depth-range clipping is applied. Confidence rejection
is optional and explicitly configured; its default is null.

The shared `reproduction.isaac61.nslv2.client.NSLv2Client` starts an isolated
NSLv2 model worker in the local `nslv2` Conda environment. The supplied profile
uses `ufm_1.8b_psv_step_00250000.pt`, its accompanying `config.yaml`, resize
preprocessing and mixed backbone precision with an FP32 head. Resolved matching
configurations share one serialized worker, avoiding three model copies for
head/left-wrist/right-wrist capture. Capture teardown releases its worker owner;
the last owner shuts down the process.

Each frame retains model/checkpoint/config/source identities, input and output
hashes, preprocessing transform, actual calibration, inference timing and
acquisition clock under `provenance.nslv2`. The configured local Python/model
paths must exist. Loading, inference, shape, dtype or output-contract failures
stop capture. There is no SGBM fallback. Other runtime backend selections are
rejected.

Earlier SGBM reports and the frozen `sl_near_range` audit remain historical
evidence for their recorded backend. They do not validate NSLv2. CPU tests
validate runtime routing, calibration, modality preservation, error handling
and cleanup; actual NSLv2 inference and depth quality require separate model
and recorded-scene validation with the new provenance.
