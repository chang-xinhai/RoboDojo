# Isaac Sim 6.1 / Isaac Lab 3.0 EA

Run these commands from the RoboDojo repository. The runtime is local to
`.venv-isaac61`; Kit data, logs, compiled kernels and caches stay under `.cache/isaac61`.
The existing Isaac Sim 5.1 environment is independent.

## Install

The lock uses Python 3.12, Isaac Sim `6.1.0.0`, PyTorch `2.11.0+cu128`, Warp
`1.16.0` and the official Isaac Lab `v3.0.0-EA` commit in `source-lock.json`.
Isaac Lab's internal Python package version differs from the release name;
the checked-out release and source commit identify this EA version.

On this workstation, CUDA 12.8 and GCC/G++ 12 build cuRobo for the RTX 5090
(`sm_120`). Install `uv`, then:

```bash
bash scripts/install.sh
bash reproduction/isaac61/run.sh python reproduction/download_assets.py --full
```

`setup.sh` installs from `uv.lock`, downloads SHA256-pinned original TacSL
assets, and checks the actual CUDA, physics and RTX camera runtime. The full
asset downloader verifies each file against the pinned Hugging Face manifest
and supports resuming interrupted files. Robot YAML paths are rewritten for
this checkout after verification. Sim wheels and all assets are large; retain
the downloads between runs.

Use `ROBODOJO_ISAAC61_ENV=/absolute/environment/path` to select another installed
environment. `ROBODOJO_CUDA_HOME` overrides the compiler toolkit. The default
evaluation entry `scripts/eval_policy.sh` selects this runtime; an explicit
`ROBODOJO_RUNTIME=legacy` keeps the older installer/evaluation behavior.

## Capture and visualize

```bash
bash reproduction/isaac61/run.sh collect --task stack_bowls --frames 8 --visualize
```

The default output is `reproduction/isaac61/outputs/multimodal-capture`. Choose
a new `--output` directory for each recording; existing frames are never
overwritten. `index.html` provides a local frame slider; PNG and MP4 previews
are generated from the saved, lossless NPZ frames. Rebuild them with:

```bash
bash reproduction/isaac61/run.sh visualize reproduction/isaac61/outputs/multimodal-capture
```

| Location | Recorded fields |
| --- | --- |
| `head`, `left_wrist`, `right_wrist` | `rgb`, `depth_gt`, `depth_sl`, `ir_left`, `ir_right`, intrinsics and world camera poses |
| `left` and `right`, fingers `0` and `1` | official TacSL `rgb`, absolute camera `depth`, SDF indentation, normal and shear forces, calibration and provenance |

`depth_gt` is renderer ground-truth distance; `depth_sl` is independently
reconstructed from projected-pattern stereo matching. Both are float32 metres.
Invalid stereo depth is NaN. IR is an active projected-light simulation obtained
from linear HDR projector-on/off subtraction; it does not model the complete
spectral response of a physical near-infrared sensor. RGB is rendered with
projectors off. Each capture keeps physics frozen through all render phases,
then verifies the same physics frame/timestamp across all modalities. Wall
acquisition times are separate metadata.

TacSL uses NVIDIA's original GelSight R1.5 geometry, calibration, SDF forces
and Taxim rendering. Its compliant pads move with each X5 finger. These force
values are simulated penalty-model estimates. Cloth-only scenes have actual
tactile camera images but no supported rigid SDF force field; those fields
are explicitly null with a reason. See [TacSL details](TACSL.md).

## Acceptance tests

Run GPU jobs sequentially on this workstation:

```bash
bash reproduction/isaac61/run.sh test tests
bash reproduction/isaac61/run.sh smoke
bash reproduction/isaac61/run.sh task-smoke --task stack_bowls
bash reproduction/isaac61/run.sh python scripts/check_structured_light_live.py
bash reproduction/isaac61/run.sh tactile-smoke
bash reproduction/isaac61/run.sh python scripts/check_scene_physics_live.py
bash reproduction/isaac61/run.sh python reproduction/isaac61/smoke_planner.py
bash reproduction/isaac61/run.sh python reproduction/isaac61/validate_migration.py --runtime-import
bash reproduction/isaac61/run.sh python reproduction/isaac61/check_tasks.py
```

The task suite starts one fresh process for each runnable task, initializes and
resets its scene, steps physics and validates all three RGBD cameras. It does
not measure policy success. `--tasks stack_bowls fold_cloth` selects a subset;
`--resume` accepts prior evidence only for matching source contents, runtime
and configuration. JSON reports, NPZ arrays, galleries and detailed logs are
saved for reviewing actual runtime results. A failed GPU assertion exits
nonzero even when Kit exits during cleanup.

Legacy particle cloth was removed by PhysX. Garments now use supported surface
FEM, live nodal state and nodal reset rather than a static substitute. Fluid
retains PBD simulation. The cloth material/solver changed, so benchmark scores
are not directly equivalent to the old particle-cloth runtime. Unsupported
legacy wind/aerodynamic and cloth-fluid settings fail explicitly.
