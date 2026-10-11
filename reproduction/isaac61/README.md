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
# Validate measured wrist motion while recording every modality:
bash reproduction/isaac61/run.sh collect --task stack_bowls --motion-demo --frames 12 --visualize --output reproduction/isaac61/outputs/moving-wrists
```

The default output is `reproduction/isaac61/outputs/multimodal-capture`. Choose
a new `--output` directory for each recording; existing frames are never
overwritten. `multimodal.html` provides a local frame slider; `multimodal.png`
and `multimodal.mp4` are generated from the saved, lossless `frame_*.npz` files.
`manifest.json` describes the recording and `validation.json` confirms the
completed runtime checks. Rebuild the previews with:

```bash
bash reproduction/isaac61/run.sh visualize reproduction/isaac61/outputs/multimodal-capture
```

Capture and task smoke entrypoints apply the same task configuration as the
production evaluator, including the conveyor and the extra Franka support arm
for interactive tasks. The requested RGBD or multimodal camera profile remains
selected unless the task explicitly overrides it. `--eval-seed` selects the
actual layout/trajectory cohort (default `0`); `--layout-id` selects its episode
and binds the matching episode seed. These settings and effective robot roles
are recorded in the runtime profile. Use `--enable-planner` to initialize
planners in a capture; sensor collection itself does not require them.

| Location | Recorded fields |
| --- | --- |
| `head`, `left_wrist`, `right_wrist` | `rgb`, `depth_gt`, `depth_sl`, `ir_left`, `ir_right`, intrinsics and world camera poses |
| `left` and `right`, fingers `0` and `1` | official TacSL `rgb`, absolute camera `depth`, `penetration_depth`, normal and shear forces, taxel world positions, units and provenance |

`depth_gt` is renderer ground-truth optical-axis depth in the RGB camera frame;
`depth_sl` is independently reconstructed from projected-pattern stereo matching
in the left IR optical frame. Both are float32 metres; compare them only after
transforming between their calibrated frames.
Invalid stereo depth is NaN. IR is an active projected-light simulation obtained
from linear HDR projector-on/off subtraction; it does not model the complete
spectral response of a physical near-infrared sensor. RGB is rendered with
projectors off. Each capture keeps physics frozen through all render phases,
then verifies the same physics frame/timestamp across all modalities. Wall
acquisition times are separate metadata.

`--motion-demo` commands the final arm joint on each wrist with a small sine
motion, bounded by its soft joint limits. Saved frames include actuator targets
and independently measured joint positions; the run fails if either wrist does
not measurably move. The default amplitude is 0.03 radians at 0.5 Hz.

TacSL uses NVIDIA's original GelSight R1.5 geometry, calibration, penalty force
model and Taxim rendering. Contact queries use native PhysX SDF on DirectGPU
scenes, or the declared triangle-mesh backend for particle readback scenes.
Its compliant pads move with each X5 finger. These force values are simulated
penalty-model estimates. Cloth-only scenes have actual
tactile camera images but no supported rigid SDF force field; those fields
are explicitly null with a reason. See [TacSL details](TACSL.md).

## Acceptance tests

Run GPU jobs sequentially on this workstation:

```bash
bash reproduction/isaac61/run.sh test tests
bash reproduction/isaac61/run.sh smoke
bash reproduction/isaac61/run.sh task-smoke --task stack_bowls --check-reset --repeat-envs 2
bash reproduction/isaac61/run.sh python scripts/check_structured_light_live.py
bash reproduction/isaac61/run.sh tactile-smoke
bash reproduction/isaac61/run.sh python scripts/check_scene_physics_live.py
# PBD fluid and FEM cloth use separate PhysX readback configurations:
bash reproduction/isaac61/run.sh python scripts/check_scene_physics_live.py --fluid-only --cpu-readback
bash reproduction/isaac61/run.sh python reproduction/isaac61/smoke_planner.py
bash reproduction/isaac61/run.sh python reproduction/isaac61/validate_migration.py --runtime-import
bash reproduction/isaac61/run.sh python reproduction/isaac61/check_tasks.py
```

The task suite starts one fresh process for each runnable task, initializes and
resets its scene, steps physics and validates all three RGBD cameras. It also
generates task instructions, executes every registered reward predicate and
the normal reward manager step, and rejects known fatal native physics signals
in the original process log. It does not measure policy success.
`--tasks stack_bowls fold_clothes` selects a subset;
`--resume` accepts prior evidence only for matching source contents, runtime
and configuration. JSON reports, NPZ arrays, galleries and detailed logs are
saved for reviewing actual runtime results. A failed GPU assertion exits
nonzero even when Kit exits during cleanup.

Legacy particle cloth was removed by PhysX. Garments now use supported surface
FEM, live nodal state and nodal reset rather than a static substitute. Fluid
retains PBD simulation. The cloth material/solver changed, so benchmark scores
are not directly equivalent to the old particle-cloth runtime. Unsupported
legacy wind/aerodynamic and cloth-fluid settings fail explicitly.

## Physics and pose compatibility

Task layouts are installed before scene construction and explicitly restored
after `TaskEnv.reset()`, which first stages existing CUDA objects offscreen.
Acceptance reports verify actual native body positions, live mesh vertices,
surface nodes or PBD particle readback rather than only counting USD prims.
Retained CUDA objects keep their original prim paths when layout metadata is
reloaded, preserving labels and functional frames. Evaluators preload the
finite remaining replay cohort's retained asset variants and maximum simultaneous
counts before the first physics warmup. Variants include the collider asset,
mass/friction settings, scale and visual configuration, so a changed physical
configuration selects its own native body. Decreasing the active count parks
unused cached bodies; an unprepared asset variant or increased cached quantity
requires a fresh TaskEnv and fails before metadata is changed.
Captures reject nonfinite state, objects more than 100 m from their environment
origin, or centers more than 5 m from the saved placement.

Rigid bodies reset linear and angular velocity together through the native
six-component tensor API; articulated objects also clear joint velocities.
X5 reset writes actual joint state and resets actuator state before assigning
drive targets. The motion/reset smoke moves both arms and real bodies, checks
exact immediate velocity/pose restoration, then exercises a full task reset.
Its repeated-environment option validates two independent scenes and RGBD
captures in one Kit app. Lab 3 owns PhysX and stage teardown; RoboDojo releases
wrappers before closing that context and creates a new pure-USD authoring
stage for the next environment.

The finite-cohort regression can also switch canonical models and query actual
PhysX masses across a separately pooled mass variant:

```bash
bash reproduction/isaac61/run.sh task-smoke --task stack_bowls \
  --replay-layout-ids 1 --probe-mass-variant --check-reset --repeat-envs 2
```

Rigid mesh bounds, Dynamic mesh samples and articulated functional frames use
live native body poses with static mesh-to-body affine transforms. Pure Xform
Dynamic assets explicitly read Fabric. USD transforms are authoritative only
before the first physics warmup. Public RoboDojo poses remain WXYZ; native
PhysX tensor poses are XYZW and are converted at the boundary.

PBD fluid needs GPU dynamics with host readback enabled. PhysX then exposes
native host tensor buffers while Lab 3 processes configured CUDA tensors.
`utils/particle_readback_bridge.py` copies real readback buffers into stable
CUDA shadows and sends actual state writes back to native host buffers. This
bridge is installed only for particle-readback contexts; ordinary DirectGPU
tasks keep the official Warp view. Host readback lacks native SDF and surface
deformable tensor support: particle tasks use the explicitly labelled real
triangle-mesh TacSL query backend described in [TacSL details](TACSL.md), and
cloth/fluid tasks use separate contexts. These force estimates cover rigid
colliders rather than fluid pressure.
