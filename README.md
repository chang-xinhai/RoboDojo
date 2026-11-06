<div align="center">

<img src="https://media.luminis-sim.com/media/challenge/posters/robodojo_logo.png"></img>

<h2 align="center">RoboDojo: A Unified Sim-and-Real Benchmark for Comprehensive Evaluation of Generalist Robot Manipulation Policies</h2>

<h2 align="center"><a href="https://robodojo-benchmark.com/">Webpage</a> | <a href="https://robodojo-benchmark.com/doc/">Document</a> | <a href="https://arxiv.org/abs/2607.04434">Paper</a> | <a href="https://robodojo-benchmark.com/community">Community</a> | <a href="https://robodojo-benchmark.com/leaderboard">Leaderboard</a></h2>

</div>

https://private-user-images.githubusercontent.com/88101805/619409345-cc074c5d-4567-4418-8a29-1385aaba9d5b.mp4

## News

- **September 16–17, 2026**
  - Fixed a single-frame discrepancy in `get_obs` and updated both the code and Hugging Face data. Rerunning evaluations across multiple models showed that the experimental results remain essentially unchanged.
  - Together with RoboTwin, fixed the channel ordering of the RGB data byte stream to keep both projects consistent. Please use this update with XPolicyLab commit [`bb9a0b5`](https://github.com/XPolicyLab/XPolicyLab/commit/bb9a0b5f5136a74503b679af830bfd0a3a837d5c) or later.
  - Updated the simulation assets for the `swap_T` task to make the objects less prone to slipping.
- **July 6, 2026** — Released the [RoboDojo paper](https://arxiv.org/abs/2607.04434) and open-sourced the code.

## ✨ Highlights

<p align="center">
  <img src="https://media.luminis-sim.com/media/home/teaser.png" width="70%"></img>
</p>

<p align="center"><em>Overview of RoboDojo. RoboDojo unifies efficient simulation evaluation and reproducible real-world testing for generalist robot manipulation, covering 42 simulation tasks, 18 real-world tasks, heterogeneous parallel simulation, RoboDojo-RealEval, XPolicyLab, and a continuously updated leaderboard.</em></p>

> RoboDojo is **eval-only** in this release: it provides the simulator client, benchmark tasks, asset/config validation, and result artifacts. Policy integration and policy servers are owned by [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab/blob/main/README.md).

- 🌐 **Unified sim-and-real benchmark** — 42 simulation tasks and 18 real-world tasks across 3 robot embodiments for generalist robot manipulation.
- 🧭 **Five capability dimensions** — Generalization, Memory, Precision, Long-Horizon, and Open, designed to probe different skills rather than simple object or layout reskins.
- 🧗 **Challenging by design** — intentionally hard, diverse, long-horizon tasks that expose failures hidden by simpler benchmarks.
- ⚡ **Heterogeneous parallel simulation** — runs different tasks, scenes, and processes concurrently on Isaac Sim for fast, scalable feedback.
- 🧱 **Physically grounded assets** — rigid, articulated, and deformable objects in a single configuration-driven scene.
- 🤖 **Integrate once, evaluate everywhere** — [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab/blob/main/README.md) unifies 40+ policies behind one interface for both simulation and real-world runs.
- 📊 **Reproducible & leaderboard-ready** — seed-controlled layouts and one-command `summarize` aggregation into a leaderboard table.

## 📚 Documentation

The [RoboDojo documentation](https://robodojo-benchmark.com/doc/) is the canonical reference. Key sections:

| Section | Description |
| :-- | :-- |
| [Usage Overview](https://robodojo-benchmark.com/doc/usage/) | End-to-end walkthrough of the evaluation workflow. |
| [Installation & Downloading (Assets and Data)](https://robodojo-benchmark.com/doc/usage/install-and-download/) | Environment setup and downloading robot/object/layout assets/training data. |
| [Quick Evaluation](https://robodojo-benchmark.com/doc/usage/quick-evaluation/) | Quickly dispatch XPolicyLab to run a policy for testing. |
| [XPolicyLab](https://robodojo-benchmark.com/doc/usage/xpolicylab/) | Integrates a large collection of policies and defines how to integrate new ones. |
| [Simulation Tasks Details](https://robodojo-benchmark.com/doc/sim-tasks/) | The 42 Isaac Sim tasks across five capability dimensions. |
| [Real Robot Tasks Details](https://robodojo-benchmark.com/doc/real-tasks/) | The 18 real-world tasks on Piper X, Piper, and ARX X5. |
| [Configurations](https://robodojo-benchmark.com/doc/usage/configurations/) | Simulator, scene, robot, and camera configuration options. |
| [Common Issues](https://robodojo-benchmark.com/doc/common-issue/) | Troubleshooting for installation, assets, GPU memory, and evaluation. |

## Native Isaac Sim 6.2 migration

This branch targets Python 3.12, Isaac Sim 6.2.0.0 and official Isaac Lab
v3.0.0 with Isaac Sim PhysX and Isaac RTX. The source submodules are fixed by
Git commits. Lab's upstream `uv.lock` owns Torch 2.11 (CUDA 13.0), Warp 1.17
and NumPy 2.x. One compatibility exception keeps NumPy `>=2,<2.5`: Lab's lock
contains NumPy 2.5.1 with Numba 0.64, whose import guard requires NumPy below
2.5. Other runtime versions follow the upstream lock.

```bash
# Requires conda on PATH; installs the simulator env, not policy server envs.
bash scripts/install.sh --install
bash scripts/robodojo.sh doctor --skip-policy
```

Existing Python 3.11 environments fail the installer guard. Use a separate
Python 3.12 environment (`export ROBODOJO_SIM_ENV=<name>`) when keeping an older
installation. Resume with `bash scripts/install.sh --from isaaclab`; set
`ROBODOJO_SIM_ENV` when using a nondefault environment. Sim and Lab install together from the
upstream lock; `--from isaacsim` is an alias. See [scripts](scripts/README.md).

Tasks use native Lab assets, actuators and cameras with a Lab-owned simulation
lifecycle. Task definitions and success criteria are retained. Garments use the
surface-FEM baseline described below. Validation uses the installed environment;
a passing doctor checks
dependencies, while replay and policy runs check simulation behavior. No fresh
environment installation or Docker build has been validated. The
[Docker recipe](docker/README.md) targets Sim 6.2, Python 3.12 and CUDA 13 using
the same installer. Policy servers retain independent environments.

Native poses use XYZW (`[x, y, z, qx, qy, qz, qw]`). Historical layouts and
asset metadata are converted at their readers without rewriting shared assets.
Native observations declare `data_format_version: v2.0`, XYZW quaternions and
optical ROS camera-to-world extrinsics. Existing XPolicyLab checkpoints retain
their v1 WXYZ poses and OpenGL extrinsics through the evaluation boundary;
joint actions and RGB channel order are unchanged. A policy designed for the
native v2 contract can set `quaternion_order: xyzw` in its deploy configuration.
Camera intrinsic matrices always describe the actual native projection.

### Cloth physics baseline

`fold_clothes` and `fold_clothes_random` now use PhysX surface FEM through Lab's
`DeformableObject`. This replaces PBD cloth and establishes a new physical
baseline; the old spring and particle coefficients are not equivalent FEM
material parameters. Asset transforms, mesh vertex IDs used by the rewards,
saved layouts and success conditions are retained. PhysX requires unique mesh
positions: exact duplicate vertices are merged only in the runtime instance,
with a mapping back to the original reward vertex IDs. Shared USD assets and
face-corner appearance data remain unchanged.

The garment baseline explicitly sets density 300 kg/m³, Young's modulus 1 MPa,
Poisson's ratio 0.3, thickness 0.001 m, static/dynamic friction 0.5, elasticity
damping 0.05 and bend damping 0.05. These replace Lab's generic surface material
defaults, including its 0.01 m thickness. Surface stretch, shear and bend stiffness
overrides remain zero, so PhysX derives them from the modulus and thickness;
zero does not disable bending resistance. Self-collision is enabled for folding.
Collider rest/contact offsets are explicitly 0.001/0.003 m; automatic offsets
produced inflated garments in the same-layout comparison. Other body solver
settings use current SDK defaults. Cloth scenes reserve at
least 4,194,304 surface contacts per garment, following the official PhysX
surface example after the default capacity overflowed on these meshes. This is
a memory budget, not a material parameter. Native overrides belong
in `physics.fem_config.material`, `.body`, `.surface` or `.collision`. Collision
overrides use native PhysX collider settings such as `rest_offset` and
`contact_offset`; omitted fields retain the garment defaults above.
The old garment PBD
configuration and shared cloth/fluid particle-system mode are not supported.
These values define this migration baseline, not a fit to measured fabric.

### Optional structured-light capture

Add `--structured-light` to `scripts/robodojo.sh eval` or `client` to capture
grayscale stereo alongside the ordinary RGB cameras. The feature is off by
default. Select another local pattern with
`--structured-light --structured-light-pattern /path/to/pattern.png`;
`scripts/eval_policy.sh` and the Python evaluation entry accept the same flags.

The default Gemini pattern comes from the supplied
[reference revision](https://github.com/chang-xinhai/RoboDojo/tree/fedfd65cb5fd434b911e985f9dd3581fd48567dc/env_cfg/camera/pattern).
The 50 mm virtual stereo baseline and projector coverage follow that reference;
these are simulation settings, not factory calibration. Each camera's native
`vision[role].structured_light` observation contains `left` and `right` uint8
grayscale images with calibration and representation metadata. Evaluation also
records separate `_ir_left` and `_ir_right` video streams.

The Sim 6.2 photometric baseline uses projector intensities of 35 for the head
and 6 for each wrist, with 2 ms exposure, ISO 400, exposure f/10 and responsivity
1.10267. The reference's 1 m projector dimensions and disabled normalization are
retained. These renderer settings were selected from measured saturation and
pattern correspondence; they do not specify measured projector power. IR
antialiasing is disabled because DLAA smoothed the dense projected dots in the
native pipeline. This override applies only to the stereo render products;
ordinary RGB keeps its current RTX settings.

Projection uses Isaac Sim's official `StructuredLightCamera`. Its output models
visible-spectrum illumination displayed in grayscale; it is not a spectral NIR
sensor model. Ordinary RGB and GT optical-Z depth retain their definitions.
Depth reconstruction and tactile sensing are not included.

## 🗂️ Repository Structure

```text
env/                   simulator backbone and managers
env_cfg/               simulator, scene, robot, and camera configs
task/RoboDojo/         task logic and task YAML configs
scripts/robodojo.sh    public RoboDojo-side eval entry
scripts/eval_policy.sh simulator client launched by XPolicyLab eval.sh
XPolicyLab/            policy server and policy integrations
Assets/                downloaded robot, object, material, and layout assets
```

## 🔌 Policy Integration

Policies live in [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab/blob/main/README.md), which owns policy structure, dependencies, checkpoint layout, and server behavior. RoboDojo only assumes a policy directory provides:

```text
XPolicyLab/policy/<POLICY_NAME>/eval.sh
XPolicyLab/policy/<POLICY_NAME>/deploy.yml
```

`eval.sh` starts the policy server and calls back into RoboDojo through `scripts/eval_policy.sh`; `deploy.yml` declares the server host, port, action mode, and policy-specific runtime settings.

## 🏆 Leaderboard

View live rankings on the [RoboDojo Leaderboard](https://robodojo-benchmark.com/leaderboard).

**Simulation.** The full evaluation stack is open source, so you can debug locally and iterate on scores. Official RoboDojo-endorsed listings are submitted through the cloud evaluation pipeline with anti-cheating verification.

**Real world.** Real-robot leaderboard entries are accepted through the same cloud evaluation process; see the public documentation for protocol, rules, and submission details.

## 📝 Citation

```bibtex
% RoboDojo
@article{chen2026robodojo,
  title={{RoboDojo}: A Unified Sim-and-Real Benchmark for Comprehensive Evaluation of Generalist Robot Manipulation Policies},
  author={Chen, Tianxing and Chen, Yue and Li, Zixuan and Tang, Junyuan and Su, Kailun and Wan, Weijie and Chen, Baijun and Lu, Haoran and Yan, Haowen and Su, Honghao and others},
  journal={arXiv preprint arXiv:2607.04434},
  year={2026}
}

% XPolicyLab
@article{community2026xpolicylab,
  title={{XPolicyLab}: A Unified Standard and Open Ecosystem for Robot Policy Evaluation and Deployment},
  author={Community, XPolicyLab and Chen, Tianxing and Chen, Yue and Nian, Tian and Cai, Zijian and Chen, Guangyu and Lin, Wenwei and Liang, Qiwei and Xiang, Peicheng and Su, Kailun and others},
  journal={arXiv preprint arXiv:2608.09892},
  year={2026}
}
```

## 🙏 Acknowledgements

RoboDojo builds on [Isaac Sim](https://developer.nvidia.com/isaac/sim), [IsaacLab](https://github.com/isaac-sim/IsaacLab), [IsaacLab-Arena](https://github.com/isaac-sim/IsaacLab-Arena), [RoboTwin 2.0](https://github.com/robotwin-Platform/robotwin), [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab), and [MagicSim](https://arxiv.org/abs/2606.17511). We thank the authors and maintainers for their open-source contributions to the robotics community.

Contact [Tianxing Chen](https://tianxingchen.github.io/) or [Yue Chen](https://yuechen0614.github.io/) if you have questions or suggestions.

## 🏫 Affiliations

RoboDojo is operated by **AI MMLab Club**, a non-profit, vendor-neutral organization, and is jointly maintained and supported by a global consortium of academic institutional partners. To preserve the fairness, neutrality, and independence of the official evaluation, RoboDojo does not involve commercial companies in its governance, operation, funding, sponsorship, compute, hardware, or other forms of project support. For inquiries from academic or non-profit partners regarding project collaboration or resource support, please contact [RoboDojoCommittee@gmail.com](mailto:RoboDojoCommittee@gmail.com).

<img src="https://media.luminis-sim.com/media/home/partners/affiliations.png"></img>

## ⚖️ License

Released under the [MIT License](LICENSE).


<p align="center">
  <img alt="Python 3.12" src="https://img.shields.io/badge/Python-3.12-475569?style=flat-square&logo=python&logoColor=white&labelColor=64748b" height="22"/>&nbsp;
  <img alt="Isaac Sim 6.2" src="https://img.shields.io/badge/Isaac_Sim-6.2-475569?style=flat-square&logo=nvidia&logoColor=76B900&labelColor=64748b" height="22"/>&nbsp;
  <img alt="Isaac Lab 3.0" src="https://img.shields.io/badge/Isaac_Lab-3.0-475569?style=flat-square&logo=nvidia&logoColor=76B900&labelColor=64748b" height="22"/>&nbsp;
  <img alt="License MIT" src="https://img.shields.io/badge/License-MIT-475569?style=flat-square&labelColor=64748b" height="22"/>
</p>
