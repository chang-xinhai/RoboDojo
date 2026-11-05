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

### Isaac Sim 6.2 migration branch

This branch uses **Isaac Sim 6.2.0.0, Isaac Lab v3.0.0, Python 3.12,
PyTorch 2.11 (CUDA 13.0), and Warp 1.17** with Isaac Sim's PhysX backend and
RTX Real-Time Path Tracing (`RealTimePathTracing`). This intentionally establishes
a new RGB rendering baseline; camera geometry, ground-truth depth definitions
and physics settings are unchanged.
The renderer uses current engine defaults with RoboDojo's explicit settings
(translucency, reflections, global illumination and DLAA). The removed Lab 2
`rendering_mode` presets are not recreated; use individual `render` settings
for overrides.
Isaac Lab, cuRobo and XPolicyLab are fixed by the recorded submodule commits.
`fold_clothes` and `fold_clothes_random` use official PhysX **surface FEM**.
This is a **new cloth physics baseline**, not numerically or behaviorally
equivalent to the removed particle-cloth model; old scores are not comparable.
Official surface cooking generates a simulation/collision mesh with
simplification disabled, welding coincident seam vertices as required by PhysX.
The original triangle mesh remains the visual skin. A checked rest-space map
returns native nodes in the original vertex order for semantic reward IDs;
textures, task actions, WXYZ layout poses and reward thresholds are retained.
Rest points are the asset's original (already shaped) garment, not a flattened
or remeshed sheet. Shared asset files are not modified.

Garments use `physics.surface_fem` overrides. Defaults are Young's modulus
1 MPa, Poisson ratio 0.3, density 300 kg/m³, thickness 1 mm, static/dynamic
friction 0.5, elasticity/bend damping 0.05, self collision enabled, 32 position
iterations, speculative CCD, and rest/contact offsets 1/3 mm. Membrane and bend
stiffness are derived by PhysX from modulus, Poisson ratio and thickness.
Total mass is explicitly the scaled rest-triangle area × density × thickness;
it is **not** the former per-particle mass. These isotropic elastic material
choices are a reproducible starting baseline, not a fit to a measured textile.

Legacy `garment_config`, `particle_material` and `particle_system` blocks in
asset/layout metadata are accepted with a warning and ignored in full (including
spring stiffness/damping, particle mass, adhesion, drag/lift and gravity scale).
They are not converted into FEM parameters. Unknown `surface_fem` keys and the
former `interaction_with_fluid` shared-particle mode raise an error. Garments
require triangle meshes with a static identity transform relative to their
Xform root. Reset stacks, animated transforms and authored root/ancestor scale,
shear or reflection are rejected; configured layout pose and scale are retained.
The shipped Top_Long assets satisfy these conditions. The former generated
Plane primitive is not supported by this loader.
Reset restores the simulation mesh and zeroes velocities. CPU readback uses
official PhysX USD state publication (including after unrendered steps), because
6.2's CPU tensor frontend does not implement deformable nodal access; CUDA
readback uses native nodal tensors. Both paths use the GPU surface FEM solver.
See NVIDIA's
[surface-deformable migration guide](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.1/dev_guide/deformables/deformable_migration.html)
and [deformable schema](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/deformables/omniphysics_deformable_schema.html).

Native validation covered both preserved cloth tasks with two CPU-readback
reset cycles each, plus two CUDA-readback reset cycles for `fold_clothes`,
with all three camera views. Native reset checks confirmed zero node velocities
and rest-space semantic mapping error below 0.000031 mm. After 80 ms of settling,
rigid-fit deformation was 8.75–9.88 mm RMS across the two garments. A separate
100-action scripted robot approach/descent/sweep/retreat produced apparent
gripper–cloth interaction and local bunching in the videos; native measurements
at action 75 showed 130.5 mm maximum displacement and 24.4 mm rigid-fit RMS
deformation relative to the first action. This was a synthesized action sequence,
not a demonstration replay; its score was zero and no successful fold was shown.

Use a supported NVIDIA RTX GPU and Linux driver 580.65.06 or newer; see the
[Isaac Sim requirements](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/requirements.html)
for GPU, memory and renderer requirements. The native installation uses conda:

```bash
bash scripts/install.sh --install
conda activate RoboDojo
bash scripts/init_assets.sh
bash scripts/robodojo.sh doctor --skip-policy
```

The installer creates a **fresh Python 3.12** `RoboDojo` environment. If that name
already contains Python 3.11, preserve or remove it after checking its consumers;
the installer will not upgrade it in place. Isaac Lab dependencies follow its
lock and official overrides, with NumPy 2.3.1 retained for Numba and Pinocchio
compatibility. cuRobo uses CUDA 13 runtime compilation,
so a separate CUDA 13 toolkit is not required. Policy servers retain their own
environments. `doctor` checks packages and paths without launching the simulator;
an actual evaluation is still required to verify GPU rendering and planning.

The existing Docker recipe targets the upstream 5.1 stack and is **not validated
for this branch**. Use the native installer above.

Validation uses the installed native environment, an RTX 5090, and headless
ARX X5 evaluation. Earlier migration checks used the legacy RTX renderer:
seven Precision tasks passed short reset/action checks;
articulation, particle-fluid and conveyor scenes were also exercised. A frozen
308-action `insert_tubes` replay retained score 100 and all reward decisions,
with maximum end-effector position difference of 0.268 mm and three-view videos.
The 320-command dataset replay completed, but did not succeed; without the full
demonstration object checkpoint it verifies robot commands, not contact replay.

In those earlier checks, with unchanged VPP2 weights/settings, seed 1 and layout 0, `insert_tubes`
succeeded in 307 actions (old-runtime control: 308); `pour_balls_into_vase`
succeeded and `plug_in_charger` reached its 400-action limit without success.
This is a representative sample, not a full-suite equivalence result. Rendering
appearance differences remain. `fill_egg_holder` layout 0 failed the original
reset stability check (30.046° rotation against a 30° limit); its thresholds
were retained. Interactive GUI evaluation remains unvalidated: the window
opened, but the bounded startup test ended during initial shader compilation.

The current RTX defaults were checked separately with 30 original `insert_tubes`
actions across two resets: actions, rewards and sampled robot/object states
matched the earlier replay, with fresh three-camera delivery and valid RGB/GT depth.
A complete VPP2 episode succeeded in 309 actions with the same weights/settings,
and a short FEM scene check passed. Both policy processes exited normally and
three 310-frame videos were recorded. This single episode does not establish
success-rate equivalence across the benchmark.

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
