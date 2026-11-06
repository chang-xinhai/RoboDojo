# RoboDojo scripts

## Public entry points

| Script | Purpose |
| --- | --- |
| [robodojo.sh](robodojo.sh) | Main CLI: `doctor`, `eval`, `client`, `smoke`, `benchmark`, `dimensions`, `summarize`, `tasks` |
| [install.sh](install.sh) | One-time environment setup (conda, Isaac Sim, submodules) |
| [init_assets.sh](init_assets.sh) | Download robot/object assets |
| [eval_policy.sh](eval_policy.sh) | Isaac Sim eval client (called by `robodojo.sh client` and XPolicyLab) |

## Simulator installation

`bash scripts/install.sh --install` requires conda on PATH and creates or
activates `RoboDojo` with Python 3.12. Export `ROBODOJO_SIM_ENV` to use another
simulator environment for both installer and doctor. An existing Python 3.11 env is rejected before Python
package installation. Policy server environments are managed separately by XPolicyLab.

The ordered steps are `system`, `conda`, `submodules`, `isaaclab`, `base_deps`,
`curobo`. Resume with `--from STEP`; `isaacsim` aliases `isaaclab` because
upstream's locked install resolves Sim and Lab together. Submodule updates
honor the recorded Git commits. Lab's official `uv.lock` and overrides own
runtime versions, including Torch cu130 and Warp. The sole compatibility
exception is NumPy `>=2,<2.5`: upstream's NumPy 2.5.1 conflicts with its locked
Numba 0.64 import guard. The Lab step skips that NumPy lock entry and immediately
installs NumPy within the supported range. Additional RoboDojo and cuRobo
dependencies resolve against that lock with the same NumPy bound. Optional Lab
extras are not selected beyond `isaacsim`; the upstream base still includes its
own packages.

`bash scripts/robodojo.sh doctor --skip-policy` checks source commit identity,
editable paths, versions, loaded Torch/NumPy and planner imports. Numba
imports check NumPy compatibility. The inspected Sim kernel/urllib3 and Sim core/fsspec
overrides are reported individually as warnings; other dependency
conflicts fail. `--skip-isaac` skips these runtime checks and cannot certify the
runtime. The doctor does not launch Kit or execute GPU kernels. End-to-end
RoboDojo runtime migration and a fresh install remain unvalidated at this stage.

## Typical eval flow

```text
robodojo.sh eval
  -> scripts/internal/run_policy_eval.sh
    -> policy server (localhost) + sim client

Split / multi-machine (see docs/SPLIT_EVAL.md):

robodojo.sh server  ->  scripts/internal/run_policy_server.sh  ->  policy server (bind 0.0.0.0)
robodojo.sh client  ->  scripts/eval_policy.sh  ->  src/eval_client/main.py
```

Run one or more official capability dimensions in a benchmark sweep:

```bash
bash scripts/robodojo.sh dimensions
bash scripts/robodojo.sh benchmark \
  --dimension memory,long-horizon \
  --policy-dir XPolicyLab/policy/<POLICY> \
  --ckpt <CHECKPOINT> \
  --policy-env <ENV> \
  --eval-num native
```

Available dimensions are `generalization`, `memory`, `precision`,
`long-horizon`, and `open`. Generalization includes both the 12 standard tasks
and their 12 runnable `_random` layout variants. Combine `--dimension` with
`--only` or `--tasks-file` to narrow a dimension further.

## Auto multi-GPU grouping

`robodojo.sh smoke` and `robodojo.sh benchmark` now support balanced
multi-GPU execution driven by the runtime table in `../optimal_8group.txt`.
Pass a concrete GPU id list and RoboDojo will partition the selected tasks
online instead of relying on a hard-coded task group.

Dry-run example:

```bash
bash scripts/robodojo.sh benchmark \
  --policy-dir XPolicyLab/policy/ACT \
  --ckpt test_ckpt \
  --policy-env RoboDojo \
  --eval-num 1 \
  --gpu-ids 0,2,5,7 \
  --dry-run
```

Launch only a subset of tasks:

```bash
bash scripts/robodojo.sh benchmark \
  --policy-dir XPolicyLab/policy/ACT \
  --ckpt test_ckpt \
  --policy-env RoboDojo \
  --eval-num native \
  --only imitate_sorting_sequence,pour_by_language,play_tic_tac_toe \
  --gpu-ids 0,1,3
```

## Internal (`internal/`)

Not intended for direct daily use. Called by `robodojo.sh` or policy utilities.

| File | Called by |
| --- | --- |
| [verify_install.sh](internal/verify_install.sh) | `robodojo.sh doctor` |
| [task_inventory.py](internal/task_inventory.py) | `robodojo.sh tasks` |
| [smoke_all_tasks.sh](internal/smoke_all_tasks.sh) | `robodojo.sh smoke` / `benchmark` |
| [summarize_result.py](internal/summarize_result.py) | `robodojo.sh summarize` |
| [stat_score_distribution.py](internal/stat_score_distribution.py) | Offline score histogram analysis (manual) |

## Docker

Container install and smoke tests live under [../docker/](../docker/), not here.

## Policy-specific scripts

Training, data prep, and per-policy `eval.sh` live in [../XPolicyLab/policy/](../XPolicyLab/policy/) (submodule).
