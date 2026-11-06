# Run RoboDojo in Docker

> Migration status: this recipe now targets Isaac Sim 6.2.0.0 / Isaac Lab
> 3.0.0, Python 3.12 and CUDA 13.0. It has only received source and shell
> checks: no image build or container GPU/evaluation run has been performed.
> Native PhysX/RTX checks on the host do not validate the container. The full
> RoboDojo runtime migration is still in progress.

This image is the **RoboDojo simulation-evaluation side** — Isaac Sim 6.2.0.0,
Isaac Lab 3.0.0, CuRobo, the RoboDojo Python stack, and the lightweight XPolicyLab client
used by `src/eval_client`. It contains **no policy, policy dependencies, or
checkpoints**: your **policy server runs outside the container** and the sim
client talks to it over the default WebSocket transport (`protocol: ws` in
`deploy.yml`), so one image can evaluate any policy.

- Base image: `nvidia/cuda:13.0.2-cudnn-devel-ubuntu22.04` (Ubuntu 22.04 + CUDA 13.0)
- Conda env inside the image: `RoboDojo` (Python 3.12)
- Project root inside the container: `/workspace/RoboDojo`

```mermaid
flowchart LR
  subgraph host [Host / another machine]
    policy["Policy server\n(your policy, or the bundled demo_policy)\nbind 0.0.0.0:PORT"]
  end
  subgraph container [RoboDojo container]
    client["scripts/robodojo.sh client\n-> Isaac Sim eval"]
  end
  client -->|"WebSocket ws://host:port"| policy
```

## TL;DR

On a machine meeting the [Isaac Sim 6.2 requirements](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/requirements.html):

```bash
sudo bash docker/install_docker_nvidia.sh   # one-time: Docker + NVIDIA Container Toolkit
newgrp docker                               # (or log out/in) so docker works without sudo
bash docker/smoke_docker.sh run             # build the image, run a demo eval, assert PASS
```

The last command prints a per-phase `PASS/FAIL` table and writes a result file.
(Step 4 uses the bundled demo server, which needs the native `RoboDojo` env. No
native env? Build the image and jump to [step 5](#5-evaluate-your-own-policy).)

Everything below explains each step and how to plug in your own policy.

## 1. Prerequisites

| Need | Details |
| --- | --- |
| NVIDIA GPU + driver | Supported RTX GPU and driver meeting both [Isaac Sim 6.2](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/requirements.html) and [CUDA 13.0.2](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-toolkit-release-notes/index.html) requirements. Check with `nvidia-smi`. |
| Disk space | Allow space for the full Isaac Sim extension cache and Docker build layers. This image target has not been measured. |
| OS | Linux; image base is Ubuntu 22.04. This container target is untested. |
| Build metadata | A clean Git checkout with initialized pinned submodules, plus a host Python 3.12 env providing `setuptools_scm` (the native `RoboDojo` env suffices). This is used only to derive cuRobo’s true source version. |
| Assets | The RoboDojo `Assets/` directory on the host (downloaded by `scripts/init_assets.sh`); it is mounted at run time, not baked into the image. |

## 2. Install Docker + the NVIDIA Container Toolkit

If Docker and the toolkit already work with `--gpus all`, skip to step 3.

The bundled installer sets up Docker Engine, the NVIDIA Container Toolkit, the
`nvidia` runtime, and adds you to the `docker` group:

```bash
sudo bash docker/install_docker_nvidia.sh
newgrp docker      # or log out and back in
```

Verify the GPU is visible inside a container:

```bash
docker run --rm --gpus all nvidia/cuda:13.0.2-cudnn-devel-ubuntu22.04 nvidia-smi
```

You should see your GPU listed. If not, see [Troubleshooting](#6-troubleshooting).

> **China networks:** the installer auto-detects a restricted network and switches
> to TUNA/USTC apt mirrors + Docker Hub registry mirrors. Force it with
> `USE_CN_MIRRORS=1` (or disable with `USE_CN_MIRRORS=0`). Details in
> [Appendix B](#appendix-b-china-networks).

## 3. Build the image

From the repository root, initialize the recorded source revisions and export
cuRobo's version from its clean checkout. Keep the metadata command's failure
visible; do not substitute an arbitrary version:

```bash
git submodule update --init -- third_party/IsaacLab third_party/curobo XPolicyLab &&
test -z "$(git -C third_party/curobo status --porcelain)" &&
test "$(git -C third_party/curobo rev-parse --is-shallow-repository)" = false &&
CUROBO_VERSION="$(conda run --no-capture-output -n "${ROBODOJO_SIM_ENV:-RoboDojo}" python -c '
from pathlib import Path
import tomllib
from setuptools_scm import get_version
root = Path("third_party/curobo")
config = tomllib.loads((root / "pyproject.toml").read_text())["tool"]["setuptools_scm"]
print(get_version(root=root, **config))
')" && docker build --build-arg "CUROBO_VERSION=$CUROBO_VERSION" -t robodojo:cuda13.0 .
```

The Dockerfile uses the same `scripts/install.sh --from isaaclab` path as the
host. Lab's lock owns the runtime dependencies, with the installer's documented
NumPy/Numba compatibility bound. No optional policy environments are installed.
The CUDA base uses a verified multi-architecture manifest digest (this does
not certify RoboDojo support on every architecture); overriding
`CUDA_IMAGE` changes that provenance.

`CUROBO_VERSION` is required and has no default. It is passed through the
[standard package-specific setuptools-scm build variable](https://setuptools-scm.readthedocs.io/v9.2.1/usage/#with-dockerpodman)
only for installation. All `.git` files and directories are excluded from the
build context; the runtime reads cuRobo's installed package metadata. Git
configuration, remotes and credentials are not copied. A source archive without
Git history needs the corresponding trusted source-derived version supplied by
its producer.

The first build downloads the simulator dependency stack; its duration and size
have not been measured for this target. The existing libXt and single Vulkan
ICD setup is retained and still needs container rendering validation.
`smoke_docker.sh run` derives and checks the source version automatically before
building. See [Appendix B](#appendix-b-china-networks) for mirror arguments.

## 4. Run the demo evaluation

Once the runtime port is ready, run the bundled end-to-end container check:

```bash
bash docker/smoke_docker.sh run
```

It runs, in order: GPU-in-container → image build → start the bundled `demo_policy`
server → a real container eval → assert the result file. A final table reports
each phase, and results land under `eval_result/`.

`demo_policy` is a zero-action dummy, so `success_rate: 0.0` is **expected** — the
point is that the pipeline renders, talks to the policy over WebSocket, steps the sim,
checks the reward, and writes a valid result:

```json
{ "success_rate": 0.0, "eval_time": 1, "score": 0.0, "details": { "0": { "success": false } } }
```

Watch progress live from a second terminal with `bash docker/smoke_docker.sh monitor`.

> **Note:** `smoke_docker.sh` launches the demo server **on the host** using the
> native `RoboDojo` conda env (from `scripts/install.sh`). If you don't have the
> native env and only want Docker, skip this step and go to
> [step 5](#5-evaluate-your-own-policy), pointing the eval at any policy server you
> run yourself — the demo server is only a convenience.

## 5. Evaluate your own policy

The container is the simulator side; you run the **policy server** yourself and the
container connects over WebSocket (`protocol: ws`, same as `deploy.yml`).

**5.1 — Start your policy server** (outside the container). Bind it to `0.0.0.0`
(all interfaces), not `127.0.0.1`, so a container can reach it. See
[XPolicyLab](../XPolicyLab/README.md) for policy servers.

**5.2 — Run the eval** in the container, pointing `--policy-host`/`--policy-port`
at your server:

```bash
docker run --rm -it \
  --gpus all --network host --ipc host \
  -v "$PWD/Assets:/workspace/RoboDojo/Assets:ro" \
  -v "$PWD/Assets:$PWD/Assets:ro" \
  -v "$PWD/eval_result:/workspace/RoboDojo/eval_result" \
  -v "$HOME/.cache/warp:/root/.cache/warp" \
  -v "$HOME/.local/share/ov:/root/.local/share/ov" \
  -v "$HOME/.cache/ov:/root/.cache/ov" \
  robodojo:cuda13.0 \
  bash scripts/robodojo.sh client \
    --task stack_bowls \
    --policy-name GR00T_N17 \
    --policy-host 127.0.0.1 --policy-port 9999 \
    --ckpt <your_ckpt> --eval-num 1
```

Required client flags: `--task`, `--policy-name` (a directory under
`XPolicyLab/policy/`), `--policy-host`, `--policy-port`. Common options: `--ckpt`,
`--eval-num N` (or `native`), `--action-type` (default `ee`), `--env-gpu` (default
`0`). Add `--dry-run` to print the resolved command without launching anything.

**Mounts explained** (the last three are strongly recommended — they prevent slow
first starts):

| Mount | Why |
| --- | --- |
| `Assets → /workspace/RoboDojo/Assets:ro` | Robot/object/layout/material assets (not baked into the image). |
| `Assets → $PWD/Assets:ro` (same host path) | curobo configs bake **absolute host paths**; this second mount makes them resolve. See [Troubleshooting](#6-troubleshooting). |
| `eval_result → …/eval_result` | Persist results on the host. |
| `~/.cache/warp` | Reuse CuRobo's compiled kernels (else it recompiles for minutes on every start). |
| `~/.local/share/ov`, `~/.cache/ov` | Reuse Isaac Sim extensions/shaders (else Kit may stall pulling them). |

### Networking: where is your policy server?

| Policy server location | Run mode | `--policy-host` |
| --- | --- | --- |
| Same host as the container | `--network host` | `127.0.0.1` |
| Same host, bridge network | `--add-host=host.docker.internal:host-gateway` | `host.docker.internal` |
| Another machine | default networking | that machine's IP / hostname |

`localhost` inside a **bridge** container is the container itself, not the host —
this is the #1 reason "the simulator can't reach the policy server." The `client`
command checks that the policy server port is reachable before the WebSocket
handshake and prints guidance if the server is unreachable.

## 6. Troubleshooting

**GPU not visible (`nvidia-smi` fails in a container).** The NVIDIA Container
Toolkit isn't active. Re-run `sudo bash docker/install_docker_nvidia.sh`, then
`sudo systemctl restart docker`.

**"Cannot reach the policy server."** Make sure the server binds `0.0.0.0`, and
match the networking table in step 5. Quick check inside the container:
`nc -vz <host> <port>`.

**First eval seems to hang for minutes (high CPU, low GPU).** A fresh container
recompiles CuRobo's Warp kernels and/or pulls Isaac Sim extensions. Mount the host
caches (step 5) so both are reused. `smoke_docker.sh` does this automatically.

**`ValueError: /abs/host/.../X5A.urdf is not a file`.** curobo robot configs
(`Assets/Robots/**/curobo.yml`, generated by
`utils/update_embodiment_config_path.py`) bake in absolute host paths. Mount
`Assets/` a second time at its original host path (step 5), both read-only.
`smoke_docker.sh` does this automatically.

**Cameras hang on the first rendered frame.** Check GPU visibility, driver
compatibility and the Vulkan ICD setup in [Appendix A](#appendix-a-configuration-reference).
The retained loader configuration has not been validated for this image target.

## Appendix A: Configuration reference

**`smoke_docker.sh` env toggles**

| Env | Effect |
| --- | --- |
| `ROBODOJO_SKIP_BUILD=1` | Reuse an existing image; ensure it was built from the intended source target. |
| `ROBODOJO_CN_MIRRORS=1/0` | Force China build mirrors on/off (default `auto`). |
| `ROBODOJO_SMOKE_TASK`, `ROBODOJO_SMOKE_PORT`, `ROBODOJO_SMOKE_POLICY`, … | Override task/port/policy/etc. (see the script header). |

**Volumes**

| Mount | Purpose |
| --- | --- |
| `-v "$PWD/Assets:/workspace/RoboDojo/Assets:ro"` | Assets (required at runtime, not baked). |
| `-v "$PWD/Assets:$PWD/Assets:ro"` | Resolve absolute host paths baked into curobo configs. |
| `-v "$PWD/eval_result:/workspace/RoboDojo/eval_result"` | Persist per-task eval artifacts on the host. |
| `-v "$HOME/.cache/warp:/root/.cache/warp"` (and `~/.local/share/ov`, `~/.cache/ov`, `~/.cache/nvidia`, `~/.nv`) | Isaac Sim / CuRobo caches — recommended for fast, offline-friendly starts. |
| `-e ROBODOJO_USD_ASSET_PREFIX=/path -v /host/usd:/path` | Point at a local Isaac Sim USD asset mirror (optional). |

**GPU selection.** `--gpus all` exposes every GPU; pin one with
`--gpus '"device=0"'`. Choose the Isaac Sim device with `--env-gpu ID`.

**Headless RTX configuration.** Tasks render through Isaac Sim’s RTX path.
The image bakes in `libxt6` (needed by Isaac Sim's MaterialX render libs) and pins a **single Vulkan ICD** via `ENV VK_ICD_FILENAMES` /
`VK_DRIVER_FILES` (the toolkit also injects one at run time; two ICDs for the same
GPU can conflict during RTX initialization). This setup still requires container
validation. It requires `--gpus all` and
`NVIDIA_DRIVER_CAPABILITIES=all` (set by default).

**Build args.** Required `CUROBO_VERSION` (exported above), `CUDA_IMAGE` (base
image), plus the China mirrors `UBUNTU_MIRROR`,
`PIP_INDEX_URL`, `MINICONDA_URL`, `CONDA_CHANNEL_MIRROR` (see the `Dockerfile`
header and [Appendix B](#appendix-b-china-networks)).

**Structural checks inside the image**

```bash
docker run --rm \
  -v "$PWD/Assets:/workspace/RoboDojo/Assets:ro" \
  robodojo:cuda13.0 \
  bash scripts/robodojo.sh doctor --skip-policy --skip-isaac
```

This deliberately skips runtime certification. The full doctor verifies Git
source identities and therefore fails those checks in this Git-free image;
do not treat its exit status as a container pass or weaken the checks. Run the
full doctor in the host Git checkout for source provenance. Container runtime
validation requires a successful GPU/render/evaluation run such as the smoke
procedure above, which has not yet been performed for this target.

## Appendix B: China networks

Everything auto-detects a restricted network; these are the manual knobs.

**Host install** (`install_docker_nvidia.sh`) — auto-uses TUNA (Docker CE) + USTC
(NVIDIA toolkit) apt mirrors and sets Docker Hub registry mirrors in
`/etc/docker/daemon.json`. Force with `USE_CN_MIRRORS=1`, disable with
`USE_CN_MIRRORS=0`.

**Image build** — `smoke_docker.sh` sets the mirror build-args when it detects (or
is told, via `ROBODOJO_CN_MIRRORS=1`) a China network. After deriving
`CUROBO_VERSION` in step 3, a manual build can use:

```bash
docker build \
  --build-arg "CUROBO_VERSION=$CUROBO_VERSION" \
  --build-arg UBUNTU_MIRROR=mirrors.tuna.tsinghua.edu.cn \
  --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  --build-arg MINICONDA_URL=https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh \
  --build-arg CONDA_CHANNEL_MIRROR=https://mirrors.tuna.tsinghua.edu.cn/anaconda \
  -t robodojo:cuda13.0 .
```
