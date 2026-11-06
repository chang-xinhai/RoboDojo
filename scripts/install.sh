#!/usr/bin/env bash
# Native Isaac Sim 6.2 / Isaac Lab 3 simulator environment. Policy envs are separate.
set -euo pipefail
export PIP_USER=0 PYTHONNOUSERSITE=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LAB_DIR="$ROOT_DIR/third_party/IsaacLab"
SIM_ENV="${ROBODOJO_SIM_ENV:-RoboDojo}"

info() { printf '\n>>> %s\n' "$*"; }
error() { printf '[install] %s\n' "$*" >&2; exit 1; }

require_python() {
  python -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else "Isaac Sim 6.2 / Lab 3 requires Python 3.12; the existing environment is not upgraded automatically.")'
}
activate_conda() {
  command -v conda >/dev/null || error "Install conda and put it on PATH before running this installer."
  eval "$(conda shell.bash hook)"
  conda activate "$SIM_ENV"
  unset VIRTUAL_ENV
  require_python
}
setup_system() {
  local missing=() pkg
  for pkg in cmake build-essential ffmpeg; do
    dpkg -s "$pkg" &>/dev/null || missing+=("$pkg")
  done
  if (( ${#missing[@]} )); then
    if (( EUID == 0 )); then
      apt-get update
      apt-get install -y "${missing[@]}"
    else
      sudo apt-get update
      sudo apt-get install -y "${missing[@]}"
    fi
  fi
}
setup_conda() {
  command -v conda >/dev/null || error "Install conda and put it on PATH before running this installer."
  eval "$(conda shell.bash hook)"
  if ! conda env list --json | python -c 'import json,pathlib,sys; sys.exit(not any(pathlib.Path(p).name == sys.argv[1] for p in json.load(sys.stdin)["envs"]))' "$SIM_ENV"; then
    conda create -n "$SIM_ENV" python=3.12 -y
  fi
  activate_conda
}
setup_submodules() {
  local sub
  cd "$ROOT_DIR"
  git submodule sync -- third_party/IsaacLab third_party/curobo XPolicyLab
  for sub in third_party/IsaacLab third_party/curobo XPolicyLab; do
    # Follow the superproject gitlinks, never the tip of a remote branch.
    git submodule update --init -- "$sub"
  done
  [[ -f "$ROOT_DIR/XPolicyLab/client_server/ws/model_client.py" ]] || error "XPolicyLab client missing."
}
require_lab() {
  [[ -f "$LAB_DIR/uv.lock" ]] || error "Initialize the pinned Lab 3 submodule first (--from submodules)."
  require_python
  command -v uv >/dev/null || python -m pip install uv
}
setup_isaaclab() {
  require_lab
  info "Installing Isaac Lab + Isaac Sim from the pinned upstream lock (CUDA 13.0 Torch)."
  # Target the conda prefix explicitly; never create a separate Lab .venv.
  # Lab defaults to managed Python; conda requires the explicit system preference.
  # --inexact preserves RoboDojo additions.
  # The official lock and overrides own runtime versions except the documented NumPy bound.
  (cd "$LAB_DIR" && UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX" uv sync \
    --locked --inexact --python "$CONDA_PREFIX/bin/python" --python-preference only-system \
    --extra isaacsim --no-install-package numpy)
  # Complete the compatible NumPy install in this step, including on a fresh env.
  install_with_lab_lock numpy
}
install_with_lab_lock() (
  require_lab
  local locked overrides
  locked="$(mktemp)"
  overrides="$(mktemp)"
  trap 'rm -f "$locked" "$overrides"' EXIT
  cd "$LAB_DIR"
  uv export --locked --extra isaacsim --no-dev --no-emit-local --no-emit-package numpy --no-hashes \
    --no-annotate --no-header --output-file "$locked" >/dev/null
  # Lab's lock has NumPy 2.5.1, but its locked numba 0.64.0 requires <2.5
  # and rejects 2.5 at import. Keep NumPy 2.x without overriding that ABI bound.
  printf '\nnumpy>=2,<2.5\n' >>"$locked"
  python - "$LAB_DIR/pyproject.toml" "$overrides" <<'PY'
from pathlib import Path
import sys
import tomllib

project = tomllib.loads(Path(sys.argv[1]).read_text())
Path(sys.argv[2]).write_text("\n".join(project["tool"]["uv"]["override-dependencies"]) + "\n")
PY
  # Unlike pip constraints, these upstream overrides reconcile Sim's older pins.
  uv pip install --python "$(command -v python)" --constraints "$locked" \
    --overrides "$overrides" --torch-backend cu130 "$@"
)
setup_base_deps() {
  info "Installing RoboDojo dependencies with upstream runtime constraints."
  install_with_lab_lock -r "$SCRIPT_DIR/requirements.txt"
}
setup_curobo() {
  info "Installing the pinned cuRobo source with its CUDA 13 extra."
  # cu13 selects cuda-core[cu13] and nvidia-cuda-runtime>=13 in this source revision.
  # The Torch build remains owned by Lab, so cu13-torch is unnecessary.
  install_with_lab_lock -e "$ROOT_DIR/third_party/curobo[cu13]"
}
usage() {
  cat <<'HELP'
Usage: bash scripts/install.sh --install | --from STEP

Steps: system, conda, submodules, isaaclab, base_deps, curobo
  --install       Run all steps in order (requires conda on PATH).
  --from STEP     Resume at STEP, activating the simulator conda env.
  --from isaacsim Alias for isaaclab: Sim and Lab are installed together.
  -h, --help      Show help.

ROBODOJO_SIM_ENV selects the simulator conda env (default: RoboDojo).
The simulator requires Python 3.12. Existing Python 3.11 envs fail explicitly;
create a separate Python 3.12 env instead of changing a policy server env.
Lab's pinned uv.lock owns Sim 6.2, Torch cu130 and Warp versions.
NumPy uses >=2,<2.5 because the locked Numba 0.64 rejects NumPy 2.5.
No optional RL extras are selected. Policy server dependencies are separate.
HELP
}
run_from() {
  local from="$1" step found=0
  [[ "$from" != isaacsim ]] || from=isaaclab
  local steps=(system conda submodules isaaclab base_deps curobo)
  for step in "${steps[@]}"; do
    [[ "$step" != "$from" ]] || found=1
  done
  (( found )) || error "Unknown step: $from"
  [[ "$from" == system || "$from" == conda ]] || activate_conda
  found=0
  for step in "${steps[@]}"; do
    [[ "$step" != "$from" ]] || found=1
    if (( found )); then
      info "$step"
      "setup_$step"
    fi
  done
  info "Install finished. Validate with: bash scripts/robodojo.sh doctor --skip-policy"
}
case "${1:-}" in
  -h|--help) usage ;;
  -i|--install) [[ $# == 1 ]] || error "Unexpected arguments"; run_from system ;;
  --from) [[ $# == 2 ]] || error "--from requires exactly one step"; run_from "$2" ;;
  *) usage; exit 2 ;;
esac
