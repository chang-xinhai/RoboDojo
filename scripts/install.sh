#!/usr/bin/env bash
set -e
export PIP_USER=0
export PYTHONNOUSERSITE=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURRENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ISAACLAB_RL_FRAMEWORK="${ISAACLAB_RL_FRAMEWORK:-none}"
ISAAC_CONSTRAINTS=""
trap '[[ -z "$ISAAC_CONSTRAINTS" ]] || rm -f "$ISAAC_CONSTRAINTS"' EXIT
# ── Helpers ────────────────────────────────────────────────────────────────────
info()  { echo -e "\e[1;32m>>> $*\e[0m"; }
warn()  { echo -e "\e[1;33m>>> $*\e[0m"; }
error() { echo -e "\e[1;31m[ERROR] $*\e[0m"; exit 1; }
pip_install() {
  prepare_isaac_constraints
  uv pip install --python "$CONDA_PREFIX/bin/python" "$@"
}
prepare_isaac_constraints() {
  [[ -n "$ISAAC_CONSTRAINTS" ]] && return
  local lab="$CURRENT_DIR/third_party/IsaacLab"
  local extras=(--extra isaacsim)
  [[ "$ISAACLAB_RL_FRAMEWORK" == none ]] || extras+=(--extra "$ISAACLAB_RL_FRAMEWORK")
  [[ -f "$lab/uv.lock" ]] || error "Initialize the pinned IsaacLab submodule before installing dependencies."
  ISAAC_CONSTRAINTS="$(mktemp)"
  uv export --project "$lab" --frozen "${extras[@]}" --no-dev --no-hashes \
    --no-emit-local --output-file "$ISAAC_CONSTRAINTS" >/dev/null
  # Lab's lock selects NumPy 2.5, which Numba cannot import and cmeel-boost
  # does not support. Isaac Sim's 2.3.1 pin satisfies all three packages.
  sed -i 's/^numpy==[^ ;]*/numpy==2.3.1/' "$ISAAC_CONSTRAINTS"
  # Isaac Lab's tested overrides reconcile Isaac Sim's older dependency pins.
  # The exported lock keeps those overrides at the versions tested upstream.
  export UV_CONSTRAINT="$ISAAC_CONSTRAINTS"
  export UV_TORCH_BACKEND=cu130
  export UV_INDEX_STRATEGY=unsafe-best-match
}
activate_conda() {
  eval "$(conda shell.bash hook)"
  conda activate RoboDojo
  [[ "$CONDA_DEFAULT_ENV" == RoboDojo ]] || error "Failed to activate conda environment 'RoboDojo'"
  python -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))' \
    || error "RoboDojo requires a fresh Python 3.12 environment; the existing environment was not changed."
  python -m pip install uv==0.13.0
}
# ── Step functions ─────────────────────────────────────────────────────────────
setup_system() {
  local missing=()
  for pkg in cmake build-essential ffmpeg; do
    dpkg -s "$pkg" &>/dev/null || missing+=("$pkg")
  done
  if [ ${#missing[@]} -gt 0 ]; then
    info "[0/7] Installing system dependencies: ${missing[*]}..."
    if [ "$(id -u)" -ne 0 ]; then
      warn "    You need sudo privileges to install system packages. Please enter your password."
      sudo apt-get update && sudo apt-get install -y "${missing[@]}"
    else
      apt-get update && apt-get install -y "${missing[@]}"
    fi
  else
    warn "[0/7] System dependencies already installed, skipping..."
  fi
}
setup_conda() {
  if ! command -v conda &>/dev/null; then
    info "[1/7] Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -q
    bash Miniconda3-latest-Linux-x86_64.sh -b -p "$HOME/miniconda3"
    rm -f Miniconda3-latest-Linux-x86_64.sh
    "$HOME/miniconda3/bin/conda" init bash
    eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
    info "    Accepting Anaconda Terms of Service..."
    "$HOME/miniconda3/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    "$HOME/miniconda3/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
  else
    warn "[1/7] Conda already installed, skipping..."
    eval "$(conda shell.bash hook)"
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
  fi
  if ! conda env list | grep -q "^RoboDojo "; then
    info "    Creating conda environment 'RoboDojo' (Python 3.12)..."
    conda create -n RoboDojo python=3.12 -y
  else
    warn "    Conda environment 'RoboDojo' already exists, skipping..."
  fi
  info "    Activating environment 'RoboDojo'..."
  activate_conda
}
setup_base_deps() {
  info "[5/7] Installing RoboDojo dependencies..."
  pip_install -r "$CURRENT_DIR/scripts/requirements.txt"
  pip_install scikit-learn
}
setup_submodules() {
  cd "$CURRENT_DIR" || exit 1
  local subs=(third_party/IsaacLab third_party/curobo XPolicyLab)
  info "[2/7] Syncing and installing pinned submodules..."
  git submodule sync "${subs[@]}"
  for sub in "${subs[@]}"; do
    info "    Checking out the recorded commit for ${sub}..."
    git submodule update --init --progress "$sub" || {
      [ "$sub" = "XPolicyLab" ] && error "Failed to clone XPolicyLab. Ensure HTTPS auth (e.g. gh auth login)."
      error "Failed to update $sub."
    }
  done
  [ -f "XPolicyLab/client_server/ws/model_client.py" ] \
    || error "XPolicyLab init failed. Check repo access."
}
setup_isaacsim() {
  info "[3/7] Installing Isaac Sim 6.2..."
  UV_OVERRIDE="$CURRENT_DIR/third_party/IsaacLab/tools/wheel_builder/uv-overrides.txt" \
    UV_PRERELEASE=allow pip_install "isaacsim[all,extscache]==6.2.0.0" \
    --extra-index-url https://pypi.nvidia.com
}
setup_isaaclab() {
  info "[4/7] Installing pinned Isaac Lab 3.0..."
  prepare_isaac_constraints
  local extras=isaacsim
  [[ "$ISAACLAB_RL_FRAMEWORK" == none ]] || extras+=",rl[$ISAACLAB_RL_FRAMEWORK]"
  export OMNI_KIT_ACCEPT_EULA=YES
  (cd "$CURRENT_DIR/third_party/IsaacLab" && \
    UV_OVERRIDE="$CURRENT_DIR/third_party/IsaacLab/tools/wheel_builder/uv-overrides.txt" \
    UV_PRERELEASE=allow ./isaaclab.sh --install "$extras")
}
setup_curobo() {
  info "[6/7] Installing pinned CuRobo with CUDA 13 runtime compilation..."
  # The default cuda.core backend uses NVRTC; no system nvcc upgrade is needed.
  CUROBO_USE_PYBIND=0 pip_install -e "$CURRENT_DIR/third_party/curobo[cu13]"
}
# ── Entry point ────────────────────────────────────────────────────────────────
usage() {
  echo "Usage: $0 [-i | --from <step>]"
  echo "  -i, --install          Full install (all steps)"
  echo "  --from <step>          Resume from a specific step:"
  echo "                           system | conda | submodules | isaacsim | isaaclab | base_deps | curobo"
  echo "  ISAACLAB_RL_FRAMEWORK  IsaacLab RL extras to install (default: none; use all/sb3/skrl/etc. if needed)"
  echo "  -h, --help             Show this help"
}
run_from() {
  local from="$1"
  local steps=(system conda submodules isaacsim isaaclab base_deps curobo)
  local start=0
  local found=0
  for i in "${!steps[@]}"; do
    if [ "${steps[$i]}" == "$from" ]; then
      start=$i
      found=1
      break
    fi
  done
  [ "$found" -eq 1 ] || error "Unknown step '$from'. Valid: ${steps[*]}"
  if [ "$from" != "system" ] && [ "$from" != "conda" ]; then
    activate_conda
  fi
  for i in "${!steps[@]}"; do
    if [ "$i" -ge "$start" ]; then
      "setup_${steps[$i]}"
    fi
  done
}
case "${1:-}" in
  -h|--help)
    usage
    ;;
  -i|--install)
    run_from system
    info "Develop environment setup completed."
    warn "Activate the environment: conda activate RoboDojo"
    ;;
  --from)
    [ -n "${2:-}" ] || { echo "Error: --from requires a step name"; usage; exit 1; }
    run_from "$2"
    info "Resumed from '$2' — done."
    warn "Activate the environment: conda activate RoboDojo"
    ;;
  *)
    usage
    exit 1
    ;;
esac

cd "$CURRENT_DIR" || exit
