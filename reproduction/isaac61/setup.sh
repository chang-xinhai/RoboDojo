#!/usr/bin/env bash
# Install the repository-pinned Isaac Lab EA / Isaac Sim 6.1 stack.
set -euo pipefail
ROBODOJO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROBODOJO_VENV="${ROBODOJO_ISAAC61_ENV:-$ROBODOJO_ROOT/.venv-isaac61}"
cd "$ROBODOJO_ROOT"
if ! command -v uv >/dev/null; then
  echo 'Install uv (https://docs.astral.sh/uv/getting-started/installation/) and run setup again.' >&2
  exit 1
fi
git submodule update --init third_party/IsaacLab3 third_party/curobo XPolicyLab
uv venv --python 3.12 --allow-existing "$ROBODOJO_VENV"
source "$ROBODOJO_ROOT/reproduction/isaac61/environment.sh"
export UV_PROJECT_ENVIRONMENT="$ROBODOJO_VENV"
uv sync --project reproduction/isaac61 --locked
"$ROBODOJO_VENV/bin/python" -m pip --version 2>/dev/null || true
uv pip freeze --python "$ROBODOJO_VENV/bin/python" > reproduction/isaac61/requirements-resolved.txt
bash reproduction/isaac61/run.sh smoke
