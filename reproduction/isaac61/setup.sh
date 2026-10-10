#!/usr/bin/env bash
# Install the repository-pinned Isaac Lab EA / Isaac Sim 6.1 stack.
set -euo pipefail
ROBODOJO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROBODOJO_VENV="${ROBODOJO_ISAAC61_ENV:-$ROBODOJO_ROOT/.venv-isaac61}"
cd "$ROBODOJO_ROOT"
ROBODOJO_UV="$(command -v uv || true)"
if [[ -z "$ROBODOJO_UV" && -x "$ROBODOJO_VENV/bin/uv" ]]; then
  ROBODOJO_UV="$ROBODOJO_VENV/bin/uv"
fi
if [[ -z "$ROBODOJO_UV" ]]; then
  echo 'Install uv (https://docs.astral.sh/uv/getting-started/installation/) and run setup again.' >&2
  exit 1
fi
git submodule update --init third_party/IsaacLab3 third_party/curobo XPolicyLab
"$ROBODOJO_UV" venv --python 3.12 --allow-existing "$ROBODOJO_VENV"
source "$ROBODOJO_ROOT/reproduction/isaac61/environment.sh"
export UV_PROJECT_ENVIRONMENT="$ROBODOJO_VENV"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-180}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-5}"
"$ROBODOJO_UV" sync --project reproduction/isaac61 --locked
"$ROBODOJO_UV" pip check --python "$ROBODOJO_VENV/bin/python"
"$ROBODOJO_UV" pip freeze --python "$ROBODOJO_VENV/bin/python" > reproduction/isaac61/requirements-resolved.txt
"$ROBODOJO_VENV/bin/python" reproduction/isaac61/fetch_tacsl_assets.py
bash reproduction/isaac61/run.sh smoke
