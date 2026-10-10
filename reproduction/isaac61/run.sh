#!/usr/bin/env bash
set -euo pipefail
ROBODOJO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROBODOJO_VENV="${ROBODOJO_ISAAC61_ENV:-$ROBODOJO_ROOT/.venv-isaac61}"
export VIRTUAL_ENV="$ROBODOJO_VENV"
export PATH="$ROBODOJO_VENV/bin:$PATH"
source "$ROBODOJO_ROOT/reproduction/isaac61/environment.sh"
cd "$ROBODOJO_ROOT"
export PYTHONPATH="$ROBODOJO_ROOT:$ROBODOJO_ROOT/XPolicyLab${PYTHONPATH:+:$PYTHONPATH}"
case "${1:-}" in
  smoke)
    shift
    python reproduction/isaac61/smoke_runtime.py --viz none --kit_args "$ROBODOJO_KIT_ARGS" "$@"
    ;;
  task-smoke)
    shift
    python reproduction/smoke_task.py --viz none --kit_args "$ROBODOJO_KIT_ARGS" "$@"
    ;;
  collect)
    shift
    python reproduction/collect_multimodal.py --viz none --kit_args "$ROBODOJO_KIT_ARGS" "$@"
    ;;
  visualize)
    shift
    python reproduction/visualize_multimodal.py "$@"
    ;;
  test)
    shift
    python -m pytest "$@"
    ;;
  client)
    shift
    python src/eval_client/main.py --viz none --kit_args "$ROBODOJO_KIT_ARGS" "$@"
    ;;
  python)
    shift
    python "$@"
    ;;
  *)
    echo 'Usage: bash reproduction/isaac61/run.sh {smoke|task-smoke|collect|visualize|test|client|python} [args...]' >&2
    exit 2
    ;;
esac
