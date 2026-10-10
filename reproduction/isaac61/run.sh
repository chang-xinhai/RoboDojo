#!/usr/bin/env bash
set -euo pipefail
ROBODOJO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROBODOJO_VENV="${ROBODOJO_ISAAC61_ENV:-$ROBODOJO_ROOT/.venv-isaac61}"
export VIRTUAL_ENV="$ROBODOJO_VENV"
export PATH="$ROBODOJO_VENV/bin:$PATH"
source "$ROBODOJO_ROOT/reproduction/isaac61/environment.sh"
cd "$ROBODOJO_ROOT"
export PYTHONPATH="$ROBODOJO_ROOT:$ROBODOJO_ROOT/XPolicyLab${PYTHONPATH:+:$PYTHONPATH}"
report_output() {
  local report_path="$1"
  shift
  while (($#)); do
    case "$1" in
      --output) report_path="${2:?--output requires a path}"; shift ;;
      --output=*) report_path="${1#--output=}" ;;
    esac
    shift
  done
  printf '%s' "$report_path"
}
case "${1:-}" in
  smoke)
    shift
    rm -f reproduction/isaac61/runtime-smoke.json
    python reproduction/isaac61/smoke_runtime.py --viz none --kit_args "$ROBODOJO_KIT_ARGS" "$@"
    python -c 'import json; assert json.load(open("reproduction/isaac61/runtime-smoke.json"))["status"] == "passed"'
    ;;
  task-smoke)
    shift
    REPORT_OUTPUT="$(report_output reproduction/isaac61/task-smoke.json "$@")"
    rm -f "$REPORT_OUTPUT"
    python reproduction/smoke_task.py --output reproduction/isaac61/task-smoke.json --viz none --kit_args "$ROBODOJO_KIT_ARGS" "$@"
    python -c 'import json, sys; assert json.load(open(sys.argv[1]))["status"] == "passed"' "$REPORT_OUTPUT"
    ;;
  collect)
    shift
    python reproduction/collect_multimodal.py --viz none --kit_args "$ROBODOJO_KIT_ARGS" "$@"
    ;;
  tactile-smoke)
    shift
    python reproduction/isaac61/smoke_tactile_contact.py --viz none --kit_args "$ROBODOJO_KIT_ARGS" "$@"
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
    echo 'Usage: bash reproduction/isaac61/run.sh {smoke|task-smoke|tactile-smoke|collect|visualize|test|client|python} [args...]' >&2
    exit 2
    ;;
esac
