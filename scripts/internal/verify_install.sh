#!/usr/bin/env bash
# Preflight checks for RoboDojo eval. Does not launch Isaac Sim or a policy server.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

policy_dir=""
env_cfg="arx_x5"
task_name="stack_bowls"
ckpt_name=""
sim_env="${ROBODOJO_SIM_ENV:-RoboDojo}"
policy_env=""
skip_isaac="false"
skip_conda="false"
skip_policy="false"
summary_path=""

usage() {
  cat <<'EOF'
Usage: bash scripts/internal/verify_install.sh [options]

Options:
  --policy-dir PATH     Optional XPolicyLab policy directory to validate
  --env-cfg NAME        env_cfg stem to validate (default: arx_x5)
  --task NAME           Task name to validate (default: stack_bowls)
  --ckpt NAME           Optional checkpoint directory under policy checkpoints
  --sim-env NAME        Simulator env (default: ROBODOJO_SIM_ENV or RoboDojo)
  --policy-env NAME     Optional policy conda env to check
  --summary PATH        Write JSON summary to PATH
  --skip-isaac          Skip native versions/source paths/dependency checks
  --skip-conda          Skip conda env existence checks
  --skip-policy         Skip policy-dir/deploy/checkpoint checks
  -h, --help            Show this help
EOF
}

need_value() {
  if [[ $# -lt 2 || "$2" == --* ]]; then
    echo "[verify_install] Missing value for $1" >&2
    exit 2
  fi
}

abs_path() {
  local path="$1"
  if [[ "${path}" = /* ]]; then
    printf '%s\n' "${path}"
  else
    printf '%s\n' "${ROOT_DIR}/${path}"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --policy-dir) need_value "$@"; policy_dir="$(abs_path "$2")"; shift 2 ;;
    --env-cfg) need_value "$@"; env_cfg="$2"; shift 2 ;;
    --task) need_value "$@"; task_name="$2"; shift 2 ;;
    --ckpt) need_value "$@"; ckpt_name="$2"; shift 2 ;;
    --sim-env) need_value "$@"; sim_env="$2"; shift 2 ;;
    --policy-env) need_value "$@"; policy_env="$2"; shift 2 ;;
    --summary) need_value "$@"; summary_path="$2"; shift 2 ;;
    --skip-isaac) skip_isaac="true"; shift ;;
    --skip-conda) skip_conda="true"; shift ;;
    --skip-policy) skip_policy="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "[verify_install] Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

RESULTS_FILE="$(mktemp)"
trap 'rm -f "${RESULTS_FILE}"' EXIT

record() {
  local status="$1"
  local name="$2"
  local message="$3"
  printf '[%s] %s - %s\n' "${status}" "${name}" "${message}"
  printf '%s\t%s\t%s\n' "${status}" "${name}" "${message}" >> "${RESULTS_FILE}"
}

check_path() {
  local kind="$1"
  local path="$2"
  local name="$3"
  if [[ "${kind}" == "dir" && -d "${path}" ]]; then
    record "PASS" "${name}" "${path}"
  elif [[ "${kind}" == "file" && -f "${path}" ]]; then
    record "PASS" "${name}" "${path}"
  else
    record "FAIL" "${name}" "missing ${kind}: ${path}"
  fi
}

echo "[verify_install] root=${ROOT_DIR}"
if [[ -n "${policy_dir}" && "${skip_policy}" != "true" ]]; then
  echo "[verify_install] policy_dir=${policy_dir}"
else
  echo "[verify_install] policy checks disabled"
fi

check_path dir "${ROOT_DIR}/env" "env package"
check_path file "${ROOT_DIR}/scripts/robodojo.sh" "robodojo launcher"
check_path file "${ROOT_DIR}/scripts/internal/task_inventory.py" "task inventory helper"
check_path file "${ROOT_DIR}/env_cfg/${env_cfg}.yml" "env_cfg"
check_path file "${ROOT_DIR}/task/RoboDojo/config/${task_name}.yml" "task config"
check_path dir "${ROOT_DIR}/Assets/Robots" "robot assets"
check_path dir "${ROOT_DIR}/Assets/Object/RoboDojo" "object assets"
check_path dir "${ROOT_DIR}/Assets/Eval_Layout/RoboDojo" "eval layouts"
check_path dir "${ROOT_DIR}/Assets/Material" "materials"

if [[ -n "${policy_dir}" && "${skip_policy}" != "true" ]]; then
  check_path file "${policy_dir}/eval.sh" "policy eval.sh"
  check_path file "${policy_dir}/deploy.yml" "policy deploy.yml"
  if [[ -n "${ckpt_name}" ]]; then
    check_path dir "${policy_dir}/checkpoints/${ckpt_name}" "policy checkpoint"
  else
    record "WARN" "policy checkpoint" "skipped; pass --ckpt to validate"
  fi
else
  record "WARN" "policy checks" "skipped; pass --policy-dir to validate a policy"
fi

if python3 "${ROOT_DIR}/scripts/internal/task_inventory.py" --format json --check >/tmp/robodojo_tasks_verify.json; then
  task_counts="$(python3 - <<'PY'
import json
with open('/tmp/robodojo_tasks_verify.json', encoding='utf-8') as f:
    payload = json.load(f)
print(payload['counts'])
PY
)"
  record "PASS" "task inventory" "${task_counts}"
else
  record "FAIL" "task inventory" "task inventory check failed"
fi
rm -f /tmp/robodojo_tasks_verify.json

if python3 - <<PY; then
from pathlib import Path
import sys
import yaml

root = Path("${ROOT_DIR}")
env_cfg = "${env_cfg}"
cfg_path = root / "env_cfg" / f"{env_cfg}.yml"
cfg = yaml.safe_load(cfg_path.read_text()) or {}
refs = cfg.get("config", {})
required = [
    root / "env_cfg" / "sim" / f"{refs.get('sim')}.yml",
    root / "env_cfg" / "scene" / f"{refs.get('scene')}.yml",
    root / "env_cfg" / "robot" / f"{refs.get('robot')}.yml",
    root / "env_cfg" / "camera" / f"{refs.get('camera')}.yml",
    root / "env_cfg" / "robot" / "_robot_info.json",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    print("\\n".join(missing), file=sys.stderr)
    raise SystemExit(1)
PY
  record "PASS" "env_cfg references" "all referenced sim/scene/robot/camera files exist"
else
  record "FAIL" "env_cfg references" "missing referenced env_cfg file"
fi

if [[ -n "${policy_dir}" && "${skip_policy}" != "true" ]]; then
if python3 - <<PY; then
from pathlib import Path
import yaml

policy_dir = Path("${policy_dir}")
cfg = yaml.safe_load((policy_dir / "deploy.yml").read_text()) or {}
processor = cfg.get("processor_path")
if processor:
    path = Path(processor).expanduser()
    if not path.is_absolute():
        path = policy_dir / path
    if not path.exists():
        raise SystemExit(f"missing processor_path: {path}")
PY
  record "PASS" "policy processor" "deploy.yml processor_path exists"
else
  record "FAIL" "policy processor" "deploy.yml processor_path is missing"
fi
else
  record "WARN" "policy processor" "skipped; pass --policy-dir to validate"
fi

if python3 - <<PY; then
from env.global_configs import BENCHMARK, ROOT_DIR
assert BENCHMARK == "RoboDojo"
assert ROOT_DIR
PY
  record "PASS" "python import" "env.global_configs imports"
else
  record "FAIL" "python import" "cannot import env.global_configs"
fi

if [[ "${skip_conda}" == "true" ]]; then
  record "WARN" "conda envs" "skipped by --skip-conda"
elif command -v conda >/dev/null 2>&1; then
  envs="$(conda env list | awk 'NF && $1 !~ /^#/ {print $1}')"
  if grep -qx "${sim_env}" <<< "${envs}"; then
    record "PASS" "sim conda env" "${sim_env}"
  else
    record "FAIL" "sim conda env" "${sim_env} not found"
  fi
  if [[ -z "${policy_env}" ]]; then
    record "WARN" "policy conda env" "skipped; pass --policy-env to validate"
  elif [[ "${policy_env}" == "uv" || "${policy_env}" == */* ]]; then
    record "WARN" "policy conda env" "policy env is path/uv; conda check skipped"
  elif grep -qx "${policy_env}" <<< "${envs}"; then
    record "PASS" "policy conda env" "${policy_env}"
  else
    record "FAIL" "policy conda env" "${policy_env} not found"
  fi
else
  record "FAIL" "conda" "conda command not found"
fi

if [[ "${skip_isaac}" == "true" ]]; then
  record "WARN" "native runtime" "skipped by --skip-isaac; runtime is not verified"
elif command -v conda >/dev/null 2>&1; then
  # conda run may consume stdin; use -c so an empty Python process cannot pass.
  native_probe="$(cat <<'PY'
import importlib
import importlib.metadata as metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tomllib
from urllib.parse import unquote, urlparse

root = Path(sys.argv[1]).resolve()
def report(status, name, message):
    print(f"{status}\t{name}\t{str(message).replace(chr(10), ' ')}", flush=True)

def check(name, function):
    try:
        report("PASS", name, function())
    except Exception as exc:
        report("FAIL", name, exc)

def python_version():
    assert sys.version_info[:2] == (3, 12), f"expected Python 3.12, got {sys.version}"
    return f"{sys.version.split()[0]}; executable={sys.executable}; prefix={sys.prefix}"

check("sim Python", python_version)
lab = root / "third_party/IsaacLab"
project = tomllib.loads((lab / "pyproject.toml").read_text())
versions = project["tool"]["isaaclab"]["versions"]

def source_identity(relative):
    path = root / relative
    actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    recorded = subprocess.check_output(["git", "-C", str(root), "ls-files", "--stage", "--", relative], text=True)
    assert recorded.startswith("160000 "), f"missing superproject gitlink: {relative}"
    expected = recorded.split()[1]
    assert actual == expected, f"checkout={actual}; recorded gitlink={expected}"
    return f"{actual}; {path}"

for relative in ("third_party/IsaacLab", "third_party/curobo", "XPolicyLab"):
    check(f"source {relative}", lambda relative=relative: source_identity(relative))

def lab_release():
    release = (lab / "VERSION").read_text().strip()
    assert release == "3.0.0", f"expected Lab source 3.0.0, got {release}"
    return f"source={release}; distribution metadata={metadata.version('isaaclab')}"

check("Lab release", lab_release)

def distribution_version(name, expected):
    version = metadata.version(name)
    assert version.split("+")[0] == expected, f"expected {expected}, got {version}"
    return f"{version}; {metadata.distribution(name).locate_file('')}"

for name, expected in (("isaacsim", versions["isaacsim"]), ("torch", versions["torch"]),
                       ("torchvision", versions["torchvision"]), ("warp-lang", versions["warp"])):
    check(f"version {name}", lambda name=name, expected=expected: distribution_version(name, expected))

def module_path(name, expected):
    spec = importlib.util.find_spec(name)
    assert spec is not None and spec.origin, f"module {name} unavailable"
    actual = Path(spec.origin).resolve()
    assert actual.is_relative_to(expected.resolve()), f"shadowed module: {actual}; expected under {expected}"
    return actual

for name, expected in (("isaacsim", Path(sys.prefix)), ("torch", Path(sys.prefix)),
                       ("numpy", Path(sys.prefix)), ("warp", Path(sys.prefix)),
                       ("isaaclab", lab / "source/isaaclab"),
                       ("isaaclab_physx", lab / "source/isaaclab_physx"),
                       ("curobo", root / "third_party/curobo")):
    check(f"module {name}", lambda name=name, expected=expected: module_path(name, expected))

def editable(name, expected):
    data = json.loads(metadata.distribution(name).read_text("direct_url.json") or "{}")
    assert data.get("dir_info", {}).get("editable"), f"{name} is not an editable source install"
    url = urlparse(data["url"])
    assert url.scheme == "file" and Path(unquote(url.path)).resolve() == expected.resolve(), data
    return data["url"]

check("editable isaaclab", lambda: editable("isaaclab", lab / "source/isaaclab"))
check("editable curobo", lambda: editable("nvidia-curobo", root / "third_party/curobo"))

def torch_runtime():
    import torch
    assert torch.__version__.split("+")[0] == versions["torch"], torch.__version__
    assert torch.version.cuda == "13.0", f"expected CUDA 13.0 Torch, got {torch.version.cuda}"
    return f"{torch.__version__}; CUDA build={torch.version.cuda}; {torch.__file__}; GPU execution not tested"

check("loaded Torch", torch_runtime)

def numpy_runtime():
    import numpy
    assert int(numpy.__version__.split('.')[0]) == 2, numpy.__version__
    return f"{numpy.__version__}; {numpy.__file__}"

check("loaded NumPy", numpy_runtime)

def numba_runtime():
    import numba
    import numpy
    # numba's own import guard checks its supported NumPy ABI range.
    return f"numba={numba.__version__}; NumPy={numpy.__version__}; import guard passed"

check("Numba/NumPy ABI", numba_runtime)

def curobo_imports():
    from curobo.batch_motion_planner import BatchMotionPlanner, MotionPlannerCfg
    from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
    from curobo.motion_planner import MotionPlanner
    from curobo.types import ToolPoseCriteria
    return "planner APIs import; CUDA kernel execution not tested"

check("cuRobo APIs", curobo_imports)

# pip check cannot understand Lab's documented resolver overrides. Report those
# mismatches explicitly, and fail every unrelated missing/incompatible dependency.
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

overrides = {canonicalize_name(r.name): r for r in map(Requirement, project["tool"]["uv"]["override-dependencies"])}
# Only these requester/dependency pairs have been inspected and confirmed as
# deliberate upstream overrides. A broad name-only waiver would hide e.g. Numba ABI failures.
known_overrides = {("isaacsim-kernel", "urllib3"), ("isaacsim-core", "fsspec")}
conflicts = 0
for dist in metadata.distributions():
    for raw in dist.requires or ():
        requirement = Requirement(raw)
        if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
            continue
        try:
            installed = metadata.version(requirement.name)
        except metadata.PackageNotFoundError:
            report("FAIL", "dependency conflict", f"{dist.metadata['Name']} requires {raw}; missing")
            conflicts += 1
            continue
        if requirement.specifier.contains(installed, prereleases=True):
            continue
        override = overrides.get(canonicalize_name(requirement.name))
        pair = (canonicalize_name(dist.metadata["Name"]), canonicalize_name(requirement.name))
        intentional = pair in known_overrides and override is not None and override.specifier.contains(installed, prereleases=True)
        report("WARN" if intentional else "FAIL", "upstream override" if intentional else "dependency conflict",
               f"{dist.metadata['Name']} requires {raw}; installed={installed}" + (f"; Lab override={override}" if intentional else ""))
        conflicts += not intentional
if conflicts == 0:
    report("PASS", "dependency audit", "no conflicts beyond the explicitly reported upstream overrides")
print("DONE\tnative probe\tfinished", flush=True)
sys.exit(0)
PY
)"
  native_output="$(mktemp)"
  if conda run --no-capture-output -n "${sim_env}" python -c "${native_probe}" "${ROOT_DIR}" >"${native_output}" 2>&1; then
    native_status=0
  else
    native_status=$?
  fi
  native_done="false"
  while IFS=$'\t' read -r status name message; do
    case "${status}" in
      PASS|WARN|FAIL) record "${status}" "${name}" "${message}" ;;
      DONE) native_done="true" ;;
      *) printf '%s\n' "${status}${name:+ ${name}}${message:+ ${message}}" ;;
    esac
  done <"${native_output}"
  rm -f "${native_output}"
  if [[ "${native_done}" != "true" || "${native_status}" != 0 ]]; then
    record "FAIL" "native runtime" "probe failed/incomplete (exit ${native_status}); see diagnostics above"
  fi
else
  record "FAIL" "native runtime" "conda unavailable; cannot check ${sim_env}"
fi

if [[ -n "${summary_path}" ]]; then
  mkdir -p "$(dirname "${summary_path}")"
  python3 - "${RESULTS_FILE}" "${summary_path}" <<'PY'
import json
from pathlib import Path
import sys

rows = []
for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    status, name, message = line.split("\t", 2)
    rows.append({"status": status, "name": name, "message": message})
payload = {
    "counts": {
        "pass": sum(row["status"] == "PASS" for row in rows),
        "warn": sum(row["status"] == "WARN" for row in rows),
        "fail": sum(row["status"] == "FAIL" for row in rows),
    },
    "checks": rows,
}
Path(sys.argv[2]).write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
  echo "[verify_install] summary=${summary_path}"
fi

fail_count="$(awk -F '\t' '$1 == "FAIL" {count++} END {print count + 0}' "${RESULTS_FILE}")"
warn_count="$(awk -F '\t' '$1 == "WARN" {count++} END {print count + 0}' "${RESULTS_FILE}")"
pass_count="$(awk -F '\t' '$1 == "PASS" {count++} END {print count + 0}' "${RESULTS_FILE}")"
echo "[verify_install] pass=${pass_count} warn=${warn_count} fail=${fail_count}"

if [[ "${fail_count}" -gt 0 ]]; then
  exit 1
fi
