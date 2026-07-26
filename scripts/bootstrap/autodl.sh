#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 0 ]]; then
    printf 'error: scripts/bootstrap/autodl.sh accepts no arguments\n' >&2
    exit 2
fi

sg_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
sg_repo_root="$(cd -- "${sg_script_dir}/../.." && pwd -P)"
cd "${sg_repo_root}"

sg_die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

for sg_command in git python3 uname getconf; do
    command -v "${sg_command}" >/dev/null 2>&1 \
        || sg_die "required command is missing: ${sg_command}"
done

sg_python_version="3.10.18"
sg_runtime_root="${sg_repo_root}/.runtime"
sg_python_install_root="${sg_runtime_root}/python"
sg_env_root="${sg_runtime_root}/envs"
sg_receipt_root="${sg_runtime_root}/receipts"
sg_bootstrap_uv_env="${sg_runtime_root}/bootstrap-uv"
sg_project_env="${sg_repo_root}/.venv"

python3 - "${sg_repo_root}" <<'PY'
from __future__ import annotations

import pathlib
import sys

project = pathlib.Path(sys.argv[1]).resolve()
runtime = project / ".runtime"
project_paths = (runtime, project / ".venv")
runtime_paths = (
    runtime / "python",
    runtime / "envs",
    runtime / "receipts",
    runtime / "bootstrap-uv",
    runtime / "envs" / "4kagent",
    runtime / "envs" / "depictqa",
    runtime / "envs" / "coz",
)
for path in (*project_paths, *runtime_paths):
    if path.is_symlink():
        raise SystemExit(f"managed runtime path must not be a symlink: {path}")
    if path.exists() and not path.is_dir():
        raise SystemExit(f"managed runtime path must be a directory: {path}")
    resolved = path.resolve(strict=False)
    boundary = project if path in project_paths else runtime.resolve(strict=False)
    if not resolved.is_relative_to(boundary):
        raise SystemExit(f"managed runtime path escapes its boundary: {path} -> {resolved}")
PY

mkdir -p \
    "${sg_python_install_root}" \
    "${sg_env_root}" \
    "${sg_receipt_root}"

sg_bootstrap_receipt="${sg_receipt_root}/bootstrap.json"
sg_write_stage_receipt() {
    local sg_status="$1"
    local sg_return_code="$2"
    local sg_temporary="${sg_receipt_root}/.bootstrap.json.tmp.$$"
    {
        printf '{\n'
        printf '  "schema_version": 1,\n'
        printf '  "status": "%s",\n' "${sg_status}"
        printf '  "updated_at_utc": "%s",\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
        printf '  "return_code": %s,\n' "${sg_return_code}"
        printf '  "claim": "This receipt supports no environment or GPU claim."\n'
        printf '}\n'
    } > "${sg_temporary}"
    mv "${sg_temporary}" "${sg_bootstrap_receipt}"
}

sg_bootstrap_failed() {
    local sg_return_code=$?
    set +e
    if [[ "${sg_return_code}" -ne 0 ]]; then
        sg_write_stage_receipt failed "${sg_return_code}"
    fi
}

sg_write_stage_receipt running 0
trap sg_bootstrap_failed EXIT

[[ "$(uname -s)" == "Linux" ]] || sg_die "the AutoDL runtime requires Linux"
[[ "$(uname -m)" == "x86_64" ]] || sg_die "the AutoDL runtime requires x86_64"
sg_glibc_record="$(getconf GNU_LIBC_VERSION 2>/dev/null)" \
    || sg_die "glibc could not be identified"
sg_glibc_version="${sg_glibc_record#glibc }"
sg_glibc_major="${sg_glibc_version%%.*}"
sg_glibc_minor="${sg_glibc_version#*.}"
sg_glibc_minor="${sg_glibc_minor%%.*}"
[[ "${sg_glibc_major}" =~ ^[0-9]+$ && "${sg_glibc_minor}" =~ ^[0-9]+$ ]] \
    || sg_die "unexpected glibc version: ${sg_glibc_record}"
if (( sg_glibc_major < 2 || (sg_glibc_major == 2 && sg_glibc_minor < 28) )); then
    sg_die "glibc 2.28 or newer is required; found ${sg_glibc_version}"
fi

sg_expected_uv="$(tr -d '[:space:]' < environments/uv.version)"
sg_uv=""
if command -v uv >/dev/null 2>&1; then
    sg_candidate_uv="$(command -v uv)"
    sg_candidate_version="$("${sg_candidate_uv}" --version | awk '{print $2}')"
    if [[ "${sg_candidate_version}" == "${sg_expected_uv}" ]]; then
        sg_uv="${sg_candidate_uv}"
    fi
fi
if [[ -z "${sg_uv}" ]]; then
    if [[ ! -x "${sg_bootstrap_uv_env}/bin/python" ]]; then
        python3 -m venv --clear "${sg_bootstrap_uv_env}"
    fi
    "${sg_bootstrap_uv_env}/bin/python" -m pip install \
        --disable-pip-version-check \
        --no-deps \
        --only-binary=:all: \
        --require-hashes \
        -r environments/bootstrap/uv.lock
    sg_uv="${sg_bootstrap_uv_env}/bin/uv"
fi
sg_actual_uv="$("${sg_uv}" --version | awk '{print $2}')"
[[ "${sg_actual_uv}" == "${sg_expected_uv}" ]] \
    || sg_die "could not provision uv ${sg_expected_uv}; found ${sg_actual_uv}"

if [[ -n "${UV_PROJECT_ENVIRONMENT:-}" \
    && "${UV_PROJECT_ENVIRONMENT}" != "${sg_project_env}" \
    && "${UV_PROJECT_ENVIRONMENT}" != ".venv" ]]
then
    sg_die "UV_PROJECT_ENVIRONMENT must target ${sg_project_env}"
fi
export UV_PROJECT_ENVIRONMENT="${sg_project_env}"
unset VIRTUAL_ENV

export UV_PYTHON_INSTALL_DIR="${sg_python_install_root}"
export UV_LINK_MODE="copy"
if [[ -z "${UV_CACHE_DIR:-}" ]]; then
    if [[ -n "${XDG_CACHE_HOME:-}" ]]; then
        export UV_CACHE_DIR="${XDG_CACHE_HOME}/uv"
    else
        export UV_CACHE_DIR="${sg_runtime_root}/cache/uv"
    fi
fi

"${sg_uv}" python install "${sg_python_version}"
"${sg_uv}" sync \
    --locked \
    --extra metrics \
    --managed-python \
    --python "${sg_python_version}"

sg_assert_python() {
    local sg_python="$1"
    local sg_label="$2"
    local sg_actual
    [[ -x "${sg_python}" ]] || sg_die "${sg_label} Python is missing: ${sg_python}"
    sg_actual="$("${sg_python}" -c 'import platform; print(platform.python_version())')"
    [[ "${sg_actual}" == "${sg_python_version}" ]] \
        || sg_die "${sg_label} requires Python ${sg_python_version}; found ${sg_actual}"
}

sg_assert_python "${sg_repo_root}/.venv/bin/python" "ScaleGuard"

"${sg_repo_root}/.venv/bin/python" scripts/upstream/materialize.py \
    upstream-lock.yaml \
    --mapping repositories \
    --project-root "${sg_repo_root}"
"${sg_repo_root}/.venv/bin/python" scripts/upstream/materialize.py \
    runtime-dependencies.yaml \
    --mapping dependencies \
    --project-root "${sg_repo_root}"

sg_prepare_env() {
    local sg_name="$1"
    local sg_env="${sg_env_root}/${sg_name}"
    local sg_python="${sg_env}/bin/python"
    if [[ -x "${sg_python}" ]]; then
        local sg_existing_version
        sg_existing_version="$("${sg_python}" -c 'import platform; print(platform.python_version())')"
        if [[ "${sg_existing_version}" != "${sg_python_version}" ]]; then
            "${sg_uv}" venv \
                --clear \
                --managed-python \
                --python "${sg_python_version}" \
                "${sg_env}"
        fi
    else
        "${sg_uv}" venv \
            --clear \
            --managed-python \
            --python "${sg_python_version}" \
            "${sg_env}"
    fi
    sg_assert_python "${sg_python}" "${sg_name}"
}

sg_sync_env() {
    local sg_name="$1"
    local sg_lock="$2"
    local sg_index="${3:-}"
    local sg_python="${sg_env_root}/${sg_name}/bin/python"
    local -a sg_sync=(
        "${sg_uv}" pip sync
        --python "${sg_python}"
        --require-hashes
        --index-strategy unsafe-best-match
    )
    if [[ -n "${sg_index}" ]]; then
        sg_sync+=(--index "${sg_index}")
    fi
    sg_sync+=("${sg_lock}")
    "${sg_sync[@]}"
}

for sg_name in 4kagent depictqa coz; do
    sg_prepare_env "${sg_name}"
done

sg_sync_env \
    4kagent \
    environments/4kagent/requirements.resolved.lock \
    https://download.pytorch.org/whl/cu126
"${sg_uv}" pip install \
    --python "${sg_env_root}/4kagent/bin/python" \
    --no-deps \
    --require-hashes \
    -r environments/4kagent/pyiqa.override.lock \
    -r environments/4kagent/hpsv2.override.lock
sg_sync_env \
    depictqa \
    environments/depictqa/requirements.resolved.lock \
    https://download.pytorch.org/whl/cu126
sg_sync_env \
    coz \
    environments/coz/requirements.resolved.lock \
    https://download.pytorch.org/whl/cu126

sg_audit_script="${sg_repo_root}/scripts/bootstrap/audit_environment.py"
"${sg_repo_root}/.venv/bin/python" "${sg_audit_script}" \
    --name scaleguard \
    --project-root "${sg_repo_root}" \
    --lock "${sg_repo_root}/uv.lock" \
    --output "${sg_receipt_root}/scaleguard.json" \
    --expected-python "${sg_python_version}" \
    --expect scaleguard-4k==0.1.0.dev0 \
    --expect pyiqa==0.1.16
"${sg_env_root}/4kagent/bin/python" "${sg_audit_script}" \
    --name 4kagent \
    --project-root "${sg_repo_root}" \
    --lock "${sg_repo_root}/environments/4kagent/requirements.resolved.lock" \
    --lock "${sg_repo_root}/environments/4kagent/pyiqa.override.lock" \
    --lock "${sg_repo_root}/environments/4kagent/hpsv2.override.lock" \
    --output "${sg_receipt_root}/4kagent.json" \
    --expected-python "${sg_python_version}" \
    --allow-4kagent-runtime-overrides
"${sg_env_root}/depictqa/bin/python" "${sg_audit_script}" \
    --name depictqa \
    --project-root "${sg_repo_root}" \
    --lock "${sg_repo_root}/environments/depictqa/requirements.resolved.lock" \
    --output "${sg_receipt_root}/depictqa.json" \
    --expected-python "${sg_python_version}"
"${sg_env_root}/coz/bin/python" "${sg_audit_script}" \
    --name coz \
    --project-root "${sg_repo_root}" \
    --lock "${sg_repo_root}/environments/coz/requirements.resolved.lock" \
    --output "${sg_receipt_root}/coz.json" \
    --expected-python "${sg_python_version}"

"${sg_repo_root}/.venv/bin/python" scripts/upstream/materialize.py \
    upstream-lock.yaml \
    --mapping repositories \
    --project-root "${sg_repo_root}" \
    --verify-only
"${sg_repo_root}/.venv/bin/python" scripts/upstream/materialize.py \
    runtime-dependencies.yaml \
    --mapping dependencies \
    --project-root "${sg_repo_root}" \
    --verify-only

"${sg_repo_root}/.venv/bin/python" - \
    "${sg_receipt_root}" \
    "${sg_expected_uv}" \
    "${sg_glibc_version}" \
    "${sg_repo_root}/src" <<'PY'
from __future__ import annotations

import datetime as dt
import hashlib
import json
import pathlib
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(sys.argv[4]).resolve()))

from scaleguard.strict_json import loads_object


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


receipt_root = pathlib.Path(sys.argv[1]).resolve()
uv_version = sys.argv[2]
glibc_version = sys.argv[3]
project_root = pathlib.Path.cwd().resolve()
environment_receipts: dict[str, Any] = {}
for name in ("scaleguard", "4kagent", "depictqa", "coz"):
    path = receipt_root / f"{name}.json"
    document = loads_object(path.read_text(encoding="utf-8"))
    if document.get("status") not in {"passed", "passed_with_audited_override"}:
        raise SystemExit(f"environment audit did not pass: {name}")
    environment_receipts[name] = {
        "path": str(path.relative_to(project_root)),
        "sha256": sha256(path),
        "status": document["status"],
    }

commit = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    cwd=project_root,
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
lock_paths = (
    pathlib.Path("uv.lock"),
    pathlib.Path("upstream-lock.yaml"),
    pathlib.Path("runtime-dependencies.yaml"),
    pathlib.Path("environments/uv.version"),
    pathlib.Path("environments/bootstrap/uv.lock"),
    pathlib.Path("environments/4kagent/requirements.lock"),
    pathlib.Path("environments/4kagent/requirements.resolved.lock"),
    pathlib.Path("environments/4kagent/pyiqa.override.lock"),
    pathlib.Path("environments/4kagent/hpsv2.override.lock"),
    pathlib.Path("environments/depictqa/requirements.lock"),
    pathlib.Path("environments/depictqa/requirements.resolved.lock"),
    pathlib.Path("environments/coz/requirements.lock"),
    pathlib.Path("environments/coz/requirements.resolved.lock"),
)
document = {
    "schema_version": 1,
    "status": "passed",
    "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "project_commit": commit,
    "python_version": "3.10.18",
    "uv_version": uv_version,
    "platform": {"system": "Linux", "machine": "x86_64", "glibc": glibc_version},
    "locks": {
        str(path): sha256(project_root / path)
        for path in lock_paths
    },
    "environments": environment_receipts,
    "claim": (
        "Locked environments and static dependency contracts passed. "
        "This receipt contains no GPU inference or quality-result claim."
    ),
}
temporary = receipt_root / ".bootstrap.json.tmp"
temporary.write_text(
    json.dumps(document, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
temporary.replace(receipt_root / "bootstrap.json")
PY

trap - EXIT
printf 'ScaleGuard AutoDL environments are locked and audited: %s\n' \
    "${sg_receipt_root}/bootstrap.json"
