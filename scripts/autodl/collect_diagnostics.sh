#!/bin/bash -p
# shellcheck shell=bash
if [[ $- != *p* ]]; then
    printf '%s\n' "error: invoke this AutoDL entry directly; an explicit Bash must use -p" >&2
    exit 2
fi
while IFS= read -r sg_imported_function; do
    builtin unset -f -- "${sg_imported_function}"
done < <(builtin compgen -A function)
builtin unset sg_imported_function
builtin set +x +v
# shellcheck source-path=SCRIPTDIR
set -Eeuo pipefail

sg_entry_source="${BASH_SOURCE[0]}"
sg_here="${sg_entry_source%/*}"
if [[ "${sg_here}" == "${sg_entry_source}" ]]; then
    sg_here="."
fi
if [[ "${sg_here}" != /* ]]; then
    sg_here="${PWD}/${sg_here}"
fi
# shellcheck source=_common.sh
# shellcheck disable=SC1091
source "${sg_here}/_common.sh"
# shellcheck disable=SC2154
sg_here="${sg_script_dir}"

sg_resolve_autodl_scheduler_envs
# Exact values remain as non-exported shell variables for redaction. System
# probes receive none of them; the sanitizer gets a dedicated read-only FD.
sg_make_sensitive_environment_private
sg_init_paths
sg_source="${SCALEGUARD_DIAGNOSTICS_SOURCE:-${SG_ARTIFACT_ROOT}}"
sg_requested_output=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source)
            [[ $# -ge 2 ]] || sg_die "--source requires a directory"
            sg_source="$2"
            shift 2
            ;;
        --output)
            [[ $# -ge 2 ]] || sg_die "--output requires a .tar.gz path"
            sg_requested_output="$2"
            shift 2
            ;;
        -h|--help)
            cat <<'EOF'
Usage: scripts/autodl/collect_diagnostics.sh [--source DIR] [--output FILE.tar.gz]

Collects allowlisted text evidence, GPU/system details, git state, and hashes.
It excludes images, weights, environment dumps, oversized files, and secret-like
values. Always inspect the archive before sharing it.
EOF
            exit 0
            ;;
        *)
            sg_die "unknown argument: $1"
            ;;
    esac
done

sg_source="$(sg_from_repo "${sg_source}")"
[[ -d "${sg_source}" ]] || sg_die "diagnostics source is not a directory: ${sg_source}"
command -v python3 >/dev/null 2>&1 || sg_die "python3 is required"
command -v tar >/dev/null 2>&1 || sg_die "tar is required"
sg_source="$(
    python3 -I -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' \
        "${sg_source}"
)"
sg_artifact_root_resolved="$(
    python3 -I -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' \
        "${SG_ARTIFACT_ROOT}"
)"
sg_repo_root_resolved="$(
    python3 -I -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' \
        "${SG_REPO_ROOT}"
)"
sg_cache_root_resolved="$(
    python3 -I -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' \
        "${SG_CACHE_ROOT}"
)"
case "${sg_artifact_root_resolved}" in
    "${sg_repo_root_resolved}"/*|"${sg_cache_root_resolved}"/*)
        ;;
    *)
        sg_die "artifact root must stay below the project or ScaleGuard cache root"
        ;;
esac
case "${sg_source}" in
    "${sg_artifact_root_resolved}"|"${sg_artifact_root_resolved}"/*)
        ;;
    *)
        sg_die \
            "diagnostics source must stay below the artifact root: ${sg_artifact_root_resolved}"
        ;;
esac

sg_new_run_dir diagnostics
sg_stage_dir="${SG_RUN_DIR}/staging"
sg_system_dir="${sg_stage_dir}/system"
mkdir -p "${sg_system_dir}"

if [[ -n "${sg_requested_output}" ]]; then
    if [[ "${sg_requested_output}" != /* ]]; then
        sg_requested_output="$(sg_from_repo "${sg_requested_output}")"
    fi
    [[ "${sg_requested_output}" == *.tar.gz ]] \
        || sg_die "--output must end in .tar.gz"
    [[ ! -e "${sg_requested_output}" && ! -L "${sg_requested_output}" ]] \
        || sg_die "diagnostics output already exists: ${sg_requested_output}"
    sg_bundle="${sg_requested_output}"
    mkdir -p "$(dirname -- "${sg_bundle}")"
else
    sg_bundle="${SG_RUN_DIR}/scaleguard-diagnostics-$(sg_timestamp).tar.gz"
fi

(
    uname -a
) > "${sg_system_dir}/uname.txt" 2>&1 || true
if [[ -r /etc/os-release ]]; then
    cp -- /etc/os-release "${sg_system_dir}/os-release.txt"
fi
if command -v lscpu >/dev/null 2>&1; then
    lscpu > "${sg_system_dir}/lscpu.txt" 2>&1 || true
fi
if command -v free >/dev/null 2>&1; then
    free -h > "${sg_system_dir}/memory.txt" 2>&1 || true
fi
df -h -- "${SG_REPO_ROOT}" "${SG_CACHE_ROOT}" \
    > "${sg_system_dir}/disk.txt" 2>&1 || true

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi > "${sg_system_dir}/nvidia-smi.txt" 2>&1 || true
    nvidia-smi \
        --query-gpu=index,uuid,name,memory.total,driver_version \
        --format=csv,noheader,nounits \
        > "${sg_system_dir}/gpu-inventory.csv" 2>&1 || true
else
    printf '%s\n' "nvidia-smi not found" > "${sg_system_dir}/nvidia-smi.txt"
fi
if command -v nvcc >/dev/null 2>&1; then
    nvcc --version > "${sg_system_dir}/nvcc.txt" 2>&1 || true
else
    printf '%s\n' "nvcc not found" > "${sg_system_dir}/nvcc.txt"
fi

python3 -I --version > "${sg_system_dir}/python.txt" 2>&1 || true
if [[ -x "${SG_REPO_ROOT}/.venv/bin/python" ]]; then
    "${SG_REPO_ROOT}/.venv/bin/python" -I -m pip freeze \
        > "${sg_system_dir}/pip-freeze.txt" 2>&1 || true
fi
if command -v conda >/dev/null 2>&1; then
    conda env list > "${sg_system_dir}/conda-envs.txt" 2>&1 || true
fi

git -C "${SG_REPO_ROOT}" rev-parse HEAD \
    > "${sg_system_dir}/git-commit.txt" 2>&1 || true
git -C "${SG_REPO_ROOT}" status --short \
    > "${sg_system_dir}/git-status.txt" 2>&1 || true
git -C "${SG_REPO_ROOT}" submodule status --recursive \
    > "${sg_system_dir}/submodules.txt" 2>&1 || true

{
    printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-<unset>}"
    printf 'SCALEGUARD_CACHE_ROOT=%s\n' "${SG_CACHE_ROOT}"
    printf 'HF_HOME=%s\n' "${HF_HOME}"
    printf 'HUGGINGFACE_HUB_CACHE=%s\n' "${HUGGINGFACE_HUB_CACHE}"
    printf 'TRANSFORMERS_CACHE=%s\n' "${TRANSFORMERS_CACHE}"
    printf 'TORCH_HOME=%s\n' "${TORCH_HOME}"
} > "${sg_system_dir}/environment-allowlist.txt"

sg_max_total_files="${SCALEGUARD_DIAGNOSTICS_MAX_TOTAL_FILES:-5000}"
sg_max_total_mib="${SCALEGUARD_DIAGNOSTICS_MAX_TOTAL_MIB:-256}"
[[ "${sg_max_total_files}" =~ ^[1-9][0-9]*$ ]] \
    || sg_die "SCALEGUARD_DIAGNOSTICS_MAX_TOTAL_FILES must be a positive integer"
[[ "${sg_max_total_mib}" =~ ^[1-9][0-9]*$ ]] \
    || sg_die "SCALEGUARD_DIAGNOSTICS_MAX_TOTAL_MIB must be a positive integer"

sg_sanitize_bounded() {
    local sg_scan_source="$1"
    local sg_scan_destination="$2"
    local sg_current_files
    local sg_current_kib
    local sg_copy_files
    local sg_copy_bytes

    sg_current_files="$(find "${sg_stage_dir}" -type f | wc -l | tr -d '[:space:]')"
    sg_current_kib="$(du -sk "${sg_stage_dir}" | awk '{print $1}')"
    # Reserve four small receipt/inventory files and 2 MiB for their contents.
    sg_copy_files="$((sg_max_total_files - sg_current_files - 4))"
    sg_copy_bytes="$((sg_max_total_mib * 1024 * 1024 - sg_current_kib * 1024 - 2 * 1024 * 1024))"
    if [[ "${sg_copy_files}" -lt 0 ]]; then
        sg_copy_files=0
    fi
    if [[ "${sg_copy_bytes}" -lt 0 ]]; then
        sg_copy_bytes=0
    fi
    python3 -I "${sg_here}/_sanitize_diagnostics.py" \
        "${sg_scan_source}" \
        "${sg_scan_destination}" \
        "${SG_REPO_ROOT}" \
        "${SG_CACHE_ROOT}" \
        --secret-fd 3 \
        --max-copied-files "${sg_copy_files}" \
        --max-copied-bytes "${sg_copy_bytes}" \
        3< <(sg_write_private_secret_stream)
}

sg_sanitize_bounded "${sg_source}" "${sg_stage_dir}"

sg_max_model_runs="${SCALEGUARD_DIAGNOSTICS_MAX_MODEL_RUNS:-8}"
[[ "${sg_max_model_runs}" =~ ^[0-9]+$ ]] \
    || sg_die "SCALEGUARD_DIAGNOSTICS_MAX_MODEL_RUNS must be a non-negative integer"
sg_model_runs_collected=0
while IFS= read -r -d '' sg_cli_result; do
    if [[ "${sg_model_runs_collected}" -ge "${sg_max_model_runs}" ]]; then
        break
    fi
    sg_model_run="$(
        python3 -I - "${sg_cli_result}" "${SG_REPO_ROOT}" "${SG_REPO_ROOT}/src" \
            2>> "${SG_RUN_DIR}/model-run-scan-warnings.txt" <<'PY'
import pathlib
import sys

repo = pathlib.Path(sys.argv[2]).resolve()
sys.path.insert(0, str(pathlib.Path(sys.argv[3]).resolve()))

from scaleguard.strict_json import StrictJSONError, loads_object

try:
    result = loads_object(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
    run_dir = pathlib.Path(str(result.get("run_dir", ""))).resolve()
    if (
        run_dir.is_dir()
        and run_dir.is_relative_to(repo)
        and (run_dir / "manifest.json").is_file()
    ):
        print(run_dir)
except (OSError, StrictJSONError) as exc:
    print(
        f"{pathlib.Path(sys.argv[1]).name}: skipped invalid CLI result ({exc})",
        file=sys.stderr,
    )
PY
    )"
    if [[ -z "${sg_model_run}" ]]; then
        continue
    fi
    printf -v sg_model_run_slot 'run-%02d' "$((sg_model_runs_collected + 1))"
    sg_sanitize_bounded \
        "${sg_model_run}" \
        "${sg_stage_dir}/model-runs/${sg_model_run_slot}"
    sg_model_runs_collected="$((sg_model_runs_collected + 1))"
done < <(
    find "${sg_source}" \
        -path '*/diagnostics/*' -prune -o \
        -type f -name cli-result.json -print0
)

sg_assert_staging_bounds() {
    local sg_total_files
    local sg_total_kib
    sg_total_files="$(find "${sg_stage_dir}" -type f | wc -l | tr -d '[:space:]')"
    sg_total_kib="$(du -sk "${sg_stage_dir}" | awk '{print $1}')"
    if [[ "${sg_total_files}" -gt "${sg_max_total_files}" ]] \
        || [[ "${sg_total_kib}" -gt "$((sg_max_total_mib * 1024))" ]]
    then
        sg_die \
            "diagnostics staging exceeds the bounded total: ${sg_total_files} files, ${sg_total_kib} KiB"
    fi
}

sg_assert_staging_bounds

python3 -I - "${sg_stage_dir}/diagnostics.json" "$(sg_timestamp)" <<'PY'
import json
import pathlib
import sys

pathlib.Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "schema_version": 1,
            "collected_at_utc": sys.argv[2],
            "status": "collected",
            "status_scope": (
                "Collection succeeded. This does not assert that GPU or integration checks passed."
            ),
            "redaction": (
                "Allowlisted text only; secrets and private roots are replaced. "
                "Manual review is still required before sharing."
            ),
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY

sg_write_file_inventory "${sg_stage_dir}" "${sg_stage_dir}/files.json"
sg_assert_staging_bounds

sg_secret_hits="${SG_RUN_DIR}/secret-scan.txt"
sg_secret_scan_error="${SG_RUN_DIR}/secret-scan-error.txt"
sg_path_scan_rc=0
python3 -I - "${sg_stage_dir}" 3< <(sg_write_private_secret_stream) \
    2> "${sg_secret_scan_error}" <<'PY' \
    || sg_path_scan_rc=$?
import os
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1]).resolve()
secret_payload = os.fdopen(3, "rb", closefd=True).read(1024 * 1024 + 1)
if len(secret_payload) > 1024 * 1024:
    raise SystemExit("private redaction stream exceeds 1 MiB")
secret_fields = secret_payload.split(b"\0")
if secret_fields[-1:] == [b""]:
    secret_fields.pop()
if len(secret_fields) % 2:
    raise SystemExit("private redaction stream is malformed")
secret_values = [
    secret_fields[index + 1]
    for index in range(0, len(secret_fields), 2)
    if secret_fields[index + 1]
]
patterns = (
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"AIza[A-Za-z0-9_-]{35}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"(?i)Bearer[ _-]+[A-Za-z0-9._~+=/-]{16,}"),
    re.compile(r"(?i)(token|api[_-]?key|password|secret)[=:][^/]+"),
)
for path in root.rglob("*"):
    relative = path.relative_to(root).as_posix()
    relative_bytes = relative.encode("utf-8", errors="surrogateescape")
    if any(pattern.search(relative) for pattern in patterns) or any(
        secret in relative_bytes for secret in secret_values
    ):
        raise SystemExit(42)
    if path.is_file():
        content = path.read_bytes()
        if any(secret in content for secret in secret_values):
            raise SystemExit(42)
PY

if [[ "${sg_path_scan_rc}" -eq 0 ]]; then
    sg_secret_scan_rc=0
    grep -RIEq \
        'hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{16,}|AIza[A-Za-z0-9_-]{35}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|Bearer[[:space:]]+[A-Za-z0-9._~+/=-]{16,}|https?://[^[][^/@[:space:]]*@|https?://[^?[:space:]]+\?[^[][^[:space:]]+|(token|api[_-]?key|password|secret)[[:space:]]*[=:][[:space:]]*[^[][^[:space:],;]+' \
        "${sg_stage_dir}" 2>> "${sg_secret_scan_error}" \
        || sg_secret_scan_rc=$?
elif [[ "${sg_path_scan_rc}" -eq 42 ]]; then
    sg_secret_scan_rc=0
else
    sg_secret_scan_rc=70
fi

if [[ "${sg_secret_scan_rc}" -eq 0 ]]; then
    printf '%s\n' "Secret-like material remained after automatic redaction." \
        > "${sg_secret_hits}"
    python3 -I - "${SG_RUN_DIR}/diagnostics-failure.json" "$(sg_timestamp)" <<'PY'
import json
import pathlib
import sys

pathlib.Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "schema_version": 1,
            "completed_at_utc": sys.argv[2],
            "status": "failed",
            "reason": "secret-like material remained after automatic redaction",
            "claim": "No diagnostics archive was produced by this attempt.",
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY
    sg_die "secret-like material remains after redaction; inspect ${sg_secret_hits} locally"
elif [[ "${sg_secret_scan_rc}" -ne 1 ]]; then
    printf '%s\n' "The bounded post-redaction scan failed to execute." \
        > "${sg_secret_hits}"
    sg_redact_stream < "${sg_secret_scan_error}" >&2
    sg_die "secret scan failed closed with exit code ${sg_secret_scan_rc}"
fi
printf '%s\n' "No secret-like values or paths matched the bounded post-redaction scan." \
    > "${sg_secret_hits}"

COPYFILE_DISABLE=1 tar -C "${sg_stage_dir}" -czf "${sg_bundle}" .
sg_bundle_hash="$(sg_sha256 "${sg_bundle}")"
printf '%s  %s\n' "${sg_bundle_hash}" "$(basename -- "${sg_bundle}")" \
    > "${sg_bundle}.sha256"

sg_note "Diagnostics archive created: ${sg_bundle}"
sg_note "SHA-256: ${sg_bundle_hash}"
sg_note "Inspect the archive before sharing; automated redaction is defense-in-depth."
