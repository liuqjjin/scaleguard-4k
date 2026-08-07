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
sg_original_argv=("$@")

[[ $# -ge 1 ]] || sg_die "internal runner requires a stage"
sg_stage="$1"
shift
case "${sg_stage}" in
    smoke)
        sg_stage_label="smoke"
        sg_input="${SCALEGUARD_SMOKE_INPUT:-}"
        sg_config="${SCALEGUARD_SMOKE_CONFIG:-}"
        ;;
    integration)
        sg_stage_label="integration"
        sg_input="${SCALEGUARD_INTEGRATION_INPUT:-}"
        sg_config="${SCALEGUARD_INTEGRATION_CONFIG:-}"
        ;;
    experiment)
        sg_stage_label="experiment"
        sg_input="${SCALEGUARD_EXPERIMENT_INPUT:-}"
        sg_config="${SCALEGUARD_EXPERIMENT_CONFIG:-}"
        ;;
    *)
        sg_die "internal runner received invalid stage: ${sg_stage}"
        ;;
esac

sg_show_help=0
sg_output=""
sg_evidence_output=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            [[ $# -ge 2 ]] || sg_die "--config requires a path"
            sg_config="$2"
            shift 2
            ;;
        --input)
            [[ $# -ge 2 ]] || sg_die "--input requires an image path"
            sg_input="$2"
            shift 2
            ;;
        --output)
            [[ $# -ge 2 ]] || sg_die "--output requires a new path"
            sg_output="$2"
            shift 2
            ;;
        --evidence-output)
            [[ $# -ge 2 ]] || sg_die "--evidence-output requires a new JSON path"
            sg_evidence_output="$2"
            shift 2
            ;;
        -h|--help)
            sg_show_help=1
            shift
            ;;
        *)
            sg_die "unknown argument: $1"
            ;;
    esac
done

if [[ "${sg_show_help}" -eq 1 ]]; then
    sg_resolve_autodl_scheduler_envs
    if [[ -n "${sg_config}" ]]; then
        if [[ "${sg_config}" != /* ]]; then
            sg_config="${SG_REPO_ROOT}/${sg_config}"
        fi
        sg_resolve_scheduler_api_key_env "${sg_config}"
        sg_register_sensitive_env_name "${SG_SCHEDULER_API_KEY_ENV}"
    fi
    sg_make_sensitive_environment_private
    cat <<EOF
Usage: scripts/autodl/run_${sg_stage}.sh --input IMAGE [--config FILE] [--output PATH]

Runs a real ScaleGuard ${sg_stage_label} invocation after GPU, upstream, and
configuration checks. Input may instead be supplied through
SCALEGUARD_$(printf '%s' "${sg_stage}" | tr '[:lower:]' '[:upper:]')_INPUT.
The output path must not already exist, preventing stale artifacts from passing.
EOF
    if [[ "${sg_stage}" == "experiment" ]]; then
        cat <<'EOF'
Experiment runs also require --evidence-output FILE. The wrapper atomically
writes the unique attempt path and a hash-bound success handoff for the suite.
The experiment group and sample id come only from the preflighted config.
EOF
    fi
    exit 0
fi

if [[ -n "${sg_evidence_output}" && "${sg_stage}" != "experiment" ]]; then
    sg_die "--evidence-output is reserved for the experiment stage"
fi
if [[ "${sg_stage}" == "experiment" && -z "${sg_evidence_output}" ]]; then
    sg_die "experiment stage requires --evidence-output FILE"
fi
if [[ -z "${sg_config}" ]]; then
    if [[ "${sg_stage}" == "smoke" \
        && -f "${SG_REPO_ROOT}/configs/runtime/autodl-smoke.yaml" ]]
    then
        sg_config="${SG_REPO_ROOT}/configs/runtime/autodl-smoke.yaml"
    elif [[ "${sg_stage}" == "experiment" ]]; then
        sg_die "experiment stage requires --config or SCALEGUARD_EXPERIMENT_CONFIG"
    else
        sg_config="${SCALEGUARD_AUTODL_CONFIG:-${SG_REPO_ROOT}/configs/runtime/autodl-2x4090.yaml}"
    fi
fi
if [[ "${sg_config}" != /* ]]; then
    sg_config="${SG_REPO_ROOT}/${sg_config}"
fi
sg_resolve_autodl_scheduler_envs
sg_resolve_scheduler_api_key_env "${sg_config}"
sg_register_sensitive_env_name "${SG_SCHEDULER_API_KEY_ENV}"
# Retain exact values only as non-exported shell variables for log redaction.
# Preflight/evidence children receive no credentials; doctor and the actual
# model command receive only the configured scheduler credential.
sg_make_sensitive_environment_private
sg_deadline_seconds="${SCALEGUARD_RUN_DEADLINE_SECONDS:-14400}"
[[ "${sg_deadline_seconds}" =~ ^[0-9]+$ ]] \
    || sg_die "SCALEGUARD_RUN_DEADLINE_SECONDS must be an integer"
if [[ "${sg_deadline_seconds}" -lt 60 || "${sg_deadline_seconds}" -gt 86400 ]]; then
    sg_die "SCALEGUARD_RUN_DEADLINE_SECONDS must be between 60 and 86400"
fi
sg_deadline_parent_valid=0
if [[ -r "/proc/${PPID}/cmdline" && -e "/proc/${PPID}/exe" ]]; then
    if python3 -I - \
        "/proc/${PPID}/cmdline" \
        "/proc/${PPID}/exe" \
        "${sg_deadline_seconds}s" \
        "${sg_here}/_run_scaleguard.sh" <<'PY'
import os
import pathlib
import sys

argv = [os.fsdecode(value) for value in pathlib.Path(sys.argv[1]).read_bytes().split(b"\0") if value]
executable = pathlib.Path(sys.argv[2]).resolve()
expected = [
    "--signal=TERM",
    "--kill-after=30s",
    sys.argv[3],
    "/bin/bash",
    "-p",
    sys.argv[4],
]
is_timeout = executable == pathlib.Path("/usr/bin/timeout").resolve()
raise SystemExit(0 if is_timeout and argv[1 : 1 + len(expected)] == expected else 1)
PY
    then
        sg_deadline_parent_valid=1
    fi
fi
if [[ "${sg_deadline_parent_valid}" -ne 1 ]]
then
    [[ -x /usr/bin/timeout ]] \
        || sg_die "/usr/bin/timeout is required to enforce the AutoDL run deadline"
    export SCALEGUARD_RUN_DEADLINE_SECONDS
    sg_export_private_credentials "${SG_SCHEDULER_API_KEY_ENV}"
    exec /usr/bin/timeout \
        --signal=TERM \
        --kill-after=30s \
        "${sg_deadline_seconds}s" \
        /bin/bash -p "${sg_here}/_run_scaleguard.sh" "${sg_original_argv[@]}"
fi
if [[ -z "${sg_input}" ]]; then
    for sg_candidate in \
        "${SG_REPO_ROOT}/examples/autodl-smoke.png" \
        "${SG_REPO_ROOT}/examples/input.png" \
        "${SG_REPO_ROOT}/assets/examples/input.png"
    do
        if [[ -f "${sg_candidate}" ]]; then
            sg_input="${sg_candidate}"
            break
        fi
    done
fi

[[ -n "${sg_input}" ]] || sg_die \
    "no input image was supplied; use --input or SCALEGUARD_$(printf '%s' "${sg_stage}" | tr '[:lower:]' '[:upper:]')_INPUT"
if [[ "${sg_input}" != /* ]]; then
    sg_input="${SG_REPO_ROOT}/${sg_input}"
fi
sg_require_file "${sg_input}" "input image"
sg_require_file "${sg_config}" "${sg_stage_label} config"
sg_init_paths
sg_lock="${SCALEGUARD_UPSTREAM_LOCK:-${SG_REPO_ROOT}/upstream-lock.yaml}"
sg_dependency_lock="${SCALEGUARD_RUNTIME_DEPENDENCIES_LOCK:-${SG_REPO_ROOT}/runtime-dependencies.yaml}"
sg_require_file "${sg_lock}" "upstream lock"
sg_require_file "${sg_dependency_lock}" "runtime dependency lock"
sg_require_clean_project

if [[ "${sg_stage}" == "experiment" ]]; then
    if [[ "${sg_evidence_output}" != /* ]]; then
        sg_evidence_output="$(sg_from_repo "${sg_evidence_output}")"
    fi
    mkdir -p "$(dirname -- "${sg_evidence_output}")"
    sg_evidence_output="$(
        python3 -I - "${sg_evidence_output}" <<'PY'
import pathlib
import sys

print(pathlib.Path(sys.argv[1]).resolve())
PY
    )"
    [[ ! -e "${sg_evidence_output}" && ! -L "${sg_evidence_output}" ]] \
        || sg_die \
            "experiment evidence output already exists and could be stale: ${sg_evidence_output}"
fi

sg_new_run_dir "${sg_stage_label}"
if [[ -z "${sg_output}" ]]; then
    sg_output="${SG_RUN_DIR}/output.png"
elif [[ "${sg_output}" != /* ]]; then
    sg_output="$(sg_from_repo "${sg_output}")"
fi
[[ ! -e "${sg_output}" && ! -L "${sg_output}" ]] \
    || sg_die "output already exists and could be mistaken for fresh evidence: ${sg_output}"
mkdir -p "$(dirname -- "${sg_output}")"

sg_log="${SG_RUN_DIR}/${sg_stage_label}.log"
sg_gpu_csv="${SG_RUN_DIR}/gpu-samples.csv"
sg_start_time="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
sg_start_epoch="$(date '+%s')"
sg_run_rc=1
sg_command_rc=1
sg_model_evidence_complete=0

sg_write_attempt_pointer() {
    [[ "${sg_stage}" == "experiment" ]] || return 0
    local sg_pointer_status="$1"
    local sg_pointer_completed_at="${2:-}"
    python3 -I - \
        "${sg_evidence_output}" \
        "${SG_RUN_DIR}" \
        "${sg_pointer_status}" \
        "${sg_start_time}" \
        "${sg_pointer_completed_at}" <<'PY'
import fcntl
import hashlib
import json
import os
import pathlib
import stat
import sys
import tempfile
from typing import Any

(
    output_text,
    attempt_text,
    status_text,
    started_at,
    completed_at,
) = sys.argv[1:]
if status_text not in {"running", "failed", "succeeded"}:
    raise SystemExit(f"invalid experiment attempt status: {status_text}")

output = pathlib.Path(output_text).resolve()
attempt = pathlib.Path(attempt_text).resolve()
if attempt.is_symlink() or not attempt.is_dir():
    raise SystemExit(f"experiment attempt directory is not a regular directory: {attempt}")


def digest(path: pathlib.Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def entry(relative: str) -> dict[str, Any] | None:
    path = attempt / relative
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise SystemExit(f"experiment evidence is not a regular file: {path}")
    return {
        "path": str(path),
        "size_bytes": metadata.st_size,
        "sha256": digest(path),
    }


expected_files = {
    "execution": "execution.json",
    "run_manifest": "scaleguard-run-manifest.json",
    "model_evidence": "model-evidence.json",
    "raw_log": "experiment.log",
    "gpu_samples": "gpu-samples.csv",
    "nvidia_smi_before": "nvidia-smi-before.txt",
    "nvidia_smi_after": "nvidia-smi-after.txt",
    "gpu_inventory": "gpu-preflight/gpu_inventory.csv",
    "gpu_preflight": "gpu-preflight/gpu_check.json",
    "files_inventory": "files.json",
    "runtime_preflight": "runtime-preflight.json",
}
files = {
    name: item
    for name, relative in expected_files.items()
    if (item := entry(relative)) is not None
}
if status_text == "succeeded" and set(files) != set(expected_files):
    missing = sorted(set(expected_files) - set(files))
    raise SystemExit(
        "successful experiment attempt is missing evidence: " + ", ".join(missing)
    )

group = None
sample_id = None
model_evidence = attempt / "model-evidence.json"
if model_evidence.is_file() and not model_evidence.is_symlink():
    summary = json.loads(model_evidence.read_text(encoding="utf-8"))
    if isinstance(summary, dict):
        group = summary.get("experiment_group")
        sample_id = summary.get("experiment_sample_id")
if status_text == "succeeded" and (
    not isinstance(group, str)
    or not group
    or not isinstance(sample_id, str)
    or not sample_id
):
    raise SystemExit("successful experiment attempt has no bound group/sample identity")

hardware = None
gpu_check = attempt / "gpu-preflight" / "gpu_check.json"
if gpu_check.is_file() and not gpu_check.is_symlink():
    gpu_document = json.loads(gpu_check.read_text(encoding="utf-8"))
    selected = gpu_document.get("selected_gpus") if isinstance(gpu_document, dict) else None
    visible = (
        gpu_document.get("cuda_visible_devices")
        if isinstance(gpu_document, dict)
        else None
    )
    if isinstance(selected, list):
        normalized = []
        for item in selected:
            if not isinstance(item, dict):
                raise SystemExit("GPU preflight selected_gpus is malformed")
            normalized.append(
                {
                    "logical_index": item.get("logical_index"),
                    "physical_index": item.get("physical_index"),
                    "uuid": item.get("uuid"),
                    "name": item.get("name"),
                    "memory_total_mib": item.get("memory_total_mib"),
                    "driver_version": item.get("driver_version"),
                }
            )
        normalized.sort(key=lambda item: int(item["logical_index"]))
        identity_payload = {
            "cuda_visible_devices": visible,
            "selected_gpus": normalized,
        }
        class_payload = {
            "selected_gpus": [
                {
                    "logical_index": item["logical_index"],
                    "name": item["name"],
                    "memory_total_mib": item["memory_total_mib"],
                    "driver_version": item["driver_version"],
                }
                for item in normalized
            ]
        }

        def canonical_sha256(value: object) -> str:
            payload = json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            return hashlib.sha256(payload).hexdigest()

        hardware = {
            "identity_sha256": canonical_sha256(identity_payload),
            "class_sha256": canonical_sha256(class_payload),
            "selected_gpu_count": len(normalized),
            "cuda_visible_devices": visible,
        }
if status_text == "succeeded" and hardware is None:
    raise SystemExit("successful experiment attempt has no stable hardware identity")

document = {
    "schema_version": 1,
    "status": status_text,
    "stage": "experiment",
    "attempt_id": attempt.name,
    "attempt_dir": str(attempt),
    "started_at_utc": started_at,
    "completed_at_utc": completed_at or None,
    "experiment_group": group,
    "experiment_sample_id": sample_id,
    "files": files,
    "hardware": hardware,
}
output.parent.mkdir(parents=True, exist_ok=True)
lock_path = output.with_name(f".{output.name}.lock")
lock_flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
lock_flags |= getattr(os, "O_NOFOLLOW", 0)
lock_descriptor = os.open(lock_path, lock_flags, 0o600)
lock_metadata = os.fstat(lock_descriptor)
if not stat.S_ISREG(lock_metadata.st_mode) or lock_metadata.st_uid != os.getuid():
    os.close(lock_descriptor)
    raise SystemExit(f"unsafe experiment handoff lock: {lock_path}")
try:
    fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    os.close(lock_descriptor)
    raise SystemExit(f"experiment handoff is being updated concurrently: {output}")

existing_identity = None
if output.exists() or output.is_symlink():
    metadata = output.lstat()
    if output.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise SystemExit(f"experiment handoff is not a regular file: {output}")
    existing = json.loads(output.read_text(encoding="utf-8"))
    if (
        not isinstance(existing, dict)
        or existing.get("attempt_dir") != str(attempt)
        or existing.get("started_at_utc") != started_at
        or existing.get("status") not in {"running", status_text}
    ):
        raise SystemExit(f"refusing to clobber another experiment handoff: {output}")
    existing_identity = (metadata.st_dev, metadata.st_ino)
elif status_text != "running":
    raise SystemExit("experiment handoff must be published as running before completion")

descriptor, temporary_text = tempfile.mkstemp(
    dir=output.parent,
    prefix=f".{output.name}.",
    suffix=".tmp",
)
temporary = pathlib.Path(temporary_text)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    if existing_identity is None:
        try:
            os.link(temporary, output, follow_symlinks=False)
        except FileExistsError as error:
            raise SystemExit(
                f"experiment handoff appeared concurrently: {output}"
            ) from error
    else:
        current = output.lstat()
        if (current.st_dev, current.st_ino) != existing_identity:
            raise SystemExit(f"experiment handoff changed concurrently: {output}")
        os.replace(temporary, output)
    directory_descriptor = os.open(output.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
finally:
    temporary.unlink(missing_ok=True)
    fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
    os.close(lock_descriptor)
PY
}

sg_finalize_failed_stage() {
    local sg_original_rc=$?
    set +e
    sg_stop_gpu_monitor
    if [[ ! -f "${SG_RUN_DIR}/execution.json" ]]; then
        python3 -I - \
            "${SG_RUN_DIR}/execution.json" \
            "${sg_stage_label}" \
            "${sg_start_time}" \
            "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
            "${sg_original_rc}" <<'PY'
import json
import pathlib
import sys

output, stage, started_at, completed_at, return_code = sys.argv[1:]
pathlib.Path(output).write_text(
    json.dumps(
        {
            "schema_version": 1,
            "stage": stage,
            "status": "failed",
            "started_at_utc": started_at,
            "completed_at_utc": completed_at,
            "return_code": int(return_code),
            "claim": "Preflight or setup failed before model execution completed.",
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY
    fi
    sg_write_file_inventory "${SG_RUN_DIR}" "${SG_RUN_DIR}/files.json"
    sg_write_attempt_pointer failed "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    sg_release_gpu_lease
}
trap sg_finalize_failed_stage EXIT
trap 'sg_stop_gpu_monitor; exit 130' INT
trap 'sg_stop_gpu_monitor; exit 143' TERM
# The canonical profile maps both physical GPUs into one process topology.
# Hold this cooperative host lease for the complete audited attempt so two
# wrappers cannot silently overlap model state, memory, or GPU evidence.
sg_acquire_gpu_lease "canonical-2gpu"
sg_write_attempt_pointer running

"${sg_here}/check_gpu.sh" --output "${SG_RUN_DIR}/gpu-preflight"
sg_resolve_cli
cd "${SG_REPO_ROOT}"

if ! sg_run_logged "${sg_log}" "${SG_CLI[@]}" upstream verify --lock "${sg_lock}"; then
    sg_die "upstream verification failed; inspect ${sg_log}"
fi
if ! sg_run_logged \
    "${sg_log}" \
    "${SG_CLI[@]}" upstream verify \
    --lock "${sg_dependency_lock}" \
    --mapping dependencies
then
    sg_die "runtime dependency verification failed; inspect ${sg_log}"
fi
sg_verify_materialized_weights \
    "${sg_log}" \
    "${SG_RUN_DIR}/materialization-verification.json"
if ! sg_run_logged_with_presence_marker \
    "${sg_log}" \
    "${SG_SCHEDULER_API_KEY_ENV}" \
    "${SG_CLI[@]}" doctor --config "${sg_config}"
then
    sg_die "ScaleGuard doctor failed; inspect ${sg_log}"
fi
sg_runtime_environments="${SG_RUN_DIR}/runtime-environments"
sg_reaudit_runtime_environments \
    "${sg_log}" \
    "${sg_runtime_environments}"
"${SG_REPO_ROOT}/.venv/bin/python" -I \
    "${sg_here}/_write_preflight_receipt.py" \
    --config "${sg_config}" \
    --materialization "${SG_RUN_DIR}/materialization-verification.json" \
    --runtime-environments "${sg_runtime_environments}" \
    --gpu-check "${SG_RUN_DIR}/gpu-preflight/gpu_check.json" \
    --stage-started-at "${sg_start_time}" \
    --output "${SG_RUN_DIR}/runtime-preflight.json"

nvidia-smi > "${SG_RUN_DIR}/nvidia-smi-before.txt" 2>&1
sg_gpu_sampling_started_at="$(date -u '+%Y-%m-%dT%H:%M:%S.%6NZ')"
sg_start_gpu_monitor "${sg_gpu_csv}" "${SG_RUN_DIR}/gpu-preflight/gpu_check.json"
if sg_run_logged_with_private_credentials \
    "${sg_log}" \
    "${SG_SCHEDULER_API_KEY_ENV}" \
    "${SG_CLI[@]}" run \
    --config "${sg_config}" \
    --input "${sg_input}" \
    --output "${sg_output}" \
    --runtime-preflight "${SG_RUN_DIR}/runtime-preflight.json"
then
    sg_command_rc=0
else
    sg_command_rc=$?
fi
sg_gpu_sampling_completed_at="$(date -u '+%Y-%m-%dT%H:%M:%S.%6NZ')"
sg_run_rc="${sg_command_rc}"
sg_stop_gpu_monitor
nvidia-smi > "${SG_RUN_DIR}/nvidia-smi-after.txt" 2>&1 || true
if ! sg_run_logged "${sg_log}" "${SG_CLI[@]}" upstream verify --lock "${sg_lock}"; then
    sg_die "post-run upstream verification failed; inspect ${sg_log}"
fi
if ! sg_run_logged \
    "${sg_log}" \
    "${SG_CLI[@]}" upstream verify \
    --lock "${sg_dependency_lock}" \
    --mapping dependencies
then
    sg_die "post-run dependency verification failed; inspect ${sg_log}"
fi
sg_verify_materialized_weights \
    "${sg_log}" \
    "${SG_RUN_DIR}/materialization-verification-after.json"
sg_require_clean_project

sg_end_epoch="$(date '+%s')"
sg_end_time="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
sg_duration_seconds="$((sg_end_epoch - sg_start_epoch))"
sg_output_present=0
if [[ -f "${sg_output}" && -s "${sg_output}" ]]; then
    sg_output_present=1
elif [[ -d "${sg_output}" ]] && find "${sg_output}" -type f -print -quit | grep -q .; then
    sg_output_present=1
fi

if [[ "${sg_command_rc}" -eq 0 && "${sg_output_present}" -eq 1 ]]; then
    sg_evidence_python="${SG_REPO_ROOT}/.venv/bin/python"
    [[ -x "${sg_evidence_python}" ]] \
        || sg_die "core evidence Python is missing: ${sg_evidence_python}"
    if sg_run_logged \
        "${sg_log}" \
        "${sg_evidence_python}" -I "${sg_here}/_extract_run_evidence.py" \
        --stage "${sg_stage}" \
        --runtime-preflight "${SG_RUN_DIR}/runtime-preflight.json" \
        "${sg_log}" \
        "${sg_output}" \
        "${sg_input}" \
        "${sg_config}" \
        "${SG_REPO_ROOT}" \
        "${sg_start_time}" \
        "${SG_RUN_DIR}/cli-result.json" \
        "${SG_RUN_DIR}/scaleguard-run-manifest.json" \
        "${SG_RUN_DIR}/output-evidence.png" \
        "${SG_RUN_DIR}/model-evidence.json"
    then
        if sg_run_logged \
            "${sg_log}" \
            "${SG_CLI[@]}" manifest validate \
            "${SG_RUN_DIR}/scaleguard-run-manifest.json"
        then
            sg_model_evidence_complete=1
        else
            sg_run_rc=65
        fi
    else
        sg_run_rc=65
    fi
fi

python3 -I - \
    "${SG_RUN_DIR}/execution.json" \
    "${sg_stage_label}" \
    "${sg_run_rc}" \
    "${sg_command_rc}" \
    "${sg_output_present}" \
    "${sg_model_evidence_complete}" \
    "${sg_start_time}" \
    "${sg_end_time}" \
    "${sg_duration_seconds}" \
    "${sg_gpu_csv}" \
    "${sg_config}" \
    "${sg_input}" \
    "${sg_output}" \
    "${sg_lock}" \
    "${sg_dependency_lock}" \
    "$(git rev-parse HEAD)" \
    "${SCALEGUARD_MIN_GPUS:-2}" \
    "${SG_RUN_DIR}/gpu-preflight/gpu_check.json" \
    "${SG_RUN_DIR}/runtime-preflight.json" \
    "${sg_deadline_seconds}" \
    "${sg_gpu_sampling_started_at}" \
    "${sg_gpu_sampling_completed_at}" \
    "${SCALEGUARD_GPU_SAMPLE_INTERVAL_SECONDS:-1}" \
    "${SG_REPO_ROOT}/src" <<'PY'
import csv
import hashlib
import json
import pathlib
import sys
from datetime import datetime, timedelta, timezone
from itertools import pairwise

sys.path.insert(0, str(pathlib.Path(sys.argv[24]).resolve()))

from scaleguard.strict_json import loads_object

(
    manifest_path,
    stage,
    return_code_text,
    command_return_code_text,
    output_present_text,
    model_evidence_complete_text,
    started_at,
    completed_at,
    duration_text,
    gpu_csv_path,
    config_path,
    input_path,
    output_path,
    lock_path,
    dependency_lock_path,
    git_commit,
    minimum_gpu_count_text,
    gpu_check_path,
    runtime_preflight_path,
    deadline_seconds_text,
    gpu_sampling_started_at,
    gpu_sampling_completed_at,
    gpu_sample_interval_text,
) = sys.argv[1:24]


def digest(path: pathlib.Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def file_entry(path_text: str) -> dict[str, object]:
    path = pathlib.Path(path_text)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": digest(path),
    }


def utc_timestamp(value: str, label: str) -> datetime:
    if not value.endswith("Z"):
        raise SystemExit(f"{label} must be a canonical UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        raise SystemExit(f"{label} must be an ISO-8601 timestamp") from None
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise SystemExit(f"{label} must be UTC")
    return parsed.astimezone(timezone.utc)


gpu_check_file = pathlib.Path(gpu_check_path).resolve()
runtime_preflight_file = pathlib.Path(runtime_preflight_path).resolve()
gpu_check = loads_object(gpu_check_file.read_text(encoding="utf-8"))
runtime_preflight = loads_object(runtime_preflight_file.read_text(encoding="utf-8"))
selected_gpus = gpu_check.get("selected_gpus")
gpu_binding = runtime_preflight.get("gpu_preflight")
if not isinstance(selected_gpus, list) or not isinstance(gpu_binding, dict):
    raise SystemExit("GPU preflight binding is missing from runtime evidence")


def normalized_gpu(item: object) -> dict[str, object]:
    if not isinstance(item, dict):
        raise SystemExit("GPU preflight selected_gpus is malformed")
    return {
        "logical_index": item.get("logical_index"),
        "physical_index": item.get("physical_index"),
        "uuid": item.get("uuid"),
        "name": item.get("name"),
        "memory_total_mib": item.get("memory_total_mib"),
        "driver_version": item.get("driver_version"),
    }


normalized_selected = [normalized_gpu(item) for item in selected_gpus]
bound_selected = gpu_binding.get("selected_gpus")
gpu_receipt_binding_complete = (
    gpu_check.get("status") == "passed"
    and gpu_check.get("git_commit") == git_commit
    and pathlib.Path(str(gpu_binding.get("path", ""))).resolve() == gpu_check_file
    and gpu_binding.get("sha256") == digest(gpu_check_file)
    and isinstance(bound_selected, list)
    and [normalized_gpu(item) for item in bound_selected] == normalized_selected
)
expected_by_uuid = {
    str(item["uuid"]): item
    for item in normalized_selected
    if isinstance(item.get("uuid"), str) and item.get("uuid")
}

peak_by_gpu: dict[str, dict[str, object]] = {}
baseline_by_uuid: dict[str, int] = {}
inventory_samples_by_uuid = {uuid: 0 for uuid in expected_by_uuid}
workload_samples_by_uuid = {uuid: 0 for uuid in expected_by_uuid}
workload_observed_by_uuid = {uuid: False for uuid in expected_by_uuid}
sample_times_by_uuid: dict[str, list[datetime]] = {
    uuid: [] for uuid in expected_by_uuid
}
sample_count = 0
sample_identity_valid = True
sample_temporal_valid = True
gpu_path = pathlib.Path(gpu_csv_path)
if gpu_path.is_file():
    with gpu_path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        expected_fields = [
            "timestamp_utc",
            "sample_kind",
            "index",
            "uuid",
            "name",
            "memory_used_mib",
            "memory_total_mib",
            "utilization_gpu_percent",
        ]
        if reader.fieldnames != expected_fields:
            sample_identity_valid = False
        for row in reader:
            sample_count += 1
            index = (row.get("index") or "").strip()
            uuid = (row.get("uuid") or "").strip()
            sample_kind = (row.get("sample_kind") or "").strip()
            expected = expected_by_uuid.get(uuid)
            try:
                sample_time = utc_timestamp(
                    (row.get("timestamp_utc") or "").strip(),
                    f"GPU sample row {sample_count}",
                )
                memory_used = int(float((row.get("memory_used_mib") or "0").strip()))
                memory_total = int(float((row.get("memory_total_mib") or "0").strip()))
                utilization = int(float((row.get("utilization_gpu_percent") or "0").strip()))
            except (ValueError, SystemExit):
                sample_identity_valid = False
                sample_temporal_valid = False
                continue
            if (
                expected is None
                or expected.get("physical_index") != index
                or expected.get("name") != (row.get("name") or "").strip()
                or expected.get("memory_total_mib") != memory_total
                or sample_kind not in {"inventory", "workload"}
            ):
                sample_identity_valid = False
                continue
            timestamps = sample_times_by_uuid[uuid]
            if timestamps and sample_time < timestamps[-1]:
                sample_temporal_valid = False
            timestamps.append(sample_time)
            current = peak_by_gpu.setdefault(
                index,
                {
                    "uuid": uuid,
                    "name": expected["name"],
                    "memory_total_mib": memory_total,
                    "peak_memory_used_mib": 0,
                    "peak_utilization_percent": 0,
                },
            )
            current["peak_memory_used_mib"] = max(
                int(current["peak_memory_used_mib"]), memory_used
            )
            current["peak_utilization_percent"] = max(
                int(current["peak_utilization_percent"]), utilization
            )
            if sample_kind == "inventory":
                inventory_samples_by_uuid[uuid] += 1
                baseline_by_uuid.setdefault(uuid, memory_used)
            else:
                workload_samples_by_uuid[uuid] += 1
                baseline = baseline_by_uuid.get(uuid)
                if baseline is not None and (utilization > 0 or memory_used > baseline + 16):
                    workload_observed_by_uuid[uuid] = True

output = pathlib.Path(output_path)
output_files = []
if output.is_file() and output.stat().st_size:
    output_files.append(file_entry(str(output)))
elif output.is_dir():
    output_files.extend(
        file_entry(str(path))
        for path in sorted(output.rglob("*"))
        if path.is_file()
    )

return_code = int(return_code_text)
command_return_code = int(command_return_code_text)
output_present = output_present_text == "1"
model_evidence_complete = model_evidence_complete_text == "1"
input_entries = {
    "runtime_config": file_entry(config_path),
    "input_image": file_entry(input_path),
    "upstream_lock": file_entry(lock_path),
    "runtime_dependencies_lock": file_entry(dependency_lock_path),
    "runtime_preflight": file_entry(
        str(pathlib.Path(manifest_path).with_name("runtime-preflight.json"))
    ),
}
model_evidence_summary = None
output_evidence_entry = None
run_manifest_entry = None
evidence_hashes_consistent = False
if model_evidence_complete:
    model_evidence_path = pathlib.Path(manifest_path).with_name("model-evidence.json")
    model_evidence_summary = loads_object(model_evidence_path.read_text(encoding="utf-8"))
    run_manifest_path = pathlib.Path(manifest_path).with_name("scaleguard-run-manifest.json")
    if run_manifest_path.is_file():
        run_manifest_entry = file_entry(str(run_manifest_path))
    output_evidence_path = pathlib.Path(
        str(model_evidence_summary.get("output_evidence_path", ""))
    )
    if output_evidence_path.is_file():
        output_evidence_entry = file_entry(str(output_evidence_path))
    evidence_hashes_consistent = (
        len(output_files) == 1
        and output_evidence_entry is not None
        and model_evidence_summary.get("final_output_sha256")
        == output_files[0]["sha256"]
        == output_evidence_entry["sha256"]
        == model_evidence_summary.get("output_evidence_sha256")
        and model_evidence_summary.get("invoked_input_sha256")
        == input_entries["input_image"]["sha256"]
        and model_evidence_summary.get("invoked_config_sha256")
        == input_entries["runtime_config"]["sha256"]
        and model_evidence_summary.get("runtime_preflight_sha256")
        == input_entries["runtime_preflight"]["sha256"]
        and pathlib.Path(
            str(model_evidence_summary.get("runtime_preflight_path", ""))
        ).resolve()
        == pathlib.Path(input_entries["runtime_preflight"]["path"]).resolve()
        and run_manifest_entry is not None
        and model_evidence_summary.get("manifest_sha256")
        == run_manifest_entry["sha256"]
    )
minimum_gpu_count = int(minimum_gpu_count_text)
gpu_sample_interval_seconds = float(gpu_sample_interval_text)
attempt_started = utc_timestamp(started_at, "execution started_at_utc")
attempt_completed = utc_timestamp(completed_at, "execution completed_at_utc")
sampling_started = utc_timestamp(
    gpu_sampling_started_at,
    "GPU sampling window_started_at_utc",
)
sampling_completed = utc_timestamp(
    gpu_sampling_completed_at,
    "GPU sampling window_completed_at_utc",
)
sampling_duration_seconds = (sampling_completed - sampling_started).total_seconds()
boundary_tolerance_seconds = max(5.0, gpu_sample_interval_seconds * 1.5)
maximum_gap_tolerance_seconds = max(1.0, gpu_sample_interval_seconds * 2.0)
maximum_observed_gap_seconds = 0.0
boundary_coverage_complete = True
for uuid in expected_by_uuid:
    timestamps = sample_times_by_uuid[uuid]
    if not timestamps:
        boundary_coverage_complete = False
        continue
    if (
        abs((timestamps[0] - sampling_started).total_seconds())
        > boundary_tolerance_seconds
        or abs((timestamps[-1] - sampling_completed).total_seconds())
        > boundary_tolerance_seconds
    ):
        boundary_coverage_complete = False
    for before, after in pairwise(timestamps):
        gap = (after - before).total_seconds()
        if gap < 0:
            sample_temporal_valid = False
            continue
        maximum_observed_gap_seconds = max(maximum_observed_gap_seconds, gap)

attempt_duration_seconds = int(duration_text)
timestamp_slack = timedelta(seconds=1)
window_binding_complete = (
    attempt_completed >= attempt_started
    and sampling_duration_seconds >= 0.0
    and sampling_started >= attempt_started - timestamp_slack
    and sampling_completed <= attempt_completed + timestamp_slack
    and sampling_duration_seconds <= attempt_duration_seconds + 1.0
)
temporal_coverage_complete = (
    sample_temporal_valid
    and window_binding_complete
    and boundary_coverage_complete
    and maximum_observed_gap_seconds <= maximum_gap_tolerance_seconds
)
expected_uuids = set(expected_by_uuid)
inventory_binding_complete = (
    gpu_receipt_binding_complete
    and sample_identity_valid
    and len(expected_uuids) == minimum_gpu_count == 2
    and set(baseline_by_uuid) == expected_uuids
    and all(inventory_samples_by_uuid.get(uuid, 0) == 1 for uuid in expected_uuids)
    and {str(item.get("uuid")) for item in peak_by_gpu.values()} == expected_uuids
)
workload_sampling_complete = (
    inventory_binding_complete
    and temporal_coverage_complete
    and all(workload_samples_by_uuid.get(uuid, 0) > 0 for uuid in expected_uuids)
    and all(workload_observed_by_uuid.get(uuid, False) for uuid in expected_uuids)
)
gpu_sampling_complete = inventory_binding_complete and workload_sampling_complete
status = (
    "passed"
    if (
        return_code == 0
        and command_return_code == 0
        and output_present
        and model_evidence_complete
        and evidence_hashes_consistent
        and gpu_sampling_complete
    )
    else "failed"
)
document = {
    "schema_version": 1,
    "stage": stage,
    "status": status,
    "status_scope": (
        (
            "The CLI command, fresh output, GPU inventory binding, and host-level "
            "workload-observation checks passed. "
            "Model reproduction level must be established from the ScaleGuard "
            "run manifest and reviewed raw logs."
        )
        if status == "passed"
        else (
            "At least one wrapper check failed. This attempt supplies failure "
            "evidence and supports no model reproduction claim."
        )
    ),
    "started_at_utc": started_at,
    "completed_at_utc": completed_at,
    "duration_seconds": attempt_duration_seconds,
    "run_deadline_seconds": int(deadline_seconds_text),
    "return_code": return_code,
    "scaleguard_command_return_code": command_return_code,
    "git_commit": git_commit,
    "inputs": input_entries,
    "outputs": output_files,
    "model_evidence": {
        "complete": model_evidence_complete and evidence_hashes_consistent,
        "helper_completed": model_evidence_complete,
        "hashes_consistent": evidence_hashes_consistent,
        "output_snapshot": output_evidence_entry,
        "run_manifest_snapshot": run_manifest_entry,
        "summary": model_evidence_summary,
    },
    "gpu_sampling": {
        "sample_count": sample_count,
        "sample_interval_seconds": gpu_sample_interval_seconds,
        "window_started_at_utc": gpu_sampling_started_at,
        "window_completed_at_utc": gpu_sampling_completed_at,
        "window_duration_seconds": sampling_duration_seconds,
        "boundary_tolerance_seconds": boundary_tolerance_seconds,
        "maximum_gap_tolerance_seconds": maximum_gap_tolerance_seconds,
        "maximum_observed_gap_seconds": maximum_observed_gap_seconds,
        "temporal_coverage_complete": temporal_coverage_complete,
        "minimum_gpu_count": minimum_gpu_count,
        "preflight_receipt_bound": gpu_receipt_binding_complete,
        "inventory_binding_complete": inventory_binding_complete,
        "workload_sampling_complete": workload_sampling_complete,
        "workload_observed_by_uuid": workload_observed_by_uuid,
        "workload_samples_by_uuid": workload_samples_by_uuid,
        "attribution_scope": "physical_gpu_host_level_not_process_attributed",
        "evidence_complete": gpu_sampling_complete,
        "peak_by_physical_index": peak_by_gpu,
        "raw_csv": pathlib.Path(gpu_csv_path).name,
    },
}
pathlib.Path(manifest_path).write_text(
    json.dumps(document, indent=2) + "\n",
    encoding="utf-8",
)
PY

if [[ "${sg_command_rc}" -eq 0 ]]; then
    sg_require_clean_project
fi
sg_write_file_inventory "${SG_RUN_DIR}" "${SG_RUN_DIR}/files.json"

if [[ "${sg_run_rc}" -ne 0 ]]; then
    sg_die "${sg_stage_label} command failed with exit code ${sg_run_rc}; evidence: ${SG_RUN_DIR}"
fi
if [[ "${sg_output_present}" -ne 1 ]]; then
    sg_die "${sg_stage_label} command returned success but produced no non-empty output: ${sg_output}"
fi
sg_execution_status="$(sg_json_get "${SG_RUN_DIR}/execution.json" status)"
if [[ "${sg_execution_status}" != "passed" ]]; then
    sg_die "${sg_stage_label} output exists, but wrapper evidence checks are incomplete: ${SG_RUN_DIR}"
fi

sg_note "${sg_stage_label} command passed. Reviewable evidence: ${SG_RUN_DIR}"
sg_write_attempt_pointer succeeded "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
sg_release_gpu_lease
trap - EXIT INT TERM
