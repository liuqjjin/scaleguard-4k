#!/usr/bin/env bash
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
source "${sg_here}/_common.sh"
sg_here="${sg_script_dir}"

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
    *)
        sg_die "internal runner received invalid stage: ${sg_stage}"
        ;;
esac

sg_show_help=0
sg_output=""
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
        -h|--help)
            sg_show_help=1
            shift
            ;;
        *)
            sg_die "unknown argument: $1"
            ;;
    esac
done

if [[ -z "${sg_config}" ]]; then
    if [[ "${sg_stage}" == "smoke" \
        && -f "${SG_REPO_ROOT}/configs/runtime/autodl-smoke.yaml" ]]
    then
        sg_config="${SG_REPO_ROOT}/configs/runtime/autodl-smoke.yaml"
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
if [[ "${sg_show_help}" -eq 1 ]]; then
    cat <<EOF
Usage: scripts/autodl/run_${sg_stage}.sh --input IMAGE [--config FILE] [--output PATH]

Runs a real ScaleGuard ${sg_stage_label} invocation after GPU, upstream, and
configuration checks. Input may instead be supplied through
SCALEGUARD_$(printf '%s' "${sg_stage}" | tr '[:lower:]' '[:upper:]')_INPUT.
The output path must not already exist, preventing stale artifacts from passing.
EOF
    exit 0
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

sg_finalize_failed_stage() {
    local sg_original_rc=$?
    set +e
    sg_stop_gpu_monitor
    if [[ ! -f "${SG_RUN_DIR}/execution.json" ]]; then
        python3 - \
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
        sg_write_file_inventory "${SG_RUN_DIR}" "${SG_RUN_DIR}/files.json"
    fi
}
trap sg_finalize_failed_stage EXIT
trap 'sg_stop_gpu_monitor; exit 130' INT
trap 'sg_stop_gpu_monitor; exit 143' TERM

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
"${SG_REPO_ROOT}/.venv/bin/python" \
    "${sg_here}/_write_preflight_receipt.py" \
    --config "${sg_config}" \
    --materialization "${SG_RUN_DIR}/materialization-verification.json" \
    --output "${SG_RUN_DIR}/runtime-preflight.json"

nvidia-smi > "${SG_RUN_DIR}/nvidia-smi-before.txt" 2>&1
sg_start_gpu_monitor "${sg_gpu_csv}"
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
        "${sg_evidence_python}" "${sg_here}/_extract_run_evidence.py" \
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

python3 - \
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
    "${SG_REPO_ROOT}/src" <<'PY'
import csv
import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(sys.argv[18]).resolve()))

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
) = sys.argv[1:18]


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


peak_by_gpu: dict[str, dict[str, object]] = {}
sample_count = 0
gpu_path = pathlib.Path(gpu_csv_path)
if gpu_path.is_file():
    with gpu_path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sample_count += 1
            index = (row.get("index") or "").strip()
            try:
                memory_used = int(float((row.get("memory_used_mib") or "0").strip()))
                utilization = int(float((row.get("utilization_gpu_percent") or "0").strip()))
            except ValueError:
                continue
            current = peak_by_gpu.setdefault(
                index,
                {
                    "uuid": (row.get("uuid") or "").strip(),
                    "name": (row.get("name") or "").strip(),
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
}
model_evidence_summary = None
output_evidence_entry = None
evidence_hashes_consistent = False
if model_evidence_complete:
    model_evidence_path = pathlib.Path(manifest_path).with_name("model-evidence.json")
    model_evidence_summary = loads_object(model_evidence_path.read_text(encoding="utf-8"))
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
    )
minimum_gpu_count = int(minimum_gpu_count_text)
gpu_sampling_complete = sample_count > 0 and len(peak_by_gpu) >= minimum_gpu_count
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
            "The CLI command, fresh output, and GPU sampling checks passed. "
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
    "duration_seconds": int(duration_text),
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
        "summary": model_evidence_summary,
    },
    "gpu_sampling": {
        "sample_count": sample_count,
        "minimum_gpu_count": minimum_gpu_count,
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
trap - EXIT INT TERM

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
