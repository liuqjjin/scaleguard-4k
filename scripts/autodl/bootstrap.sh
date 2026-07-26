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

sg_resolve_autodl_scheduler_envs
sg_config="${SG_AUTODL_RUNTIME_CONFIG}"
if [[ -n "${SCALEGUARD_BOOTSTRAP_COMMAND:-}" ]]; then
    sg_assert_no_secret_args bash -lc "${SCALEGUARD_BOOTSTRAP_COMMAND}"
fi
# Bootstrap, installation hooks, and source verification never need account
# credentials. Strip them in this process so every descendant inherits the
# least-privilege environment even when this entry point is invoked directly.
# shellcheck disable=SC2119
sg_unset_sensitive_environment
sg_init_paths
sg_new_run_dir bootstrap
sg_log="${SG_RUN_DIR}/bootstrap.log"
sg_skip_gpu_check=0
sg_installation_scope="not-started"
sg_doctor_status="not-run"
sg_bootstrap_started_at="$(sg_timestamp)"
sg_failure_phase="initialization"
sg_child_return_code=""
sg_runtime_receipt_validation=""
sg_runtime_receipt_validation_sha256=""
sg_runtime_aggregate_receipt_sha256=""

sg_finalize_failed_bootstrap() {
    local sg_original_rc=$?
    set +e
    if [[ ! -f "${SG_RUN_DIR}/bootstrap.json" ]] && command -v python3 >/dev/null 2>&1; then
        python3 - \
            "${SG_RUN_DIR}/bootstrap.json" \
            "${sg_bootstrap_started_at}" \
            "$(sg_timestamp)" \
            "${sg_original_rc}" \
            "${sg_installation_scope}" \
            "${sg_failure_phase}" \
            "${sg_child_return_code}" <<'PY'
import json
import pathlib
import sys

(
    output,
    started_at,
    completed_at,
    return_code,
    installation_scope,
    failure_phase,
    child_return_code,
) = sys.argv[1:]
pathlib.Path(output).write_text(
    json.dumps(
        {
            "schema_version": 1,
            "started_at_utc": started_at,
            "completed_at_utc": completed_at,
            "status": "failed",
            "return_code": int(return_code),
            "failure_phase": failure_phase,
            "child_return_code": (
                int(child_return_code) if child_return_code else None
            ),
            "installation_scope": installation_scope,
            "upstream_runtime_installation_claimed": False,
            "claim": "This attempt supplies failure evidence and supports no environment validation claim.",
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
trap sg_finalize_failed_bootstrap EXIT

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-gpu-check)
            sg_skip_gpu_check=1
            shift
            ;;
        -h|--help)
            cat <<'EOF'
Usage: scripts/autodl/bootstrap.sh [--skip-gpu-check]

Creates cache directories, verifies the two-GPU host, installs the locked
project environment, verifies upstream checkouts, and records an environment
receipt. Use --skip-gpu-check only for preparing a non-GPU host.
EOF
            trap - EXIT
            exit 0
            ;;
        *)
            sg_die "unknown argument: $1"
            ;;
    esac
done

cd "${SG_REPO_ROOT}"
sg_write_cache_env "${SG_RUN_DIR}/cache.env"
sg_installation_scope="core-only"

for sg_command in git python3; do
    command -v "${sg_command}" >/dev/null 2>&1 \
        || sg_die "required command is missing: ${sg_command}"
done
sg_require_clean_project

sg_weight_cache="${SG_CACHE_ROOT}/weights"
sg_weight_link="${SG_REPO_ROOT}/weights"
mkdir -p "${sg_weight_cache}"
if [[ -L "${sg_weight_link}" ]]; then
    sg_actual_weight_target="$(
        python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' \
            "${sg_weight_link}"
    )"
    sg_expected_weight_target="$(
        python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' \
            "${sg_weight_cache}"
    )"
    [[ "${sg_actual_weight_target}" == "${sg_expected_weight_target}" ]] \
        || sg_die \
            "weights symlink points to ${sg_actual_weight_target}; expected ${sg_expected_weight_target}"
elif [[ -e "${sg_weight_link}" ]]; then
    sg_die \
        "refusing to replace the real path ${sg_weight_link}; move it only after auditing its contents"
else
    ln -s "${sg_weight_cache}" "${sg_weight_link}"
fi

if [[ "${sg_skip_gpu_check}" -eq 0 ]]; then
    "${sg_here}/check_gpu.sh" --output "${SG_RUN_DIR}/gpu-preflight"
fi

if [[ -n "${SCALEGUARD_BOOTSTRAP_COMMAND:-}" ]]; then
    sg_note "Using the explicit SCALEGUARD_BOOTSTRAP_COMMAND override."
    sg_failure_phase="custom-bootstrap-command"
    if sg_run_logged "${sg_log}" bash -lc "${SCALEGUARD_BOOTSTRAP_COMMAND}"; then
        :
    else
        sg_child_return_code=$?
        sg_die "custom bootstrap command failed; inspect ${sg_log}"
    fi
elif [[ -f "${SG_REPO_ROOT}/scripts/bootstrap/autodl.sh" ]]; then
    sg_failure_phase="project-autodl-hook"
    if sg_run_logged \
        "${sg_log}" \
        bash "${SG_REPO_ROOT}/scripts/bootstrap/autodl.sh"
    then
        :
    else
        sg_child_return_code=$?
        sg_die "project AutoDL bootstrap failed; inspect ${sg_log}"
    fi
    sg_installation_scope="project-autodl-hook"
elif [[ -f "${SG_REPO_ROOT}/uv.lock" ]]; then
    sg_note "No project AutoDL hook found; installing the ScaleGuard core environment only."
    command -v uv >/dev/null 2>&1 \
        || sg_die "uv.lock exists but uv is unavailable; install uv or set SCALEGUARD_BOOTSTRAP_COMMAND"
    if ! sg_run_logged "${sg_log}" uv sync --locked; then
        sg_die "uv sync failed; inspect ${sg_log}"
    fi
elif [[ -f "${SG_REPO_ROOT}/pyproject.toml" ]]; then
    sg_note "No project AutoDL hook found; installing the ScaleGuard core environment only."
    sg_python="${SCALEGUARD_BOOTSTRAP_PYTHON:-python3}"
    if [[ ! -x "${SG_REPO_ROOT}/.venv/bin/python" ]]; then
        if ! sg_run_logged "${sg_log}" "${sg_python}" -m venv "${SG_REPO_ROOT}/.venv"; then
            sg_die "virtual environment creation failed; inspect ${sg_log}"
        fi
    fi
    if [[ -f "${SG_REPO_ROOT}/requirements.lock" ]]; then
        if ! sg_run_logged \
            "${sg_log}" \
            "${SG_REPO_ROOT}/.venv/bin/python" -m pip install \
            --require-hashes -r "${SG_REPO_ROOT}/requirements.lock"
        then
            sg_die "locked dependency installation failed; inspect ${sg_log}"
        fi
        sg_install_args=(-e . --no-deps)
    else
        sg_note "No uv.lock or requirements.lock found; pyproject dependencies will be resolved now."
        sg_install_args=(-e .)
    fi
    if ! sg_run_logged \
        "${sg_log}" \
        "${SG_REPO_ROOT}/.venv/bin/python" -m pip install "${sg_install_args[@]}"
    then
        sg_die "ScaleGuard installation failed; inspect ${sg_log}"
    fi
else
    sg_die "no project installer found; expected scripts/bootstrap/autodl.sh, uv.lock, or pyproject.toml"
fi

if [[ "${sg_installation_scope}" == "project-autodl-hook" ]]; then
    sg_failure_phase="runtime-receipt-validation"
    sg_runtime_receipt_source="${SG_REPO_ROOT}/.runtime/receipts/bootstrap.json"
    sg_runtime_receipt_validation="${SG_RUN_DIR}/runtime-receipts/validation.json"
    sg_require_file "${sg_runtime_receipt_source}" "project AutoDL aggregate receipt"
    if sg_run_logged \
        "${sg_log}" \
        "${SG_REPO_ROOT}/.venv/bin/python" "${sg_here}/_validate_bootstrap_receipt.py" \
        --source "${sg_runtime_receipt_source}" \
        --project-root "${SG_REPO_ROOT}" \
        --git-commit "$(git -C "${SG_REPO_ROOT}" rev-parse HEAD)" \
        --not-before "${sg_bootstrap_started_at}" \
        --destination "${SG_RUN_DIR}/runtime-receipts"
    then
        :
    else
        sg_child_return_code=$?
        sg_die "project environment receipt validation failed; inspect ${sg_log}"
    fi
    sg_runtime_receipt_validation_sha256="$(
        sg_sha256 "${sg_runtime_receipt_validation}"
    )"
    sg_runtime_aggregate_receipt_sha256="$(
        sg_json_get \
            "${sg_runtime_receipt_validation}" \
            aggregate_receipt \
            sha256
    )"
fi

sg_failure_phase="production-preflight"
sg_resolve_cli
sg_lock="${SCALEGUARD_UPSTREAM_LOCK:-${SG_REPO_ROOT}/upstream-lock.yaml}"
sg_dependency_lock="${SCALEGUARD_RUNTIME_DEPENDENCIES_LOCK:-${SG_REPO_ROOT}/runtime-dependencies.yaml}"
sg_require_file "${sg_config}" "AutoDL runtime config"
sg_require_file "${sg_lock}" "upstream lock"
sg_require_file "${sg_dependency_lock}" "runtime dependency lock"

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
sg_doctor_rc=0
sg_doctor_json="${SG_RUN_DIR}/doctor.json"
sg_doctor_stderr="${SG_RUN_DIR}/doctor.stderr.txt"
sg_assert_no_secret_args "${SG_CLI[@]}" doctor --config "${sg_config}" --json
{
    printf 'command:'
    printf ' %q' "${SG_CLI[@]}" doctor --config "${sg_config}" --json
    printf '\n'
} | sg_redact_stream | tee -a "${sg_log}"
if "${SG_CLI[@]}" doctor --config "${sg_config}" --json \
    > "${sg_doctor_json}" 2> "${sg_doctor_stderr}"
then
    :
else
    sg_doctor_rc=$?
fi
{
    cat "${sg_doctor_json}"
    cat "${sg_doctor_stderr}"
} | sg_redact_stream | tee -a "${sg_log}"

python3 - \
    "${sg_doctor_json}" \
    "${SG_RUN_DIR}/doctor-summary.json" \
    "${sg_doctor_rc}" \
    "${sg_skip_gpu_check}" \
    "${SG_REPO_ROOT}/src" <<'PY'
import json
import pathlib
import sys

input_path, output_path, return_code_text, skip_gpu_text = sys.argv[1:5]
sys.path.insert(0, str(pathlib.Path(sys.argv[5]).resolve()))

from scaleguard.strict_json import StrictJSONError, loads

try:
    checks = loads(pathlib.Path(input_path).read_text(encoding="utf-8"))
except (OSError, StrictJSONError) as exc:
    raise SystemExit(f"bootstrap doctor emitted invalid JSON: {exc}") from None
if not isinstance(checks, list) or not all(isinstance(item, dict) for item in checks):
    raise SystemExit("bootstrap doctor output must be a JSON list of checks")

failures = [
    str(item.get("name"))
    for item in checks
    if item.get("status") == "fail"
]
deferred_weight_checks = {
    "coz_model",
    "coz_qwen_model",
    "coz_sr_lora",
    "coz_vae",
    "coz_vlm_lora",
    "4kagent_toolbox",
    "4kagent_hps",
    "4kagent_quality_model",
    "4kagent_perception_model",
}
deferred_credential_checks = {"4kagent_api_key"}
allowed_deferred_checks = set(deferred_weight_checks | deferred_credential_checks)
if skip_gpu_text == "1":
    allowed_deferred_checks.add("gpu_inventory")
unexpected = sorted(set(failures) - allowed_deferred_checks)
return_code = int(return_code_text)
if unexpected:
    raise SystemExit(
        "bootstrap doctor has non-deferred failures: " + ", ".join(unexpected)
    )
if return_code == 0 and failures:
    raise SystemExit("bootstrap doctor returned success while reporting failed checks")
if return_code != 0 and not failures:
    raise SystemExit(
        f"bootstrap doctor exited {return_code} without a structured failed check"
    )
if failures and return_code != 1:
    raise SystemExit(
        f"bootstrap doctor exited {return_code}; only the normal failed-check code 1 "
        "can defer missing weights"
    )

deferred_weights = sorted(set(failures) & deferred_weight_checks)
status = "passed"
if failures:
    status = (
        "passed_with_deferred_weights"
        if set(failures) <= deferred_weight_checks
        else "passed_with_deferred_checks"
    )
warnings = [
    {
        "name": str(item.get("name")),
        "detail": str(item.get("detail", "")),
    }
    for item in checks
    if item.get("status") == "warn"
]
document = {
    "schema_version": 1,
    "status": status,
    "doctor_return_code": return_code,
    "deferred_checks": sorted(failures),
    "deferred_weight_checks": deferred_weights,
    "weights_ready": not deferred_weights,
    "warnings": warnings,
    "claim": (
        (
            "No doctor check failed. Any warnings are recorded and remain "
            "non-passing research caveats."
        )
        if not failures
        else (
            "Only explicitly allowlisted preflight checks were deferred. Smoke and "
            "integration wrappers require a full doctor pass."
        )
    ),
}
pathlib.Path(output_path).write_text(
    json.dumps(document, indent=2) + "\n",
    encoding="utf-8",
)
PY
sg_doctor_status="$(sg_json_get "${SG_RUN_DIR}/doctor-summary.json" status)"
sg_weights_ready="$(sg_json_get "${SG_RUN_DIR}/doctor-summary.json" weights_ready)"
sg_require_clean_project

"${SG_CLI[@]}" --version > "${SG_RUN_DIR}/scaleguard-version.txt" 2>&1 || true
git rev-parse HEAD > "${SG_RUN_DIR}/git-commit.txt"
git status --short > "${SG_RUN_DIR}/git-status.txt"
git submodule status --recursive > "${SG_RUN_DIR}/submodules.txt" 2>&1 || true
"${SG_REPO_ROOT}/.venv/bin/python" -m pip freeze \
    > "${SG_RUN_DIR}/pip-freeze.txt" 2>&1 || true
sg_sha256 "${sg_lock}" > "${SG_RUN_DIR}/upstream-lock.sha256"
sg_sha256 "${sg_dependency_lock}" > "${SG_RUN_DIR}/runtime-dependencies.sha256"
sg_sha256 "${sg_config}" > "${SG_RUN_DIR}/runtime-config.sha256"

python3 - \
    "${SG_RUN_DIR}/bootstrap.json" \
    "${sg_skip_gpu_check}" \
    "$(sg_timestamp)" \
    "$(git rev-parse HEAD)" \
    "${sg_config#"${SG_REPO_ROOT}"/}" \
    "${sg_lock#"${SG_REPO_ROOT}"/}" \
    "${sg_dependency_lock#"${SG_REPO_ROOT}"/}" \
    "${sg_installation_scope}" \
    "${sg_doctor_status}" \
    "${sg_weights_ready}" \
    "${sg_weight_link}" \
    "${sg_weight_cache}" \
    "${sg_runtime_receipt_validation}" \
    "${sg_runtime_receipt_validation_sha256}" \
    "${sg_runtime_aggregate_receipt_sha256}" <<'PY'
import json
import pathlib
import sys

(
    output,
    skipped,
    timestamp,
    commit,
    config,
    lock,
    dependency_lock,
    installation_scope,
    doctor_status,
    weights_ready,
    weight_link,
    weight_cache,
    runtime_receipt_validation,
    runtime_receipt_validation_sha256,
    runtime_aggregate_receipt_sha256,
) = sys.argv[1:]
document = {
    "schema_version": 1,
    "completed_at_utc": timestamp,
    "status": "passed",
    "gpu_preflight_skipped": skipped == "1",
    "git_commit": commit,
    "runtime_config": config,
    "upstream_lock": lock,
    "runtime_dependencies_lock": dependency_lock,
    "installation_scope": installation_scope,
    "upstream_runtime_installation_claimed": installation_scope == "project-autodl-hook",
    "doctor_status": doctor_status,
    "weights_ready_at_bootstrap": weights_ready == "true",
    "weights_link": weight_link,
    "weights_cache": weight_cache,
    "runtime_receipt_validation": runtime_receipt_validation or None,
    "runtime_receipt_validation_sha256": runtime_receipt_validation_sha256 or None,
    "runtime_aggregate_receipt_sha256": runtime_aggregate_receipt_sha256 or None,
}
pathlib.Path(output).write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
PY

sg_write_file_inventory "${SG_RUN_DIR}" "${SG_RUN_DIR}/files.json"
trap - EXIT
sg_note "Bootstrap completed. Evidence: ${SG_RUN_DIR}"
