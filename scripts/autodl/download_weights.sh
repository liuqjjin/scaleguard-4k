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
# Keep credential values available to the shell redactor, but remove their
# export attribute. Only the downloader child receives Hugging Face auth.
sg_make_sensitive_environment_private
sg_init_paths
sg_manifest="${SCALEGUARD_WEIGHTS_MANIFEST:-}"
sg_weight_root="${SCALEGUARD_WEIGHTS_ROOT:-${SG_CACHE_ROOT}/weights}"
sg_include_optional="${SCALEGUARD_DOWNLOAD_OPTIONAL_WEIGHTS:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --manifest)
            [[ $# -ge 2 ]] || sg_die "--manifest requires a JSON path"
            sg_manifest="$2"
            shift 2
            ;;
        --weight-root)
            [[ $# -ge 2 ]] || sg_die "--weight-root requires a directory"
            sg_weight_root="$2"
            shift 2
            ;;
        --include-optional)
            sg_include_optional=1
            shift
            ;;
        -h|--help)
            cat <<'EOF'
Usage: scripts/autodl/download_weights.sh [--manifest FILE] [--weight-root DIR] [--include-optional]

Downloads weights from an immutable JSON manifest. Hugging Face entries require
a 40-character revision SHA; public HTTPS entries require a sha256. Credentials
may come from HF_TOKEN/HUGGING_FACE_HUB_TOKEN or a login under HF_HOME; they
must never be supplied as command arguments. Manual entries are never fetched,
and optional entries are skipped unless --include-optional is set.
EOF
            exit 0
            ;;
        *)
            sg_die "unknown argument: $1"
            ;;
    esac
done

if [[ -z "${sg_manifest}" ]]; then
    for sg_candidate in \
        "${SG_REPO_ROOT}/weights-lock.json" \
        "${SG_REPO_ROOT}/configs/weights-lock.json" \
        "${SG_REPO_ROOT}/external_gate/weights_manifest.json"
    do
        if [[ -f "${sg_candidate}" ]]; then
            sg_manifest="${sg_candidate}"
            break
        fi
    done
fi

[[ -n "${sg_manifest}" ]] || sg_die \
    "no weight manifest found; pass --manifest or set SCALEGUARD_WEIGHTS_MANIFEST"
sg_manifest="$(sg_from_repo "${sg_manifest}")"
sg_weight_root="$(sg_from_repo "${sg_weight_root}")"
sg_require_file "${sg_manifest}" "weight manifest"
sg_project_python="${SG_REPO_ROOT}/.venv/bin/python"
[[ -x "${sg_project_python}" ]] \
    || sg_die "project Python is missing; run scripts/autodl/bootstrap.sh first"
sg_require_clean_project
sg_hf_runtime_bin="${SG_REPO_ROOT}/.runtime/envs/4kagent/bin"
if [[ ! -x "${sg_hf_runtime_bin}/hf" ]] \
    && [[ ! -x "${sg_hf_runtime_bin}/huggingface-cli" ]]
then
    sg_die "pinned Hugging Face CLI is missing; run scripts/autodl/bootstrap.sh first"
fi
PATH="${sg_hf_runtime_bin}:${PATH}"
export PATH
[[ "${sg_include_optional}" == "0" || "${sg_include_optional}" == "1" ]] \
    || sg_die "SCALEGUARD_DOWNLOAD_OPTIONAL_WEIGHTS must be 0 or 1"

sg_new_run_dir weight-download
sg_log="${SG_RUN_DIR}/download.log"
cp -- "${sg_manifest}" "${SG_RUN_DIR}/weights-manifest.json"
sg_git_commit="$(git -C "${SG_REPO_ROOT}" rev-parse HEAD 2>/dev/null)" \
    || sg_die "repository has no committed HEAD; weight evidence requires an immutable project commit"

sg_download_rc=0
sg_download_command=(
    "${sg_project_python}" "${sg_here}/_download_weights.py"
    "${SG_RUN_DIR}/weights-manifest.json"
    "${sg_weight_root}"
    "${SG_RUN_DIR}/weights-receipt.json"
    --git-commit "${sg_git_commit}"
)
if [[ "${sg_include_optional}" == "1" ]]; then
    sg_download_command+=(--include-optional)
fi
if sg_run_logged_with_private_credentials \
    "${sg_log}" \
    "HF_TOKEN HF_HUB_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACE_TOKEN" \
    "${sg_download_command[@]}"
then
    :
else
    sg_download_rc=$?
fi

sg_materializer="${SG_REPO_ROOT}/scripts/weights/materialize.py"
sg_materialization_receipt="${SG_RUN_DIR}/materialization-receipt.json"
sg_materialization_marker="${sg_weight_root}/.scaleguard-materialization.json"
if [[ "${sg_download_rc}" -eq 0 ]]; then
    if [[ ! -f "${sg_materializer}" ]]; then
        printf 'error: project weight materializer not found: %s\n' \
            "${sg_materializer}" | tee -a "${sg_log}" >&2
        sg_download_rc=66
    elif sg_run_logged \
        "${sg_log}" \
        "${sg_project_python}" "${sg_materializer}" \
        --weights-root "${sg_weight_root}" \
        --receipt "${SG_RUN_DIR}/weights-receipt.json" \
        --output "${sg_materialization_receipt}"
    then
        :
    else
        sg_download_rc=$?
    fi
    if [[ "${sg_download_rc}" -eq 0 ]] && ! sg_validate_materialization_pair \
        "${sg_materialization_receipt}" \
        "${sg_materialization_marker}" \
        "${SG_RUN_DIR}/weights-receipt.json" \
        "${sg_weight_root}" \
        "${sg_git_commit}"
    then
        printf '%s\n' \
            "error: materialization receipt and fixed marker are inconsistent" \
            | tee -a "${sg_log}" >&2
        sg_download_rc=65
    fi
fi

if [[ "${sg_download_rc}" -eq 0 ]]; then
    sg_sha256 "${sg_materialization_marker}" \
        > "${SG_RUN_DIR}/materialization-marker.sha256"
    sg_require_clean_project
fi

if [[ "${sg_download_rc}" -ne 0 ]]; then
    sg_failure_status="failed"
    if [[ -f "${SG_RUN_DIR}/weights-receipt.json" ]]; then
        sg_receipt_status="$(
            sg_json_get "${SG_RUN_DIR}/weights-receipt.json" status
        )"
        if [[ "${sg_receipt_status}" == "external_gate" ]]; then
            sg_failure_status="external_gate"
        fi
    fi
    "${sg_project_python}" - \
        "${SG_RUN_DIR}/weights-failure.json" \
        "$(sg_timestamp)" \
        "${sg_failure_status}" <<'PY'
import json
import pathlib
import sys

pathlib.Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "schema_version": 1,
            "completed_at_utc": sys.argv[2],
            "status": sys.argv[3],
            "claim": "No weight set is considered verified from this failed attempt.",
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY
    "${sg_project_python}" --version > "${SG_RUN_DIR}/python-version.txt" 2>&1 || true
    sg_write_file_inventory "${SG_RUN_DIR}" "${SG_RUN_DIR}/files.json"
    if [[ "${sg_failure_status}" == "external_gate" ]]; then
        printf 'error: required manual weight access is an external gate; inspect %s\n' \
            "${SG_RUN_DIR}" >&2
        exit 3
    fi
    printf 'error: weight download failed; inspect %s\n' "${SG_RUN_DIR}" >&2
    exit "${sg_download_rc}"
fi

"${sg_project_python}" --version > "${SG_RUN_DIR}/python-version.txt" 2>&1
if command -v hf >/dev/null 2>&1; then
    hf --version > "${SG_RUN_DIR}/huggingface-cli-version.txt" 2>&1 || true
elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli --version > "${SG_RUN_DIR}/huggingface-cli-version.txt" 2>&1 || true
fi
sg_write_file_inventory "${SG_RUN_DIR}" "${SG_RUN_DIR}/files.json"
sg_note "Weight acquisition and materialization completed. Evidence: ${SG_RUN_DIR}"
