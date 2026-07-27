#!/bin/bash -p
# shellcheck shell=bash
if [[ $- != *p* ]]; then
    printf '%s\n' "error: invoke this gate directly; an explicit Bash must use -p" >&2
    exit 2
fi
while IFS= read -r sg_imported_function; do
    builtin unset -f -- "${sg_imported_function}"
done < <(builtin compgen -A function)
builtin unset sg_imported_function
builtin set +x +v
set -Eeuo pipefail

sg_gate_source="${BASH_SOURCE[0]}"
sg_gate_dir="${sg_gate_source%/*}"
if [[ "${sg_gate_dir}" == "${sg_gate_source}" ]]; then
    sg_gate_dir="."
fi
if [[ "${sg_gate_dir}" != /* ]]; then
    sg_gate_dir="${PWD}/${sg_gate_dir}"
fi
cd -- "${sg_gate_dir}/.."
sg_repo_root="${PWD}"
cd "${sg_repo_root}"
# shellcheck source=scripts/autodl/_common.sh
# shellcheck disable=SC1091
source "${sg_repo_root}/scripts/autodl/_common.sh"

sg_collect_on_failure() {
    local sg_gate_rc=$?
    if [[ "${sg_gate_rc}" -ne 0 ]]; then
        set +e
        scripts/autodl/collect_diagnostics.sh
        set -e
    fi
    trap - EXIT
    exit "${sg_gate_rc}"
}
trap sg_collect_on_failure EXIT

: "${SCALEGUARD_SMOKE_INPUT:?Set SCALEGUARD_SMOKE_INPUT to an authorized image}"
: "${SCALEGUARD_INTEGRATION_INPUT:?Set SCALEGUARD_INTEGRATION_INPUT to an authorized image}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export SCALEGUARD_CACHE_ROOT="${SCALEGUARD_CACHE_ROOT:-/root/autodl-tmp/scaleguard-4k/cache}"
export SCALEGUARD_ARTIFACT_ROOT="${SCALEGUARD_ARTIFACT_ROOT:-${sg_repo_root}/artifacts/autodl}"
export SCALEGUARD_WEIGHTS_MANIFEST="${SCALEGUARD_WEIGHTS_MANIFEST:-${sg_repo_root}/weights-lock.json}"
export HF_HOME="${HF_HOME:-${SCALEGUARD_CACHE_ROOT}/huggingface}"
sg_resolve_autodl_scheduler_envs

sg_run_sanitized scripts/autodl/check_gpu.sh
sg_run_sanitized scripts/autodl/bootstrap.sh
[[ -f "${SCALEGUARD_WEIGHTS_MANIFEST}" ]] || {
    printf 'error: weight manifest not found: %s\n' "${SCALEGUARD_WEIGHTS_MANIFEST}" >&2
    exit 1
}
sg_hf_token_path="${HF_TOKEN_PATH:-${HF_HOME}/token}"
if [[ -z "${HF_TOKEN:-}" \
    && -z "${HUGGING_FACE_HUB_TOKEN:-}" \
    && ! -s "${sg_hf_token_path}" ]]
then
    printf '%s\n' \
        "error: export a Hugging Face token or log in under the configured HF_HOME" \
        >&2
    exit 1
fi
sg_run_with_download_credentials \
    scripts/autodl/download_weights.sh --manifest "${SCALEGUARD_WEIGHTS_MANIFEST}"
if [[ -z "${!SG_AUTODL_SMOKE_SCHEDULER_ENV:-}" ]]; then
    printf 'error: enter %s with hidden shell input before model execution\n' \
        "${SG_AUTODL_SMOKE_SCHEDULER_ENV}" >&2
    exit 1
fi
if [[ -z "${!SG_AUTODL_INTEGRATION_SCHEDULER_ENV:-}" ]]; then
    printf 'error: enter %s with hidden shell input before model execution\n' \
        "${SG_AUTODL_INTEGRATION_SCHEDULER_ENV}" >&2
    exit 1
fi
sg_run_with_scheduler_credential \
    "${SG_AUTODL_SMOKE_SCHEDULER_ENV}" \
    scripts/autodl/run_smoke.sh --input "${SCALEGUARD_SMOKE_INPUT}"
sg_run_with_scheduler_credential \
    "${SG_AUTODL_INTEGRATION_SCHEDULER_ENV}" \
    scripts/autodl/run_integration.sh --input "${SCALEGUARD_INTEGRATION_INPUT}"
scripts/autodl/collect_diagnostics.sh

printf '%s\n' "Gate commands completed. Review every manifest and the diagnostics archive;"
printf '%s\n' "do not infer a reproduction level from this wrapper's exit code alone."
trap - EXIT
