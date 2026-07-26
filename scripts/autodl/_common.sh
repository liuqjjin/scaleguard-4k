#!/usr/bin/env bash

# Shared, source-only helpers for the AutoDL entry points.

sg_die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

sg_note() {
    printf '%s\n' "$*"
}

sg_common_source="${BASH_SOURCE[0]}"
sg_script_dir="${sg_common_source%/*}"
if [[ "${sg_script_dir}" == "${sg_common_source}" ]]; then
    sg_script_dir="."
fi
if [[ "${sg_script_dir}" != /* ]]; then
    sg_script_dir="${PWD}/${sg_script_dir}"
fi
sg_common_saved_pwd="${PWD}"
cd -- "${sg_script_dir}"
sg_script_dir="${PWD}"
cd -- "${sg_common_saved_pwd}"
SG_REPO_ROOT="${SCALEGUARD_REPO_ROOT:-${sg_script_dir}/../..}"
if [[ "${SG_REPO_ROOT}" != /* ]]; then
    SG_REPO_ROOT="${sg_common_saved_pwd}/${SG_REPO_ROOT}"
fi
cd -- "${SG_REPO_ROOT}"
SG_REPO_ROOT="${PWD}"
cd -- "${sg_common_saved_pwd}"
unset sg_common_source sg_common_saved_pwd
SG_ARTIFACT_ROOT="${SCALEGUARD_ARTIFACT_ROOT:-${SG_REPO_ROOT}/artifacts/autodl}"
if [[ "${SG_ARTIFACT_ROOT}" != /* ]]; then
    SG_ARTIFACT_ROOT="${SG_REPO_ROOT}/${SG_ARTIFACT_ROOT}"
fi

sg_from_repo() {
    local sg_path="$1"
    if [[ "${sg_path}" == /* ]]; then
        printf '%s\n' "${sg_path}"
    else
        printf '%s\n' "${SG_REPO_ROOT}/${sg_path}"
    fi
}

SG_COMMON_SENSITIVE_ENV_NAMES=(
    HF_TOKEN
    HF_HUB_TOKEN
    HUGGING_FACE_HUB_TOKEN
    HUGGINGFACE_TOKEN
    OPENAI_API_KEY
    AZURE_OPENAI_API_KEY
    ANTHROPIC_API_KEY
    DASHSCOPE_API_KEY
    GOOGLE_API_KEY
    GEMINI_API_KEY
    COHERE_API_KEY
    MISTRAL_API_KEY
    GROQ_API_KEY
    TOGETHER_API_KEY
    NVIDIA_API_KEY
    WANDB_API_KEY
    REPLICATE_API_TOKEN
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    AWS_SESSION_TOKEN
    GITHUB_TOKEN
    GH_TOKEN
    GITLAB_TOKEN
    CI_JOB_TOKEN
)
# Bash 3.2 with `set -u` rejects expansion of a genuinely empty array.
# A reserved, valid sentinel keeps every array expansion portable.
SG_EXTRA_SENSITIVE_ENV_NAMES=(SCALEGUARD_INTERNAL_NO_EXTRA_SENSITIVE_ENV)

sg_register_sensitive_env_name() {
    local sg_name="$1"
    local sg_existing

    [[ "${sg_name}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] \
        || sg_die "invalid sensitive environment variable name: ${sg_name}"
    for sg_existing in \
        "${SG_COMMON_SENSITIVE_ENV_NAMES[@]}" \
        "${SG_EXTRA_SENSITIVE_ENV_NAMES[@]}"
    do
        [[ "${sg_existing}" != "${sg_name}" ]] || return 0
    done
    SG_EXTRA_SENSITIVE_ENV_NAMES+=("${sg_name}")
}

sg_resolve_scheduler_api_key_env() {
    local sg_config_path="$1"
    local sg_in_fourkagent=0
    local sg_fourkagent_sections=0
    local sg_api_key_fields=0
    local sg_line
    local sg_trimmed
    local sg_value="OPENAI_API_KEY"

    if [[ -f "${sg_config_path}" ]]; then
        while IFS= read -r sg_line || [[ -n "${sg_line}" ]]; do
            sg_trimmed="${sg_line#"${sg_line%%[![:space:]]*}"}"
            if [[ -z "${sg_trimmed}" || "${sg_trimmed}" == \#* ]]; then
                continue
            fi
            if [[ "${sg_line}" == "${sg_trimmed}" ]]; then
                if [[ "${sg_trimmed}" == "---" || "${sg_trimmed}" == "..." ]]; then
                    continue
                fi
                if [[ "${sg_line}" =~ ^(runtime|fourkagent|coz|metrics|controller):[[:space:]]*(#.*)?$ ]]
                then
                    if [[ "${BASH_REMATCH[1]}" == "fourkagent" ]]; then
                        sg_fourkagent_sections="$((sg_fourkagent_sections + 1))"
                        [[ "${sg_fourkagent_sections}" -eq 1 ]] \
                            || sg_die \
                                "AutoDL config has duplicate fourkagent mappings: ${sg_config_path}"
                        sg_in_fourkagent=1
                    else
                        sg_in_fourkagent=0
                    fi
                    continue
                fi
                sg_die \
                    "AutoDL config root must use canonical unquoted block mappings: ${sg_config_path}"
            fi
            if [[ "${sg_in_fourkagent}" -eq 0 ]]; then
                continue
            fi
            if [[ "${sg_trimmed}" == \<\<:* \
                    || "${sg_trimmed}" == \"* \
                    || "${sg_trimmed}" == \'* \
                    || "${sg_trimmed}" == \?* \
                    || "${sg_trimmed}" == \!* \
                    || "${sg_trimmed}" == \{* \
                    || "${sg_trimmed}" == \&* \
                    || "${sg_trimmed}" == \** ]]
            then
                sg_die \
                    "AutoDL config must declare api_key_env once with canonical unquoted syntax: ${sg_config_path}"
            fi
            if [[ "${sg_line}" =~ ^[[:space:]]+api_key_env:[[:space:]]*(.*)$ ]]
            then
                sg_api_key_fields="$((sg_api_key_fields + 1))"
                [[ "${sg_api_key_fields}" -eq 1 ]] \
                    || sg_die \
                        "AutoDL config has duplicate fourkagent.api_key_env fields: ${sg_config_path}"
                sg_value="${BASH_REMATCH[1]}"
                sg_value="${sg_value%%#*}"
                sg_value="${sg_value#"${sg_value%%[![:space:]]*}"}"
                sg_value="${sg_value%"${sg_value##*[![:space:]]}"}"
                if [[ "${sg_value}" == \"*\" && "${sg_value}" == *\" ]]; then
                    sg_value="${sg_value:1:${#sg_value}-2}"
                elif [[ "${sg_value}" == \'*\' && "${sg_value}" == *\' ]]; then
                    sg_value="${sg_value:1:${#sg_value}-2}"
                fi
                continue
            fi
            if [[ "${sg_trimmed}" == api_key_env* ]]
            then
                sg_die \
                    "AutoDL config must declare api_key_env once with canonical unquoted syntax: ${sg_config_path}"
            fi
        done < "${sg_config_path}"
    fi
    [[ "${sg_value}" =~ ^[A-Z][A-Z0-9_]*_(API_KEY|TOKEN|CREDENTIAL|SECRET)$ ]] \
        || sg_die \
            "AutoDL config fourkagent.api_key_env must be uppercase and end in _API_KEY, _TOKEN, _CREDENTIAL, or _SECRET"
    # shellcheck disable=SC2034 # consumed by scripts that source this library
    SG_SCHEDULER_API_KEY_ENV="${sg_value}"
}

sg_resolve_autodl_scheduler_envs() {
    SG_AUTODL_RUNTIME_CONFIG="${SCALEGUARD_AUTODL_CONFIG:-${SG_REPO_ROOT}/configs/runtime/autodl-2x4090.yaml}"
    if [[ "${SG_AUTODL_RUNTIME_CONFIG}" != /* ]]; then
        SG_AUTODL_RUNTIME_CONFIG="${SG_REPO_ROOT}/${SG_AUTODL_RUNTIME_CONFIG}"
    fi
    sg_resolve_scheduler_api_key_env "${SG_AUTODL_RUNTIME_CONFIG}"
    SG_AUTODL_RUNTIME_SCHEDULER_ENV="${SG_SCHEDULER_API_KEY_ENV}"
    sg_register_sensitive_env_name "${SG_AUTODL_RUNTIME_SCHEDULER_ENV}"

    SG_AUTODL_SMOKE_CONFIG="${SCALEGUARD_SMOKE_CONFIG:-}"
    if [[ -z "${SG_AUTODL_SMOKE_CONFIG}" ]]; then
        if [[ -f "${SG_REPO_ROOT}/configs/runtime/autodl-smoke.yaml" ]]; then
            SG_AUTODL_SMOKE_CONFIG="${SG_REPO_ROOT}/configs/runtime/autodl-smoke.yaml"
        else
            SG_AUTODL_SMOKE_CONFIG="${SG_AUTODL_RUNTIME_CONFIG}"
        fi
    elif [[ "${SG_AUTODL_SMOKE_CONFIG}" != /* ]]; then
        SG_AUTODL_SMOKE_CONFIG="${SG_REPO_ROOT}/${SG_AUTODL_SMOKE_CONFIG}"
    fi
    sg_resolve_scheduler_api_key_env "${SG_AUTODL_SMOKE_CONFIG}"
    SG_AUTODL_SMOKE_SCHEDULER_ENV="${SG_SCHEDULER_API_KEY_ENV}"
    sg_register_sensitive_env_name "${SG_AUTODL_SMOKE_SCHEDULER_ENV}"

    SG_AUTODL_INTEGRATION_CONFIG="${SCALEGUARD_INTEGRATION_CONFIG:-${SG_AUTODL_RUNTIME_CONFIG}}"
    if [[ "${SG_AUTODL_INTEGRATION_CONFIG}" != /* ]]; then
        SG_AUTODL_INTEGRATION_CONFIG="${SG_REPO_ROOT}/${SG_AUTODL_INTEGRATION_CONFIG}"
    fi
    sg_resolve_scheduler_api_key_env "${SG_AUTODL_INTEGRATION_CONFIG}"
    SG_AUTODL_INTEGRATION_SCHEDULER_ENV="${SG_SCHEDULER_API_KEY_ENV}"
    sg_register_sensitive_env_name "${SG_AUTODL_INTEGRATION_SCHEDULER_ENV}"
}

sg_sensitive_name_is_kept() {
    local sg_name="$1"
    shift
    local sg_keep
    for sg_keep in "$@"; do
        [[ "${sg_name}" != "${sg_keep}" ]] || return 0
    done
    return 1
}

sg_unset_sensitive_environment() {
    local sg_name
    for sg_name in \
        "${SG_COMMON_SENSITIVE_ENV_NAMES[@]}" \
        "${SG_EXTRA_SENSITIVE_ENV_NAMES[@]}"
    do
        if ! sg_sensitive_name_is_kept "${sg_name}" "$@"; then
            unset "${sg_name}"
        fi
    done
}

sg_make_sensitive_environment_private() {
    local sg_name
    local sg_value
    for sg_name in \
        "${SG_COMMON_SENSITIVE_ENV_NAMES[@]}" \
        "${SG_EXTRA_SENSITIVE_ENV_NAMES[@]}"
    do
        sg_value="${!sg_name:-}"
        unset "${sg_name}"
        if [[ -n "${sg_value}" ]]; then
            printf -v "${sg_name}" '%s' "${sg_value}"
        fi
    done
}

sg_write_private_secret_stream() {
    local sg_name
    local sg_value
    for sg_name in \
        "${SG_COMMON_SENSITIVE_ENV_NAMES[@]}" \
        "${SG_EXTRA_SENSITIVE_ENV_NAMES[@]}"
    do
        sg_value="${!sg_name:-}"
        if [[ -n "${sg_value}" ]]; then
            printf '%s\0%s\0' "${sg_name}" "${sg_value}"
        fi
    done
}

sg_export_private_credentials() {
    local sg_name
    for sg_name in "$@"; do
        [[ "${sg_name}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] \
            || sg_die "invalid credential environment variable name: ${sg_name}"
        if [[ -n "${!sg_name:-}" ]]; then
            export "${sg_name?}"
        fi
    done
}

sg_run_sanitized() {
    (
        sg_unset_sensitive_environment
        exec "$@"
    )
}

sg_run_with_download_credentials() {
    (
        sg_unset_sensitive_environment \
            HF_TOKEN \
            HF_HUB_TOKEN \
            HUGGING_FACE_HUB_TOKEN \
            HUGGINGFACE_TOKEN
        exec "$@"
    )
}

sg_run_with_scheduler_credential() {
    local sg_scheduler_env="$1"
    shift
    sg_register_sensitive_env_name "${sg_scheduler_env}"
    (
        sg_unset_sensitive_environment "${sg_scheduler_env}"
        exec "$@"
    )
}

sg_init_paths() {
    local sg_default_cache_root

    if [[ -n "${SCALEGUARD_CACHE_ROOT:-}" ]]; then
        sg_default_cache_root="${SCALEGUARD_CACHE_ROOT}"
    elif [[ -d /root/autodl-tmp && -w /root/autodl-tmp ]]; then
        sg_default_cache_root="/root/autodl-tmp/scaleguard-4k/cache"
    else
        sg_default_cache_root="${SG_REPO_ROOT}/work/autodl/cache"
    fi
    if [[ "${sg_default_cache_root}" != /* ]]; then
        sg_default_cache_root="${SG_REPO_ROOT}/${sg_default_cache_root}"
    fi

    SG_CACHE_ROOT="${sg_default_cache_root}"
    export SCALEGUARD_CACHE_ROOT="${SG_CACHE_ROOT}"
    export HF_HOME="${HF_HOME:-${SG_CACHE_ROOT}/huggingface}"
    export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
    export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${SG_CACHE_ROOT}/transformers}"
    export TORCH_HOME="${TORCH_HOME:-${SG_CACHE_ROOT}/torch}"
    export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SG_CACHE_ROOT}/xdg}"
    export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${SG_CACHE_ROOT}/pip}"
    if [[ -z "${HF_TOKEN:-}" && -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
        HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
    fi

    umask 077
    mkdir -p \
        "${SG_ARTIFACT_ROOT}" \
        "${SG_CACHE_ROOT}" \
        "${HF_HOME}" \
        "${HUGGINGFACE_HUB_CACHE}" \
        "${TRANSFORMERS_CACHE}" \
        "${TORCH_HOME}" \
        "${XDG_CACHE_HOME}" \
        "${PIP_CACHE_DIR}"
}

sg_timestamp() {
    date -u '+%Y%m%dT%H%M%SZ'
}

sg_new_run_dir() {
    local sg_stage="$1"
    local sg_run_id="${SCALEGUARD_RUN_ID:-$(sg_timestamp)-$$}"
    [[ "${sg_stage}" =~ ^[a-z0-9][a-z0-9_-]*$ ]] || sg_die "invalid stage name: ${sg_stage}"
    [[ "${sg_run_id}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
        || sg_die "SCALEGUARD_RUN_ID may contain only letters, numbers, '.', '_' and '-'"

    SG_RUN_DIR="${SG_ARTIFACT_ROOT}/${sg_stage}/${sg_run_id}"
    if [[ -e "${SG_RUN_DIR}" ]]; then
        SG_RUN_DIR="${SG_RUN_DIR}-attempt-$(sg_timestamp)-$$"
    fi
    mkdir -p "${SG_RUN_DIR}"
    export SG_RUN_DIR
}

sg_sha256() {
    local sg_path="$1"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum -- "${sg_path}" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 -- "${sg_path}" | awk '{print $1}'
    else
        python3 - "${sg_path}" <<'PY'
import hashlib
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
digest = hashlib.sha256()
with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
print(digest.hexdigest())
PY
    fi
}

sg_json_get() {
    local sg_document="$1"
    shift
    python3 - "${SG_REPO_ROOT}/src" "${sg_document}" "$@" <<'PY'
import pathlib
import sys

source_root = pathlib.Path(sys.argv[1]).resolve()
document_path = pathlib.Path(sys.argv[2])
sys.path.insert(0, str(source_root))

from scaleguard.strict_json import StrictJSONError, loads_object

try:
    value = loads_object(document_path.read_bytes())
    for key in sys.argv[3:]:
        if not isinstance(value, dict) or key not in value:
            raise KeyError(key)
        value = value[key]
except (OSError, KeyError, StrictJSONError) as error:
    raise SystemExit(f"cannot read strict JSON value from {document_path}: {error}") from None

if isinstance(value, bool):
    print(str(value).lower())
elif isinstance(value, (str, int, float)):
    print(value)
else:
    raise SystemExit(f"strict JSON value at {document_path} is not scalar")
PY
}

sg_redact_stream() {
    local sg_line
    local sg_secret_name
    local sg_secret_value
    while IFS= read -r sg_line || [[ -n "${sg_line}" ]]; do
        for sg_secret_name in \
            "${SG_COMMON_SENSITIVE_ENV_NAMES[@]}" \
            "${SG_EXTRA_SENSITIVE_ENV_NAMES[@]}"
        do
            sg_secret_value="${!sg_secret_name:-}"
            if [[ -n "${sg_secret_value}" ]]; then
                sg_line="${sg_line//"${sg_secret_value}"/"[REDACTED:${sg_secret_name}]"}"
            fi
        done
        printf '%s\n' "${sg_line}" \
            | sed -E \
                -e 's/hf_[A-Za-z0-9]{20,}/[REDACTED:HF_TOKEN]/g' \
                -e 's/sk-[A-Za-z0-9_-]{16,}/[REDACTED:API_KEY]/g' \
                -e 's/AIza[A-Za-z0-9_-]{35}/[REDACTED:GOOGLE_API_KEY]/g' \
                -e 's/gh[pousr]_[A-Za-z0-9]{20,}/[REDACTED:GITHUB_TOKEN]/g' \
                -e 's/github_pat_[A-Za-z0-9_]{20,}/[REDACTED:GITHUB_TOKEN]/g' \
                -e 's/AKIA[0-9A-Z]{16}/[REDACTED:AWS_ACCESS_KEY_ID]/g' \
                -e 's/(Bearer[[:space:]]+)[A-Za-z0-9._~+\/=-]{16,}/\1[REDACTED]/Ig' \
                -e 's#(https?://)[^/@[:space:]]+@#\1[REDACTED_USERINFO]@#Ig' \
                -e "s@(https?://[^?[:space:]\"'<>]+)[?][^[:space:]\"'<>)]+@\\1?[REDACTED_QUERY]@Ig" \
                -e 's/(token|api[_-]?key|password|secret)[=:][^[:space:]]+/\1=[REDACTED]/Ig'
    done
}

sg_assert_no_secret_args() {
    local sg_arg
    local sg_secret_name
    local sg_secret_value
    for sg_arg in "$@"; do
        for sg_secret_name in \
            "${SG_COMMON_SENSITIVE_ENV_NAMES[@]}" \
            "${SG_EXTRA_SENSITIVE_ENV_NAMES[@]}"
        do
            sg_secret_value="${!sg_secret_name:-}"
            if [[ -n "${sg_secret_value}" && "${sg_arg}" == *"${sg_secret_value}"* ]]; then
                sg_die "a secret was placed in a command argument; pass credentials through the environment"
            fi
        done
    done
}

sg_run_logged() {
    local sg_log_path="$1"
    shift
    local sg_rc
    local sg_arg
    local -a sg_pipeline_status

    sg_assert_no_secret_args "$@"
    {
        printf 'command:'
        for sg_arg in "$@"; do
            printf ' %q' "${sg_arg}"
        done
        printf '\n'
    } | sg_redact_stream | tee -a "${sg_log_path}"

    set +e
    "$@" 2>&1 | sg_redact_stream | tee -a "${sg_log_path}"
    sg_pipeline_status=("${PIPESTATUS[@]}")
    set -e
    sg_rc="${sg_pipeline_status[0]}"
    if [[ "${sg_rc}" -eq 0 \
        && ( "${sg_pipeline_status[1]}" -ne 0 || "${sg_pipeline_status[2]}" -ne 0 ) ]]
    then
        sg_rc=74
    fi
    return "${sg_rc}"
}

sg_run_logged_with_private_credentials() {
    local sg_log_path="$1"
    local sg_credential_names="$2"
    shift 2
    local sg_rc
    local sg_arg
    local -a sg_pipeline_status
    local -a sg_credentials=()

    read -r -a sg_credentials <<< "${sg_credential_names}"
    sg_assert_no_secret_args "$@"
    {
        printf 'command:'
        for sg_arg in "$@"; do
            printf ' %q' "${sg_arg}"
        done
        printf '\n'
    } | sg_redact_stream | tee -a "${sg_log_path}"

    set +e
    (
        sg_export_private_credentials "${sg_credentials[@]}"
        exec "$@"
    ) 2>&1 | sg_redact_stream | tee -a "${sg_log_path}"
    sg_pipeline_status=("${PIPESTATUS[@]}")
    set -e
    sg_rc="${sg_pipeline_status[0]}"
    if [[ "${sg_rc}" -eq 0 \
        && ( "${sg_pipeline_status[1]}" -ne 0 || "${sg_pipeline_status[2]}" -ne 0 ) ]]
    then
        sg_rc=74
    fi
    return "${sg_rc}"
}

sg_run_logged_with_presence_marker() {
    local sg_log_path="$1"
    local sg_credential_name="$2"
    shift 2
    local sg_rc
    local sg_arg
    local -a sg_pipeline_status

    sg_register_sensitive_env_name "${sg_credential_name}"
    sg_assert_no_secret_args "$@"
    {
        printf 'command:'
        for sg_arg in "$@"; do
            printf ' %q' "${sg_arg}"
        done
        printf '\n'
    } | sg_redact_stream | tee -a "${sg_log_path}"

    set +e
    (
        sg_unset_sensitive_environment
        # shellcheck disable=SC2163 # dynamic name is validated above
        export "${sg_credential_name}=SCALEGUARD_DOCTOR_CREDENTIAL_PRESENT"
        exec "$@"
    ) 2>&1 | sg_redact_stream | tee -a "${sg_log_path}"
    sg_pipeline_status=("${PIPESTATUS[@]}")
    set -e
    sg_rc="${sg_pipeline_status[0]}"
    if [[ "${sg_rc}" -eq 0 \
        && ( "${sg_pipeline_status[1]}" -ne 0 || "${sg_pipeline_status[2]}" -ne 0 ) ]]
    then
        sg_rc=74
    fi
    return "${sg_rc}"
}

sg_resolve_cli() {
    if [[ -n "${SCALEGUARD_CLI:-}" ]]; then
        [[ "${SCALEGUARD_CLI}" != *[[:space:]]* ]] \
            || sg_die "SCALEGUARD_CLI must be one executable path, not a shell command"
        [[ -x "${SCALEGUARD_CLI}" ]] \
            || command -v "${SCALEGUARD_CLI}" >/dev/null 2>&1 \
            || sg_die "SCALEGUARD_CLI is not executable: ${SCALEGUARD_CLI}"
        SG_CLI=("${SCALEGUARD_CLI}")
    elif [[ -x "${SG_REPO_ROOT}/.venv/bin/scaleguard" ]]; then
        SG_CLI=("${SG_REPO_ROOT}/.venv/bin/scaleguard")
    elif command -v scaleguard >/dev/null 2>&1; then
        SG_CLI=("$(command -v scaleguard)")
    else
        local sg_python="${SCALEGUARD_PYTHON:-python3}"
        command -v "${sg_python}" >/dev/null 2>&1 \
            || sg_die "could not find scaleguard or Python; run scripts/autodl/bootstrap.sh"
        SG_CLI=("${sg_python}" -m scaleguard.cli)
    fi
    : "${SG_CLI[*]}"
}

sg_require_file() {
    local sg_path="$1"
    local sg_label="$2"
    [[ -f "${sg_path}" ]] || sg_die "${sg_label} not found: ${sg_path}"
}

sg_require_clean_project() {
    local sg_status
    git -C "${SG_REPO_ROOT}" rev-parse --verify HEAD >/dev/null 2>&1 \
        || sg_die "repository has no committed HEAD; reproducible evidence requires one"
    sg_status="$(git -C "${SG_REPO_ROOT}" status --porcelain --untracked-files=all)" \
        || sg_die "could not inspect the project working tree"
    if [[ -n "${sg_status}" ]]; then
        printf '%s\n' "error: project working tree is not clean:" >&2
        printf '%s\n' "${sg_status}" | sed 's/^/  /' >&2
        sg_die "commit or intentionally remove project changes before collecting evidence"
    fi
}

sg_write_cache_env() {
    local sg_destination="$1"
    {
        printf '# Generated by ScaleGuard-4K. Contains paths only; never add credentials.\n'
        printf 'export SCALEGUARD_CACHE_ROOT=%q\n' "${SG_CACHE_ROOT}"
        printf 'export HF_HOME=%q\n' "${HF_HOME}"
        printf 'export HUGGINGFACE_HUB_CACHE=%q\n' "${HUGGINGFACE_HUB_CACHE}"
        printf 'export TRANSFORMERS_CACHE=%q\n' "${TRANSFORMERS_CACHE}"
        printf 'export TORCH_HOME=%q\n' "${TORCH_HOME}"
        printf 'export XDG_CACHE_HOME=%q\n' "${XDG_CACHE_HOME}"
        printf 'export PIP_CACHE_DIR=%q\n' "${PIP_CACHE_DIR}"
    } > "${sg_destination}"
    chmod 600 "${sg_destination}"
}

sg_start_gpu_monitor() {
    local sg_destination="$1"
    local sg_interval="${SCALEGUARD_GPU_SAMPLE_INTERVAL_SECONDS:-1}"
    local sg_visible_devices="${CUDA_VISIBLE_DEVICES:-}"
    [[ "${sg_interval}" =~ ^[0-9]+([.][0-9]+)?$ ]] \
        || sg_die "SCALEGUARD_GPU_SAMPLE_INTERVAL_SECONDS must be numeric"
    if [[ -n "${sg_visible_devices}" \
        && ! "${sg_visible_devices}" =~ ^[^,[:space:]]+(,[^,[:space:]]+)*$ ]]
    then
        sg_die \
            "CUDA_VISIBLE_DEVICES must be a comma-separated list without empty or whitespace selectors"
    fi
    command -v nvidia-smi >/dev/null 2>&1 \
        || sg_die "nvidia-smi is required for GPU evidence collection"

    (
        printf 'timestamp_utc,index,uuid,name,memory_used_mib,memory_total_mib,utilization_gpu_percent\n'
        while :; do
            local sg_sample_time
            sg_sample_time="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
            if [[ -n "${sg_visible_devices}" ]]; then
                nvidia-smi \
                    -i "${sg_visible_devices}" \
                    --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu \
                    --format=csv,noheader,nounits \
                    | sed "s/^/${sg_sample_time},/"
            else
                nvidia-smi \
                    --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu \
                    --format=csv,noheader,nounits \
                    | sed "s/^/${sg_sample_time},/"
            fi
            sleep "${sg_interval}"
        done
    ) > "${sg_destination}" 2>/dev/null &
    SG_GPU_MONITOR_PID=$!
    export SG_GPU_MONITOR_PID

    for _ in {1..50}; do
        if [[ -s "${sg_destination}" ]] \
            && [[ "$(wc -l < "${sg_destination}")" -ge 2 ]]
        then
            return 0
        fi
        sleep 0.1
    done
    sg_stop_gpu_monitor
    sg_die "GPU monitor did not produce an nvidia-smi sample: ${sg_destination}"
}

sg_stop_gpu_monitor() {
    if [[ -n "${SG_GPU_MONITOR_PID:-}" ]] && kill -0 "${SG_GPU_MONITOR_PID}" 2>/dev/null; then
        kill "${SG_GPU_MONITOR_PID}" 2>/dev/null || true
        wait "${SG_GPU_MONITOR_PID}" 2>/dev/null || true
    fi
    SG_GPU_MONITOR_PID=""
}

sg_write_file_inventory() {
    local sg_root="$1"
    local sg_destination="$2"
    python3 - "${sg_root}" "${sg_destination}" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
destination = pathlib.Path(sys.argv[2]).resolve()
files = []
for path in sorted(root.rglob("*")):
    if path.is_symlink():
        raise SystemExit(f"evidence directory contains a symbolic link: {path}")
    if not path.is_file() or path.resolve() == destination:
        continue
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    files.append(
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
        }
    )
destination.write_text(
    json.dumps({"schema_version": 1, "root": root.name, "files": files}, indent=2) + "\n",
    encoding="utf-8",
)
PY
}

sg_find_bound_weight_receipt() {
    local sg_marker="$1"
    local sg_override="${SCALEGUARD_WEIGHT_RECEIPT:-}"
    if [[ -n "${sg_override}" ]]; then
        sg_override="$(sg_from_repo "${sg_override}")"
    fi
    python3 - \
        "${sg_marker}" \
        "${SG_ARTIFACT_ROOT}" \
        "${sg_override}" \
        "${SG_REPO_ROOT}/src" <<'PY'
import hashlib
import pathlib
import re
import sys

marker_path, artifact_root_text, override_text = sys.argv[1:4]
sys.path.insert(0, str(pathlib.Path(sys.argv[4]).resolve()))

from scaleguard.strict_json import StrictJSONError, loads_object


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


try:
    marker = loads_object(pathlib.Path(marker_path).read_text(encoding="utf-8"))
except (OSError, StrictJSONError) as exc:
    raise SystemExit(f"invalid materialization marker {marker_path}: {exc}") from None
expected = marker.get("source_weights_receipt_sha256")
if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
    raise SystemExit("materialization marker has no valid source receipt SHA-256")

if override_text:
    candidates = [pathlib.Path(override_text)]
else:
    artifact_root = pathlib.Path(artifact_root_text)
    candidates = sorted(
        artifact_root.glob("weight-download/*/weights-receipt.json"),
        reverse=True,
    )
matches = [
    path.resolve()
    for path in candidates
    if path.is_file() and sha256(path) == expected
]
if not matches:
    location = override_text or str(pathlib.Path(artifact_root_text) / "weight-download")
    raise SystemExit(
        f"no download receipt below {location} matches materialization SHA-256 {expected}"
    )
print(matches[0])
PY
}

sg_validate_materialization_pair() {
    local sg_materialization_receipt="$1"
    local sg_marker="$2"
    local sg_weight_receipt="$3"
    local sg_weight_root="$4"
    local sg_git_commit="$5"
    python3 - \
        "${sg_materialization_receipt}" \
        "${sg_marker}" \
        "${sg_weight_receipt}" \
        "${sg_weight_root}" \
        "${sg_git_commit}" \
        "${SG_REPO_ROOT}/src" <<'PY'
import hashlib
import pathlib
import sys

materialization_path, marker_path, weights_path, root_text, git_commit = sys.argv[1:6]
sys.path.insert(0, str(pathlib.Path(sys.argv[6]).resolve()))

from scaleguard.strict_json import StrictJSONError, loads_object


def load(path_text: str) -> object:
    path = pathlib.Path(path_text)
    try:
        return loads_object(path.read_text(encoding="utf-8"))
    except (OSError, StrictJSONError) as exc:
        raise SystemExit(f"invalid JSON receipt {path}: {exc}") from None


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


for evidence_path, label in (
    (pathlib.Path(materialization_path), "materialization receipt"),
    (pathlib.Path(marker_path), "materialization marker"),
    (pathlib.Path(weights_path), "download receipt"),
):
    if evidence_path.is_symlink():
        raise SystemExit(f"{label} must not be a symbolic link: {evidence_path}")

materialization = load(materialization_path)
marker = load(marker_path)
if not isinstance(materialization, dict) or not isinstance(marker, dict):
    raise SystemExit("materialization receipt and marker must be JSON objects")
if materialization != marker:
    raise SystemExit("attempt materialization receipt differs from the fixed marker")
if materialization.get("schema_version") != 1:
    raise SystemExit("materialization receipt schema_version must be 1")
if materialization.get("status") != "passed":
    raise SystemExit("materialization receipt status is not passed")
if pathlib.Path(str(materialization.get("weights_root", ""))).resolve() != pathlib.Path(
    root_text
).resolve():
    raise SystemExit("materialization receipt is bound to another weight root")
if materialization.get("source_git_commit") != git_commit:
    raise SystemExit("materialization receipt is bound to another project commit")
if materialization.get("checkout_mutations") is not False:
    raise SystemExit("materialization receipt does not prove checkout_mutations=false")
if materialization.get("errors") != []:
    raise SystemExit("materialization receipt contains errors")
layouts = materialization.get("layouts")
if not isinstance(layouts, list) or not layouts:
    raise SystemExit("materialization receipt has no verified layouts")
weights_receipt = pathlib.Path(weights_path)
if (
    materialization.get("source_weights_receipt_sha256")
    != sha256(weights_receipt)
):
    raise SystemExit("materialization receipt is not bound to the supplied download receipt")
weights_document = load(str(weights_receipt))
if not isinstance(weights_document, dict):
    raise SystemExit("download receipt must be a JSON object")
if weights_document.get("status") != "passed":
    raise SystemExit("bound download receipt status is not passed")
if weights_document.get("git_commit") != git_commit:
    raise SystemExit("bound download receipt is tied to another project commit")
if pathlib.Path(str(weights_document.get("weight_root", ""))).resolve() != pathlib.Path(
    root_text
).resolve():
    raise SystemExit("bound download receipt is tied to another weight root")
if weights_document.get("manual_gates") != []:
    raise SystemExit("bound download receipt still contains manual gates")
artifacts = weights_document.get("artifacts")
if not isinstance(artifacts, list) or not artifacts:
    raise SystemExit("bound download receipt has no artifact records")
weight_root = pathlib.Path(root_text).resolve()


def artifact_destination(value: object) -> pathlib.Path:
    if not isinstance(value, str) or not value:
        raise SystemExit("completed artifact has no destination")
    relative = pathlib.PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise SystemExit(f"unsafe artifact destination in receipt: {value!r}")
    unresolved = weight_root / pathlib.Path(*relative.parts)
    current = unresolved
    while current != weight_root:
        if current.is_symlink():
            raise SystemExit(f"artifact destination contains a symlink: {value!r}")
        current = current.parent
    destination = unresolved.resolve()
    if not destination.is_relative_to(weight_root):
        raise SystemExit(f"artifact destination escapes the weight root: {value!r}")
    return destination


def current_inventory(destination: pathlib.Path) -> list[dict[str, object]]:
    if destination.is_file():
        paths = [destination]
        inventory_root = destination.parent
    elif destination.is_dir():
        discovered = [
            path
            for path in sorted(destination.rglob("*"))
            if ".cache" not in path.relative_to(destination).parts
            and ".git" not in path.relative_to(destination).parts
        ]
        symlinks = [path for path in discovered if path.is_symlink()]
        if symlinks:
            raise SystemExit(f"artifact inventory contains a symlink: {symlinks[0]}")
        paths = [path for path in discovered if path.is_file()]
        inventory_root = destination
    else:
        raise SystemExit(f"artifact destination is missing: {destination}")
    return [
        {
            "path": path.relative_to(inventory_root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in paths
    ]


for artifact in artifacts:
    if not isinstance(artifact, dict):
        raise SystemExit("bound download receipt contains a malformed artifact record")
    if artifact.get("required", True) and artifact.get("status") not in {
        "downloaded",
        "recorded_manual",
    }:
        raise SystemExit(
            f"required artifact {artifact.get('id')!r} is not present in the bound receipt"
        )
    if artifact.get("status") in {"downloaded", "recorded_manual"}:
        recorded_files = artifact.get("files")
        if not isinstance(recorded_files, list) or not recorded_files:
            raise SystemExit(
                f"completed artifact {artifact.get('id')!r} has no file inventory"
            )
        destination = artifact_destination(artifact.get("destination"))
        if current_inventory(destination) != recorded_files:
            raise SystemExit(
                f"artifact {artifact.get('id')!r} no longer matches its download receipt"
            )
if not all(isinstance(layout, dict) and layout for layout in layouts):
    raise SystemExit("materialization receipt contains a malformed layout record")
PY
}

sg_verify_materialized_weights() {
    local sg_log_path="$1"
    local sg_verification_receipt="$2"
    local sg_weight_root="${SCALEGUARD_WEIGHTS_ROOT:-${SG_CACHE_ROOT}/weights}"
    local sg_materializer="${SG_REPO_ROOT}/scripts/weights/materialize.py"
    local sg_project_python="${SG_REPO_ROOT}/.venv/bin/python"
    local sg_marker
    local sg_weight_receipt
    local sg_git_commit

    sg_weight_root="$(sg_from_repo "${sg_weight_root}")"
    sg_marker="${sg_weight_root}/.scaleguard-materialization.json"
    sg_require_file "${sg_materializer}" "project weight materializer"
    [[ -x "${sg_project_python}" ]] \
        || sg_die "project Python is missing: ${sg_project_python}"
    sg_require_file "${sg_marker}" "fixed materialization marker"
    [[ ! -L "${sg_marker}" ]] \
        || sg_die "fixed materialization marker must not be a symbolic link: ${sg_marker}"
    sg_weight_receipt="$(sg_find_bound_weight_receipt "${sg_marker}")" \
        || sg_die "could not resolve the download receipt bound to ${sg_marker}"
    sg_git_commit="$(git -C "${SG_REPO_ROOT}" rev-parse HEAD 2>/dev/null)" \
        || sg_die "repository has no committed HEAD; materialization verification requires one"

    if ! sg_run_logged \
        "${sg_log_path}" \
        "${sg_project_python}" "${sg_materializer}" \
        --weights-root "${sg_weight_root}" \
        --receipt "${sg_weight_receipt}" \
        --output "${sg_verification_receipt}" \
        --verify-only
    then
        sg_die "weight materialization verification failed; inspect ${sg_log_path}"
    fi
    if ! sg_validate_materialization_pair \
        "${sg_verification_receipt}" \
        "${sg_marker}" \
        "${sg_weight_receipt}" \
        "${sg_weight_root}" \
        "${sg_git_commit}"
    then
        sg_die "weight materialization receipts are inconsistent; inspect ${sg_log_path}"
    fi
}
