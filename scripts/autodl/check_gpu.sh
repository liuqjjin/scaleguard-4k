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
# Hardware inventory has no credential-bearing operation.
# shellcheck disable=SC2119
sg_unset_sensitive_environment
sg_init_paths

sg_output_dir=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output)
            [[ $# -ge 2 ]] || sg_die "--output requires a directory"
            sg_output_dir="$2"
            shift 2
            ;;
        -h|--help)
            cat <<'EOF'
Usage: scripts/autodl/check_gpu.sh [--output DIR]

Checks the selected GPUs, driver/CUDA visibility, and cache-volume free space.
The defaults require two GPUs whose names contain "4090", at least 24000 MiB
each, NVIDIA driver 560.28.03 or newer, and 150 GiB free. Override with
SCALEGUARD_MIN_GPUS, SCALEGUARD_GPU_NAME_PATTERN,
SCALEGUARD_MIN_GPU_MEMORY_MIB, SCALEGUARD_MIN_NVIDIA_DRIVER, and
SCALEGUARD_MIN_DISK_GIB.
EOF
            exit 0
            ;;
        *)
            sg_die "unknown argument: $1"
            ;;
    esac
done

if [[ -n "${sg_output_dir}" ]]; then
    sg_output_dir="$(sg_from_repo "${sg_output_dir}")"
    [[ ! -L "${sg_output_dir}" ]] \
        || sg_die "GPU evidence directory must not be a symlink: ${sg_output_dir}"
    if [[ -d "${sg_output_dir}" ]] \
        && find "${sg_output_dir}" -mindepth 1 -print -quit | grep -q .
    then
        sg_die "GPU evidence directory must be empty: ${sg_output_dir}"
    fi
    [[ ! -e "${sg_output_dir}" || -d "${sg_output_dir}" ]] \
        || sg_die "GPU evidence path is not a directory: ${sg_output_dir}"
    SG_RUN_DIR="${sg_output_dir}"
    mkdir -p "${SG_RUN_DIR}"
else
    sg_new_run_dir gpu-check
fi

sg_errors="${SG_RUN_DIR}/errors.txt"
: > "${sg_errors}"

sg_min_gpus="${SCALEGUARD_MIN_GPUS:-2}"
sg_min_memory_mib="${SCALEGUARD_MIN_GPU_MEMORY_MIB:-24000}"
sg_name_pattern="${SCALEGUARD_GPU_NAME_PATTERN:-4090}"
sg_min_driver="${SCALEGUARD_MIN_NVIDIA_DRIVER:-560.28.03}"
sg_min_disk_gib="${SCALEGUARD_MIN_DISK_GIB:-150}"

[[ "${sg_min_gpus}" =~ ^[1-9][0-9]*$ ]] \
    || sg_die "SCALEGUARD_MIN_GPUS must be a positive integer"
[[ "${sg_min_memory_mib}" =~ ^[1-9][0-9]*$ ]] \
    || sg_die "SCALEGUARD_MIN_GPU_MEMORY_MIB must be a positive integer"
[[ "${sg_min_driver}" =~ ^[0-9]+([.][0-9]+)*$ ]] \
    || sg_die "SCALEGUARD_MIN_NVIDIA_DRIVER must be a dotted numeric version"
[[ "${sg_min_disk_gib}" =~ ^[1-9][0-9]*$ ]] \
    || sg_die "SCALEGUARD_MIN_DISK_GIB must be a positive integer"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf '%s\n' "nvidia-smi is not installed or not on PATH" >> "${sg_errors}"
    : > "${SG_RUN_DIR}/gpu_inventory.csv"
    : > "${SG_RUN_DIR}/nvidia-smi.txt"
else
    nvidia-smi > "${SG_RUN_DIR}/nvidia-smi.txt" 2>&1 || {
        printf '%s\n' "nvidia-smi failed; inspect nvidia-smi.txt" >> "${sg_errors}"
    }
    nvidia-smi \
        --query-gpu=index,uuid,name,memory.total,driver_version \
        --format=csv,noheader,nounits \
        > "${SG_RUN_DIR}/gpu_inventory.csv" 2>> "${SG_RUN_DIR}/nvidia-smi.txt" || {
            printf '%s\n' "GPU inventory query failed" >> "${sg_errors}"
        }
fi

{
    printf 'cache_root=%s\n' "${SG_CACHE_ROOT}"
    df -Pk -- "${SG_CACHE_ROOT}"
} > "${SG_RUN_DIR}/disk.txt" 2>&1 || {
    printf '%s\n' "could not inspect free space at ${SG_CACHE_ROOT}" >> "${sg_errors}"
}

if command -v nvcc >/dev/null 2>&1; then
    nvcc --version > "${SG_RUN_DIR}/nvcc.txt" 2>&1 || true
else
    printf '%s\n' "nvcc not found; runtime CUDA may still be supplied by the driver/containers" \
        > "${SG_RUN_DIR}/nvcc.txt"
fi

sg_visible_devices="${CUDA_VISIBLE_DEVICES:-}"
if [[ -n "${sg_visible_devices}" \
    && ! "${sg_visible_devices}" =~ ^[^,[:space:]]+(,[^,[:space:]]+)*$ ]]
then
    printf '%s\n' \
        "CUDA_VISIBLE_DEVICES must not contain empty or whitespace selectors" \
        >> "${sg_errors}"
fi
sg_available_kib="$(
    df -Pk -- "${SG_CACHE_ROOT}" 2>/dev/null | awk 'NR == 2 {print $4}'
)"
sg_available_kib="${sg_available_kib:-0}"
sg_git_commit="$(git -C "${SG_REPO_ROOT}" rev-parse HEAD 2>/dev/null || true)"

python3 - \
    "${SG_RUN_DIR}/gpu_inventory.csv" \
    "${sg_errors}" \
    "${SG_RUN_DIR}/gpu_check.json" \
    "${sg_min_gpus}" \
    "${sg_min_memory_mib}" \
    "${sg_name_pattern}" \
    "${sg_min_driver}" \
    "${sg_visible_devices}" \
    "${sg_min_disk_gib}" \
    "${sg_available_kib}" \
    "${SG_CACHE_ROOT}" \
    "${sg_git_commit}" <<'PY'
import csv
import datetime as dt
import json
import pathlib
import re
import sys

(
    inventory_path,
    errors_path,
    output_path,
    min_gpus_text,
    min_memory_text,
    name_pattern,
    min_driver_text,
    visible_text,
    min_disk_text,
    available_kib_text,
    cache_root,
    git_commit,
) = sys.argv[1:]

errors_file = pathlib.Path(errors_path)
errors = [line.strip() for line in errors_file.read_text(encoding="utf-8").splitlines() if line.strip()]
inventory = []
path = pathlib.Path(inventory_path)
if path.exists():
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.reader(handle):
            if len(row) < 5:
                if row:
                    errors.append(f"malformed nvidia-smi inventory row: {row!r}")
                continue
            try:
                memory_mib = int(float(row[3].strip()))
            except ValueError:
                errors.append(f"invalid GPU memory value: {row[3].strip()!r}")
                continue
            inventory.append(
                {
                    "physical_index": row[0].strip(),
                    "uuid": row[1].strip(),
                    "name": row[2].strip(),
                    "memory_total_mib": memory_mib,
                    "driver_version": row[4].strip(),
                }
            )

min_gpus = int(min_gpus_text)
selected = []
if visible_text:
    raw_selectors = visible_text.split(",")
    selectors = [item.strip() for item in raw_selectors]
    if any(not item or item != raw for item, raw in zip(selectors, raw_selectors)):
        errors.append(
            "CUDA_VISIBLE_DEVICES contains an empty selector or surrounding whitespace"
        )
        selectors = []
    if len(selectors) < min_gpus:
        errors.append(
            f"CUDA_VISIBLE_DEVICES exposes {len(selectors)} device(s), but {min_gpus} are required"
        )
    resolved_selectors = []
    selected_uuids = set()
    for selector in selectors:
        match = next(
            (
                gpu
                for gpu in inventory
                if gpu["physical_index"] == selector or gpu["uuid"] == selector
            ),
            None,
        )
        if match is None:
            errors.append(f"CUDA_VISIBLE_DEVICES selector {selector!r} is absent from nvidia-smi")
        elif match["uuid"] in selected_uuids:
            errors.append(
                f"CUDA_VISIBLE_DEVICES selects GPU {match['uuid']} more than once"
            )
        else:
            selected_uuids.add(match["uuid"])
            resolved_selectors.append(match)
    selected = resolved_selectors[:min_gpus]
else:
    selected = inventory[:min_gpus]

if len(inventory) < min_gpus:
    errors.append(f"found {len(inventory)} NVIDIA GPU(s), but {min_gpus} are required")
if len(selected) < min_gpus:
    errors.append(f"could validate only {len(selected)} selected GPU(s)")

try:
    compiled_pattern = re.compile(name_pattern, re.IGNORECASE)
except re.error as exc:
    errors.append(f"invalid SCALEGUARD_GPU_NAME_PATTERN: {exc}")
    compiled_pattern = re.compile(r"(?!x)x")

min_memory_mib = int(min_memory_text)


def parse_driver_version(value: str) -> tuple[int, ...]:
    parts = value.split(".")
    if not parts or any(not part.isdigit() for part in parts):
        raise ValueError(value)
    return tuple(int(part) for part in parts)


required_driver = parse_driver_version(min_driver_text)
for logical_index, gpu in enumerate(selected):
    gpu["logical_index"] = logical_index
    if gpu["memory_total_mib"] < min_memory_mib:
        errors.append(
            f"GPU {logical_index} has {gpu['memory_total_mib']} MiB; "
            f"{min_memory_mib} MiB is required"
        )
    if not compiled_pattern.search(gpu["name"]):
        errors.append(
            f"GPU {logical_index} name {gpu['name']!r} does not match {name_pattern!r}"
        )
    try:
        installed_driver = parse_driver_version(gpu["driver_version"])
    except ValueError:
        errors.append(
            f"GPU {logical_index} has invalid NVIDIA driver version "
            f"{gpu['driver_version']!r}"
        )
    else:
        width = max(len(installed_driver), len(required_driver))
        installed_key = installed_driver + (0,) * (width - len(installed_driver))
        required_key = required_driver + (0,) * (width - len(required_driver))
        if installed_key < required_key:
            errors.append(
                f"GPU {logical_index} uses NVIDIA driver {gpu['driver_version']}; "
                f"{min_driver_text} or newer is required for the CUDA 12.6 runtime"
            )

available_kib = int(available_kib_text or "0")
available_gib = available_kib / 1024 / 1024
min_disk_gib = int(min_disk_text)
if available_gib < min_disk_gib:
    errors.append(
        f"cache volume has {available_gib:.1f} GiB free; {min_disk_gib} GiB is required"
    )

deduplicated_errors = list(dict.fromkeys(errors))
errors_file.write_text(
    "".join(f"{item}\n" for item in deduplicated_errors),
    encoding="utf-8",
)
document = {
    "schema_version": 1,
    "checked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "status": "passed" if not deduplicated_errors else "failed",
    "git_commit": git_commit or None,
    "requirements": {
        "minimum_gpu_count": min_gpus,
        "minimum_memory_mib_per_gpu": min_memory_mib,
        "gpu_name_pattern": name_pattern,
        "minimum_nvidia_driver": min_driver_text,
        "minimum_free_disk_gib": min_disk_gib,
    },
    "cuda_visible_devices": visible_text or None,
    "selected_gpus": selected,
    "all_visible_nvidia_gpus": inventory,
    "cache_volume": {
        "path": cache_root,
        "free_gib": round(available_gib, 3),
    },
    "errors": deduplicated_errors,
}
pathlib.Path(output_path).write_text(
    json.dumps(document, indent=2) + "\n",
    encoding="utf-8",
)
PY

sg_write_file_inventory "${SG_RUN_DIR}" "${SG_RUN_DIR}/files.json"

sg_status="$(sg_json_get "${SG_RUN_DIR}/gpu_check.json" status)"
if [[ "${sg_status}" != "passed" ]]; then
    sg_note "GPU preflight failed. Evidence: ${SG_RUN_DIR}"
    sed 's/^/  - /' "${sg_errors}" >&2
    exit 1
fi

sg_note "GPU preflight passed with real nvidia-smi evidence: ${SG_RUN_DIR}"
