#!/usr/bin/env bash
set -Eeuo pipefail

sg_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
sg_repo_root="$(cd -- "${sg_script_dir}/../.." && pwd -P)"
cd "${sg_repo_root}"

sg_audit=(
    uv run
    --locked
    --extra dev
    pip-audit
    --disable-pip
    --no-deps
)

"${sg_audit[@]}" \
    --requirement environments/4kagent/requirements.resolved.lock \
    --ignore-vuln PYSEC-2026-1215 \
    --ignore-vuln PYSEC-2026-2447
"${sg_audit[@]}" \
    --requirement environments/depictqa/requirements.resolved.lock
"${sg_audit[@]}" \
    --requirement environments/coz/requirements.resolved.lock
