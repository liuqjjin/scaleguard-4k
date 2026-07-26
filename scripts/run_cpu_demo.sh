#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly script_dir
project_root="$(cd -- "${script_dir}/.." && pwd -P)"
readonly project_root
readonly config_path="${project_root}/configs/runtime/cpu-mock.yaml"
readonly temp_base="${RUNNER_TEMP:-${TMPDIR:-/tmp}}"

if ! command -v uv >/dev/null 2>&1; then
  printf 'error: uv is required; install the version in environments/uv.version\n' >&2
  exit 127
fi

if [[ ! -f "${project_root}/uv.lock" ]]; then
  printf 'error: uv.lock is missing from %s\n' "${project_root}" >&2
  exit 2
fi

mkdir -p -- "${temp_base}"
demo_root="$(mktemp -d "${temp_base%/}/scaleguard-cpu-demo.XXXXXXXX")"
readonly demo_root
readonly input_path="${demo_root}/input/fixture.png"
readonly output_path="${demo_root}/artifacts/final.png"
readonly runtime_config_path="${demo_root}/config/cpu-mock.yaml"
readonly run_id="cpu-demo"
readonly run_dir="${demo_root}/runs/${run_id}"
readonly manifest_path="${run_dir}/manifest.json"

export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SCALEGUARD_TEST_MODE=cpu
export SCALEGUARD_PROJECT_ROOT="${project_root}"

on_error() {
  local status=$?
  printf 'CPU demo failed (exit %d). Partial artifacts: %s\n' \
    "${status}" "${demo_root}" >&2
  exit "${status}"
}
trap on_error ERR

printf 'CPU demo workspace: %s\n' "${demo_root}"
cd -- "${demo_root}"

printf '[1/5] Generate deterministic fixture\n'
uv run --locked --project "${project_root}" \
  python "${project_root}/examples/make_fixture.py" "${input_path}"

printf '[2/5] Validate CPU/mock configuration and isolate its run root\n'
uv run --locked --project "${project_root}" \
  scaleguard config validate "${config_path}"
uv run --locked --project "${project_root}" \
  python - "${config_path}" "${runtime_config_path}" "${demo_root}/runs" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import yaml

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
run_root = Path(sys.argv[3]).resolve()
config = yaml.safe_load(source.read_text(encoding="utf-8"))
config["runtime"]["run_root"] = str(run_root)
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY
uv run --locked --project "${project_root}" \
  scaleguard config validate "${runtime_config_path}"

printf '[3/5] Run public ScaleGuard CLI\n'
uv run --locked --project "${project_root}" \
  scaleguard run \
  --config "${runtime_config_path}" \
  --input "${input_path}" \
  --output "${output_path}" \
  --run-id "${run_id}"

printf '[4/5] Validate run manifest contract\n'
uv run --locked --project "${project_root}" \
  scaleguard manifest validate "${manifest_path}"

printf '[5/5] Verify evidence labels and final artifact hash\n'
uv run --locked --project "${project_root}" \
  python - "${manifest_path}" "${output_path}" "${project_root}/src" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[3]).resolve()))

from scaleguard.strict_json import loads_object

manifest_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
manifest = loads_object(manifest_path.read_text(encoding="utf-8"))

allowed_statuses = {"succeeded", "succeeded_with_rollback"}
if manifest.get("status") not in allowed_statuses:
    raise SystemExit(f"unexpected run status: {manifest.get('status')!r}")
if manifest.get("completion_level") != "STATIC_READY":
    raise SystemExit(f"unexpected completion level: {manifest.get('completion_level')!r}")
if manifest.get("mock") is not True:
    raise SystemExit("CPU demo manifest must have mock=true")

final_image = manifest.get("final_image")
if not isinstance(final_image, dict) or final_image.get("mock") is not True:
    raise SystemExit("CPU demo final artifact must have mock=true")
internal_path = Path(str(final_image.get("path"))).resolve()
if internal_path != (manifest_path.parent / "final.png").resolve():
    raise SystemExit("manifest final artifact is not the immutable run artifact")
internal_digest = hashlib.sha256(internal_path.read_bytes()).hexdigest()
if final_image.get("sha256") != internal_digest:
    raise SystemExit("manifest final artifact SHA-256 does not match the run artifact")
digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
if digest != internal_digest:
    raise SystemExit("published output bytes differ from the immutable run artifact")

print(
    json.dumps(
        {
            "status": manifest["status"],
            "completion_level": manifest["completion_level"],
            "mock": manifest["mock"],
            "run_dir": str(manifest_path.parent.resolve()),
            "output": str(output_path.resolve()),
            "output_sha256": digest,
        },
        sort_keys=True,
    )
)
PY

trap - ERR
printf 'CPU demo passed. Artifacts retained at: %s\n' "${demo_root}"
