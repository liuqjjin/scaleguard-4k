# Reproduction

ScaleGuard-4K currently supports reproduction of its static, CPU, mock, and
process contracts. No retained project evidence yet shows a real 4KAgent run,
a real CoZ run, dual-GPU integration, calibrated trusted-scale behavior, or
research metrics. The highest supported project level is `STATIC_READY`.

## Evidence ladder

| Level | Minimum evidence |
| --- | --- |
| `STATIC_READY` | locked source tree, static checks, deterministic CPU/mock run, contract tests, and no unmarked mock artifacts |
| `COMPONENT_REPRODUCED` | 4KAgent and CoZ each run with real locked source and weights; commands, logs, hashes, environment, and hardware are retained |
| `AB_INTEGRATED` | one real terminal 4KAgent → CoZ path succeeds with a fresh output and per-GPU evidence |
| `SCALEGUARD_VALIDATED` | real candidates exercise reviewed accept, stop, and rollback behavior with a matching calibration receipt |
| `RESEARCH_EVALUATED` | complete paired A-only, B-only, A→B-fixed, and ScaleGuard experiments plus declared ablations and failure analysis |

Levels are cumulative. A syntax check is not an installation; a mock run is not
a component reproduction; an output image is not a validated controller; and a
CSV is not a research conclusion.

## Reproduce `STATIC_READY`

Start from a clean checkout with Python 3.10 or newer and `uv`:

```bash
uv sync --locked --extra dev
uv run --locked scaleguard config validate configs/runtime/cpu-mock.yaml
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy src/scaleguard
uv run --locked pytest --cov=scaleguard --cov-report=term-missing -q
```

The CI workflow runs the locked CPU suite on Python 3.10 through 3.14 with
CUDA hidden and the Hugging Face/Transformers offline flags set.

Exercise the public CLI:

```bash
uv run --locked python -I examples/make_fixture.py /tmp/scaleguard-input.jpg
uv run --locked scaleguard run \
  --config configs/runtime/cpu-mock.yaml \
  --input /tmp/scaleguard-input.jpg \
  --output /tmp/scaleguard-output.png \
  --run-id cpu-contract

uv run --locked scaleguard manifest validate runs/cpu-contract/manifest.json
```

Confirm rather than assume the evidence boundary:

```bash
uv run --locked python -I - <<'PY'
import json
from pathlib import Path

manifest = json.loads(Path("runs/cpu-contract/manifest.json").read_text())
assert manifest["status"] in {"succeeded", "succeeded_with_rollback"}
assert manifest["completion_level"] == "STATIC_READY"
assert manifest["mock"] is True
assert manifest["final_image"]["mock"] is True
print(manifest["run_id"], manifest["status"], manifest["completion_level"])
PY
```

The output is useful for contract review only. Do not publish its proxy score
as an image-quality metric.

## Reproduce source identity

Materialize and verify the exact two core repositories:

```bash
uv run --locked python -I scripts/upstream/materialize.py \
  upstream-lock.yaml \
  --mapping repositories \
  --project-root "$PWD"
uv run --locked scaleguard upstream verify --lock upstream-lock.yaml
```

Materialize the selected 4KAgent transitive service dependency separately:

```bash
uv run --locked python -I scripts/upstream/materialize.py \
  runtime-dependencies.yaml \
  --mapping dependencies \
  --project-root "$PWD"
```

The second command obtains DepictQA for 4KAgent. It does not introduce a third
core algorithm. AgenticIR is not checked out.

Verification covers commits, root trees, ordered patch hashes, patch
application state, final hashes for every patch-modified file, and unexpected
checkout changes. It is still static evidence: no model is loaded.

## Reproduce runtime environment identity

The AutoDL environment contract is Linux `x86_64` with glibc 2.28 or newer and
a system `python3` with `venv`. The user does not need to preinstall the pinned
uv. The project hook clears `.runtime/bootstrap-uv`, reconstructs uv 0.11.16
from the hashed wheel in `environments/bootstrap/uv.lock`, and rejects its
executable unless it matches `environments/bootstrap/uv-binary.sha256`. That
private uv reinstalls the exact managed CPython 3.10.18 archive declared in
`environments/python-downloads.json`; a same-version executable on `PATH` is
never evidence-valid. It then installs ScaleGuard and each of the three
isolated 4KAgent, DepictQA, and CoZ environments. It materializes both source
locks, installs hash-resolved base locks plus the two hash-pinned 4KAgent
override wheels, and writes:

```text
.runtime/receipts/{scaleguard,4kagent,depictqa,coz}.json
.runtime/receipts/bootstrap.json
```

The aggregate receipt binds the project commit, platform, Python and uv
versions, source locks, resolved environment locks, and the four receipt
hashes. The normal status is `passed` for ScaleGuard, DepictQA, and CoZ.
4KAgent instead must report `passed_with_audited_override` with exactly four
declared metadata observations:

```text
pyiqa 0.1.13: transformers ==4.37.2 -> installed 5.5.0
hpsv2 1.2.0: protobuf <4 -> installed 6.33.5
hpsv2 1.2.0: pytest ==7.2.0 -> omitted
hpsv2 1.2.0: pytest-split ==0.8.0 -> omitted
```

Any other missing or incompatible distribution fails the audit. This narrowly
recorded exception follows the pinned upstream environment; it is not evidence
that PyIQA, Qwen, or restoration inference ran correctly. See
[ADR 0005](adr/0005-audited-inference-metadata-overrides.md).

Immediately before each real smoke or integration run, the wrapper reruns the
same four audits into the attempt's `runtime-environments/` directory. The
schema-v2 runtime preflight binds those receipt hashes and requires their full
distribution maps, Python identities, locks, import/entrypoint probes,
overrides, status, and empty issue lists to equal the bootstrap baselines. A
fresh receipt must belong to the same attempt and be no older than that
attempt's recorded start. Historical manifest review verifies the bound hashes
without applying a wall-clock expiry.

The aggregate receipt starts with `status: running` and becomes `passed` only
after the complete hook succeeds. An ordinary failure rewrites it with
`status: failed`, a return code, and the statement that it supports no
environment or GPU claim. Treat missing, `running`, and `failed` aggregates
identically for promotion: none supports an environment validation,
component reproduction, GPU claim, or higher completion level. Do not salvage
a claim from partial environment receipts.

## Real-runtime external gate

Real reproduction requires account and hardware actions that the repository
cannot perform:

- provision the declared Linux `x86_64`, glibc 2.28+, dual-4090 host;
- provide a working system `python3` with `venv`; the hook provisions uv
  0.11.16 and Python 3.10.18;
- accept the gated Stable Diffusion 3 terms;
- provide Hugging Face authentication through the environment;
- provide the remote 4KAgent scheduler credential through the environment
  variable declared by the runtime configuration;
- obtain the required no-publisher-digest DepictQA delta;
- provide authorized smoke/integration images; and
- later provide authorized evaluation data and split manifests.

After those conditions exist, run:

```bash
export CUDA_VISIBLE_DEVICES=0,1
export SCALEGUARD_SMOKE_INPUT=/authorized-data/smoke.png
export SCALEGUARD_INTEGRATION_INPUT=/authorized-data/integration.png
external_gate/commands.sh
```

The complete user-owned steps, success criteria, and redaction rules are in:

- [external gate request](../external_gate/REQUEST.md);
- [expected artifacts](../external_gate/expected_artifacts.json);
- [result template](../external_gate/RESULT_TEMPLATE.json); and
- [AutoDL deployment guide](autodl.md).

Do not edit the result template before evidence exists. Do not fill hardware,
memory, runtime, output, metric, or completion fields from expectations.

## Review a GPU attempt

Wrapper success is necessary but not sufficient. Review the same attempt's
evidence in this order:

1. wrapper `bootstrap.json` and its `runtime-receipts/bootstrap.json` snapshot
   of `.runtime/receipts/bootstrap.json`: exact project commit, non-skipped GPU
   preflight, Linux `x86_64`/glibc identity, hash-pinned uv 0.11.16 bootstrap
   identity, Python 3.10.18, both materialized source locks, resolved lock
   hashes, environment-receipt hashes, upstream verification, and doctor
   outcome. Both bootstrap receipts must be `passed`. The AutoDL scripts must
   invoke the CLI through the lexical project entry
   `.venv/bin/python -I -m scaleguard.cli`; `SCALEGUARD_CLI`,
   `SCALEGUARD_PYTHON`, and `PATH` resolution are not evidence-valid. Inspect
   all four copied environment receipts; the 4KAgent receipt must contain only
   the exact audited PyIQA override described above.
2. four per-attempt `runtime-environments/*.json` receipts and the
   schema-v2 `runtime-preflight.json`: exact match to the bootstrap baselines,
   same-attempt paths, fresh timestamps, and bound hashes.
3. `weights-receipt.json`: fixed revisions, known digest checks, per-file
   measured hashes, and `recorded_manual` for the no-digest DepictQA delta.
4. `materialization-receipt.json`: binding to the weight receipt and project
   commit, fixed layouts, no checkout mutation, and no errors.
5. `execution.json`: fresh output, successful command, complete GPU samples,
   and extracted non-mock model evidence.
6. copied ScaleGuard `manifest.json`: real backends, completed 4KAgent event,
   at least one real CoZ candidate when required, explicit decisions, final
   output hash, and no hidden failure.
7. raw stdout/stderr and CoZ protocol logs: no ignored model, device, OOM, or
   service-lifecycle errors.
8. diagnostics archive: checksum passes and a human reviewed redaction before
   transfer.

A failed attempt remains evidence. Keep its directory, diagnose the recorded
cause, and rerun into a new attempt instead of overwriting it.

## Run-manifest interpretation

The main manifest is updated atomically throughout a run. Key fields are:

- `status`: execution outcome;
- `completion_level`: the run's asserted evidence tier;
- `mock`: whether either algorithmic backend is fake;
- `config` and `provenance`: exact settings and backend identities;
- `input_image`, `restored_image`, `final_image`: hashed artifacts;
- `steps`: trusted state, candidate, metrics, decision, acceptance, reason,
  process evidence, and timestamps; and
- `events` and `error`: lifecycle or failure record.

Never promote the project from `completion_level` alone. The manifest must
also have a successful status, matching files and hashes, `mock: false`, and
the raw evidence required by the level. The wrapper's
`highest_supported_completion_level` is filled only after review.

## Calibration and evaluation reproduction

Trusted thresholds are a separate evidence stage. A calibration receipt:

- excludes mock samples;
- verifies every trusted and candidate artifact hash;
- requires a label for every metric-bearing step;
- binds metric identity and measurement model;
- records quantiles, bootstrap settings, sample counts, and issues; and
- must match the exact runtime thresholds.

Controller construction records the receipt's resolved path, size, SHA-256, and
semantic verification in the manifest. Paired-summary review independently
reopens the same byte snapshot and repeats the check. Rewriting thresholds and
nearby self-digests does not preserve eligibility. Measurement value and
canonical forward-model identity must appear together only when the observation
gate is enabled.

Follow [the evaluation protocol](evaluation-protocol.md). A receipt with
`status: insufficient_data`, any issue, a proxy backend, a hash mismatch, or
threshold mismatch does not support `SCALEGUARD_VALIDATED`.

## Reproducibility notes

- CoZ seeds are explicit, but upstream VLM/planning and CUDA kernels may still
  vary. Retain actual plans, prompts, versions, and outputs.
- The current 4KAgent issue record includes environment, toolbox, PyIQA, and
  evaluation-setting reproduction caveats.
- The manual DepictQA delta has no published digest; its receipt proves local
  byte identity only after acquisition.
- Model and package licenses can restrict use even when every digest matches.

The fixed source identities and issue review are in
[upstream-audit.md](upstream-audit.md). Current claims and missing evidence are
listed in [results/STATUS.md](results/STATUS.md).
