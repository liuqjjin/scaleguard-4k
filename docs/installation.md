# Installation

ScaleGuard-4K supports a reproducible CPU/mock development environment and a
completed dual-GPU 4KAgent → CoZ research path on a Linux dual-RTX-4090 host.
The project evidence level is `RESEARCH_EVALUATED`. CPU/mock runs remain
`STATIC_READY` and `mock: true`.

## CPU and development environment

Requirements:

- Git;
- Python 3.10–3.14; and
- [uv](https://docs.astral.sh/uv/).

From the repository root:

```bash
uv sync --locked --extra dev
uv run --locked scaleguard --version
uv run --locked scaleguard config validate configs/runtime/cpu-mock.yaml
uv run --locked pytest
```

`uv.lock` is the reproducible dependency source. Do not replace `--locked`
with an unlocked resolve when preparing release evidence.

Create and run a deterministic CPU fixture:

```bash
uv run --locked python -I examples/make_fixture.py /tmp/scaleguard-input.jpg
uv run --locked scaleguard run \
  --config configs/runtime/cpu-mock.yaml \
  --input /tmp/scaleguard-input.jpg \
  --output /tmp/scaleguard-output.png
```

The command writes an image and a run manifest, but the run is explicitly
mock. It establishes controller and artifact contracts, not restoration
quality or GPU readiness.

## Optional quality metrics

The research metric extra pins PyIQA 0.1.16:

```bash
uv sync --locked --extra metrics
```

PyIQA and the selected MUSIQ weight carry non-commercial licenses separate
from this repository's Apache-2.0 source license. Installing the package does
not calibrate a controller threshold. Read [NOTICE](../NOTICE) and
[ADR 0003](adr/0003-gradient-proxy-cpu-contract-only.md) before enabling it.

## Materialize audited source checkouts

The main repository does not vendor upstream source. Materialize only the two
locked core checkouts:

```bash
uv run --locked python -I scripts/upstream/materialize.py \
  upstream-lock.yaml \
  --mapping repositories \
  --project-root "$PWD"

uv run --locked scaleguard upstream verify --lock upstream-lock.yaml
```

The materializer:

- clones only into ignored project paths;
- detaches at exact commits;
- verifies root trees and patch SHA-256 values;
- applies CoZ patches in lock order;
- refuses to switch a dirty checkout; and
- disables the checkout's push URL.

The selected 4KAgent runtime also needs its pinned DepictQA perception
dependency:

```bash
uv run --locked python -I scripts/upstream/materialize.py \
  runtime-dependencies.yaml \
  --mapping dependencies \
  --project-root "$PWD"
```

DepictQA is a 4KAgent transitive service dependency. This command does not make
it a third ScaleGuard core project. AgenticIR is not materialized or executed.

For an existing installation, add `--verify-only` to either materializer
command. Manual clone and patch commands are documented in
[`third_party/README.md`](../third_party/README.md).

## Real runtime environments

Do not merge ScaleGuard, 4KAgent, DepictQA, and CoZ into one Python
environment. Their dependency stacks and GPU lifetimes are intentionally
separate:

```text
.venv/                         ScaleGuard CLI and controller
.runtime/envs/4kagent/         4KAgent
.runtime/envs/depictqa/        4KAgent's DepictQA service
.runtime/envs/coz/             Chain-of-Zoom
```

The project-specific AutoDL hook has a narrower platform contract than CPU
development:

- Linux on `x86_64`;
- glibc 2.28 or newer;
- a system `python3` with the standard-library `venv` module; and
- uv-managed CPython exactly 3.10.18.

The installed uv identity is still exactly 0.11.16, as recorded in
`environments/uv.version`, but the user does not need to preinstall it. The
hook always clears `.runtime/bootstrap-uv` and installs the hash-pinned Linux
wheel from `environments/bootstrap/uv.lock`; it does not trust a same-version
binary found on `PATH`. The installed executable must match
`environments/bootstrap/uv-binary.sha256`.

The provisioned uv reinstalls the exact Python 3.10.18 Linux build declared in
`environments/python-downloads.json`, whose archive URL, build, and SHA-256 are
committed. It reinstalls the ScaleGuard `.venv` from `uv.lock` with the metric
extra and synchronizes the three runtime environments above with
`--reinstall`. It materializes both source locks before installation:
`upstream-lock.yaml` with the `repositories` mapping and
`runtime-dependencies.yaml` with the `dependencies` mapping.

Each runtime environment is synchronized with `--require-hashes` from its
resolved lock under `environments/`. Successful bootstrap writes
per-environment package files, interpreter, standard-library, import-origin,
lock-hash, and dependency-audit receipts plus an aggregate receipt:

```text
.runtime/receipts/
  scaleguard.json
  4kagent.json
  depictqa.json
  coz.json
  bootstrap.json
```

These receipts establish the installed dependency contract only. They contain
no GPU inference or quality-result claim.

Smoke and integration do not trust these historical bytes alone: immediately
before model execution they rerun the same four audits into the attempt
directory and bind an exact baseline comparison into a schema-v2 runtime
preflight receipt.

These checks detect and reject ordinary in-host drift. They do not defend
against a malicious administrator who can replace the repository,
interpreters, loader, and unsigned evidence together; see
[ADR 0009](adr/0009-bind-runtime-bytes-at-each-real-attempt.md).

The aggregate `bootstrap.json` is written as `running` before provisioning and
is replaced with `passed` only after every lock, checkout, environment, and
audit check succeeds. Any ordinary hook failure replaces it with `failed`, the
return code, and an explicit no-claim statement. A missing, `running`, or
`failed` aggregate receipt supports no environment, GPU, component, or project
completion claim, even if partial per-environment receipts remain on disk.

### Audited 4KAgent inference metadata overrides

The pinned 4KAgent environment retains upstream PyIQA 0.1.13 and HPSv2 1.2.0
while security-updating Transformers to 5.5.0 and Protobuf to 6.33.5. Their
legacy wheel metadata is therefore not resolver-clean.

`pyiqa.override.lock` and `hpsv2.override.lock` pin both exact wheel hashes and
install them without dependencies after the fully hashed base environment. The
environment audit permits only these four records:

```text
pyiqa 0.1.13: transformers ==4.37.2 -> installed 5.5.0
hpsv2 1.2.0: protobuf <4 -> installed 6.33.5
hpsv2 1.2.0: pytest ==7.2.0 -> omitted
hpsv2 1.2.0: pytest-split ==0.8.0 -> omitted
```

The `4kagent.json` receipt must have
`status: passed_with_audited_override` and four matching `audited_overrides`
entries. A changed, absent, duplicated, or additional dependency mismatch
fails bootstrap. ScaleGuard's separate `.venv` continues to use its own PyIQA
0.1.16 metric extra. See
[ADR 0005](adr/0005-audited-inference-metadata-overrides.md).

Do not install unreviewed “latest” packages over these environments or edit an
audited checkout to solve a conflict. Record a deliberate lock update instead.
The process-local compatibility shims and offline symbol-import receipts are
defined in
[ADR 0008](adr/0008-minimal-inference-compatibility-shims.md); they preserve
the security-updated stack without adding training-only DeepSpeed.

## Models and weights

Models are not distributed with ScaleGuard. `weights-lock.json` records
immutable Hugging Face revisions, known file digests, destinations, license
metadata, and one required manual artifact.

Three user-authorized actions are unavoidable for the checked-in real-runtime
profile:

1. accept the Stable Diffusion 3 model terms on Hugging Face and authenticate
   on the target machine; and
2. create an Alibaba Cloud Model Studio API key in the Beijing region and
   expose it under the configured `DASHSCOPE_API_KEY` variable; and
3. obtain the 4KAgent DepictQA degradation delta from its pinned upstream
   Google Drive object.

The publisher provides no digest for the manual delta. ScaleGuard records its
locally measured SHA-256 but does not call it publisher-authenticated.

Weight acquisition and materialization are handled by:

```bash
scripts/autodl/download_weights.sh
```

Public downloads are pinned and hashed. Optional artifacts are skipped unless
`--include-optional` is supplied. That option adds the locked LPIPS v0.1
linear-layer checkpoint and content-addressed OpenAI RN50 file used by the
offline CLIPIQA adapter. LPIPS also needs the separate Torchvision AlexNet
ImageNet backbone supplied through `--pyiqa-backbone`; because its publisher
URL identifies only a short hash prefix, it is treated as an explicit
user-provided evaluation input and its full measured SHA-256 is bound in the
metric receipt rather than implied to be prepared by the downloader. The
wrapper then invokes
`scripts/weights/materialize.py` without mutating the audited checkouts. That
hook derives only the expanded 4KAgent toolbox and the DQ495K copy under
`weights/4kagent/runtime/`, then records their inventories in a materialization
receipt. HPSv2 and PyIQA continue to use their locked download destinations;
they are not copied into the derived runtime tree.

Never add tokens to command arguments, URLs, `.env` files, Git, issues, or
diagnostic archives.

AutoDL entry points enforce stage-specific credential visibility. GPU checks,
bootstrap/install hooks, and source verification receive none; weight download
receives only Hugging Face authentication; doctor receives a non-secret
presence marker; model execution receives only the variable named by
`fourkagent.api_key_env`. The diagnostics collector
privatizes exact values before launching probes and passes them only to its
redactor through a private file descriptor.

## AutoDL dual-4090 entry point

The prepared real-runtime target is Linux `x86_64` with glibc 2.28 or newer,
a system `python3` with `venv`, NVIDIA driver 560.28.03 or newer, two visible
RTX 4090 GPUs, at least 24000 MiB reported per GPU, and at least 150 GiB of
free cache space. The locked GPU environments use the official PyTorch 2.10.0
CUDA 12.6 wheel family. The hook hash-bootstraps uv 0.11.16 when necessary and
installs Python 3.10.18. These are preflight requirements, not measured project
results.

After provisioning the host and accepting gated terms:

```bash
read -rsp 'HF token: ' HF_TOKEN && printf '\n'
export HF_TOKEN
read -rsp 'DashScope API key: ' DASHSCOPE_API_KEY && printf '\n'
export DASHSCOPE_API_KEY
export CUDA_VISIBLE_DEVICES=0,1

scripts/autodl/check_gpu.sh
scripts/autodl/bootstrap.sh

mkdir -p weights/4kagent/depictqa/delta
# In an authorized browser, obtain the upstream manual object:
# https://drive.google.com/file/d/1o-PN1iXctWl62Tdb8fZs1eD1Ehv6HBMh/view
# Upload it to the exact path below, then require a non-empty file:
test -s weights/4kagent/depictqa/delta/degra_eval.pt

scripts/autodl/download_weights.sh
```

The publisher supplies no digest for that manual object. ScaleGuard records
the bytes it receives but cannot authenticate their publisher origin.
Bootstrap can therefore run before credentials are exported; its doctor receipt
records the absent `4kagent_api_key` as a deferred external check. The shown
single-shell flow remains supported because bootstrap strips both exported
credentials before any child process.

Run the smoke and integration wrappers only with authorized images:

```bash
scripts/autodl/run_smoke.sh \
  --config configs/runtime/autodl-2x4090.yaml \
  --input /authorized-data/smoke.png

scripts/autodl/run_integration.sh \
  --config configs/runtime/autodl-2x4090.yaml \
  --input /authorized-data/integration.png
```

The wrappers reject stale outputs, rerun GPU/upstream/weight checks, preserve
logs, sample both GPUs, and require a fresh non-mock manifest. Their success
still requires evidence review before the project status is raised.

Unset both credential variables after the attempt:

```bash
unset HF_TOKEN DASHSCOPE_API_KEY
```

The complete cache layout, recovery procedure, diagnostics collection, and
one-command gate are in the [AutoDL guide](autodl.md) and
[external gate request](../external_gate/REQUEST.md).

## Readiness checks

For CPU/mock development:

```bash
uv run --locked scaleguard doctor --config configs/runtime/cpu-mock.yaml
```

For the real runtime:

```bash
uv run --locked scaleguard upstream verify --lock upstream-lock.yaml
uv run --locked scaleguard doctor --config configs/runtime/autodl-2x4090.yaml
```

`doctor` checks paths, service launchability, GPU inventory, and whether a
calibration receipt is configured. It does not start the models or by itself
re-establish that thresholds are statistically valid. Research reproduction
should point doctor at the same calibration receipt used by the published
study.

## Common installation failures

- **Wrong upstream commit or patch state:** rerun the materializer in
  `--verify-only` mode; do not silently use a different revision.
- **Gated download denied:** confirm the account accepted the exact model
  terms and that `HF_TOKEN` is exported without printing it.
- **Manual delta gate:** place the complete file at its locked destination and
  rerun the downloader; preserve the measured-hash receipt.
- **Runtime dependency conflict:** rebuild the isolated environment from its
  lock instead of changing the ScaleGuard core environment.
- **Platform or bootstrap rejection:** use Linux `x86_64` with glibc 2.28+
  and a system Python whose `python3 -m venv` works. Inspect the failed
  aggregate receipt; do not bypass the locked uv self-bootstrap.
- **4KAgent audit rejection:** inspect `.runtime/receipts/4kagent.json`; only
  the four exact records in ADR 0005 are accepted.
- **GPU preflight failure:** inspect the emitted JSON and `nvidia-smi` evidence;
  do not lower requirements merely to turn a failed attempt green.
- **CUDA out of memory:** retain the failed evidence, change only recorded
  tile/offload settings, and rerun as a new attempt.

No installation command proves a model or research result. The evidence ladder
and promotion rules are in [reproduction.md](reproduction.md).
