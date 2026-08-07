# AutoDL dual-RTX-4090 deployment

This path is for real CUDA validation. Running the scripts locally, passing
shell syntax checks, or exercising a fake backend does not establish a model
reproduction level.

## Host and account prerequisites

Provision a Linux instance with:

- two RTX 4090 GPUs visible to one process, each reporting at least 24000 MiB;
- NVIDIA driver 560.28.03 or newer and `nvidia-smi`, for the locked CUDA 12.6
  PyTorch runtime;
- at least 150 GiB free on the cache volume;
- Git and a host `python3` 3.10 or newer with `venv` support; the bootstrap
  installs the evidence runtime at Python 3.10.18;
- an authorized smoke image and integration image.

The project hook never trusts `uv` from `PATH`. It clears its private bootstrap
environment, installs the exact version in `environments/uv.version` from the
hash-locked wheel in `environments/bootstrap/uv.lock`, verifies the installed
executable against `environments/bootstrap/uv-binary.sha256`, and reinstalls
the managed CPython archive pinned by `environments/python-downloads.json`.
No unpinned installer or same-version host binary is evidence-valid.

Accept the Stable Diffusion 3 model terms on Hugging Face before connecting to
the instance. Enter the token interactively:

```bash
read -rsp 'HF token: ' HF_TOKEN && printf '\n'
export HF_TOKEN
read -rsp 'DashScope API key: ' DASHSCOPE_API_KEY && printf '\n'
export DASHSCOPE_API_KEY
export CUDA_VISIBLE_DEVICES=0,1
```

The scripts read tokens only from the environment. Never add `--token`, embed a
token in a URL, save it in the repository, or enable `set -x`. Exporting both
values before the one-command gate does not expose them to every stage:
GPU/bootstrap/source-verification children receive no credentials, the weight
downloader receives only Hugging Face authentication, doctor receives a fixed
non-secret presence marker, and only model execution receives the configured
scheduler credential.

## Cache and artifact locations

On AutoDL the recommended data-disk cache is
`/root/autodl-tmp/scaleguard-4k/cache`. Every path can be overridden before
bootstrap:

```bash
export SCALEGUARD_CACHE_ROOT=/root/autodl-tmp/scaleguard-4k/cache
export SCALEGUARD_ARTIFACT_ROOT="$PWD/artifacts/autodl"
export HF_HOME="$SCALEGUARD_CACHE_ROOT/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$SCALEGUARD_CACHE_ROOT/transformers"
export TORCH_HOME="$SCALEGUARD_CACHE_ROOT/torch"
export PIP_CACHE_DIR="$SCALEGUARD_CACHE_ROOT/pip"
```

`bootstrap.sh` writes the resolved, credential-free cache exports to its run
directory as `cache.env`. Model files, run outputs, logs, diagnostics and
generated environment files are ignored by Git.

The downloader stores model snapshots below
`$SCALEGUARD_CACHE_ROOT/weights/models`. Bootstrap creates a repository
`weights` symlink to `$SCALEGUARD_CACHE_ROOT/weights`, so production config can
use stable paths such as
`weights/models/stabilityai/stable-diffusion-3-medium-diffusers`. Bootstrap is
idempotent when that link is correct. It aborts instead of replacing a real
`weights` directory or repointing an existing link.

## Bootstrap

Run all commands from any directory; each script resolves the repository root
from its own location. Evidence-producing stages require a committed `HEAD` and
a clean main-project worktree, so the recorded commit cannot conceal local code
or configuration changes. Runtime environments, checkouts, caches, weights and
artifacts must remain under the ignored paths supplied by the repository.

Invoke these public scripts directly. They require privileged Bash startup,
clear imported functions and startup hooks, disable tracing, and reject an
ambient repository-root override that does not resolve to the tree containing
the script. Do not use ordinary `bash scripts/autodl/...`; if an explicit Bash
is required, use `/bin/bash -p`. Each entry also creates a fresh private HOME
below `.runtime/isolated-homes`, disables account and system Git configuration,
and ignores user-level pip, uv, and Python startup configuration.

```bash
scripts/autodl/check_gpu.sh
scripts/autodl/bootstrap.sh
```

The preflight defaults are deliberately strict: at least two GPUs, a name
matching `4090`, 24000 MiB per selected GPU, NVIDIA driver 560.28.03 or newer,
and 150 GiB of free cache space.
Adjust only for an intentional, recorded experiment:

```bash
export SCALEGUARD_MIN_GPUS=2
export SCALEGUARD_GPU_NAME_PATTERN='4090'
export SCALEGUARD_MIN_GPU_MEMORY_MIB=24000
export SCALEGUARD_MIN_NVIDIA_DRIVER=560.28.03
export SCALEGUARD_MIN_DISK_GIB=150
```

Bootstrap prefers the repository's `scripts/bootstrap/autodl.sh` when present.
That project hook checks out the locked upstream commits and applies every
`upstream-lock.yaml` patch in listed order before environment installation.
It separately materializes `runtime-dependencies.yaml`; DepictQA is recorded
only as a 4KAgent runtime dependency, never as a third algorithm upstream.
Otherwise it uses a locked `uv.lock`, a hashed `requirements.lock`, or finally
the local `pyproject.toml` in that order. These fallbacks install the ScaleGuard
core only and `bootstrap.json` labels them `core-only`; they do not claim that
the 4KAgent or CoZ runtime environments were installed. It then runs:

```text
scaleguard upstream verify --lock upstream-lock.yaml
scaleguard upstream verify --lock runtime-dependencies.yaml --mapping dependencies
scaleguard doctor --config configs/runtime/autodl-2x4090.yaml
```

Because bootstrap intentionally runs before gated/manual weight acquisition and
model execution, it may defer `4kagent_api_key` plus only
`coz_sr_lora`, `coz_vae`, `coz_vlm_lora`,
`4kagent_toolbox`, `4kagent_hps`, `4kagent_quality_model`, and
`4kagent_perception_model` presence failures.
`doctor-summary.json` records a credential deferral as
`passed_with_deferred_checks` and a weights-only deferral as
`passed_with_deferred_weights`; any other failed doctor check aborts bootstrap.
Doctor warnings are retained as non-passing research caveats. When
`--skip-gpu-check` is explicitly used, only `gpu_inventory` may additionally be
deferred and the state is `passed_with_deferred_checks`.
After weights are present, both smoke and integration wrappers require the full
doctor to report no failed check before starting a model process; warnings
remain visible in evidence and are not promoted to validation claims.

For the project hook, the outer wrapper validates the aggregate environment
receipt against the current commit and every declared lock, snapshots all four
environment receipts into the attempt directory, and includes them in the file
inventory. This receipt establishes environment preparation only; it contains
no GPU inference or quality-result claim.

An unfrozen `pyproject.toml` fallback is recorded in the bootstrap log and
should not be used for a release claim. Set `SCALEGUARD_BOOTSTRAP_COMMAND` only
for a reviewed project-specific installer; never put credentials in that
string. `--skip-gpu-check` exists for non-GPU image preparation and cannot
count as GPU validation.

## Weight manifest and download

Weight acquisition is separate from environment setup because Stable Diffusion
3 is gated. The audited root `weights-lock.json` is used by default; override it
only with another reviewed manifest:

```bash
scripts/autodl/download_weights.sh
```

The manifest has `schema_version: 1` and a non-empty `artifacts` array.
Hugging Face artifacts require an immutable 40-character commit SHA:

```json
{
  "schema_version": 1,
  "artifacts": [
    {
      "id": "coz-sd3",
      "provider": "huggingface",
      "repo_id": "organization/model",
      "revision": "40-character-audited-commit-sha-goes-here",
      "destination": "models/organization/model",
      "gated": true,
      "include": ["model_index.json", "transformer/**"],
      "exclude": []
    }
  ]
}
```

Replace the illustrative fields with values reviewed into the project's weight
lock; the shown text is not a usable model record. Public single-file downloads
use provider `https`, reject signed/query URLs, and require `url`,
file-valued `destination`, and an expected 64-character `sha256`.

Entries with `provider: "manual"` are never downloaded. If `required` is true,
the script writes an `external_gate` receipt containing the entry's destination
and `instructions`/`notes`, then exits non-zero while the file is absent. Once
the user supplies the complete declared file, a rerun records it as
`recorded_manual` and measures its SHA-256 without claiming an upstream digest
match. If `required` is false and absent, the receipt records it as `skipped`.
A null `known_sha256` on a Hugging Face or manual entry is accepted as “no
upstream digest published”; downloaded regular files still receive measured
SHA-256 values in the receipt.

Artifacts with `required: false` are skipped by default. Include them only for a
recorded optional experiment with `--include-optional` or
`SCALEGUARD_DOWNLOAD_OPTIONAL_WEIGHTS=1`.

The current lock has one required manual artifact. Run bootstrap first so the
safe `weights` symlink exists. In an authorized browser, open the upstream
Google Drive object below, review its terms, download it locally, and transfer
the obtained file to the exact destination:

[4KAgent-listed DepictQA degradation delta](https://drive.google.com/file/d/1o-PN1iXctWl62Tdb8fZs1eD1Ehv6HBMh/view)

```bash
mkdir -p weights/4kagent/depictqa/delta
test -s weights/4kagent/depictqa/delta/degra_eval.pt
scripts/autodl/download_weights.sh
```

The upstream guide publishes no digest for that Google Drive object. Preserve
the receipt's measured hash and do not describe it as source-authenticated.
`download_weights.sh` invokes `scripts/weights/materialize.py`, stores the
attempt receipt, and checks it against the fixed marker; do not run an
unrecorded extraction in its place.

The downloader uses `hf download` or `huggingface-cli download` without a token
argument. It accepts either the token environment variables above or credentials
already stored by `hf auth login` under the configured `HF_HOME`; no credential
value enters the receipt. It searches the normal `PATH` and the project
`.venv`, 4KAgent and CoZ runtime `bin` directories, so the bootstrap hook does
not need to leak shell activation state into later commands. It writes
`weights-receipt.json` with source revisions
and SHA-256 for every regular model file. The script makes all credential
variables non-exported before setup: only the actual download child receives
Hugging Face auth, while the materializer and receipt checks receive none.

After that receipt is complete, the wrapper requires the repository's
`scripts/weights/materialize.py` hook. The hook safely expands the large
4KAgent toolbox into the cache and derives the locked DepictQA DQ495K layout
without changing either audited checkout. HPSv2 and PyIQA use their locked
download destinations directly. The hook writes both
`materialization-receipt.json` and the fixed
`$SCALEGUARD_CACHE_ROOT/weights/.scaleguard-materialization.json` marker.
Archive members with absolute or parent-traversal paths, links, devices or
other unsafe types are rejected. A missing hook or inconsistent marker fails
the attempt; the wrapper never silently falls back to upstream implicit
downloads.

A failed attempt writes failure evidence and does not claim that a complete
weight layout is verified.

## Smoke and integration runs

Smoke and integration use the same public CLI contract but separate inputs,
configs, artifact directories and evidence:

```bash
scripts/autodl/run_smoke.sh \
  --config configs/runtime/autodl-2x4090.yaml \
  --input /authorized-data/smoke.png

scripts/autodl/run_integration.sh \
  --config configs/runtime/autodl-2x4090.yaml \
  --input /authorized-data/integration.png
```

The smoke wrapper uses `configs/runtime/autodl-smoke.yaml` automatically when it
exists; otherwise it uses `autodl-2x4090.yaml`. Equivalent environment
variables are `SCALEGUARD_SMOKE_CONFIG`, `SCALEGUARD_SMOKE_INPUT`,
`SCALEGUARD_INTEGRATION_CONFIG`, and `SCALEGUARD_INTEGRATION_INPUT`.

For each run the wrapper:

1. repeats the real GPU preflight;
2. verifies `upstream-lock.yaml`;
3. separately verifies the 4KAgent runtime dependencies in
   `runtime-dependencies.yaml`;
4. before and after model execution, resolves the download receipt bound to the
   fixed materialization marker, reruns the project materializer in strict
   `--verify-only` mode, and rehashes every completed artifact against the
   bound download receipt;
5. runs `scaleguard doctor`;
6. reruns the locked distribution, dependency, offline import, and 4KAgent
   tool-entrypoint audit with each of the four isolated interpreters, without
   account credentials;
7. writes a schema-v2 preflight receipt bound to the current Git commit,
   config, lock hashes, fresh per-attempt environment receipts, bootstrap
   baselines, and materialization verification;
8. starts per-GPU memory/utilization sampling;
9. executes `scaleguard run --config … --input … --output …
   --runtime-preflight …`;
10. requires a fresh, non-empty non-symlink output, snapshots its exact bytes and
   cross-checks both hashes;
11. snapshots and validates the CLI-named ScaleGuard manifest, rejects
   `mock: true`, verifies a
   completed 4KAgent event and at least one non-mock CoZ candidate, and checks
   the final output hash against that manifest;
12. writes `execution.json`, the validated manifest/output snapshots, the raw log and a
   complete evidence inventory.

The wrapper privatizes all ambient credentials before preflight.
`scaleguard doctor` receives a fixed non-secret presence marker under the name
declared by `fourkagent.api_key_env`; only the actual `scaleguard run` child
receives that variable's real value. GPU probes, upstream verification,
environment re-audit, materialization verification, evidence extraction, and
post-run checks receive no account credentials.

All AutoDL bootstrap, smoke, and integration CLI calls are pinned to the
lexical project entry `.venv/bin/python -I -m scaleguard.cli`. The wrappers do
not search `PATH` and reject `SCALEGUARD_CLI` or `SCALEGUARD_PYTHON` overrides.
Before issuing an evidence command they verify that `sys.executable` is that
lexical venv entry, `sys.prefix` is the project `.venv`, and
`scaleguard.cli` resolves to the tracked `src/scaleguard/cli.py`. Rerun
`scripts/autodl/bootstrap.sh` if this attestation fails.

Model commands, managed services, and persistent CoZ workers own fresh process
groups. A returned leader cannot leave an ordinary same-group helper running:
the wrapper waits only a short shutdown grace and then uses bounded TERM/KILL
cleanup. Configured upstream commands must not daemonize through `setsid`.
The canonical two-GPU topology also holds one cooperative, host-local `flock`
lease for the complete wrapper attempt. A concurrent ScaleGuard wrapper fails
before GPU preflight instead of sharing model memory or producing ambiguous GPU
evidence. The lease holder exits with its parent even after abnormal shutdown;
this coordinates ScaleGuard runs, not unrelated host processes.

The per-attempt receipts live under
`runtime-environments/{scaleguard,4kagent,depictqa,coz}.json`. Preflight rejects
missing, stale, relocated, or symlinked receipts and any difference from the
bootstrap baseline in Python identity, lock inventory, complete installed
distribution map, import/entrypoint probes, audited overrides, status, or
issues. This catches an install, uninstall, or import breakage introduced after
bootstrap before a real model process starts.

An explicit output path must not already exist. This prevents a stale image from
turning a no-op into a false pass. Default output paths are unique to each run.
Set `SCALEGUARD_GPU_SAMPLE_INTERVAL_SECONDS` to change the default one-second
sampling period; accepted values are 0.1 through 60 seconds. The runtime receipt
binds the monitor to the two UUIDs selected by GPU preflight. `execution.json`
reports inventory binding separately from host-level workload observation; the
latter is deliberately not described as per-process GPU attribution. The
runtime validator reopens the attempt-local GPU receipt, verifies its hash,
commit, topology, and CUDA selectors, then carries that normalized identity in
the runtime execution binding. Set
`SCALEGUARD_RUN_DEADLINE_SECONDS` to change the full-wrapper deadline from its
14,400-second default (allowed range: 60 through 86,400 seconds). The deadline
covers preflight, model execution, and post-run evidence checks and has a bounded
TERM/KILL shutdown. If the matching download receipt was moved outside the
artifact tree, set `SCALEGUARD_WEIGHT_RECEIPT` to its reviewed path; its SHA-256
must still match the fixed marker.

`execution.json.status == "passed"` means only that the command and wrapper
artifact checks passed. Review the ScaleGuard run manifest to confirm real
backends, exact weight revisions, terminal CoZ execution, per-scale decisions
and rollback state before raising the project completion level.

## Evidence and diagnostics

Artifacts are grouped under:

```text
artifacts/autodl/
  bootstrap/<attempt>/
  gpu-check/<attempt>/
  weight-download/<attempt>/
  smoke/<attempt>/
  integration/<attempt>/
  diagnostics/<attempt>/
```

Collect an allowlisted diagnostics bundle after either success or failure:

```bash
scripts/autodl/collect_diagnostics.sh
for checksum in artifacts/autodl/diagnostics/*/*.tar.gz.sha256; do
  (cd "$(dirname "$checksum")" && sha256sum -c "$(basename "$checksum")")
done
```

The collector includes textual run manifests/logs, Git identity and status,
package versions, disk information and NVIDIA evidence. For up to eight
successful CLI results it also follows a run directory only when it remains
inside the repository, collecting its bounded worker stdout/stderr and protocol
text. Set `SCALEGUARD_DIAGNOSTICS_MAX_MODEL_RUNS` to change that bound. It
excludes images, weights, archives, symlinks, large files and a full environment
dump. Credential values, including a nonstandard configured scheduler variable,
are made non-exported before any system probe and delivered only to the
sanitizer through a private file descriptor. Exact values, common token forms,
repository roots and cache roots are redacted. Direct `--input`/`--output`
paths are derived from execution receipts, and both text content and
archive-relative path names receive a final exact-value and pattern scan.
The 4KAgent overlay records only image byte count and SHA-256 in its chat log;
it does not put the source bytes there. Any copied text that nevertheless
contains a parameterized base64 image data URL is excluded from the archive.
Automated redaction is not infallible:
follow [external_gate/REDACTION.md](../external_gate/REDACTION.md) and inspect
the archive before transfer.

## One-command gate execution

After setting only authorized input paths and credential environment variables:

```bash
export SCALEGUARD_SMOKE_INPUT=/authorized-data/smoke.png
export SCALEGUARD_INTEGRATION_INPUT=/authorized-data/integration.png
external_gate/commands.sh
unset HF_TOKEN DASHSCOPE_API_KEY
```

The exact request, pass conditions and result record are in
`external_gate/REQUEST.md`, `expected_artifacts.json`, and
`RESULT_TEMPLATE.json`. Fill the result template from evidence; never pre-fill
GPU names, timings, memory peaks, metrics or completion levels. The gate wrapper
also attempts a redacted diagnostics collection on any failed stage while
preserving the original non-zero exit code.

## Failure recovery

Every attempt receives a new directory, so rerunning is safe and retains earlier
evidence. Keep the failed log and manifest, collect diagnostics, fix the
specific cause, then rerun the failed stage.

- GPU preflight failure: inspect `gpu_check.json`, `nvidia-smi.txt` and
  `disk.txt`; confirm `CUDA_VISIBLE_DEVICES=0,1`.
- Gated download failure: verify that the account accepted the exact model
  terms and that a token is exported; never print the token.
- Materialization failure: inspect `materialization-receipt.json`; do not
  extract the 4KAgent archive manually over an audited checkout or bypass an
  unsafe-member rejection.
- Revision error: replace a branch or tag with the audited 40-character model
  commit SHA.
- Upstream verification failure: restore the checkout specified by
  `upstream-lock.yaml`; do not silently run a different commit.
- Runtime re-audit failure: inspect the attempt log and the partial
  `runtime-environments/` receipts, rerun the locked bootstrap to repair
  distribution drift, and do not install packages over an audited environment.
- CUDA OOM: keep the raw failure, then change only the recorded tile/offload
  config. Do not lower the GPU preflight thresholds to hide an OOM.
- Command success without output: treat the run as failed and inspect the
  ScaleGuard run manifest and stderr.

No script deletes checkouts, caches, weights, outputs or failed evidence.
