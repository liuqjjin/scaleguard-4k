# Development

ScaleGuard-4K favors a narrow, evidence-driven architecture over framework
growth. Contributions should strengthen the 4KAgent + CoZ boundary, trusted
scale control, reproducibility, or evaluation without adding another
algorithmic core.

## Setup

Use the uv version in `environments/uv.version` when changing locks or
bootstrap behavior. The current AutoDL contract installs uv 0.11.16 and
Python 3.10.18; routine CPU development remains supported on Python 3.10 or
newer. AutoDL users need a system `python3` with `venv`, not a preinstalled
exact uv.

```bash
uv sync --locked --extra dev
uv run --locked scaleguard config validate configs/runtime/cpu-mock.yaml
uv run --locked pytest
```

The optional metric environment is separate:

```bash
uv sync --locked --extra dev --extra metrics
```

Read [NOTICE](../NOTICE) before installing PyIQA or model weights.

## Repository map

```text
src/scaleguard/
  backends/       narrow 4KAgent, CoZ, command, and fake adapters
  controller/     factor policy and trusted-scale state machine
  evaluation/     hash-bound calibration and paired summaries
  imaging/        declared forward observation models
  metrics/        same-size quality and cross-scale checks
  runtime/        subprocess, service, and GPU-phase lifecycles
configs/          runtime contracts and executable experiment protocol
environments/     dependency, uv-binary, and managed-Python identities
third_party/      ScaleGuard overlays, patches, and ignored checkouts
scripts/          upstream, bootstrap, experiment, and AutoDL entry points
tests/            unit, contract, integration, and evaluation tests
docs/             decisions, protocols, deployment, status, and limits
```

## Non-negotiable invariants

- 4KAgent and Chain-of-Zoom are the only two core algorithmic upstreams.
- AgenticIR is citation/lineage context and is never a runtime checkout.
- DepictQA is a 4KAgent transitive perception service, not a third core.
- 4KAgent owns degradation planning and native-scale restoration.
- The outer pipeline contains one terminal generative SR phase.
- CoZ produces exactly one explicit 4× candidate per session request.
- A second 4× transition is requested only after accepting the first.
- Quality is compared at equal pixel dimensions.
- Cross-scale and measurement errors remain separate gates.
- Final color processing happens once, followed by final re-scoring.
- Mock artifacts are always marked mock and never support research claims.
- Completion levels are evidence labels, not aspirations.

Changes that intentionally alter an invariant need a concise ADR, source or
experimental evidence, updated tests, and updated public documentation.

## Checks before a pull request

Run the same classes of checks as CI:

```bash
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy src/scaleguard
uv run --locked pytest --cov=scaleguard --cov-report=term-missing -q
```

Coverage is branch-aware and has an 85% floor. Tests must remain CPU-only and
offline by default; do not make CI download weights, contact an API, or require
CUDA.

Also validate the public mock path when changing CLI, configuration, images,
controller state, or manifests:

```bash
uv run --locked python -I examples/make_fixture.py /tmp/scaleguard-dev-input.jpg
uv run --locked scaleguard run \
  --config configs/runtime/cpu-mock.yaml \
  --input /tmp/scaleguard-dev-input.jpg \
  --output /tmp/scaleguard-dev-output.png \
  --run-id dev-contract
uv run --locked scaleguard manifest validate runs/dev-contract/manifest.json
```

Use a fresh run id or remove only your own prior test artifacts outside a
reviewed evidence tree.

## Test layers

- **Unit tests** cover factor policy, metrics, decisions, images, process
  helpers, configuration, doctor checks, and upstream verification.
- **Contract tests** exercise argument vectors, isolated working directories,
  service lifetime, adapter artifacts, CoZ session operations, timeouts, and
  errors without loading real models.
- **Integration tests** run the complete public CLI with fake workers and
  inspect manifests and outputs.
- **Evaluation tests** reject missing labels, mock calibration, mixed backends,
  hash mismatches, incomplete ablation pairs, and forged receipts.
- **Environment tests** enforce the exact audited dependency exception and
  reject unrelated or changed package mismatches.
- **External GPU runs** are not CI. They use AutoDL wrappers and retained raw
  evidence.

When fixing a failure mode, add the smallest test that would have caught it at
the lowest appropriate layer.

## Configuration changes

Runtime YAML is strict: unknown sections and fields fail. Do not add environment
variable interpolation to model configs; it makes manifests ambiguous. Resolve
secrets outside YAML and pass them only through the process environment.
For AutoDL changes, preserve the stage boundary: no credentials during
bootstrap/source verification, Hugging Face auth only in the downloader child,
a non-secret presence marker in doctor, and the configured scheduler variable
only in the model child.
Keep the [annotated configuration reference](configuration.md) synchronized
with defaults, accepted values, path rules, and cross-field constraints.

When adding a field:

1. add it to the correct frozen dataclass;
2. validate its exact type, domain, and cross-field requirements;
3. ensure paths serialize into the manifest;
4. update doctor/readiness behavior;
5. add positive and negative tests; and
6. update installation, architecture, or evaluation documentation as needed.

Checked-in research thresholds must not be described as calibrated until a
valid receipt matches their exact values and backend.

## Adapter and process changes

Keep backend interfaces narrow:

```text
4KAgent: restore(source, destination, bridge_factor, run_dir)
CoZ: session() → upscale_once / accept / rollback
```

External commands must:

- use argument arrays without a shell;
- use a private working directory;
- normalize and validate input/output artifacts;
- preserve stdout and stderr;
- have deadlines and process-group termination;
- redact credential-bearing arguments;
- record process and available GPU evidence; and
- fail closed on ambiguous output or protocol state.

Do not import 4KAgent or CoZ into the ScaleGuard core environment. Their
overlays execute inside isolated upstream environments.

## Upstream and dependency updates

Never edit a checkout and then update a lock to match accidentally. For a
deliberate update:

1. review the upstream commit, root tree, release state, license, and relevant
   issues;
2. test existing patches with `git apply --check`;
3. keep patches minimal and content-hashed;
4. update `upstream-lock.yaml` or `runtime-dependencies.yaml`;
5. materialize both locks in a clean temporary root, using the `repositories`
   and `dependencies` mappings respectively;
6. run upstream-verifier unit tests and the CPU suite;
7. update the upstream audit and NOTICE if the boundary changed; and
8. record an ADR for a material architectural deviation.

Do not copy an unlicensed or ambiguously licensed upstream file into this
repository. Do not turn a normal library, metric, or 4KAgent transitive
dependency into a third runtime project.

## Runtime environment lock changes

The AutoDL hook targets Linux `x86_64` with glibc 2.28 or newer and installs
uv 0.11.16 plus uv-managed Python 3.10.18. Its host prerequisite is a system
`python3` with `venv`. The hook never trusts a same-version uv from `PATH`: it
clears `.runtime/bootstrap-uv`, installs the hash-pinned wheel, and checks the
committed executable SHA-256. It then reinstalls the committed
Python-build-standalone archive and creates `.venv` plus three isolated runtime
environments under `.runtime/envs/`: `4kagent`, `depictqa`, and `coz`.

Treat the files under `environments/` as a reviewed set:

- `requirements.lock` files declare the intended direct runtime inputs;
- `requirements.resolved.lock` files contain the complete hashed solutions;
- `uv.version` fixes the resolver/installer identity;
- `bootstrap/uv.lock` pins the Linux `x86_64` uv wheel used for self-bootstrap;
- `bootstrap/uv-binary.sha256` pins the executable extracted from that wheel;
- `python-downloads.json` pins the uv-managed CPython archive URL, build, and
  digest; and
- `4kagent/{pyiqa,hpsv2}.override.lock` contain exact wheel hashes for the
  audited upstream inference exceptions.

4KAgent uses PyIQA 0.1.13 and HPSv2 1.2.0 with a security-updated inference
stack. The audit accepts precisely the four metadata observations in
[ADR 0005](adr/0005-audited-inference-metadata-overrides.md) and records them
in `.runtime/receipts/4kagent.json` with status
`passed_with_audited_override`. ScaleGuard's `.venv` uses its separate PyIQA
0.1.16 extra. Do not generalize the exceptions, suppress dependency auditing,
or add another `--no-deps` install. Any change to an exact observation requires
updating the locks, tests, import contract, and ADR.

The only accepted source-compatibility shims are the three narrow inference
adapters in
[ADR 0008](adr/0008-minimal-inference-compatibility-shims.md). Do not broaden
them into a general monkey-patch layer or install DepictQA's training-only
DeepSpeed path.

After a lock change, run the environment-audit unit tests and exercise the
public privileged entry `scripts/autodl/bootstrap.sh` on the declared platform.
Do not invoke the source-only `scripts/bootstrap/autodl.sh` directly. Review
`.runtime/receipts/bootstrap.json` and all four environment receipts. A
successful receipt validates installation identity only; it does not promote
the project evidence level or prove GPU model behavior.

The real-run wrapper must also rerun all four audits with isolated interpreters
and bind the fresh receipts into schema-v2 runtime preflight evidence. Do not
replace this check with a historical bootstrap hash or a partial `pip freeze`;
the complete distribution map and declared import/entrypoint probes are the
runtime contract.

Run `bash scripts/security/audit_runtime_locks.sh` before review. The two exact
unfixed advisories and their isolation boundaries are documented in
[ADR 0007](adr/0007-isolate-unpatched-legacy-dependencies.md); do not add a
scanner ignore without an equally narrow runtime mitigation and ADR update.

The aggregate receipt is the authority over partial receipts. It starts as
`running`, becomes `passed` only after all checks, and is rewritten to `failed`
with a return code after an ordinary hook error. Never derive a claim from a
missing, `running`, or `failed` aggregate, even when one or more individual
environment receipts exist.

## Weights and data

No model weight or evaluation dataset belongs in Git. Weight manifests require
immutable revisions or expected hashes. Manual artifacts without publisher
digests must remain explicit external gates and receive measured-hash receipts.

Dataset changes require an authorized source, license/terms record, immutable
file list, split manifest, and preprocessing definition. Never replace a
missing upstream split silently.

## Evidence and result changes

Do not hand-edit run manifests, receipts, paired summaries, or GPU result
templates to make a check pass. Preserve failed attempts.

Before changing `docs/results/STATUS.md`:

- identify the exact completion-level definition;
- link or name the retained evidence set;
- verify all hashes and non-mock flags;
- review raw logs and failures;
- confirm no metric is copied from an upstream paper; and
- keep unsupported higher levels explicitly pending.

Numerical tables need executable metric definitions and raw artifacts.

## Documentation and release hygiene

- Use relative repository links and portable commands.
- Do not commit private absolute paths, hostnames, usernames, tokens, signed
  URLs, or account identifiers.
- Distinguish implemented, tested, externally blocked, and planned behavior.
- Update [CHANGELOG](../CHANGELOG.md) for user-visible changes.
- Update [limitations](limitations.md) when a new risk is discovered.
- Follow [SECURITY](../SECURITY.md) for sensitive reports.

Release preparation also requires a clean tree, passing locked CI, valid
`CITATION.cff`, current NOTICE/license boundaries, and a status file containing
only reviewed evidence.
