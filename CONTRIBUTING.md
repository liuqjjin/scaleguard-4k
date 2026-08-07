# Contributing to ScaleGuard-4K

ScaleGuard-4K welcomes focused contributions to restoration/SR integration,
trusted-scale control, reproducibility, evaluation, tests, and documentation.
The project is pre-release and currently `STATIC_READY`; contributions must not
turn planned GPU or research work into unsupported claims.

## Before starting

For a bug fix or small documentation change, open a focused pull request. For a
new public interface, upstream revision, metric, forward model, or architectural
change, start with an issue describing:

- the concrete problem;
- why the existing two-upstream design cannot handle it;
- source, issue, or experimental evidence;
- expected tests and migration impact; and
- license and model/data implications.

Report security-sensitive findings privately as described in
[SECURITY.md](SECURITY.md), not in a public issue.

## Scope

The core runtime is intentionally limited to:

- 4KAgent for degradation perception and native-scale restoration; and
- Chain-of-Zoom for terminal generative super-resolution.

AgenticIR is lineage/citation context only. DepictQA is a 4KAgent transitive
service. A proposal that adds another agent, VLM project, restoration project,
or SR project as a runtime fallback is out of scope for ScaleGuard-4K.

Normal libraries, testing tools, metrics, and declared observation operators
are not automatically “third upstreams,” but they still need a clear purpose,
maintenance cost, license review, and reproducible identity.

## Development setup

Use Python 3.10–3.14 and `uv`:

```bash
uv sync --locked --extra dev
uv run --locked scaleguard config validate configs/runtime/cpu-mock.yaml
uv run --locked pytest
```

The optional PyIQA extra has separate non-commercial licensing:

```bash
uv sync --locked --extra dev --extra metrics
```

Read [docs/development.md](docs/development.md) for repository structure,
invariants, adapter contracts, and evidence rules.

## Required checks

Before submitting:

```bash
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy src/scaleguard
uv run --locked pytest --cov=scaleguard --cov-report=term-missing -q
```

Add tests for behavior changes. Tests in normal CI must be deterministic,
CPU-only, and offline. Never make CI accept gated terms, download model
weights, contact a paid API, or require an NVIDIA device.

## Change guidelines

Keep changes small and direct:

- preserve one terminal generative SR phase;
- expose each CoZ transition as one explicit 4× candidate;
- keep quality, cross-scale, and measurement metrics separate;
- preserve failure, rollback, and post-color final-score evidence;
- use argument vectors and isolated working directories for subprocesses;
- reject unknown configuration and ambiguous artifacts;
- mark every fake-derived artifact `mock: true`; and
- avoid speculative abstractions or generated boilerplate.

An important deviation needs a short ADR in `docs/adr/`, relevant tests, and
updated architecture/limitations documentation.

## Upstream, weight, and data policy

Do not vendor full upstream repositories, model weights, or datasets.

For an upstream change:

- pin an immutable commit and root tree;
- review its license and current issues;
- keep patches minimal and SHA-256 locked;
- verify patches in order against a clean checkout; and
- update the audit and NOTICE when attribution or risk changes.

For a model or weight:

- use an immutable repository revision or expected content hash;
- retain the model's own license/gate metadata;
- keep manual acquisition explicit when account action is required; and
- never reinterpret a locally measured hash as a publisher digest.

For evaluation data, document authorization, exact files, hashes, split, and
preprocessing. Do not silently fill incomplete upstream datasets.

## Evidence and result policy

Pull requests must not include:

- fabricated GPU names, VRAM, runtime, metrics, or outputs;
- copied upstream paper numbers presented as project measurements;
- edited manifests or receipts that did not come from the recorded tool;
- mock results described as model evidence; or
- private paths, credentials, signed URLs, weights, or restricted data.

Failed runs are useful evidence. Preserve and explain them rather than
overwriting them.

Changes to `docs/results/STATUS.md` require the complete evidence for the new
level and must leave unsupported higher levels explicit.

## Pull request checklist

- [ ] The change has one clear purpose and stays within the two-core boundary.
- [ ] User-visible behavior and configuration are documented.
- [ ] New behavior and failure paths have tests.
- [ ] Ruff, format, mypy, tests, and coverage pass.
- [ ] No secret, private absolute path, model weight, or dataset was added.
- [ ] License/NOTICE/CITATION implications were reviewed.
- [ ] Mock and real evidence remain unambiguous.
- [ ] `CHANGELOG.md` describes the user-visible change.
- [ ] An ADR is included if an architectural invariant changed.

## Licensing contributions

Unless stated otherwise in the pull request and accepted by the maintainers,
contributions are submitted under the repository's Apache License 2.0. You must
have the right to contribute the code, tests, documentation, and assets. Do not
copy source from an upstream whose license is missing, incompatible, or
unclear.
