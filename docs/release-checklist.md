# Release checklist

ScaleGuard-4K is currently a `0.1.0.dev0` pre-release with an evidence level of
`STATIC_READY`. This checklist makes the source package and its public claims
reproducible; it does not promote GPU or research evidence. The authoritative
claim boundary is [results/STATUS.md](results/STATUS.md).

## 1. Freeze the release candidate

- Start from a reviewed, committed revision with no unintended working-tree
  changes.
- Confirm that 4KAgent and Chain-of-Zoom remain the only core algorithm
  upstreams. Keep AgenticIR as lineage only and DepictQA as a pinned 4KAgent
  transitive runtime dependency.
- Review version, date, and public metadata in `pyproject.toml`, `CHANGELOG.md`,
  `CITATION.cff`, `NOTICE`, `SECURITY.md`, and this status document. Keep the
  package version identical in `src/scaleguard/_version.py` and in the
  environment expectations in `src/scaleguard/provenance.py`,
  `scripts/autodl/_common.sh`, and `scripts/bootstrap/autodl.sh`.
- Keep the canonical repository URL, issue tracker, and source metadata bound
  to `https://github.com/liuqjjin/scaleguard-4k` in `pyproject.toml` and
  `CITATION.cff`.
- Verify that `uv.lock`, `upstream-lock.yaml`, `runtime-dependencies.yaml`,
  `weights-lock.json`, `environments/python-downloads.json`,
  `environments/bootstrap/uv-binary.sha256`, patches, overlays, and environment
  locks are part of the candidate revision.
- Ensure that credentials, datasets, model weights, upstream checkouts, local
  environments, run outputs, receipts, diagnostics, and private absolute paths
  are absent from the candidate.

Do not convert the development version into a numbered release until its scope,
compatibility, changelog entry, and tag are final.

## 2. Run the locked source checks

Use uv 0.11.16, as recorded in `environments/uv.version`:

```bash
test "$(uv --version | awk '{print $2}')" = "$(cat environments/uv.version)"
uv sync --locked --extra dev
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy src/scaleguard

pycache="$(mktemp -d)"
PYTHONPYCACHEPREFIX="$pycache" \
  uv run --locked python -I -m compileall -q \
    src scripts examples tests third_party/overlays

find scripts external_gate -type f -name '*.sh' -print0 | xargs -0 shellcheck
bash scripts/security/audit_runtime_locks.sh

CUDA_VISIBLE_DEVICES="" \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
SCALEGUARD_TEST_MODE=cpu \
  uv run --locked python -I -m pytest \
    --cov=scaleguard --cov-report=term-missing -q

bash scripts/run_cpu_demo.sh
```

Inspect the demo's printed temporary directory. Its manifest must be
`STATIC_READY`, `mock: true`, and successful; its final artifact must also be
mock-labelled and hash-valid. The demo must not create `runs/`, `outputs/`, or
other generated evidence in the repository.

Run the same checks on every supported Python version in CI. A failed,
network-dependent, or locally modified check is not a release pass.

## 3. Build and inspect distributions

Build outside the repository so stale files cannot enter the result:

```bash
release_dist="$(mktemp -d)"
uv build --no-build-isolation --out-dir "$release_dist"
find "$release_dist" -maxdepth 1 -type f -print

wheel="$(find "$release_dist" -maxdepth 1 -type f -name '*.whl' -print -quit)"
unzip -Z1 "$wheel" > "$release_dist/wheel-files.txt"
grep -Eq '^scaleguard_4k-[^/]+\.dist-info/licenses/third_party/licenses/Chain-of-Zoom-MIT\.txt$' \
  "$release_dist/wheel-files.txt"
```

`--no-build-isolation` is intentional: the locked development environment
contains the exact Hatchling version declared by `build-system`, so the release
build does not resolve a newer backend outside `uv.lock`.

Require one source distribution and one wheel. Inspect both archives and reject
the candidate if either contains weights, upstream checkouts, run artifacts,
credentials, caches, diagnostics, or undeclared generated files. Install the
wheel into a fresh environment, change to a temporary directory outside the
repository, and unset `SCALEGUARD_PROJECT_ROOT`. The archive assertion above
protects the Chain-of-Zoom MIT notice referenced by `NOTICE`; it must remain
under the wheel's standard `.dist-info/licenses` directory.

```bash
release_smoke="$(mktemp -d)"
uv venv "$release_smoke/venv"
uv pip install --python "$release_smoke/venv/bin/python" "$wheel"
cd "$release_smoke"
unset SCALEGUARD_PROJECT_ROOT
"$release_smoke/venv/bin/scaleguard" --version
"$release_smoke/venv/bin/scaleguard" config validate \
  /path/to/repository/configs/runtime/cpu-mock.yaml
"$release_smoke/venv/bin/scaleguard" manifest validate \
  /absolute/path/to/valid/manifest.json
"$release_smoke/venv/bin/scaleguard" evaluation verify \
  --receipt /absolute/path/to/valid/calibration-receipt.json \
  --config /absolute/path/to/runtime-config.yaml
```

All three file-validation commands must complete without locating the source
tree. Keep their inputs outside the installed package and use absolute paths.

Record SHA-256 digests for the exact distributions selected for publication.
Do not rebuild after recording them.

## 4. Review documentation and claims

- Check every README and documentation link from the built revision.
- Before uploading to PyPI, replace or render every repository-relative link in
  the package long description as an absolute URL under the selected canonical
  GitHub repository. Render the exact built `METADATA` with `readme-renderer`
  and reject any `pypi.org/...` resolution of a repository file. The canonical
  source URL is `https://github.com/liuqjjin/scaleguard-4k`.
- Confirm that the README and [results/STATUS.md](results/STATUS.md) name the
  same highest evidence level.
- Keep mock output visibly labelled and exclude it from quality, latency, VRAM,
  or research-result tables.
- Do not copy upstream paper numbers into ScaleGuard result fields.
- Keep unsupported evidence levels and external gates explicit.
- Review licensing and redistribution terms for code, weights, data, metrics,
  example inputs, and any media included in release notes.

Raising the evidence level requires the immutable records and review described
in [reproduction.md](reproduction.md). A successful package build or GPU
process exit is not sufficient.

## 5. Publish or stop

For a numbered release, require a configured canonical remote, tag the exact
reviewed commit, publish only the digest-matched distributions, attach concise
release notes, and link the evidence status. Verify the public archive and
installed CLI after publication.

Stop the release if any lock, archive, hash, manifest, license, mock label, or
public claim is ambiguous. Preserve failed real-runtime attempts; never repair
their evidence by hand.
