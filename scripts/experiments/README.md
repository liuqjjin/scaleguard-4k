# Evaluation utilities

The metric, calibration, and summary programs only consume completed ScaleGuard
manifests. They never synthesize missing measurements or turn mock runs into
research evidence. `run_ablation.py` is the separate executable orchestration
entry point.

Plan or execute the paired four-group suite:

```bash
.venv/bin/python -I scripts/experiments/run_ablation.py \
  --base-config configs/runtime/autodl-2x4090.yaml \
  --input /authorized-data/evaluation/image-001.png \
  --seed 20250727 \
  --output-dir /root/autodl-tmp/scaleguard-4k/ablation/suite-001 \
  --plan-only
```

Remove `--plan-only` to execute. Every job goes through the fixed AutoDL
experiment wrapper and receives a fresh runtime preflight. The output directory
must be new or empty; failures remain in the atomic `suite-receipt.json` and do
not suppress later jobs.

Measure one or more aligned manifest/reference pairs:

```bash
uv run --locked python -I scripts/experiments/evaluate_metrics.py \
  --manifest runs/one/manifest.json \
  --reference data/references/one.png \
  --manifest runs/two/manifest.json \
  --reference data/references/two.png \
  --metric psnr \
  --metric ssim \
  --output artifacts/metrics/batch.json \
  --artifact-root "$PWD"
```

The default metrics are RGB PSNR and SSIM. There is no resize or color
conversion: output and reference must be aligned 8-bit RGB rasters. Optional
LPIPS, MUSIQ, and CLIPIQA use the locked PyIQA extra and require explicit local
weights. See [the evaluation protocol](../../docs/evaluation-protocol.md) for
the exact formulas, offline learned-metric flags, receipt schema, and exit
semantics. The receipt self-hash is only an integrity field: the summary reader
reopens all source evidence and recomputes built-in scores before aggregation.
Learned scores are used only when their hash-locked local weights are still
available and the score can be replayed offline.

Create `labels.csv` with exactly one row for every metric-bearing scale step:

```csv
run_id,step_index,acceptable
image-001-scaleguard,1,true
image-002-scaleguard,1,false
```

Then produce a deterministic receipt:

```bash
uv run --locked python -I scripts/experiments/calibrate_gates.py \
  --runs runs/calibration \
  --labels labels.csv \
  --output artifacts/calibration/receipt.json \
  --artifact-root "$PWD"
```

The default minimum is 20 acceptable, non-mock input-image clusters. Multiple
recursive steps or seeds for one image remain one bootstrap cluster. Fewer
clusters still produce an auditable `insufficient_data` receipt and exit with
status 1; they do not produce a valid calibration claim. Every labeled
candidate and trusted state must still exist and match the SHA256 recorded by
its fully validated manifest.
Runtime use and paired-summary review independently bind the exact receipt
path, size, SHA-256, semantics, backend, forward-model identity, and thresholds.

Build a paired ablation table:

```bash
uv run --locked python -I scripts/experiments/summarize_ablation.py \
  --a-only /root/autodl-tmp/scaleguard-4k/ablation/suite-001/jobs/a-only \
  --b-only /root/autodl-tmp/scaleguard-4k/ablation/suite-001/jobs/b-only \
  --ab-fixed /root/autodl-tmp/scaleguard-4k/ablation/suite-001/jobs/ab-fixed \
  --scaleguard /root/autodl-tmp/scaleguard-4k/ablation/suite-001/jobs/scaleguard \
  --suite-receipt /root/autodl-tmp/scaleguard-4k/ablation/suite-001/suite-receipt.json \
  --metric-receipt artifacts/metrics/full-reference.json \
  --metric-receipt artifacts/metrics/no-reference.json \
  --output-csv artifacts/ablation/paired.csv \
  --output-json artifacts/ablation/paired.json \
  --artifact-root "$PWD"
```

Rows are paired by the suite's deterministic experiment sample ID, which binds
the full verified input-image SHA-256 and seed. Missing groups and mock runs
remain in the outputs with explicit issue flags. A pair is never
research-eligible without an independently revalidated passed suite receipt
whose exact manifest paths, hashes, and hardware identities match the supplied
runs. It also revalidates every configured calibration receipt from one byte
snapshot; measurement evidence is required only for measurement-enabled runs.
The reader requires the suite's recorded clean project commit to remain
checked out and its original raw evidence paths to remain available. Omitting
the receipt still produces a diagnostic summary, but every pair is marked
`research_eligible: false`. The utility retains every observed or missing
per-run value and computes only the predeclared paired aggregates:
ScaleGuard-minus-baseline deltas, paired Cohen dz, input-cluster bootstrap 95%
intervals, missing rates, and systems summaries. Its JSON commit marker binds
the exact CSV bytes. Systems output separates manifest-validated CoZ
initialization/step timing and worker allocator peaks from independently
replayed, UUID-bound `gpu-samples.csv` host peaks; the latter are explicitly
non-process-attributed. Insufficient eligible input clusters produce an
unavailable interval rather than a fabricated number.

Every metric sample is joined by resolved manifest path, manifest SHA-256, and
run ID. Complementary metric sets for one manifest are merged; duplicate
metric names, source drift, and definition conflicts fail the summary. Missing
and unreplayable values remain explicit and are excluded from effects. A-only stays at native resolution: its 4× full-reference scores are
`not_applicable`, never produced by resizing or imputation. Omit
`--metric-receipt` to create a controller-only diagnostic summary with no
external metric effects.
