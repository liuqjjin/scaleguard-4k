# Evaluation utilities

These programs only consume completed ScaleGuard manifests. They never invoke a
restoration model, synthesize missing measurements, or turn mock runs into
research evidence.

Measure one or more aligned manifest/reference pairs:

```bash
uv run --locked python scripts/experiments/evaluate_metrics.py \
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
semantics.

Create `labels.csv` with exactly one row for every metric-bearing scale step:

```csv
run_id,step_index,acceptable
image-001-scaleguard,1,true
image-002-scaleguard,1,false
```

Then produce a deterministic receipt:

```bash
uv run --locked python scripts/experiments/calibrate_gates.py \
  --runs runs/calibration \
  --labels labels.csv \
  --output artifacts/calibration/receipt.json \
  --artifact-root "$PWD"
```

The default minimum is 20 acceptable, non-mock steps. Fewer samples still
produce an auditable `insufficient_data` receipt and exit with status 1; they do
not produce a valid calibration claim. Every labeled candidate and trusted
state must still exist and match the SHA256 recorded by its manifest.

Build a paired ablation table:

```bash
uv run --locked python scripts/experiments/summarize_ablation.py \
  --a-only runs/ablation/a-only \
  --b-only runs/ablation/b-only \
  --ab-fixed runs/ablation/ab-fixed \
  --scaleguard runs/ablation/scaleguard \
  --output-csv artifacts/ablation/paired.csv \
  --output-json artifacts/ablation/paired.json \
  --artifact-root "$PWD"
```

Rows are paired by the verified input-image SHA256. Missing groups and mock runs
remain in the outputs with explicit issue flags. The utility copies only
observed per-run metrics and deliberately computes no aggregate headline
numbers.
