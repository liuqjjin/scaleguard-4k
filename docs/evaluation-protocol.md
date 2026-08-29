# Evaluation protocol

Status: the four-group dual-GPU study has been executed on the declared AutoDL
host. This document is the protocol; numerical results are reported in the
README.

This protocol separates controller calibration from final evaluation and keeps
all comparisons paired by exact input bytes.

## Research questions

The evaluation is designed to answer:

1. Does degradation-aware 4KAgent restoration improve the input before
   terminal SR?
2. Does a fixed 4KAgent → CoZ chain improve over either component alone?
3. Do explicit trusted-scale gates reduce harmful scale transitions without
   suppressing useful ones?
4. What is the systems cost of gating and persistent CoZ execution?
5. When a declared observation model exists, does measurement consistency add
   information beyond same-size quality and cross-scale checks?

## Data and splits

Use only authorized data. Before running a model, create a versioned split
manifest containing:

- dataset name, source, terms, and acquisition date;
- immutable file list and SHA-256 per image;
- input/reference pairing and any generated degradation parameters;
- calibration, evaluation, and failure-analysis membership;
- exclusions with reasons; and
- preprocessing, color space, bit depth, and resize kernel.

The calibration and evaluation splits must be disjoint. Do not tune thresholds
on the reported evaluation set. If a public upstream split is incomplete or
ambiguous, mark that dataset blocked rather than reconstructing the missing
members silently.

For a research-eligible paired summary, this separation is also checked in
code: every evaluation input SHA-256 is compared with the input identities in
the source-replayed calibration receipt, and any overlap excludes the pair.

## Paired experiment groups

`configs/experiments/ablation.yaml` declares four groups:

| Group | Path |
| --- | --- |
| A-only | 4KAgent native restoration without terminal CoZ |
| B-only | CoZ applied directly to the observed input |
| AB-fixed | 4KAgent restoration followed by a fixed CoZ step count |
| ScaleGuard | 4KAgent restoration followed by gated one-step CoZ states |

This document defines the executable protocol. Numerical results are reported
in the README. The runner requires a
clean, fixed Git HEAD and a real base configuration with
`controller.target_factor: 4`, at least one configured CoZ step, upstream
4KAgent, and persistent CoZ. It generates these exact group semantics:

| Group | 4KAgent | Target | CoZ steps | Acceptance |
| --- | --- | ---: | ---: | --- |
| A-only | upstream | 1 | 0 | fixed |
| B-only | identity observation | 4 | 1 | fixed |
| AB-fixed | upstream | 4 | 1 | fixed |
| ScaleGuard | upstream | 4 | 1 | trusted |

A-only is the native-resolution restoration-component baseline. Its output is
not secretly resized to 4x: full-reference metrics at the 4x target are marked
not applicable for that group.

### Coverage boundary of the 4x suite

Every group in this suite runs a single terminal transition
(`target_factor: 4`, `max_coz_steps: 1`). With one planned step the controller
reaches `stop` or `rollback` on the first candidate, so `continue` is not
exercised and no candidate is ever promoted and then re-evaluated at a higher
scale.

The 4× suite therefore supports conclusions about gating a single terminal
transition. Recursive scale control uses a separate 16× protocol
(`target_factor: 16`, `max_coz_steps: 2`), where the first candidate can be
promoted by `continue` and the second is judged against it. That 16× study
was executed on the same dual-GPU host; its reach rates, conditional quality,
and systems measurements are reported in the README.

Plan an authorized suite without starting a model:

```bash
.venv/bin/python -I scripts/experiments/run_ablation.py \
  --base-config configs/runtime/autodl-2x4090.yaml \
  --input /authorized-data/evaluation/image-001.png \
  --seed 20250727 \
  --output-dir /root/autodl-tmp/scaleguard-4k/ablation/plan-001 \
  --plan-only
```

Omit `--plan-only` to execute. Repeat `--input` and `--seed` to declare the
Cartesian product. Each input is copied once into a hash-named immutable suite
snapshot; the deterministic sample ID is `sha256[:16]-s<seed>` and is shared by
all four groups. Duplicate sample IDs, a non-empty output directory, unsafe
root destinations, a dirty or changed HEAD, and changed protocol, base config,
runner, input, config, manifest, or run artifacts fail the suite.

Every job directly invokes the fixed
`scripts/autodl/run_experiment.sh` wrapper. That wrapper creates a new runtime
preflight for the generated config. The orchestrator has no API-key option,
does not serialize credentials, and leaves credential scoping and redaction to
the wrapper.

`suite-receipt.json` is atomically replaced before and after every job. A
non-zero job or malformed evidence is retained and does not stop later jobs.
The receipt records exact input/config/runner/protocol hashes, argv and return
code, clean commits before and after execution, full manifest and run-artifact
inventory hashes, and the per-job runtime evidence digests. At suite completion
it fully revalidates each manifest and enforces within-sample equality of input
evidence, quality configuration, project commit, runtime execution binding,
environment installation identity, and stable weight/materialization/source
identity. Fresh preflight and environment-receipt file hashes are retained per
job but are not incorrectly required to match across jobs.

For every input, keep constant:

- source bytes and preprocessing;
- upstream commits, patches, runtime locks, and weights;
- CoZ seed per corresponding scale step;
- prompt type and semantic anchor policy;
- quality metric revision and preprocessing;
- target-factor policy;
- hardware class and measured software environment; and
- reference metrics and evaluation scripts.

Record group-specific differences explicitly. Do not “help” a missing group by
copying another output or imputing a metric.

## Gate definitions

### Same-resolution quality gain

For previous trusted state \(I_{t-1}\), candidate \(I_t\), and deterministic
bicubic baseline \(B_t\) resized to the candidate dimensions:

\[
\Delta Q_t = Q(I_t) - Q(B_t)
\]

Larger values are always treated as better; lower-is-better PyIQA metrics are
sign-inverted by the adapter. The quality gate is:

\[
\Delta Q_t \ge \tau_Q
\]

The CPU `gradient_proxy_v1` is not eligible for a research conclusion.

### Cross-scale consistency

The implementation low-pass filters and downsamples \(I_t\) to the dimensions
of \(I_{t-1}\), then records:

\[
E_\text{scale}^{\mathrm{rgb}} =
\frac{\operatorname{RMSE}(D(I_t), I_{t-1})}
{\operatorname{RMS}(I_{t-1}) + 10^{-6}}
\]

and a mean absolute horizontal/vertical gradient disagreement. Both must stay
below separately calibrated maxima. They are not added to quality gain.

### Measurement consistency

When the observation operator \(H_\theta\) is declared:

\[
E_\text{meas} =
\frac{\operatorname{RMSE}(H_\theta(I_t), I_\text{obs})}
{\operatorname{RMS}(I_\text{obs}) + 10^{-6}}
\]

Use only an operator justified by the data-generation process or a controlled
synthetic experiment. The implemented resize, Gaussian PSF, JPEG,
Poisson–Gaussian, and uniform-haze operators are simplified models. Do not
describe them as recovered camera physics.

## Calibration procedure

Human review produces one CSV row for every metric-bearing scale step:

```csv
run_id,step_index,acceptable
calibration-001,1,true
calibration-002,1,false
```

The default calibration algorithm uses acceptable, non-mock samples only. Its
minimum and resampling unit is one unique input-image SHA-256 cluster, not an
individual recursive scale step or a repeated seed:

- minimum acceptable input-image clusters: 20;
- quality lower quantile: 0.05;
- error upper quantile: 0.95;
- linear quantile estimator;
- 2,000 bootstrap samples;
- 95% bootstrap interval; and
- bootstrap seed `20250727`.

Generate the receipt from run directories:

```bash
uv run --locked python -I scripts/experiments/calibrate_gates.py \
  --runs runs/calibration \
  --labels artifacts/calibration/labels.csv \
  --output artifacts/calibration/receipt.json \
  --artifact-root "$PWD"
```

Use `--include-measurement` only when every accepted calibration sample has the
same declared measurement model and a measurement score.

The tool fully validates every source manifest, verifies artifact hashes and
complete label coverage, and rejects mock or ineligible run states. Fewer than
20 acceptable real input-image clusters write `status: insufficient_data` and
exit non-zero. Mixed evaluator identities are rejected.

Copy the resulting threshold values exactly into a dedicated runtime
configuration, set `metrics.calibration_receipt` to the receipt, and verify the
binding:

```bash
uv run --locked scaleguard evaluation verify \
  --receipt artifacts/calibration/receipt.json \
  --config configs/runtime/calibrated.yaml
```

Verification reopens the recorded labels and manifests, reruns calibration from
those exact bytes, and compares the rebuilt receipt. It also checks evaluator
and weight identity, complete forward-model parameters, cluster-bootstrap
settings, intervals, and exact threshold equality. A self-hash or a file merely
present at the configured path is not enough; run this verifier.

The controller repeats that verification when it is constructed and records
the resolved receipt path, size, SHA-256, and semantic result in the run
manifest. Measurement value plus canonical forward-model identity are required
exactly when measurement consistency is enabled and are forbidden when it is
disabled.

Quantile calibration defines an acceptable-sample envelope. It does not prove
optimal classification, causality, or human preference generalization. Report
sensitivity to quantiles, minimum sample count, seeds, datasets, and labeler
agreement.

## Final evaluation metrics

The declared protocol includes:

- full-reference: PSNR, SSIM, and LPIPS when aligned references exist;
- no-reference: MUSIQ and CLIPIQA;
- controller: quality gain, scale NRMSE, scale edge MAE, and optional
  measurement NRMSE;
- decisions: accept, stop, rollback, and failure rates; and
- systems: success rate, wall time, CoZ initialization versus first/steady-step
  time, worker allocator peaks, and host-level sampled memory/utilization for
  each preflight-bound physical GPU.

Research question 3 asks whether gating reduces *harmful* transitions. The
implemented decision metrics are `stop_rate` and `rollback_rate`, which count
what the controller decided, not whether each decision was correct. A harmful
transition is a candidate that a reviewer judges worse than its trusted input;
scoring one requires a per-candidate reviewer label, and the summary has no such
field today. The calibration corpus carries an `acceptable` label per scale
step, but it feeds threshold estimation and is not joined into the paired
summary. Until a labelled decision-outcome join exists, report the decision
rates as decision rates and do not present them as a harm reduction.

The metric harness consumes completed run manifests. PSNR, SSIM, and LPIPS
require one aligned reference per manifest; MUSIQ and CLIPIQA can run without a
reference. It fully validates each manifest and verifies the declared input,
final-image, reference, and weight bytes before measuring anything. A single
full-reference pair can be measured with:

```bash
uv run --locked scaleguard evaluation metrics \
  --manifest runs/example/manifest.json \
  --reference data/references/example.png \
  --output artifacts/metrics/example.json
```

Repeat `--manifest` and `--reference` in corresponding order for a
full-reference batch. The receipt retains one sample record per pair and does
not impute scores:

```bash
uv run --locked scaleguard evaluation metrics \
  --manifest runs/one/manifest.json \
  --reference data/references/one.png \
  --manifest runs/two/manifest.json \
  --reference data/references/two.png \
  --metric psnr \
  --metric ssim \
  --crop-border 4 \
  --output artifacts/metrics/batch.json
```

The built-in reference metrics have this fixed contract:

- decoded Pillow mode must be `RGB`, which is 8 bits per channel;
- stored RGB code values are divided by 255; no transfer-function
  linearization is applied;
- output and reference dimensions must match exactly; there is no resize;
- EXIF orientation must be absent or identity; the evaluator never rotates
  stored pixels implicitly;
- ICC profiles are not applied and their byte digests must match;
- the declared border is cropped from all four sides before reference metrics;
- PSNR is `10 log10(1 / MSE)` over all retained RGB samples, with exact matches
  represented by the JSON string `"infinity"`; and
- SSIM uses an 11×11 Gaussian window, sigma 1.5, valid support, `K1=0.01`,
  `K2=0.03`, data range 1, Gaussian population moments, and a mean over
  positions and RGB channels.

LPIPS, MUSIQ, and CLIPIQA are optional adapters to the locked PyIQA `0.1.16`
environment. Every requested learned metric requires an explicit local weight.
The fixed profiles are LPIPS v0.1 with AlexNet, MUSIQ KonIQ-10k, and vanilla
CLIPIQA with OpenAI RN50; they are recorded in every request/result.
The process blocks Python socket access and isolates the Torch/Hugging Face
caches during model construction and scoring; a missing or incompatible weight
produces a failed metric record rather than a download or value. LPIPS also
requires its local AlexNet checkpoint:

```bash
uv run --locked scaleguard evaluation metrics \
  --manifest runs/example/manifest.json \
  --reference data/references/example.png \
  --metric lpips \
  --metric musiq \
  --metric clipiqa \
  --pyiqa-weight lpips=weights/metrics/lpips/LPIPS_v0.1_alex-df73285e.pth \
  --pyiqa-backbone lpips=weights/metrics/user-provided/alexnet-owt-7be5be79.pth \
  --pyiqa-weight musiq=weights/metrics/pyiqa/musiq_koniq_ckpt-e95806b9.pth \
  --pyiqa-weight clipiqa=weights/metrics/clipiqa/RN50.pt \
  --device cuda:0 \
  --output artifacts/metrics/example-learned.json
```

CLIPIQA accepts only the OpenAI RN50 checkpoint whose SHA-256 is fixed by the
pinned PyIQA implementation. `download_weights.sh --include-optional` prepares
the content-addressed LPIPS linear-layer and CLIPIQA RN50 files. The separate
244 MB Torchvision AlexNet ImageNet backbone is not in the project lock because
the publisher URL exposes only a short hash prefix; supply it explicitly,
record its complete locally measured SHA-256 in the metric receipt, and review
its terms before the study. Learned metrics require `--crop-border 0` because
the adapter will not create unbound temporary crops. Every result records the
metric name, backend and version, native score direction, requested/observed
device when initialization succeeds, parameters, and local weight hashes.

The JSON output is atomically replaced and carries its own canonical
`receipt_sha256`, but that self-hash is not a trust decision. A clean receipt
exits 0. Missing weights, mock runs, failed
run status, metric exceptions, or invalid evidence remain in `issues`, produce
`completed_with_issues`, and exit 1. Counts distinguish measured, failed, and
not-run metrics. Invalid command structure exits 2.

The metric command does not aggregate across samples. Aggregate comparisons
come only from the paired-summary implementation described below. On
consumption it reopens the exact manifest, input, output, reference, and weight
paths, verifies their recorded hashes and the complete manifest contract, and
recomputes PSNR/SSIM. A learned score is admitted only when its recorded local
checkpoint remains available, hash-identical, and can be rerun offline with
the locked implementation. A missing learned-metric runtime remains visible as
`unverified`; its reported value is never used. Keeping these definitions
fixed is mandatory because upstream issue history documents PSNR/SSIM setting
mismatches.

## Paired summary

After all four manifests exist for each exact input, generate non-imputed
tables:

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

Rows are paired by `experiment_sample_id`, deterministically derived from the
verified full input SHA-256 and seed. The summary:

- verifies input and final-image hashes;
- reopens each configured calibration receipt as one byte snapshot and repeats
  its semantic verification against the manifest metric configuration;
- retains missing groups, failures, mock runs, and missing metrics as issues;
- independently revalidates the passed suite, every raw wrapper attempt,
  manifest path/hash binding, and within-sample hardware identity;
- maps each repeated `--metric-receipt` sample by the exact resolved manifest
  path, manifest SHA-256, and run ID; complementary metric sets for one
  manifest are merged, while a duplicate metric name, identity drift, or
  conflicting definition fails the summary;
- marks a pair research-eligible only when all four real successful groups are
  complete, issue-free, and exactly present in that verified suite receipt;
- records all source-manifest hashes; and
- computes the same predeclared ScaleGuard-minus-baseline paired effects for
  controller consistency and source-replayed PSNR, SSIM, LPIPS, MUSIQ, and
  CLIPIQA scores, with improvement-oriented deltas, paired Cohen dz,
  input-cluster bootstrap 95% intervals, exclusion rates, and per-group
  systems aggregates; and
- independently replays the wrapper's UUID-bound `gpu-samples.csv` against its
  execution summary, reports GPU UUIDs only as SHA-256 identities, and keeps
  those host-level peaks separate from worker allocator evidence.

External score states are explicit: `measured`, `missing`, `failed`,
`unverified`, or `not_applicable`. In particular, A-only is a native-resolution
restoration baseline. Its output is never resized or imputed for the 4×
full-reference comparison, so PSNR, SSIM, and LPIPS are recorded as
`not_applicable` for that comparison. No metric receipts are required to build
a diagnostic summary, but such a summary contains no external metric claim.

The aggregation unit is the unique input SHA-256 cluster. Runs or seeds from
the same input are averaged within that cluster before equal-weight bootstrap
resampling, preventing repeated scale steps or seeds from masquerading as
independent images. The JSON receipt binds the exact CSV bytes and is the commit
marker for the two-file summary. Statistics with insufficient eligible input
clusters remain explicitly unavailable; the command never invents a value.

The suite reader requires the recorded clean project commit to remain checked
out and every original raw evidence path to remain available. Omitting
`--suite-receipt` is allowed for diagnostic table generation, but every pair is
then explicitly marked `research_eligible: false`.

## Statistical reporting

The summary already implements paired effects, standardized effect sizes,
cluster-bootstrap intervals, missing rates, per-manifest completion indicators, wall time, CoZ
initialization/first/steady-step timing, worker allocator peaks, and replayed
host-level per-GPU aggregates. For a completed study, reporting must still:

- report sample counts before and after exclusions;
- use paired differences at the input level;
- provide confidence intervals and an effect-size definition;
- separate datasets and degradation regimes before any justified aggregate;
- report success, failure, stop, and rollback counts;
- include threshold and tile-size sensitivity;
- preserve outliers and representative failure cases;
- distinguish first-load from steady-state time; and
- report every physical GPU's sampled peak and sampling interval as host-level,
  non-process-attributed evidence, never as a component-specific VRAM figure.

Do not select the best seed after viewing evaluation outputs. If multiple seeds
are part of the protocol, declare them in advance and report their distribution.

The independently replayed suite reader currently promotes only an all-passed
suite. Its `success_rate` field is therefore a per-manifest diagnostic, not a
study-wide stability estimate. Failed jobs remain in the raw suite receipt and
must be reported separately until failed-job receipts have an independent
replay contract.

## Promotion criteria

`SCALEGUARD_VALIDATED` requires real, hash-verified manifests demonstrating the
intended decisions with a valid matching calibration receipt. One accepted
image is insufficient.

`RESEARCH_EVALUATED` additionally requires complete paired groups, implemented
and reviewed metric definitions, declared datasets and splits, ablations,
systems evidence, failures, and limitations. Generating the protocol files or
paired CSV does not meet that level by itself.

The published study meets `RESEARCH_EVALUATED`. Numerical results are in the
README. The authoritative current status is
[results/STATUS.md](results/STATUS.md).
