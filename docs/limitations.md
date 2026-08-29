# Limitations

ScaleGuard-4K is a research-engineering project. Its current evidence level is
`RESEARCH_EVALUATED`: the dual-GPU study on the declared AutoDL host is
complete, and the CPU/mock contracts remain available for development and CI.
The limitations below are part of that completed result.

## Restricted algorithm scope

The system intentionally uses only 4KAgent and Chain-of-Zoom as core
algorithmic upstreams. This keeps attribution and causality tractable but means
ScaleGuard does not compare or fall back to another restoration, SR, agent, or
VLM project at runtime.

DepictQA is present only as a transitive perception service required by the
selected 4KAgent profile. AgenticIR is lineage context only. The project does
not claim to reproduce AgenticIR or its paper's internal DepictQA behavior.

## Discrete scale policy

Supported requested factors are 1×, 2×, 4×, 8×, and 16×. At most two CoZ 4×
steps are allowed. There is no arbitrary target-dimension optimizer, fractional
scale policy, or unbounded recursive zoom.

The 2× and 8× paths use 4KAgent's existing 2× bridge. The audited allowlist
admits `swinir_2x_gan` and `swinir_2x_psnr`, and the locked profile expresses a
perception preference, so the selected bridge tool may be perception-oriented
rather than fidelity-preserving. The bridge and CoZ may also differ in
fidelity/perception behavior; both must be evaluated, not assumed equivalent
across factors.

## Quality gate and the CPU proxy

The gradient proxy is designed only for deterministic CPU plumbing. It can
reward noise or sharpening and has no validated relationship to human
preference, identity preservation, or hallucination risk. Research runs use the
versioned PyIQA MUSIQ/CLIPIQA path, subject to its separate non-commercial
licenses.

PyIQA support fixes package and metric identity and normalizes score direction,
but availability alone does not validate a threshold. The research study binds
a disjoint, labeled calibration split. Checked-in AutoDL operational defaults
are not a substitute for that receipt.

No single no-reference IQA metric is sufficient to establish faithful
restoration. Domain shift, text, faces, scientific images, and structured
artifacts need separate failure analysis.

## Cross-scale checks are low-level

Current cross-scale consistency uses low-pass RGB reconstruction error and
gradient disagreement. It can detect large color or structural drift, but it
does not explicitly measure:

- text correctness;
- face or biometric identity;
- object count and geometry;
- semantic prompt drift;
- uncertainty; or
- localized artifacts hidden by image-wide averages.

It is a fidelity floor, not a proof that new high-frequency detail is true.

## Observation models are simplified and declared

Resize, Gaussian blur, JPEG, Poisson–Gaussian noise, and uniform haze operators
support controlled experiments. They are not blind estimators of an unknown
camera, microscope, satellite, or medical acquisition pipeline.

Measurement consistency is meaningful only when the configured operator and
parameters match the data-generation protocol. Without that evidence, the
project should be described as low-level vision and generative SR engineering,
not physically validated computational imaging.

## Terminal generation can hallucinate

CoZ is a generative diffusion SR system guided by VLM prompts. It can synthesize
plausible detail that is absent or contradicted by the observation. Gating can
reject some inconsistent candidates, but it cannot prove provenance of every
pixel or recover information that was never observed.

Do not use outputs as sole evidence in medicine, forensics, remote-sensing
measurement, archival recovery, or other consequential decisions.

## Memory still scales with image area

The CoZ patch streams processed latent tiles into Gaussian accumulators instead
of retaining every output tile. Full latent and accumulation tensors remain
resident. Memory is therefore not independent of resolution.

The pinned full-image implementation places the transformer and VAE in FP32 on
its second visible device. ScaleGuard therefore rejects a contradictory
precision label and records requested precision alongside actual component
placement. Measured host-level peak VRAM and wall time for the dual-4090 study
are reported in the README; those samples are not a guarantee of an OOM-free
resolution on a different host, tile setting, or image size.

## Persistent-worker caveats

Persistent CoZ avoids mandatory reloads between accepted scales, but it also
keeps heavyweight model state resident. The protocol has health, timeout,
accept, rollback, close, and process-group termination contracts. Dual-GPU
runs record:

- cleanup after ordinary completion and failure;
- initialization versus per-step duration;
- host-level GPU sampling across the execution window; and
- worker-reported allocator peaks, kept separate from host samples.

They do not eliminate allocator fragmentation, CUDA-kernel nondeterminism, or
numerical equivalence to an unpatched or one-shot reference.

The online PyIQA gate runs on CPU while CoZ owns the two configured GPUs.
Learned metrics may use a GPU only as a separate offline evaluation after the
run. This avoids hidden CUDA residency at the cost of added controller latency.

## Upstream variability and reproducibility

Planning can vary with remote model/API behavior, environment, tool availability,
and metric versions. The canonical scheduler is a dated Qwen snapshot on
DashScope with thinking disabled and temperature zero, but the service is still
not mathematically deterministic and dated snapshots have finite lifetimes. Its
request contains task labels inferred from the authorized image, so it is not a
zero-disclosure path even though image bytes remain local. The upstream issue
history also records
evaluation-setting and dataset-completeness concerns. CoZ device mapping may
vary with the installed Transformers/Accelerate stack.

Seeds, source commits, model revisions, and dependency locks reduce variation;
they do not eliminate nondeterministic CUDA kernels or VLM generation. Actual
plans, CoZ generation prompts, redacted remote-scheduler metadata, logs,
versions, and outputs must be retained; raw scheduler prompts are intentionally
excluded from receipts.

The required DepictQA degradation delta is a manual Google Drive artifact with
no publisher digest. A ScaleGuard receipt can bind the exact local bytes after
acquisition but cannot authenticate them against an unavailable upstream hash.

## Evaluation harness scope

Calibration, paired-manifest summarization, and hash-bound PSNR, SSIM, LPIPS,
MUSIQ, and CLIPIQA execution are implemented and were used for the published
study. The learned metrics are optional PyIQA adapters: they require explicit
local weights and block implicit network access.

The declared ablation protocol and four-group evidence orchestrator are
executable. Paired effects, input-cluster bootstrap intervals, CoZ
initialization/step timing, and replayed host-level GPU sampling summaries are
part of the published report. A suite or metric receipt still demonstrates
execution and provenance in addition to the reported statistics. Host GPU
samples are explicitly not process-attributed and must not be presented as a
component's allocator peak.

A valid receipt needs at least the configured number of acceptable real
samples, but that minimum is not a guarantee of statistical power or population
coverage. Human-label guidelines and inter-rater agreement still need a
declared study design.

## Color and image representation

Inputs and worker outputs are normalized to RGB PNG. This is convenient for
hashing and contracts but can discard source metadata, alpha, higher bit depth,
raw sensor values, ICC profiles, or domain-specific channels.

The optional final AdaIN operation may change color statistics. ScaleGuard
re-scores the final bytes, but color correction itself is not validated for all
domains.

## Security and privacy

Images, prompts, tool logs, and model services may expose sensitive content.
The diagnostics collector is allowlisted and redacts common secret patterns,
but automated redaction is not a guarantee. Manual review is required before
sharing.

The project does not sandbox upstream model code. A pinned hash establishes
identity, not safety. Run models and untrusted images in an isolated account or
host with least privilege and no unnecessary credentials.

## License and distribution limits

ScaleGuard's original source is Apache-2.0, but the complete runtime is not
uniformly Apache-licensed. Stable Diffusion 3, Qwen, Vicuna, PyIQA, MUSIQ
weights, embedded 4KAgent tools, and other weights have separate terms,
including non-commercial restrictions.

The repository makes no claim that the complete research runtime, its outputs,
or every upstream checkpoint is commercially usable. A digest verifies content
identity; it does not grant rights. See [NOTICE](../NOTICE).

## Status consequence

These limitations remain after `RESEARCH_EVALUATED`. They constrain how the
dual-GPU results should be interpreted; they do not retract the completed
study. Reproduction steps are listed in
[reproduction.md](reproduction.md),
[evaluation-protocol.md](evaluation-protocol.md), and
[results/STATUS.md](results/STATUS.md).
