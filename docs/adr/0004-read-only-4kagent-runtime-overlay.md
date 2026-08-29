# ADR 0004: Run a minimal 4KAgent toolbox through a read-only union view

- Status: Accepted
- Date: 2026-07-27

## Context

The pinned 4KAgent checkout assumes that its repository is also a mutable
runtime directory. Installation scripts copy a large toolbox into that tree,
several modules derive paths from module-level repository globals, and the
default profiles expose tools whose Conda environments are outside the
ScaleGuard runtime boundary. Its scheduler also parses structured model output
with `eval`, may trigger implicit face-model downloads, and evaluates a
mutable bundle of PyIQA metrics.

Editing the audited checkout or installing every upstream research tool would
weaken provenance and exceed the two-GPU deployment's controlled surface.

## Decision

ScaleGuard keeps the locked checkout unchanged. For each 4KAgent attempt, its
overlay constructs a run-local union view from the audited source tree and the
materialized toolbox. A collision between non-identical files fails closed.
Only a dependency-complete subset of upstream tools that runs in the locked
4KAgent base environment is visible:

- SwinIR, MPRNet, Restormer, DehazeFormer, and FBCNN restoration entries;
- the three upstream classical brightening entries; and
- at most one existing controlled SwinIR 2× bridge when the scale policy asks
  for it.

Generative 4×/16× SR, special-environment tools, face restoration, and
old-photo restoration are removed from scheduling. CoZ remains the only
generative SR runtime and stays terminal.

The overlay redirects upstream path globals to the union view, invokes only
the selected tool modules through argument vectors without a shell, disables
implicit model downloads, and replaces scheduler response parsing with
`ast.literal_eval` plus schema normalization. The local Qwen model and HPSv2
root are fixed by configuration. The upstream toolbox's audited BPE vocabulary
is hash-checked and copied only into the two run-local import locations needed
by CLIB-FIQA and HPSv2; neither the checkout nor site-packages is modified.
HPSv2 receives the locked v2.1 checkpoint explicitly, and its redundant LAION
pretraining request is suppressed before the strict HPS state-dict load so
reflection cannot fall back to an implicit download. Reflection uses one
locked MUSIQ checkpoint rather than a mutable metric bundle. The scheduler
credential is read only from the configured environment variable and is never
written to an argument, file, or evidence record. ADR 0012 later replaces the
provider-specific request loop with a bounded, provider-bound transport.

ScaleGuard still owns the sole final AdaIN color-alignment step.

## Consequences

- Source provenance can be verified before and after every upstream attempt.
- The executable toolbox is smaller than the upstream installation archive;
  this is an intentional deployment profile, not a claim that every 4KAgent
  experiment is reproduced.
- The remote scheduler remains an external service. A dated snapshot and
  request receipt reduce drift but cannot guarantee server-side determinism.
- Model imports and GPU behavior are validated on the dual-GPU host; this
  ADR records the overlay contract, not those measurements.

## Evidence

Contract tests inspect argument construction, path isolation, service
lifecycle, clean-checkout enforcement, run-local BPE binding, explicit HPS
checkpoint use, and the absence of remote pretraining fallback. The overlay
compiles locally. Those checks do not replace the dual-GPU study.
