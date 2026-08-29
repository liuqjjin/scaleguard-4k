# ADR 0008: Keep security-updated inference compatibility in narrow shims

- Status: Accepted
- Date: 2026-07-27

## Context

The pinned upstreams predate the security-updated inference stack. Three import
assumptions no longer hold:

- CLIB-FIQA and DepictQA import `packaging` through the removed
  `pkg_resources` compatibility namespace.
- BasicSR imports `rgb_to_grayscale` from
  `torchvision.transforms.functional_tensor`, removed by Torchvision 0.25.
- importing DepictQA's `model` package eagerly imports its training-only
  DeepSpeed agent even when the service imports only `model.depictqa`.

Downgrading Setuptools, Torchvision, or Transformers would undo the reviewed
security upgrade. Installing DeepSpeed in the inference service would add a
large unused execution surface.

## Decision

ScaleGuard applies three process-local shims before importing upstream code:

1. when genuine `pkg_resources` is absent, expose only the separately pinned
   `packaging` module under that name;
2. for Torchvision 0.25 only, expose only `rgb_to_grayscale` at the legacy
   BasicSR module path; and
3. create a DepictQA `model` package namespace rooted at the pinned source
   directory without executing its training-only `__init__.py`.

The shims do not edit either checkout and do not replace restoration, quality,
VLM, or SR logic. Unexpected versions, early imports, missing symbols, and
unsafe paths fail closed.

The CLIB-FIQA/HPS vocabulary is copied into the run-local 4KAgent view only
after its SHA-256 matches
`924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a`.
Bootstrap performs offline symbol-import contracts in every installed
environment and records the exact checks in its receipts.

## Consequences

- Security-updated dependencies remain isolated and reproducible without
  carrying training-only DeepSpeed into inference.
- A dependency upgrade that restores or changes any affected API must remove
  or deliberately revise the corresponding shim, tests, locks, and this ADR.
- Passing import contracts establishes compatibility only. Checkpoint loading,
  CUDA execution, numerical quality, and VRAM are recorded by the dual-GPU
  study, not by this ADR.
