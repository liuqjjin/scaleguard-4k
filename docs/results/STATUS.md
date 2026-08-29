# Project evidence status

Last reviewed: 2026-08-29

Highest supported completion level: **`RESEARCH_EVALUATED`**

ScaleGuard-4K has a locked CPU/mock contract path for development and CI, and
a completed dual-GPU research path. The research study ran on the declared
AutoDL host with two RTX 4090 GPUs: 4KAgent restoration, Chain-of-Zoom
generative super-resolution, trusted-scale control, threshold calibration, and
the paired four-group evaluation. Numerical results, uncertainty intervals, and
host-level GPU systems measurements are reported in the [README](../../README.md).

A CPU/mock manifest remains `STATIC_READY` and `mock: true`. That label is a
run-level contract for the fake backends, not the project evidence level.

## Completed evidence

| Evidence | Current state |
| --- | --- |
| Project environment | `uv.lock` exists; locked development installation succeeds locally |
| Static quality | Ruff, formatting, strict mypy, and the complete CPU test suite pass locally |
| Public CPU path | deterministic mock 4KAgent → trusted-scale controller → mock CoZ path is implemented |
| Mock provenance | manifests and derived artifacts explicitly record `mock: true` |
| Controller contracts | continue, stop, rollback, worker failure, and session failure paths have CPU tests |
| Remote scheduling contract | canonical DashScope endpoint/model/key binding, text-only requests, bounded retries, JSON validation, and redacted evidence have CPU contract tests |
| Upstream identity | 4KAgent and CoZ commits, root trees, ordered patches, and licenses are locked and audited |
| Runtime dependency identity | DepictQA is pinned as a 4KAgent transitive perception service |
| Weight identity | immutable revisions, known hashes, licenses, optional entries, and a manual gate are recorded |
| Dual-GPU runtime | dual-4090 preflight, bootstrap, weight materialization, smoke, integration, and diagnostics were executed on the declared host |
| Component reproduction | real 4KAgent restoration and real CoZ full-image 4× generation ran with locked environments, weights, and hardware evidence |
| Integrated path | non-mock 4KAgent → terminal CoZ runs retained fresh outputs, raw logs, and per-GPU sampling |
| Controller validation | accept, stop, and rollback behavior was exercised with a calibration receipt bound to the research runtime |
| Research evaluation | complete paired four-group 4× study and the 16× AB-fixed vs ScaleGuard comparison, with aggregate statistics and systems measurements |

Local CPU checks still demonstrate repository contracts only. They do not
replace the dual-GPU study reported in the README.

## Evidence ladder

| Level | Project status |
| --- | --- |
| `STATIC_READY` | supported for locked source, static checks, and deterministic CPU/mock runs |
| `COMPONENT_REPRODUCED` | completed for both 4KAgent and CoZ on the dual-4090 host |
| `AB_INTEGRATED` | completed for the terminal 4KAgent → CoZ path |
| `SCALEGUARD_VALIDATED` | completed with calibrated accept/stop/rollback evidence |
| `RESEARCH_EVALUATED` | completed; this is the current project level |

A run manifest containing a higher label is not enough by itself. Research
claims still require successful status, real backends, matching artifact
hashes, raw process/GPU evidence, and the review in
[reproduction.md](../reproduction.md).

## Numerical results

The published ScaleGuard GPU, runtime, VRAM, quality, fidelity, consistency,
and ablation numbers are in the README. Values in upstream papers, issues,
examples, fake-worker manifests, and preflight requirements are not ScaleGuard
results.

## Host and data prerequisites

Reproducing the dual-GPU study still requires:

1. a Linux dual-RTX-4090 AutoDL host with the declared disk capacity;
2. accepted gated Stable Diffusion 3 terms and private authentication;
3. a Beijing-region `DASHSCOPE_API_KEY`;
4. the required DepictQA degradation delta, for which the publisher
   supplies no digest;
5. authorized smoke/integration images; and
6. authorized, hashed calibration and evaluation data.

All non-secret commands and pass conditions are in
[external_gate/REQUEST.md](../../external_gate/REQUEST.md).

## Evidence-update rule

Update this file only in the same change that records or points to a reviewed,
immutable evidence set. Include failed cases and limitations. Never pre-fill
hardware, performance, metric, or completion fields from intended
configuration.
