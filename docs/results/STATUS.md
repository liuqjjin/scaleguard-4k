# Project evidence status

Last reviewed: 2026-07-27

Highest supported completion level: **`STATIC_READY`**

ScaleGuard-4K has a tested CPU/mock implementation, immutable upstream and
weight metadata, deployment entry points, and evidence tooling. It has no
project-generated GPU result or real image-quality measurement. This file is
the authoritative claim boundary until a later evidence review updates it.

## What `STATIC_READY` means here

| Evidence | Current state |
| --- | --- |
| Project environment | `uv.lock` exists; locked development installation succeeds locally |
| Static quality | Ruff, formatting, strict mypy, and the complete CPU test suite pass locally |
| Public CPU path | deterministic mock 4KAgent → trusted-scale controller → mock CoZ path is implemented |
| Mock provenance | manifests and derived artifacts explicitly record `mock: true` |
| Controller contracts | continue, stop, rollback, worker failure, and session failure paths have CPU tests |
| Upstream identity | 4KAgent and CoZ commits, root trees, ordered patches, and licenses are locked and audited |
| Runtime dependency identity | DepictQA is pinned as a 4KAgent transitive perception service |
| Weight identity | immutable revisions, known hashes, licenses, optional entries, and a manual gate are recorded |
| Deployment preparation | dual-4090 preflight, fresh runtime re-attestation, bootstrap, weight, smoke, integration, diagnostics, and external-gate contracts exist |
| Evaluation preparation | hash-bound RGB PSNR/SSIM and offline PyIQA receipt harnesses, calibration receipts, and paired non-imputed summary tooling exist |

Local checks demonstrate repository contracts only. The configured AutoDL
hardware requirements and upstream paper numbers are not local measurements.

## Unsupported levels

| Level | Why it is not yet supported |
| --- | --- |
| `COMPONENT_REPRODUCED` | no retained real 4KAgent component output and no retained real CoZ full-image output with locked environment/weight/hardware evidence |
| `AB_INTEGRATED` | no reviewed non-mock 4KAgent → terminal CoZ run with fresh output, raw logs, and per-GPU sampling |
| `SCALEGUARD_VALIDATED` | no real accept/stop/rollback evidence and no valid calibration receipt bound to a real runtime config |
| `RESEARCH_EVALUATED` | no complete paired four-group study, authorized evaluation evidence, aggregate statistics, systems analysis, or failure analysis |

A run manifest containing a higher label is not enough by itself. Promotion
also requires successful status, real backends, matching artifact hashes, raw
process/GPU evidence, and the level-specific review in
[reproduction.md](../reproduction.md).

## Numerical results

There are currently **no ScaleGuard GPU, runtime, VRAM, quality, fidelity,
consistency, or ablation numbers to report**.

The repository intentionally contains no placeholder result table. Values in
upstream papers, issues, examples, runtime thresholds, fake-worker manifests,
and preflight requirements are not ScaleGuard results.

## External gates

The remaining user-owned prerequisites are:

1. provision a Linux dual-RTX-4090 AutoDL host with the declared disk capacity;
2. accept the gated Stable Diffusion 3 terms and authenticate privately;
3. provide the remote 4KAgent scheduler credential privately;
4. obtain the required DepictQA degradation delta, for which the publisher
   supplies no digest;
5. provide authorized smoke/integration images; and
6. provide authorized, hashed calibration and evaluation data.

All non-secret commands and pass conditions are prepared in
[external_gate/REQUEST.md](../../external_gate/REQUEST.md). Once access exists,
the real attempt must retain bootstrap, weight, materialization, execution,
manifest, GPU, log, and diagnostics evidence.

## Promotion checklist

The next possible promotion is `COMPONENT_REPRODUCED`. It requires, for both
4KAgent and CoZ independently:

- dispatched project commit and verified upstream lock;
- environment and package inventory;
- weight receipt and materialization receipt;
- exact input/output hashes;
- command, stdout, stderr, and exit status;
- physical GPU inventory and sampled memory; and
- an honest comparison with the upstream example, including failure or drift.

Only after both component records pass may the terminal integration attempt be
reviewed for `AB_INTEGRATED`.

Threshold calibration and the paired study follow
[evaluation-protocol.md](../evaluation-protocol.md); they do not block honest
component reproduction, but they do block controller-validation and research
claims.

## Evidence-update rule

Update this file only in the same change that records or points to a reviewed,
immutable evidence set. Include failed cases and limitations. Never pre-fill
hardware, performance, metric, or completion fields from intended
configuration.
