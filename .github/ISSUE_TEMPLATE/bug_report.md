---
name: Runtime or contract bug
about: Report a reproducible ScaleGuard failure with redacted evidence
title: '[bug] '
labels: bug
assignees: ''
---

## Failure boundary

Name the failing phase: configuration, restoration, CoZ session, controller,
manifest verification, evaluation, or deployment wrapper. State whether the
failure is reproducible in CPU mock mode.

## Minimal reproduction

- ScaleGuard commit or version:
- Configuration profile and changed non-secret fields:
- Redacted command (replace local paths and data names):
- First failing step or exception class:
- Expected contract:
- Observed contract:

## Environment

- OS and architecture:
- Python and uv versions:
- Mode: CPU mock or real runtime
- For real runtime only: GPU model/count, driver, and CUDA runtime:

## Redacted evidence

Include the run ID, manifest schema/status, and the smallest relevant log or
validation excerpt. Do not attach raw manifests, logs, configurations, images,
datasets, or diagnostic archives. Remove credentials, private paths, hostnames,
signed URLs, image content, and dataset identifiers by following
[`external_gate/REDACTION.md`](https://github.com/liuqjjin/scaleguard-4k/blob/main/external_gate/REDACTION.md).

- [ ] I checked the excerpt for secrets and private or licensed data.
- [ ] The reproduction uses data I am authorized to disclose, or no data is attached.
