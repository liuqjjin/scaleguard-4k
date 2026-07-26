# ADR 0006: Fail closed at PyTorch checkpoint boundaries

- Status: Accepted
- Date: 2026-07-27

## Context

4KAgent uses DepictQA as a local degradation evaluator. Its locked launch
script loads a manually acquired PyTorch delta checkpoint and starts a Flask
development server on every interface with debug mode enabled. The source
publisher does not provide a digest for the Google Drive checkpoint.

PyTorch security advisory GHSA-53q9-r3pm-6pq6 identified remote code execution
in `torch.load(..., weights_only=True)` through 2.5.1. A later advisory,
GHSA-63cw-57p8-fm3p / CVE-2026-24747, covers every release through 2.9.1 and
names 2.10.0 as the first patched line. A local content receipt detects later
mutation but cannot authenticate bytes that were already malicious when
acquired.

## Decision

All three isolated GPU environments use the official PyTorch 2.10.0 CUDA 12.6
wheel family, torchvision 0.25.0, and Triton 3.6.0; 4KAgent additionally uses
torchaudio 2.10.0. Before importing the locked 4KAgent, DepictQA, or
Chain-of-Zoom runtime, each owned overlay rejects older PyTorch versions and
wraps every `torch.load` call so
`weights_only=True` cannot be disabled. The DepictQA overlay also overrides
the upstream Flask launch arguments to bind only
`127.0.0.1:5001`, with debug and the reloader disabled. Production
configuration cannot select a different DepictQA endpoint.

The upstream checkout remains unchanged. The external weight receipt and
materialization inventory remain mandatory because safe deserialization does
not establish model provenance or scientific equivalence.

## Consequences

- The manually acquired delta is treated as untrusted data rather than code.
- DepictQA is reachable only during the managed 4KAgent phase and only over
  loopback.
- CUDA 12.6 raises the declared Linux host floor to glibc 2.28 and NVIDIA
  driver 560.28.03. The GPU preflight enforces the driver floor.
- Compatibility of the pinned upstream commits with the security-updated
  PyTorch stack still requires an external AutoDL GPU smoke test; no GPU result
  is inferred from dependency resolution or local unit tests.

## References

- [PyTorch GHSA-53q9-r3pm-6pq6](https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6)
- [PyTorch GHSA-63cw-57p8-fm3p](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p)
- [Official PyTorch 2.10.0 wheel matrix](https://pytorch.org/get-started/previous-versions/#v2100)
- [CUDA 12.8 release notes, including the CUDA 12.6 driver table](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html#cuda-toolkit-and-minimum-required-driver-version-for-cuda-minor-version-compatibility)
