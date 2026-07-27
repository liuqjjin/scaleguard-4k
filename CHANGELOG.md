# Changelog

All notable changes to ScaleGuard-4K will be documented here.

The project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and intends to use semantic versioning after its first public release.

## [Unreleased]

### Added

- Deterministic Trusted Scale Controller with explicit continue, stop, and
  rollback decisions for one-step CoZ states.
- Discrete 1×/2×/4×/8×/16× factor policy with at most one 4KAgent fidelity
  bridge and two CoZ transitions.
- Fake CPU backends, command adapters, pinned-upstream adapters, and persistent
  CoZ JSON-lines session.
- Same-resolution quality gain, low-pass cross-scale consistency, optional
  declared observation models, and one final color/re-score stage.
- Atomic run manifests with artifact hashes, decisions, process evidence,
  failures, mock provenance, and GPU phase events.
- Hash-bound gate calibration receipts and paired, non-imputed ablation
  summaries.
- Executable, fail-continuing A-only, B-only, AB-fixed, and ScaleGuard suites
  with exact input/seed pairing, raw attempt retention, hardware identity
  checks, and an independently revalidated suite receipt required for research
  eligibility.
- Strict YAML configuration, CLI, doctor, upstream verification, and manifest
  validation.
- Locked source identities for 4KAgent and Chain-of-Zoom; DepictQA is pinned as
  a 4KAgent transitive service.
- Immutable weight manifest with gated, optional, manual, and known-digest
  handling.
- AutoDL dual-4090 preflight, bootstrap, weight materialization, smoke,
  integration, diagnostics, and external-gate evidence contracts.
- Per-attempt runtime re-attestation that binds current distributions,
  dependency checks, offline imports, and 4KAgent tool-entrypoint probes to
  schema-v2 preflight evidence, with single-open regular-file snapshots and
  private atomic no-clobber receipt publication.
- Byte-pinned Linux bootstrap identities for the private uv executable and
  managed CPython archive, plus reinstall-only construction of every declared
  runtime environment.
- Runtime-bound calibration receipts and conditional observation-model
  evidence, including independent paired-summary verification.
- CPU unit, contract, integration, and evaluation tests plus locked CI.
- Architecture, installation, reproduction, evaluation, limitation,
  development, security, contribution, citation, notice, and status material.

### Changed

- Reserved 4KAgent's outer generative SR for one terminal CoZ phase while
  retaining native restoration, reflection, rollback, and an optional 2×
  fidelity bridge.
- Patched the pinned CoZ full-image path to initialize one-step scheduling,
  move VLM inputs to the model device, restrict VAE state loading, and stream
  Gaussian latent fusion.
- Kept 4KAgent, DepictQA, CoZ, and ScaleGuard in separate runtime environments.
- Scoped AutoDL credentials to acquisition/model boundaries, replaced doctor
  access with a non-secret presence marker, sanitized transitive system probes,
  and rejected ambiguous YAML credential mappings and unsafe environment names.
- Added fail-closed, process-local compatibility shims for the pinned
  inference sources plus offline symbol-import receipts for every runtime
  environment.
- Made the four ablation modes executable and explicit: B-only uses an
  observation-preserving identity restoration boundary, fixed groups disclose
  their fixed acceptance policy, and only ScaleGuard uses the trusted
  controller.

### Security

- Added argument redaction, shell-free process templates, timeouts,
  process-group termination, service-port ownership checks, safe model-archive
  materialization, receipt hashing, and allowlisted diagnostic collection.
- Hardened public AutoDL startup against shell-function/startup poisoning and
  ambient repository redirection; isolated every stage in a fresh HOME with
  user/system tool configuration disabled; bounded every owned process group
  after leader exit and bounded partial CoZ protocol responses.
- Prevented 4KAgent diagnostic logs from embedding source-image bytes and made
  diagnostic collection reject parameterized base64 image data URLs.
- Security-updated every GPU environment to PyTorch 2.10.0 with official CUDA
  12.6 wheels and rejected older runtimes at the overlay boundary for
  CVE-2026-24747. Enforced the CUDA 12.6 minimum NVIDIA driver separately.
- Upgraded vulnerable inference dependencies, added lock-level vulnerability
  auditing, and isolated the two unpatched legacy cache/SLURM boundaries.

### Known limitations

- Highest evidence level is `STATIC_READY`.
- No ScaleGuard GPU, runtime, VRAM, quality, or research result is published.
- Real model access, manual DepictQA weight acquisition, authorized data,
  threshold calibration, and paired evaluation remain external or future
  evidence gates.
