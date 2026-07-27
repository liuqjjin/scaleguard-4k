# Security policy

ScaleGuard-4K executes large upstream model stacks and handles credentials,
images, weights, subprocesses, and diagnostic logs. Treat the current
`0.1.0.dev0` code as research pre-release software, not a hardened multi-tenant
service.

## Supported versions

Security fixes are made on the current development branch. No older release
line is supported yet.

| Version | Supported |
| --- | --- |
| current development branch / `0.1.0.dev0` | best-effort |
| older snapshots | no |

## Report a vulnerability

Do not disclose a credential leak, command-execution path, unsafe archive,
path traversal, model-cache poisoning issue, or sensitive-image exposure in a
public issue.

Use this repository's private **Report a vulnerability** feature when it is
available. If private reporting is unavailable, open a public issue containing
only a request for a private security contact; include no exploit, secret,
private path, or affected data.

Include privately:

- affected commit and platform;
- minimal reproduction without real credentials or private images;
- security impact and trust boundary crossed;
- whether model weights, caches, logs, or diagnostics are involved; and
- a proposed mitigation if known.

There is no guaranteed response SLA during pre-release. Maintainers will
acknowledge and triage a reproducible report as capacity permits.

## High-priority report classes

- command or argument injection across adapter/process boundaries;
- path traversal or link/device extraction from model archives;
- writes outside declared run, cache, checkout, or output roots;
- credential exposure in argv, logs, receipts, manifests, or diagnostics;
- accepting an unexpected upstream commit, patch, model revision, or digest;
- stale or forged output accepted as a fresh successful run;
- public network exposure of a local model service;
- failure to terminate a service/process group after an error or timeout; and
- diagnostics that include weights, images, environment dumps, or secrets
  despite their exclusion policy.

Model hallucination, poor restoration, or a metric disagreement is normally a
research-quality issue rather than a security vulnerability. It becomes a
security issue when it enables boundary bypass, data disclosure, code
execution, or a false evidence/identity claim.

## Operational guidance

### Credentials

- Pass Hugging Face and API credentials through the process environment or a
  private provider login.
- Never put a token in a URL, command argument, YAML/JSON config, `.env`, shell
  transcript, issue, or Git.
- Disable shell tracing while credentials exist and unset them after the gated
  operation.
- Do not give model processes credentials they do not need.
- AutoDL stages are credential-scoped: bootstrap/source checks receive none,
  download receives only Hugging Face auth, doctor receives a non-secret
  presence marker, and only model execution receives the configured scheduler
  key.
- Execute public AutoDL entry points directly. They require privileged Bash
  startup, clear imported functions and startup hooks, disable tracing, use a
  fixed system path, and derive the repository root from their own checked-out
  location. An ambient `SCALEGUARD_REPO_ROOT` cannot redirect credential-bearing
  stages to another tree. If an explicit interpreter is unavoidable, use
  `/bin/bash -p`; ordinary `bash script` invocation is rejected.
- Each entry creates a fresh, owned `0700` HOME below the non-symlinked
  project runtime directory and disables user/system Git, pip, uv, and Python
  startup configuration before invoking tools. Do not reuse those temporary
  homes as persistent account state.

### Source and models

- Run only checkouts verified against `upstream-lock.yaml`.
- Run only weights bound to `weights-lock.json` and a reviewed receipt.
- Keep the per-attempt runtime environment re-audit enabled. It reruns with
  isolated interpreters and no account credentials, then binds complete
  distribution and import-probe identity into schema-v2 preflight evidence.
  Receipt readers bind strict JSON and hashes to one regular-file snapshot.
  Receipt writers use private exclusive temporary files and publish a
  validated inode without clobbering a concurrently created target.
- Bootstrap must reconstruct uv from its hash-pinned wheel, verify the
  committed executable digest, and reinstall the committed managed-Python
  archive and package locks. Do not replace this with a version-only `PATH`
  check.
- Remember that a valid hash proves identity, not safety.
- Upstream Python/model loading is not sandboxed; use an isolated account or
  disposable host with least privilege and no unrelated secrets.
- Do not load an untrusted pickle or checkpoint merely because its filename
  matches an expected path.
- Keep the isolated GPU environments on PyTorch 2.10.0 or newer. Releases
  through 2.9.1 are rejected at runtime because CVE-2026-24747 affects the
  `weights_only` unpickler.
- Do not pass SLURM variables into the 4KAgent worker. Keep its Outlines cache
  run-local, non-symlinked, and mode `0700`; the overlay disables that cache
  after import. These are the explicit mitigations for the two unfixed legacy
  dependency advisories recorded in ADR 0007.
- Run `bash scripts/security/audit_runtime_locks.sh` after every environment
  lock change. New advisories are release blockers until fixed or bounded by a
  reviewed ADR.

The byte-identity checks are designed for reproducibility and ordinary
in-host drift. They do not establish a secure-boot chain against a malicious
administrator who controls the repository, interpreters, loader, and unsigned
receipts together. Use an externally verified read-only image or independent
verifier for that threat model; see
[ADR 0009](docs/adr/0009-bind-runtime-bytes-at-each-real-attempt.md).

### Services and processes

- Keep DepictQA bound to loopback.
- Do not expose worker protocols to a public interface.
- Preserve timeouts and process-group termination.
- Worker, service, and persistent-session leaders start a new session. Normal
  leader exit, startup failure, protocol timeout, and context-manager close all
  require their owned process group to quiesce or be terminated with bounded
  TERM/KILL escalation. Upstream commands must not deliberately escape that
  ownership with a second `setsid`.
- Refuse to take ownership of an already occupied service port.
- Keep ScaleGuard, 4KAgent, DepictQA, and CoZ environments separate.

### Images and outputs

Input images and parsed evidence are inspected from one regular-file byte
snapshot so recorded metadata and SHA-256 refer to the same bytes.

Input images, VLM prompts, outputs, and logs may contain personal, medical,
location, or proprietary information. Use authorized data, private storage,
least-retention policies, and human review before sharing.

Generative SR can create plausible but false details. Do not use an output as
sole evidence for a medical, forensic, identity, scientific-measurement, or
other consequential decision.

### Diagnostics

`collect_diagnostics.sh` privatizes credential values before system probes,
passes exact values only to its sanitizer over a private file descriptor,
copies an allowlist of bounded text, and performs exact-value plus
secret-pattern scans. The 4KAgent overlay replaces logging-side image encoding
with only the source byte count and SHA-256; the model-request encoder remains
unchanged. As a second boundary, the collector rejects any text containing a
parameterized base64 image data URL instead of trying to redact the image.
This is defense in depth, not a guarantee. Follow
[external_gate/REDACTION.md](external_gate/REDACTION.md),
inspect every archive manually, and share only the reviewed archive plus its
SHA-256.

## Public incident handling

After a fix is available, maintainers should document affected versions,
impact, mitigation, and any credential/model-cache rotation required without
publishing unnecessary sensitive details. A security fix must not silently
alter upstream or weight identity; update locks and audit records explicitly.
