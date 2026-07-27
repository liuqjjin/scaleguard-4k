# ADR 0009: Bind runtime bytes at each real attempt

- Status: Accepted
- Date: 2026-07-27

## Context

Package names and versions do not detect modified wheel files, stray
site-packages, a replaced virtual-environment interpreter, or a changed
standard library. Historical bootstrap receipts are also insufficient after a
runtime has been reused.

## Decision

The Linux bootstrap always installs the hash-pinned uv wheel into a freshly
cleared private environment. Its binary SHA-256 is committed. uv then
reinstalls the one committed Python-build-standalone archive and every locked
package; the Python archive URL, build, and SHA-256 are supplied through a
committed uv downloads file.

Environment receipts bind RECORD-owned files, fixed virtual-environment
metadata, interpreter and `pyvenv.cfg` bytes, the managed base interpreter,
standard-library contents, executable aliases, and offline import origins.
Immediately before a real run, ScaleGuard creates four fresh receipts and
independently repeats the audits in credential-free subprocesses. The run
preflight must match the bootstrap baseline and the independent observations.

## Consequences

- Re-running bootstrap repairs ordinary changes to locked interpreter and
  package files instead of accepting them as a new baseline.
- Unknown package files, unsafe symlinks, changed imports, and post-bootstrap
  drift fail closed.
- This is reproducibility and accidental-drift protection, not a claim of
  resistance to a malicious host administrator. Receipts are not signed, and a
  process with write control over the repository, loader, interpreters, and
  evidence can defeat an in-host verifier. That stronger threat model requires
  an external read-only image or independent verifier.
