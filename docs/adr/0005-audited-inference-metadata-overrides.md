# ADR 0005: Isolate and audit 4KAgent inference metadata overrides

- Status: Accepted
- Date: 2026-07-27

## Context

The pinned 4KAgent installation uses two wheels whose metadata combines old
training and inference constraints:

- PyIQA 0.1.13 requires Transformers 4.37.2.
- HPSv2 1.2.0 requires Protobuf below 4, Pytest 7.2.0, and
  pytest-split 0.8.0.

Those versions carry known vulnerabilities or conflict with the
security-updated Qwen runtime. Pytest and pytest-split are not imported by the
HPS scoring path. HPS itself is required for 4KAgent reflection, so removing the
wheel would change upstream behavior. Ignoring dependency validation globally
would hide unrelated missing or incompatible packages.

## Decision

The 4KAgent base environment is resolved with hashes at Transformers 5.5.0 and
Protobuf 6.33.5. Pytest and pytest-split are omitted from this inference-only
environment. Exact PyIQA 0.1.13 and HPSv2 1.2.0 wheels are installed with
`--no-deps` from separate one-line locks containing their SHA-256 digests.

Bootstrap runs a metadata audit over every installed distribution. It permits
exactly four observations:

```text
pyiqa 0.1.13: transformers ==4.37.2 -> installed 5.5.0
hpsv2 1.2.0: protobuf <4 -> installed 6.33.5
hpsv2 1.2.0: pytest ==7.2.0 -> omitted
hpsv2 1.2.0: pytest-split ==0.8.0 -> omitted
```

The parent version, required specifier, dependency, and installed version must
all match. If an observation disappears, changes, occurs twice, or another
dependency is unsatisfied, bootstrap fails. The audit result and all three
lock hashes are recorded in the environment receipt.

ScaleGuard's controller remains in a separate environment and uses its own
locked PyIQA 0.1.16 optional dependency. The exceptions do not leak into CoZ or
DepictQA.

## Consequences

- The environment retains the two pinned upstream components without turning
  dependency validation off globally.
- Regenerating any 4KAgent lock requires re-evaluating this ADR and the exact
  observations.
- `pip check` reports the known metadata conflicts; the project audit
  distinguishes those exact records from every other dependency error.
- Installation success is not evidence of Qwen, MUSIQ, or restoration behavior
  on a GPU.

## Evidence

All resolved requirements and both override wheels carry hashes. CPU tests
cover exact-match, missing-dependency, and unexpected-mismatch behavior.
Bootstrap also runs import contracts against the materialized upstream trees.
Real checkpoint loading and inference remain part of the external AutoDL gate.
