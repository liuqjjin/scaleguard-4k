# ADR 0010: Make ablation modes explicit and executable

- Status: Accepted
- Date: 2026-07-27

## Context

The original experiment file described four groups but did not execute them.
Ad-hoc configuration edits would make it easy to change seeds, weights,
quality gates, or preprocessing between groups, and a naive “fixed” ablation
could be mistaken for ScaleGuard's trusted acceptance policy.

## Decision

One executable harness generates every input × seed × group job from a single
base configuration:

- `A-only`: 4KAgent observation, no CoZ transition;
- `B-only`: identity observation, one persistent CoZ transition;
- `AB-fixed`: 4KAgent followed by one persistent CoZ transition, accepted by a
  disclosed fixed evaluation policy; and
- `ScaleGuard`: the same two upstream runtimes with trusted metric gates.

Identity restoration and fixed acceptance are reserved for these declared
ablations. They cannot promote an `AB_INTEGRATED` runtime claim. Each job gets
a fresh preflight and a hash-bound AutoDL attempt; the suite compares input
bytes, seed, configuration controls, hardware identity, source/weight/runtime
bindings, and raw evidence across the four paired groups. Failed jobs are
preserved, never imputed.

## Consequences

- Component baselines are executable without adding a third algorithmic
  project or duplicating upstream implementations.
- A legitimate ScaleGuard gate rollback remains an observation, including a
  rejected CoZ candidate or a final post-color gate rollback.
- The harness and its receipts establish experiment integrity only. No GPU
  metric or ablation conclusion is claimed until authorized data, model access,
  and real attempt artifacts exist.
