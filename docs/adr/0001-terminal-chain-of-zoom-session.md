# ADR 0001: Keep Chain-of-Zoom outside the 4KAgent tool registry

- Status: Accepted
- Date: 2026-07-27

## Context

4KAgent's `Tool` abstraction assumes one image in, one image out, a hard-coded
tool-to-Conda mapping, and one integer `tool_run_gpu_id`. Its SR profile also
filters tools by fixed names. Registering a new `chain_of_zoom` tool would
therefore require invasive changes merely to select the CoZ environment and
make two GPUs visible.

More importantly, 4KAgent represents 16× enlargement as two equal
`super-resolution` strings. Rollback and rescheduling convert plans to sets in
several places, so repeated scale steps are not a reliable scale state machine.
The initial SR agenda may also be shuffled with restoration work. CoZ needs an
explicit ordered session in which every 4× result can be inspected before the
next step.

## Decision

4KAgent remains the sole high-level controller for degradation perception,
native-resolution restoration, reflection, and non-SR rollback. A thin overlay
removes its generative SR tasks and permits at most one controlled 2× bridge.

After 4KAgent produces a trusted native-scale image, ScaleGuard starts one
terminal CoZ session. ScaleGuard, rather than the 4KAgent tool registry, owns
the session process, GPU visibility, per-scale state, and
continue/stop/rollback decisions.

CoZ is still the only generative SR runtime. This boundary does not add another
agent, VLM, restoration method, or SR method.

## Consequences

- Generative SR cannot be shuffled ahead of degradation restoration.
- Each CoZ scale is a first-class state with its own image, metrics, decision,
  seed, logs, and failure evidence.
- Two-GPU visibility and process lifetime are expressed without widening
  4KAgent's single-GPU `Tool` contract.
- CoZ failure returns control to the last trusted image instead of entering
  4KAgent's set-based SR rollback path.
- The overlay must be audited whenever the pinned 4KAgent pipeline changes.

## Evidence

The decision is based on static inspection of the commits in
`upstream-lock.yaml`; see `docs/upstream-audit.md`. CPU contracts and mock
integration tests cover this ADR. Dual-GPU execution of both upstreams is
recorded by the research evaluation, not by this decision.
