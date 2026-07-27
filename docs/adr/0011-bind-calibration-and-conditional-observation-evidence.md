# ADR 0011: Bind calibration and conditional observation evidence

- Status: Accepted
- Date: 2026-07-27

## Context

A threshold mapping copied into a runtime config is not proof that it came from
the declared calibration run. Likewise, always emitting or always omitting an
observation-consistency value makes it impossible to distinguish a disabled
forward model from a missing measurement.

## Decision

Trusted controller construction verifies the configured calibration receipt
against the complete runtime metric configuration. The run manifest records
the receipt's resolved path, size, SHA-256, and verified semantic result.
Paired-summary validation reopens that exact receipt as one regular-file
snapshot, repeats semantic verification, and binds it into the pairing
fingerprint.

`measurement_nrmse` and the canonical forward-model identity are required
exactly when measurement consistency is enabled; both must be absent when it
is disabled. The identity is constructed by the same forward-model factory
used for evaluation rather than copied from a free-form label.

## Consequences

- Editing thresholds and recomputing nearby self-digests cannot make a run
  research-eligible without the original valid calibration evidence.
- Missing optional measurement evidence is distinguishable from an intentionally
  disabled observation model.
- Calibration and measurement bindings improve evidence integrity; they do not
  establish that the chosen observation model matches a real camera.
