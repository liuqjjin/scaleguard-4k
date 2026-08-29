# ADR 0003: Restrict the gradient proxy to CPU contracts

- Status: Accepted
- Date: 2026-07-27

## Context

Controller tests need a deterministic, lightweight score with a
higher-is-better direction. A luma-gradient proxy satisfies that engineering
need without model downloads. It is not a validated no-reference image quality
assessment method: sharpening noise or ringing can increase it, and its scale
has no established relationship to human preference, fidelity, or safe
hallucination control.

Using its current thresholds as research evidence would make mock plumbing look
like a calibrated quality gate.

## Decision

`gradient_proxy_v1` is only a CPU contract fixture. It may test
same-resolution baseline construction, metric direction, manifest recording,
and continue/stop/rollback wiring.

Non-mock configurations reject the proxy unless an operator explicitly marks
the run as a calibration experiment. Such an override does not make the score
validated and must remain visible in the run manifest.

Research runs use a versioned IQA backend such as the optional PyIQA 0.1.16
MUSIQ/CLIPIQA path, subject to its separate non-commercial licenses. Thresholds
must be calibrated on a declared validation split, with direction, color
space, resize kernel, confidence or sensitivity analysis, and failure cases
recorded before the quality gate can support a result.

## Consequences

- CPU CI remains deterministic and model-free.
- A passing fake or CPU run demonstrates contracts, not restoration quality.
- Default numerical proxy thresholds are test fixtures and must not appear as
  paper findings.
- PyIQA availability alone does not validate a controller threshold; model
  weights, version, dataset split, and calibration evidence are all required.
- Research-quality gates use the versioned PyIQA path and a bound calibration
  receipt. This proxy remains a CPU contract fixture and is not itself a
  paper metric.

## Evidence

Unit tests cover only the proxy's numerical and state-transition contracts.
GPU measurements, calibrated thresholds, and paper metrics belong to the
dual-GPU evaluation evidence, not this decision.
