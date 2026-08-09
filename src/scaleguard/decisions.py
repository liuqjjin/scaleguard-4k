"""Canonical scale-step decisions shared by execution and evidence replay."""

from __future__ import annotations

from scaleguard.config import MetricConfig
from scaleguard.contracts import Decision, MetricRecord


def decide_scale_step(
    metrics: MetricRecord,
    thresholds: MetricConfig,
    *,
    acceptance_policy: str,
    step_index: int,
    total_steps: int,
) -> tuple[Decision, bool, str]:
    """Return the canonical decision for one generated scale candidate."""

    if not 1 <= step_index <= total_steps:
        raise ValueError("step_index must identify one of the planned scale steps")
    if acceptance_policy == "fixed":
        decision = Decision.STOP if step_index == total_steps else Decision.CONTINUE
        return (
            decision,
            True,
            (
                "fixed_policy_accepted: fixed ablation policy accepted the candidate; "
                "gates were recorded but not enforced"
            ),
        )
    if acceptance_policy != "trusted":
        raise ValueError(f"unsupported acceptance policy: {acceptance_policy!r}")
    if metrics.scale_nrmse > thresholds.max_scale_nrmse:
        return (
            Decision.ROLLBACK,
            False,
            (
                "scale_nrmse_exceeded: "
                f"scale_nrmse={metrics.scale_nrmse:.6f} exceeds "
                f"{thresholds.max_scale_nrmse:.6f}"
            ),
        )
    if metrics.scale_edge_mae > thresholds.max_scale_edge_mae:
        return (
            Decision.ROLLBACK,
            False,
            (
                "scale_edge_mae_exceeded: "
                f"scale_edge_mae={metrics.scale_edge_mae:.6f} exceeds "
                f"{thresholds.max_scale_edge_mae:.6f}"
            ),
        )
    if (
        metrics.measurement_nrmse is not None
        and metrics.measurement_nrmse > thresholds.max_measurement_nrmse
    ):
        return (
            Decision.ROLLBACK,
            False,
            (
                "measurement_nrmse_exceeded: "
                f"measurement_nrmse={metrics.measurement_nrmse:.6f} exceeds "
                f"{thresholds.max_measurement_nrmse:.6f}"
            ),
        )
    if metrics.quality_gain < thresholds.min_quality_gain:
        return (
            Decision.STOP,
            False,
            (
                "quality_gain_below_minimum: "
                f"quality_gain={metrics.quality_gain:.6f} is below "
                f"{thresholds.min_quality_gain:.6f}"
            ),
        )
    if step_index == total_steps:
        return Decision.STOP, True, "target_scale_accepted: target scale accepted"
    return Decision.CONTINUE, True, "all_gates_passed: all gates passed"
