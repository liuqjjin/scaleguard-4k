from __future__ import annotations

from pathlib import Path

import pytest

from scaleguard.backends.fake import FakeRestorationBackend, FakeScaleBackend
from scaleguard.config import ControllerConfig, MetricConfig, PipelineConfig, RuntimeConfig
from scaleguard.contracts import Decision, MetricRecord
from scaleguard.controller.trusted_scale import TrustedScaleController


def controller(tmp_path: Path) -> TrustedScaleController:
    config = PipelineConfig(
        runtime=RuntimeConfig(run_root=tmp_path / "runs"),
        metrics=MetricConfig(
            min_quality_gain=0.0,
            max_scale_nrmse=0.1,
            max_scale_edge_mae=0.05,
            max_measurement_nrmse=0.08,
        ),
        controller=ControllerConfig(target_factor=16, color_strategy="none"),
    )
    return TrustedScaleController(config, FakeRestorationBackend(), FakeScaleBackend())


def metrics(**overrides: float | None) -> MetricRecord:
    values: dict[str, float | str | None] = {
        "quality_baseline": 0.3,
        "quality_candidate": 0.4,
        "quality_gain": 0.1,
        "quality_backend": "test",
        "quality_identity_sha256": "a" * 64,
        "scale_nrmse": 0.02,
        "scale_edge_mae": 0.01,
        "measurement_nrmse": None,
        "measurement_model": None,
    }
    values.update(overrides)
    return MetricRecord(**values)  # type: ignore[arg-type]


def test_all_gates_passed_continues_before_the_target(tmp_path: Path) -> None:
    decision, accepted, reason = controller(tmp_path)._decide(metrics(), 1, 2)

    assert decision is Decision.CONTINUE
    assert accepted is True
    assert reason == "all gates passed"


def test_all_gates_passed_stops_and_accepts_at_the_target(tmp_path: Path) -> None:
    decision, accepted, reason = controller(tmp_path)._decide(metrics(), 2, 2)

    assert decision is Decision.STOP
    assert accepted is True
    assert reason == "target scale accepted"


def test_low_quality_gain_stops_without_accepting_the_candidate(tmp_path: Path) -> None:
    decision, accepted, reason = controller(tmp_path)._decide(
        metrics(quality_gain=-0.01),
        1,
        2,
    )

    assert decision is Decision.STOP
    assert accepted is False
    assert "quality_gain" in reason


@pytest.mark.parametrize(
    "overrides",
    [
        {"scale_nrmse": 0.11},
        {"scale_edge_mae": 0.06},
        {"measurement_nrmse": 0.09},
    ],
)
def test_consistency_failure_rolls_back(
    tmp_path: Path,
    overrides: dict[str, float],
) -> None:
    decision, accepted, _reason = controller(tmp_path)._decide(metrics(**overrides), 1, 2)

    assert decision is Decision.ROLLBACK
    assert accepted is False
