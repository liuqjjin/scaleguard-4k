from __future__ import annotations

import pytest

from scaleguard.controller.policy import build_scale_plan
from scaleguard.errors import ConfigurationError


@pytest.mark.parametrize(
    ("target_factor", "bridge_factor", "coz_steps"),
    [
        (1, 1, 0),
        (2, 2, 0),
        (4, 1, 1),
        (8, 2, 1),
        (16, 1, 2),
    ],
)
def test_factor_policy_uses_only_the_2x_bridge_and_4x_coz_steps(
    target_factor: int,
    bridge_factor: int,
    coz_steps: int,
) -> None:
    plan = build_scale_plan(target_factor)

    assert plan.target_factor == target_factor
    assert plan.bridge_factor == bridge_factor
    assert plan.coz_steps == coz_steps
    assert plan.realized_factor == target_factor


@pytest.mark.parametrize("target_factor", [0, 3, 6, 32])
def test_factor_policy_rejects_unsupported_targets(target_factor: int) -> None:
    with pytest.raises(ConfigurationError, match="unsupported target factor"):
        build_scale_plan(target_factor)


def test_factor_policy_rejects_a_target_beyond_the_session_limit() -> None:
    with pytest.raises(ConfigurationError, match=r"needs 2 CoZ steps.*max_coz_steps=1"):
        build_scale_plan(16, max_coz_steps=1)
