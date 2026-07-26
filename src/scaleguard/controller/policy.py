"""Factor policy separating 4KAgent's 2x bridge from CoZ's 4x recursion."""

from __future__ import annotations

from dataclasses import dataclass

from scaleguard.errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class ScalePlan:
    target_factor: int
    bridge_factor: int
    coz_steps: int

    @property
    def realized_factor(self) -> int:
        return int(self.bridge_factor * (4**self.coz_steps))


def build_scale_plan(target_factor: int, max_coz_steps: int = 2) -> ScalePlan:
    policies = {
        1: ScalePlan(target_factor=1, bridge_factor=1, coz_steps=0),
        2: ScalePlan(target_factor=2, bridge_factor=2, coz_steps=0),
        4: ScalePlan(target_factor=4, bridge_factor=1, coz_steps=1),
        8: ScalePlan(target_factor=8, bridge_factor=2, coz_steps=1),
        16: ScalePlan(target_factor=16, bridge_factor=1, coz_steps=2),
    }
    try:
        plan = policies[target_factor]
    except KeyError as error:
        raise ConfigurationError(f"unsupported target factor: {target_factor}") from error
    if plan.coz_steps > max_coz_steps:
        raise ConfigurationError(
            f"target factor {target_factor} needs {plan.coz_steps} CoZ steps, "
            f"but max_coz_steps={max_coz_steps}"
        )
    return plan
