"""Explicit phase ledger for the two-GPU runtime."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum

from scaleguard.contracts import utc_now
from scaleguard.errors import WorkerError


class GpuPhase(str, Enum):
    IDLE = "idle"
    PERCEPTION = "perception"
    RESTORATION = "restoration"
    COZ = "coz"
    EVALUATION = "evaluation"


@dataclass(frozen=True, slots=True)
class PhaseEvent:
    at: str
    phase: GpuPhase
    action: str
    devices: str


class GpuLifecycle:
    """Prevent overlapping heavyweight phases and record allocation intent."""

    def __init__(self, sink: Callable[[PhaseEvent], None] | None = None) -> None:
        self.phase = GpuPhase.IDLE
        self._sink = sink

    @contextmanager
    def enter(self, phase: GpuPhase, devices: str) -> Iterator[None]:
        if self.phase is not GpuPhase.IDLE:
            raise WorkerError(
                f"cannot enter GPU phase {phase.value}; {self.phase.value} has not been released"
            )
        self.phase = phase
        self._emit(phase, "acquire", devices)
        try:
            yield
        finally:
            self._emit(phase, "release", devices)
            self.phase = GpuPhase.IDLE

    def _emit(self, phase: GpuPhase, action: str, devices: str) -> None:
        if self._sink is not None:
            self._sink(PhaseEvent(at=utc_now(), phase=phase, action=action, devices=devices))
