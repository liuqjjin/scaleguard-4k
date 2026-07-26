"""Narrow adapter interfaces around the two algorithmic upstreams."""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import Protocol

from scaleguard.contracts import WorkerResult


class RestorationBackend(Protocol):
    name: str
    mock: bool

    def restore(
        self,
        source: Path,
        destination: Path,
        *,
        bridge_factor: int,
        run_dir: Path,
    ) -> WorkerResult:
        """Run 4KAgent's native-scale restoration and optional fidelity bridge."""


class ScaleSession(Protocol):
    name: str
    mock: bool

    def __enter__(self) -> ScaleSession: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    def upscale_once(
        self,
        source: Path,
        destination: Path,
        *,
        step_index: int,
        seed: int,
    ) -> WorkerResult:
        """Produce exactly one 4x scale state."""

    def accept(self, candidate: WorkerResult, *, step_index: int) -> None:
        """Promote a candidate to the session's trusted scale."""

    def rollback(self, *, step_index: int) -> None:
        """Discard the pending candidate and retain the previous trusted scale."""


class ScaleBackend(Protocol):
    name: str
    mock: bool

    def session(self, run_dir: Path) -> ScaleSession:
        """Create an explicit scale-autoregressive session."""
