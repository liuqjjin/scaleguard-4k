"""Construct only the two configured algorithmic backends."""

from __future__ import annotations

from pathlib import Path

from scaleguard.backends.base import RestorationBackend, ScaleBackend
from scaleguard.backends.command import CommandRestorationBackend, CommandScaleBackend
from scaleguard.backends.coz import CoZBackend
from scaleguard.backends.fake import FakeRestorationBackend, FakeScaleBackend
from scaleguard.backends.fourkagent import FourKAgentBackend
from scaleguard.config import PipelineConfig


def build_backends(
    config: PipelineConfig,
    *,
    project_root: Path,
) -> tuple[RestorationBackend, ScaleBackend]:
    if config.fourkagent.mode == "fake":
        restoration: RestorationBackend = FakeRestorationBackend()
    elif config.fourkagent.mode == "command":
        restoration = CommandRestorationBackend(config.fourkagent, config.runtime)
    else:
        restoration = FourKAgentBackend(
            config.fourkagent,
            config.runtime,
            project_root=project_root,
        )

    if config.coz.mode == "fake":
        scale: ScaleBackend = FakeScaleBackend()
    elif config.coz.mode == "command":
        scale = CommandScaleBackend(config.coz, config.runtime)
    else:
        scale = CoZBackend(config.coz, config.runtime, project_root=project_root)
    return restoration, scale
