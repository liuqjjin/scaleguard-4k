"""Non-algorithmic observation pass-through for the B-only ablation."""

from __future__ import annotations

from pathlib import Path

from scaleguard.contracts import WorkerResult
from scaleguard.errors import WorkerError
from scaleguard.images import inspect_image, normalize_to_png


class IdentityRestorationBackend:
    """Preserve observed pixels without introducing a third restoration method."""

    name = "scaleguard_identity_observation"
    mock = False

    def restore(
        self,
        source: Path,
        destination: Path,
        *,
        bridge_factor: int,
        run_dir: Path,
    ) -> WorkerResult:
        del run_dir
        if bridge_factor != 1:
            raise WorkerError("identity restoration supports only a 1x fidelity bridge")
        normalize_to_png(source, destination)
        return WorkerResult(
            image=inspect_image(
                destination,
                mock=False,
                stage="identity_observation",
            ),
            metadata={
                "backend": self.name,
                "bridge_factor": 1,
                "algorithmic_restoration": False,
                "experiment_group": "B-only",
            },
        )
