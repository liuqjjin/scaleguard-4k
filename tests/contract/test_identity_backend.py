from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from scaleguard.backends.identity import IdentityRestorationBackend
from scaleguard.errors import WorkerError
from scaleguard.images import file_sha256


def test_identity_backend_preserves_observation_without_an_algorithmic_claim(
    tmp_path: Path,
    make_image: Callable[..., Path],
) -> None:
    source = make_image(tmp_path / "source.jpg", size=(13, 7), image_format="JPEG")
    destination = tmp_path / "observation.png"
    backend = IdentityRestorationBackend()

    result = backend.restore(
        source,
        destination,
        bridge_factor=1,
        run_dir=tmp_path / "run",
    )

    assert result.image.path == destination.resolve()
    assert result.image.sha256 == file_sha256(destination)
    assert (result.image.width, result.image.height) == (13, 7)
    assert result.image.mock is False
    assert result.metadata == {
        "backend": "scaleguard_identity_observation",
        "bridge_factor": 1,
        "algorithmic_restoration": False,
        "experiment_group": "B-only",
    }
    assert result.process is None


def test_identity_backend_rejects_a_controlled_bridge(
    tmp_path: Path,
    make_image: Callable[..., Path],
) -> None:
    source = make_image(tmp_path / "source.png")

    with pytest.raises(WorkerError, match="requires bridge_factor=1"):
        IdentityRestorationBackend().restore(
            source,
            tmp_path / "output.png",
            bridge_factor=2,
            run_dir=tmp_path / "run",
        )
