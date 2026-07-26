from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture(autouse=True)
def disable_gpu_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """CPU tests must never invoke nvidia-smi, even on a GPU runner."""

    monkeypatch.setattr("scaleguard.runtime.process.shutil.which", lambda _name: None)


@pytest.fixture
def make_image() -> Callable[..., Path]:
    def _make_image(
        path: Path,
        *,
        size: tuple[int, int] = (9, 6),
        color: tuple[int, int, int] = (72, 116, 164),
        image_format: str | None = None,
    ) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, color).save(path, format=image_format)
        return path

    return _make_image
