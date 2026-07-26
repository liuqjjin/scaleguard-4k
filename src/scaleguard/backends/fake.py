"""Deterministic CPU workers for contract tests and demos."""

from __future__ import annotations

import time
from pathlib import Path
from types import TracebackType

from PIL import Image, ImageEnhance, ImageFilter

from scaleguard.contracts import WorkerResult
from scaleguard.images import inspect_image, normalize_to_png


class FakeRestorationBackend:
    name = "fake_4kagent"
    mock = True

    def restore(
        self,
        source: Path,
        destination: Path,
        *,
        bridge_factor: int,
        run_dir: Path,
    ) -> WorkerResult:
        del run_dir
        if bridge_factor == 1:
            normalize_to_png(source, destination)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(source) as image:
                restored = image.convert("RGB").resize(
                    (image.width * bridge_factor, image.height * bridge_factor),
                    Image.Resampling.LANCZOS,
                )
                restored.save(destination, "PNG")
        return WorkerResult(
            image=inspect_image(destination, mock=True, stage="4kagent_restoration"),
            metadata={"backend": self.name, "bridge_factor": bridge_factor, "mock": True},
        )


class FakeScaleSession:
    name = "fake_coz_session"
    mock = True

    def __enter__(self) -> FakeScaleSession:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback

    def upscale_once(
        self,
        source: Path,
        destination: Path,
        *,
        step_index: int,
        seed: int,
    ) -> WorkerResult:
        started = time.monotonic()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as image:
            candidate = image.convert("RGB").resize(
                (image.width * 4, image.height * 4), Image.Resampling.BICUBIC
            )
            candidate = candidate.filter(
                ImageFilter.UnsharpMask(radius=1.2, percent=55, threshold=3)
            )
            candidate = ImageEnhance.Contrast(candidate).enhance(1.005)
            candidate.save(destination, "PNG")
        return WorkerResult(
            image=inspect_image(destination, mock=True, stage=f"coz_scale_{step_index}"),
            metadata={
                "backend": self.name,
                "seed": seed,
                "step_index": step_index,
                "duration_seconds": time.monotonic() - started,
                "mock": True,
            },
        )

    def accept(self, candidate: WorkerResult, *, step_index: int) -> None:
        del candidate, step_index

    def rollback(self, *, step_index: int) -> None:
        del step_index


class FakeScaleBackend:
    name = "fake_coz"
    mock = True

    def session(self, run_dir: Path) -> FakeScaleSession:
        del run_dir
        return FakeScaleSession()
