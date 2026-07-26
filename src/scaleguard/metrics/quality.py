"""Same-resolution quality scoring.

The bundled gradient proxy exists for CPU contract tests and threshold calibration
plumbing. It is deliberately not presented as a validated no-reference IQA model.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt
from PIL import Image


class QualityEvaluator(Protocol):
    name: str
    is_proxy: bool

    def score(self, image: Path) -> float:
        """Return a score where larger is better."""


def _luma(path: Path) -> npt.NDArray[np.float32]:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    luma = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
    return cast(npt.NDArray[np.float32], luma)


class GradientQualityEvaluator:
    """Small deterministic proxy for tests, not a research-quality IQA model."""

    name = "gradient_proxy_v1"
    is_proxy = True

    def score(self, image: Path) -> float:
        luma = _luma(image)
        if min(luma.shape) < 2:
            raise ValueError(
                "gradient quality proxy requires both image dimensions to be at least 2"
            )
        dx = np.diff(luma, axis=1)
        dy = np.diff(luma, axis=0)
        gradient = float(np.sqrt(np.mean(dx * dx) + np.mean(dy * dy)))
        clipped = float(np.mean((luma < 1.0 / 255.0) | (luma > 254.0 / 255.0)))
        # Log compression limits the incentive to amplify noise.
        return float(np.log1p(32.0 * gradient) - 0.25 * clipped)


class PyiqaQualityEvaluator:
    """Versioned PyIQA metric with direction normalized to higher-is-better."""

    is_proxy = False

    def __init__(
        self,
        metric_name: str,
        device: str,
        model_path: Path | None = None,
    ) -> None:
        try:
            pyiqa: Any = importlib.import_module("pyiqa")
        except ImportError as error:
            raise RuntimeError(
                "PyIQA quality backend is not installed; install scaleguard-4k[metrics]"
            ) from error
        self.name = f"pyiqa:{metric_name}"
        options: dict[str, object] = {}
        if model_path is not None:
            resolved = model_path.resolve()
            if not resolved.is_file():
                raise FileNotFoundError(f"PyIQA model weight is missing: {resolved}")
            options["pretrained_model_path"] = str(resolved)
        self._metric = pyiqa.create_metric(metric_name, device=device, **options)
        self._lower_better = bool(getattr(self._metric, "lower_better", False))

    def score(self, image: Path) -> float:
        value = self._metric(str(image))
        scalar = float(value.detach().cpu().item())
        return -scalar if self._lower_better else scalar


def build_quality_evaluator(
    backend: str,
    metric_name: str,
    device: str,
    model_path: Path | None = None,
) -> QualityEvaluator:
    if backend == "gradient_proxy":
        return GradientQualityEvaluator()
    if backend == "pyiqa":
        return PyiqaQualityEvaluator(metric_name, device, model_path)
    raise ValueError(f"unknown quality backend: {backend}")


def bicubic_baseline(source: Path, size: tuple[int, int], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.convert("RGB").resize(size, Image.Resampling.BICUBIC).save(destination, "PNG")
