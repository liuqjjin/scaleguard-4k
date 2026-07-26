"""Low-pass cross-scale consistency metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageFilter


@dataclass(frozen=True, slots=True)
class ScaleConsistency:
    nrmse: float
    edge_mae: float


def _rgb(path: Path, size: tuple[int, int] | None = None) -> npt.NDArray[np.float32]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        if size is not None and rgb.size != size:
            ratio = max(rgb.width / size[0], rgb.height / size[1])
            radius = max(0.5, ratio / 2.0)
            rgb = rgb.filter(ImageFilter.GaussianBlur(radius=radius)).resize(
                size, Image.Resampling.LANCZOS
            )
        return np.asarray(rgb, dtype=np.float32) / 255.0


def _gradient(
    array: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    return np.diff(array, axis=1), np.diff(array, axis=0)


def evaluate_scale_consistency(candidate: Path, trusted: Path) -> ScaleConsistency:
    with Image.open(trusted) as reference:
        target_size = reference.size
    if min(target_size) < 2:
        raise ValueError(
            "cross-scale edge consistency requires both trusted dimensions to be at least 2"
        )
    reference_rgb = _rgb(trusted)
    candidate_rgb = _rgb(candidate, target_size)
    residual = candidate_rgb - reference_rgb
    denominator = float(np.sqrt(np.mean(reference_rgb * reference_rgb)) + 1e-6)
    nrmse = float(np.sqrt(np.mean(residual * residual)) / denominator)
    candidate_dx, candidate_dy = _gradient(candidate_rgb)
    reference_dx, reference_dy = _gradient(reference_rgb)
    edge_mae = float(
        0.5
        * (
            np.mean(np.abs(candidate_dx - reference_dx))
            + np.mean(np.abs(candidate_dy - reference_dy))
        )
    )
    return ScaleConsistency(nrmse=nrmse, edge_mae=edge_mae)
