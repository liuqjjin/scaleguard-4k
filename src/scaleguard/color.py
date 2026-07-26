"""A single, explicit final color-alignment policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image


def apply_adain(candidate: Path, reference: Path, destination: Path) -> None:
    """Match candidate channel statistics to an upsampled trusted reference once."""

    with Image.open(candidate) as candidate_image, Image.open(reference) as reference_image:
        candidate_rgb = np.asarray(candidate_image.convert("RGB"), dtype=np.float32)
        reference_rgb = np.asarray(
            reference_image.convert("RGB").resize(candidate_image.size, Image.Resampling.BICUBIC),
            dtype=np.float32,
        )
    candidate_mean = candidate_rgb.mean(axis=(0, 1), keepdims=True)
    candidate_std = candidate_rgb.std(axis=(0, 1), keepdims=True)
    reference_mean = reference_rgb.mean(axis=(0, 1), keepdims=True)
    reference_std = reference_rgb.std(axis=(0, 1), keepdims=True)
    aligned = (candidate_rgb - candidate_mean) / np.maximum(candidate_std, 1e-6) * np.maximum(
        reference_std, 1e-6
    ) + reference_mean
    destination.parent.mkdir(parents=True, exist_ok=True)
    output_array = np.uint8(np.clip(aligned, 0.0, 255.0))
    Image.fromarray(cast(Any, output_array), mode="RGB").save(destination, "PNG")
