"""Recorded forward operators for controlled computational-imaging experiments."""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import PIL
from PIL import Image, ImageFilter

from scaleguard.errors import ConfigurationError


class ForwardModel(Protocol):
    @property
    def name(self) -> str:
        """Stable identifier recorded in the run manifest."""

    @property
    def identity(self) -> dict[str, Any]:
        """Canonical implementation and parameter identity."""

    def apply(self, image: Image.Image, output_size: tuple[int, int]) -> Image.Image:
        """Map a reconstruction into observation space."""


@dataclass(frozen=True, slots=True)
class ResizeModel:
    name: str = "resize_lanczos"

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "implementation": "scaleguard.pillow",
            "version": 1,
            "dependency_versions": {"pillow": PIL.__version__},
            "preprocessing": {"color_mode": "RGB", "resize": "LANCZOS"},
            "parameters": {},
        }

    def apply(self, image: Image.Image, output_size: tuple[int, int]) -> Image.Image:
        return image.convert("RGB").resize(output_size, Image.Resampling.LANCZOS)


@dataclass(frozen=True, slots=True)
class GaussianPSFModel:
    sigma: float = 1.2
    name: str = "gaussian_psf_resize"

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "implementation": "scaleguard.pillow",
            "version": 1,
            "dependency_versions": {"pillow": PIL.__version__},
            "preprocessing": {"color_mode": "RGB", "resize": "LANCZOS"},
            "parameters": {"sigma": self.sigma},
        }

    def apply(self, image: Image.Image, output_size: tuple[int, int]) -> Image.Image:
        return (
            image.convert("RGB")
            .filter(ImageFilter.GaussianBlur(radius=self.sigma))
            .resize(output_size, Image.Resampling.LANCZOS)
        )


@dataclass(frozen=True, slots=True)
class JPEGModel:
    quality: int = 75
    name: str = "jpeg_resize"

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "implementation": "scaleguard.pillow",
            "version": 1,
            "dependency_versions": {"pillow": PIL.__version__},
            "preprocessing": {"color_mode": "RGB", "resize": "LANCZOS"},
            "parameters": {
                "quality": self.quality,
                "subsampling": 0,
            },
        }

    def apply(self, image: Image.Image, output_size: tuple[int, int]) -> Image.Image:
        resized = image.convert("RGB").resize(output_size, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        resized.save(buffer, format="JPEG", quality=self.quality, subsampling=0)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB").copy()


@dataclass(frozen=True, slots=True)
class PoissonGaussianModel:
    peak_photons: float = 60.0
    read_noise_std: float = 0.01
    seed: int = 0
    name: str = "poisson_gaussian_resize"

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "implementation": "scaleguard.numpy-pillow",
            "version": 1,
            "dependency_versions": {
                "numpy": np.__version__,
                "pillow": PIL.__version__,
            },
            "preprocessing": {"color_mode": "RGB", "resize": "LANCZOS"},
            "parameters": {
                "peak_photons": self.peak_photons,
                "read_noise_std": self.read_noise_std,
                "seed": self.seed,
            },
        }

    def apply(self, image: Image.Image, output_size: tuple[int, int]) -> Image.Image:
        resized = image.convert("RGB").resize(output_size, Image.Resampling.LANCZOS)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        rng = np.random.default_rng(self.seed)
        shot = rng.poisson(array * self.peak_photons) / self.peak_photons
        noisy = shot + rng.normal(0.0, self.read_noise_std, array.shape)
        output = np.uint8(np.clip(noisy, 0.0, 1.0) * 255.0)
        return Image.fromarray(cast(Any, output), mode="RGB")


@dataclass(frozen=True, slots=True)
class HazeModel:
    transmission: float = 0.75
    atmospheric_light: float = 0.9
    name: str = "uniform_haze_resize"

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "implementation": "scaleguard.numpy-pillow",
            "version": 1,
            "dependency_versions": {
                "numpy": np.__version__,
                "pillow": PIL.__version__,
            },
            "preprocessing": {"color_mode": "RGB", "resize": "LANCZOS"},
            "parameters": {
                "transmission": self.transmission,
                "atmospheric_light": self.atmospheric_light,
            },
        }

    def apply(self, image: Image.Image, output_size: tuple[int, int]) -> Image.Image:
        resized = image.convert("RGB").resize(output_size, Image.Resampling.LANCZOS)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        hazy = array * self.transmission + self.atmospheric_light * (1.0 - self.transmission)
        output = np.uint8(np.clip(hazy, 0.0, 1.0) * 255.0)
        return Image.fromarray(cast(Any, output), mode="RGB")


@dataclass(frozen=True, slots=True)
class MeasurementConsistency:
    nrmse: float
    model: str


def _number(parameters: dict[str, Any], key: str, default: float) -> float:
    value = parameters.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"measurement parameter '{key}' must be numeric and finite")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigurationError(f"measurement parameter '{key}' must be numeric and finite")
    return result


def _reject_unknown(parameters: dict[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(parameters) - allowed)
    if unknown:
        raise ConfigurationError("unknown measurement parameters: " + ", ".join(unknown))


def build_forward_model(name: str, parameters: dict[str, Any]) -> ForwardModel:
    if name == "resize":
        _reject_unknown(parameters, set())
        return ResizeModel()
    if name == "gaussian_psf":
        _reject_unknown(parameters, {"sigma"})
        sigma = _number(parameters, "sigma", 1.2)
        if sigma < 0:
            raise ConfigurationError("Gaussian PSF sigma must be non-negative")
        return GaussianPSFModel(sigma=sigma)
    if name == "jpeg":
        _reject_unknown(parameters, {"quality"})
        quality_value = _number(parameters, "quality", 75)
        if not quality_value.is_integer():
            raise ConfigurationError("JPEG quality must be an integer")
        quality = int(quality_value)
        if not 1 <= quality <= 100:
            raise ConfigurationError("JPEG quality must be between 1 and 100")
        return JPEGModel(quality=quality)
    if name == "poisson_gaussian":
        _reject_unknown(parameters, {"peak_photons", "read_noise_std", "seed"})
        peak_photons = _number(parameters, "peak_photons", 60.0)
        read_noise_std = _number(parameters, "read_noise_std", 0.01)
        seed = parameters.get("seed", 0)
        if peak_photons <= 0:
            raise ConfigurationError("peak_photons must be positive")
        if read_noise_std < 0:
            raise ConfigurationError("read_noise_std must be non-negative")
        if type(seed) is not int or not 0 <= seed <= 2**63 - 1:
            raise ConfigurationError("measurement seed must be an integer between 0 and 2^63-1")
        return PoissonGaussianModel(
            peak_photons=peak_photons,
            read_noise_std=read_noise_std,
            seed=seed,
        )
    if name == "haze":
        _reject_unknown(parameters, {"transmission", "atmospheric_light"})
        transmission = _number(parameters, "transmission", 0.75)
        atmospheric_light = _number(parameters, "atmospheric_light", 0.9)
        if not 0 <= transmission <= 1:
            raise ConfigurationError("haze transmission must be between 0 and 1")
        if not 0 <= atmospheric_light <= 1:
            raise ConfigurationError("atmospheric_light must be between 0 and 1")
        return HazeModel(
            transmission=transmission,
            atmospheric_light=atmospheric_light,
        )
    raise ConfigurationError(
        f"unknown measurement model '{name}'; expected resize, gaussian_psf, jpeg, "
        "poisson_gaussian, or haze"
    )


def evaluate_measurement_consistency(
    candidate: Path,
    observation: Path,
    model: ForwardModel,
) -> MeasurementConsistency:
    with Image.open(candidate) as high_resolution, Image.open(observation) as observed:
        observed_rgb = observed.convert("RGB")
        predicted = model.apply(high_resolution, observed_rgb.size)
        predicted_array = np.asarray(predicted, dtype=np.float32) / 255.0
        observed_array = np.asarray(observed_rgb, dtype=np.float32) / 255.0
    residual = predicted_array - observed_array
    denominator = float(np.sqrt(np.mean(observed_array * observed_array)) + 1e-6)
    return MeasurementConsistency(
        nrmse=float(np.sqrt(np.mean(residual * residual)) / denominator),
        model=model.name,
    )
