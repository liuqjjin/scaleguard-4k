"""Offline, hash-bound image metric execution."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import math
import os
import platform
import shutil
import socket
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast
from unittest import mock

import numpy as np
import numpy.typing as npt
import PIL
from PIL import Image, UnidentifiedImageError

from scaleguard import __version__
from scaleguard.contracts import utc_now
from scaleguard.errors import ScaleGuardError
from scaleguard.evaluation.evidence import (
    EvaluationEvidenceError,
    canonical_sha256,
    load_json_object,
    require_text,
    resolved_distinct_paths,
    sha256_file,
    verify_artifact,
    write_json_atomic,
)
from scaleguard.manifest import ManifestValidationError, validate_run_manifest
from scaleguard.provenance import load_regular_file_snapshot

METRIC_RECEIPT_SCHEMA = "scaleguard.metric-receipt/v2"
SUPPORTED_METRICS = ("psnr", "ssim", "lpips", "musiq", "clipiqa")
PYIQA_METRICS = ("lpips", "musiq", "clipiqa")
FULL_REFERENCE_METRICS = ("psnr", "ssim", "lpips")
NO_REFERENCE_METRICS = ("musiq", "clipiqa")
PYIQA_VERSION = "0.1.16"
_PYIQA_WEIGHT_SHA256 = {
    "lpips": "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0",
    "musiq": "e95806b9eae5f3814c410f574ba8e552362bd5bc63d758ed5b97860f5d6185aa",
    "clipiqa": "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762",
}
_PYIQA_PROFILES: dict[str, dict[str, object]] = {
    "lpips": {
        "model": "LPIPS",
        "net": "alex",
        "version": "0.1",
        "spatial": False,
        "normalize_input": True,
    },
    "musiq": {
        "model": "MUSIQ",
        "pretrained_profile": "koniq10k",
    },
    "clipiqa": {
        "model": "CLIPIQA",
        "model_type": "clipiqa",
        "backbone": "RN50",
    },
}

_RGBArray = npt.NDArray[np.uint8]
_FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class _LoadedImage:
    pixels: _RGBArray
    evidence: dict[str, Any]


class _LearnedMetric(Protocol):
    metadata: dict[str, Any]

    def score(self, output_path: Path, reference_path: Path | None) -> float:
        """Return the native PyIQA score."""

    def close(self) -> None:
        """Release temporary cache storage."""


def _load_rgb8(path: Path, *, role: str) -> _LoadedImage:
    """Load an encoded 8-bit RGB image without implicit conversion or orientation."""

    try:
        with Image.open(path) as image:
            image.load()
            if image.mode != "RGB":
                raise EvaluationEvidenceError(
                    f"{role} must use Pillow mode RGB (8 bits/channel), observed {image.mode!r}: "
                    f"{path}"
                )
            orientation = image.getexif().get(274)
            if orientation not in (None, 1):
                raise EvaluationEvidenceError(
                    f"{role} has non-identity EXIF orientation {orientation}: {path}"
                )
            raw_profile = image.info.get("icc_profile")
            if raw_profile is not None and not isinstance(raw_profile, bytes):
                raise EvaluationEvidenceError(f"{role} has an invalid ICC profile: {path}")
            pixels = np.array(image, dtype=np.uint8, copy=True)
            if pixels.ndim != 3 or pixels.shape[2] != 3:
                raise EvaluationEvidenceError(f"{role} is not an HxWx3 RGB raster: {path}")
            profile_hash = None if raw_profile is None else hashlib.sha256(raw_profile).hexdigest()
            evidence = {
                "width": int(image.width),
                "height": int(image.height),
                "mode": image.mode,
                "bits_per_channel": 8,
                "exif_orientation": orientation,
                "icc_profile_sha256": profile_hash,
                "observed_code_min": int(pixels.min()),
                "observed_code_max": int(pixels.max()),
            }
    except EvaluationEvidenceError:
        raise
    except (OSError, UnidentifiedImageError) as error:
        raise EvaluationEvidenceError(f"cannot decode {role} image {path}: {error}") from error
    return _LoadedImage(pixels=pixels, evidence=evidence)


def _validate_pair(
    output_image: _LoadedImage,
    reference_image: _LoadedImage,
    *,
    crop_border: int,
    require_ssim_window: bool,
) -> tuple[_RGBArray, _RGBArray]:
    output_pixels = output_image.pixels
    reference_pixels = reference_image.pixels
    if output_pixels.shape != reference_pixels.shape:
        output_size = (output_pixels.shape[1], output_pixels.shape[0])
        reference_size = (reference_pixels.shape[1], reference_pixels.shape[0])
        raise EvaluationEvidenceError(
            f"output/reference dimensions differ: {output_size} != {reference_size}; "
            "implicit resize is forbidden"
        )
    if (
        output_image.evidence["icc_profile_sha256"]
        != reference_image.evidence["icc_profile_sha256"]
    ):
        raise EvaluationEvidenceError(
            "output/reference ICC profiles differ; implicit color conversion is forbidden"
        )
    height, width = output_pixels.shape[:2]
    if crop_border < 0:
        raise EvaluationEvidenceError("crop_border must be non-negative")
    if crop_border * 2 >= min(height, width):
        raise EvaluationEvidenceError(
            f"crop_border={crop_border} removes all pixels from {width}x{height} images"
        )
    if crop_border:
        output_pixels = output_pixels[
            crop_border : height - crop_border,
            crop_border : width - crop_border,
            :,
        ]
        reference_pixels = reference_pixels[
            crop_border : height - crop_border,
            crop_border : width - crop_border,
            :,
        ]
    if require_ssim_window and min(output_pixels.shape[:2]) < 11:
        raise EvaluationEvidenceError(
            "SSIM requires each post-crop dimension to be at least 11 pixels"
        )
    return output_pixels, reference_pixels


def _validate_declared_image(
    raw_artifact: Any,
    evidence: Mapping[str, Any],
    *,
    role: str,
) -> tuple[_LoadedImage, bool]:
    if not isinstance(raw_artifact, dict):
        raise EvaluationEvidenceError(f"{role} must be an artifact object")
    loaded = _load_rgb8(Path(str(evidence["verified_path"])), role=role)
    for dimension in ("width", "height"):
        declared = raw_artifact.get(dimension)
        observed = loaded.evidence[dimension]
        if type(declared) is not int or declared <= 0:
            raise EvaluationEvidenceError(f"{role}.{dimension} must be a positive integer")
        if declared != observed:
            raise EvaluationEvidenceError(
                f"{role}.{dimension} mismatch: declared {declared}, observed {observed}"
            )
    mock_artifact = raw_artifact.get("mock")
    if type(mock_artifact) is not bool:
        raise EvaluationEvidenceError(f"{role}.mock must be boolean")
    return loaded, mock_artifact


def _validate_metric_arrays(
    output: _RGBArray,
    reference: _RGBArray,
    *,
    minimum_dimension: int,
) -> None:
    for role, pixels in (("output", output), ("reference", reference)):
        if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
            raise EvaluationEvidenceError(f"{role} metric array must be uint8 HxWx3 RGB")
        if min(pixels.shape[:2]) < minimum_dimension:
            raise EvaluationEvidenceError(
                f"{role} metric array dimensions must be at least "
                f"{minimum_dimension}x{minimum_dimension}"
            )
    if output.shape != reference.shape:
        raise EvaluationEvidenceError("output/reference metric array dimensions differ")


def psnr_rgb(output: _RGBArray, reference: _RGBArray) -> float | str:
    """Return RGB PSNR in dB over code values normalized to [0, 1]."""

    _validate_metric_arrays(output, reference, minimum_dimension=1)
    difference = output.astype(np.float64) / 255.0 - reference.astype(np.float64) / 255.0
    mse = float(np.mean(difference * difference, dtype=np.float64))
    if mse == 0.0:
        return "infinity"
    return float(10.0 * math.log10(1.0 / mse))


def _gaussian_kernel() -> _FloatArray:
    positions = np.arange(-5, 6, dtype=np.float64)
    kernel = np.exp(-(positions * positions) / (2.0 * 1.5 * 1.5))
    return cast(_FloatArray, kernel / kernel.sum())


def _filter_valid(image: _FloatArray, kernel: _FloatArray) -> _FloatArray:
    height, width, channels = image.shape
    radius = kernel.size - 1
    horizontal = np.zeros((height, width - radius, channels), dtype=np.float64)
    for offset, weight in enumerate(kernel):
        horizontal += weight * image[:, offset : offset + horizontal.shape[1], :]
    vertical = np.zeros(
        (height - radius, horizontal.shape[1], channels),
        dtype=np.float64,
    )
    for offset, weight in enumerate(kernel):
        vertical += weight * horizontal[offset : offset + vertical.shape[0], :, :]
    return cast(_FloatArray, vertical)


def ssim_rgb(output: _RGBArray, reference: _RGBArray) -> float:
    """Return channel-averaged RGB SSIM with the fixed 11x11 Gaussian window."""

    _validate_metric_arrays(output, reference, minimum_dimension=11)
    x = output.astype(np.float64) / 255.0
    y = reference.astype(np.float64) / 255.0
    kernel = _gaussian_kernel()
    mu_x = _filter_valid(x, kernel)
    mu_y = _filter_valid(y, kernel)
    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x_sq = np.maximum(_filter_valid(x * x, kernel) - mu_x_sq, 0.0)
    sigma_y_sq = np.maximum(_filter_valid(y * y, kernel) - mu_y_sq, 0.0)
    sigma_xy = _filter_valid(x * y, kernel) - mu_xy
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    score = float(np.mean(numerator / denominator, dtype=np.float64))
    if not math.isfinite(score):
        raise EvaluationEvidenceError("SSIM produced a non-finite score")
    return score


def _blocked_network(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError("network access is disabled during PyIQA metric execution")


@contextmanager
def _offline_environment(torch_home: Path) -> Iterator[None]:
    environment = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": str(torch_home / "huggingface"),
        "HUGGINGFACE_HUB_CACHE": str(torch_home / "huggingface" / "hub"),
        "TRANSFORMERS_CACHE": str(torch_home / "huggingface" / "transformers"),
        "TORCH_HOME": str(torch_home),
        "XDG_CACHE_HOME": str(torch_home / "xdg"),
        "TORCH_FORCE_WEIGHTS_ONLY_LOAD": "1",
    }
    with (
        mock.patch.dict(os.environ, environment, clear=False),
        mock.patch.object(socket.socket, "connect", _blocked_network),
        mock.patch.object(socket, "create_connection", _blocked_network),
    ):
        os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
        yield


def _stage_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


class _PyiqaMetricRunner:
    def __init__(
        self,
        metric_name: str,
        *,
        weight_path: Path,
        backbone_path: Path | None,
        device: str,
    ) -> None:
        try:
            version = importlib.metadata.version("pyiqa")
        except importlib.metadata.PackageNotFoundError as error:
            raise EvaluationEvidenceError(
                "PyIQA is not installed; install the locked scaleguard-4k[metrics] extra"
            ) from error
        if version != PYIQA_VERSION:
            raise EvaluationEvidenceError(
                f"PyIQA version mismatch: expected {PYIQA_VERSION}, observed {version}"
            )
        expanded_weight = weight_path.expanduser()
        expanded_backbone = backbone_path.expanduser() if backbone_path is not None else None
        if not expanded_weight.is_file():
            raise EvaluationEvidenceError(
                f"explicit local PyIQA weight is unavailable for {metric_name}: {weight_path}"
            )
        if metric_name == "lpips" and (
            expanded_backbone is None or not expanded_backbone.is_file()
        ):
            raise EvaluationEvidenceError(
                "LPIPS requires --pyiqa-backbone lpips=PATH for the local AlexNet weights"
            )
        if metric_name != "lpips" and backbone_path is not None:
            raise EvaluationEvidenceError(
                f"{metric_name} does not accept a separate PyIQA backbone"
            )

        resolved_weight = expanded_weight.resolve()
        weight_hash = sha256_file(resolved_weight)
        resolved_backbone = expanded_backbone.resolve() if expanded_backbone is not None else None
        backbone_hash = sha256_file(resolved_backbone) if resolved_backbone is not None else None
        expected_weight_hash = _PYIQA_WEIGHT_SHA256[metric_name]
        if weight_hash != expected_weight_hash:
            raise EvaluationEvidenceError(
                f"{metric_name} weight must match the pinned canonical checkpoint "
                f"{expected_weight_hash}, observed {weight_hash}"
            )

        self._temporary = tempfile.TemporaryDirectory(prefix="scaleguard-pyiqa-")
        torch_home = Path(self._temporary.name) / "torch"
        self._source_files = [("weight", resolved_weight, weight_hash)]
        if resolved_backbone is not None and backbone_hash is not None:
            self._source_files.append(("backbone", resolved_backbone, backbone_hash))
        options: dict[str, object] = {}
        try:
            if metric_name in {"lpips", "musiq"}:
                options["pretrained_model_path"] = str(resolved_weight)
            if metric_name == "lpips":
                if resolved_backbone is None:
                    raise AssertionError("LPIPS backbone validation did not hold")
                _stage_file(
                    resolved_backbone,
                    torch_home / "hub" / "checkpoints" / "alexnet-owt-7be5be79.pth",
                )
            elif metric_name == "clipiqa":
                _stage_file(resolved_weight, torch_home / "hub" / "clip" / "RN50.pt")
            with _offline_environment(torch_home):
                pyiqa: Any = importlib.import_module("pyiqa")
                self._metric = pyiqa.create_metric(metric_name, device=device, **options)
                self._verify_source_files()
        except Exception as error:
            self._temporary.cleanup()
            raise EvaluationEvidenceError(
                f"cannot initialize offline PyIQA {metric_name}: {type(error).__name__}: {error}"
            ) from error
        lower_better = bool(getattr(self._metric, "lower_better", metric_name == "lpips"))
        self._metric_name = metric_name
        self._torch_home = torch_home
        actual_device = str(getattr(self._metric, "device", device))
        self.metadata = {
            "name": metric_name,
            "backend": "pyiqa",
            "backend_version": version,
            "implementation": "pyiqa.create_metric",
            "device": actual_device,
            "requested_device": device,
            "direction": "lower_is_better" if lower_better else "higher_is_better",
            "dependency_versions": {
                "pillow": PIL.__version__,
                "pyiqa": version,
                "torch": _installed_version("torch"),
            },
            "preprocessing": {
                "image_mode": "RGB uint8",
                "exif_orientation": "identity_only",
                "color_management": "no_implicit_conversion",
                "crop_border": 0,
                "reference_required": metric_name == "lpips",
                "implicit_resize": False,
            },
            "parameters": {
                "offline": True,
                "implicit_downloads": False,
                "profile": dict(_PYIQA_PROFILES[metric_name]),
                "weight": {
                    "path": str(resolved_weight),
                    "available": True,
                    "sha256": weight_hash,
                    "size_bytes": resolved_weight.stat().st_size,
                },
                "backbone": (
                    None
                    if resolved_backbone is None
                    else {
                        "path": str(resolved_backbone),
                        "available": True,
                        "sha256": backbone_hash,
                        "size_bytes": resolved_backbone.stat().st_size,
                    }
                ),
            },
        }
        self.metadata["identity_sha256"] = _metric_identity_sha256(self.metadata)

    def _verify_source_files(self) -> None:
        for role, path, expected_hash in self._source_files:
            observed_hash = sha256_file(path)
            if observed_hash != expected_hash:
                raise EvaluationEvidenceError(
                    f"PyIQA {role} changed during execution: "
                    f"expected {expected_hash}, observed {observed_hash}"
                )

    def score(self, output_path: Path, reference_path: Path | None) -> float:
        try:
            self._verify_source_files()
            with _offline_environment(self._torch_home):
                if self._metric_name == "lpips":
                    if reference_path is None:
                        raise EvaluationEvidenceError("LPIPS requires an aligned reference")
                    raw = self._metric(str(output_path), str(reference_path))
                else:
                    raw = self._metric(str(output_path))
            self._verify_source_files()
            detached = raw.detach().cpu() if hasattr(raw, "detach") else raw
            scalar = float(detached.item() if hasattr(detached, "item") else detached)
        except Exception as error:
            raise EvaluationEvidenceError(
                f"offline PyIQA {self._metric_name} failed: {type(error).__name__}: {error}"
            ) from error
        if not math.isfinite(scalar):
            raise EvaluationEvidenceError(
                f"offline PyIQA {self._metric_name} produced a non-finite score"
            )
        return scalar

    def close(self) -> None:
        self._temporary.cleanup()


def _metric_definition(name: str, *, crop_border: int, device: str) -> dict[str, Any]:
    if name == "psnr":
        return {
            "name": "psnr",
            "backend": "scaleguard.numpy",
            "backend_version": __version__,
            "implementation": "scaleguard.evaluation.metrics.psnr_rgb",
            "device": "cpu",
            "direction": "higher_is_better",
            "dependency_versions": {
                "numpy": np.__version__,
                "pillow": PIL.__version__,
            },
            "preprocessing": {
                "image_mode": "RGB uint8",
                "exif_orientation": "identity_only",
                "color_management": "matching_icc_digest",
                "crop_border": crop_border,
                "reference_required": True,
                "implicit_resize": False,
            },
            "parameters": {
                "channels": "RGB",
                "data_range": 1.0,
                "crop_border": crop_border,
                "aggregation": "MSE over all retained RGB samples",
                "exact_match_json_value": "infinity",
            },
        }
    if name == "ssim":
        return {
            "name": "ssim",
            "backend": "scaleguard.numpy",
            "backend_version": __version__,
            "implementation": "scaleguard.evaluation.metrics.ssim_rgb",
            "device": "cpu",
            "direction": "higher_is_better",
            "dependency_versions": {
                "numpy": np.__version__,
                "pillow": PIL.__version__,
            },
            "preprocessing": {
                "image_mode": "RGB uint8",
                "exif_orientation": "identity_only",
                "color_management": "matching_icc_digest",
                "crop_border": crop_border,
                "reference_required": True,
                "implicit_resize": False,
            },
            "parameters": {
                "channels": "RGB",
                "data_range": 1.0,
                "crop_border": crop_border,
                "window": "11x11 Gaussian",
                "sigma": 1.5,
                "k1": 0.01,
                "k2": 0.03,
                "padding": "valid",
                "covariance_normalization": "Gaussian population moments",
                "aggregation": "mean over spatial positions and RGB channels",
            },
        }
    return {
        "name": name,
        "backend": "pyiqa",
        "backend_version": PYIQA_VERSION,
        "implementation": "pyiqa.create_metric",
        "device": device,
        "direction": "lower_is_better" if name == "lpips" else "higher_is_better",
        "dependency_versions": {
            "pillow": PIL.__version__,
            "pyiqa": _installed_version("pyiqa"),
            "torch": _installed_version("torch"),
        },
        "preprocessing": {
            "image_mode": "RGB uint8",
            "exif_orientation": "identity_only",
            "color_management": "no_implicit_conversion",
            "crop_border": 0,
            "reference_required": name == "lpips",
            "implicit_resize": False,
        },
        "parameters": {
            "offline": True,
            "implicit_downloads": False,
            "profile": dict(_PYIQA_PROFILES[name]),
        },
    }


def _provided_file(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    evidence: dict[str, Any] = {
        "path": str(resolved),
        "available": resolved.is_file(),
        "sha256": None,
        "size_bytes": None,
    }
    if resolved.is_file():
        try:
            payload, digest = load_regular_file_snapshot(resolved, "metric model file")
            evidence["sha256"] = digest
            evidence["size_bytes"] = len(payload)
        except (ScaleGuardError, OSError) as error:
            evidence["error"] = f"{type(error).__name__}: {error}"
    return evidence


def _metric_identity_sha256(definition: Mapping[str, Any]) -> str:
    identity = dict(definition)
    identity.pop("identity_sha256", None)
    return canonical_sha256(identity)


def _request_definition(
    name: str,
    *,
    crop_border: int,
    device: str,
    weight: Path | None,
    backbone: Path | None,
) -> dict[str, Any]:
    definition = _metric_definition(name, crop_border=crop_border, device=device)
    if name in PYIQA_METRICS:
        definition["parameters"] = {
            **definition["parameters"],
            "weight": _provided_file(weight),
            "backbone": _provided_file(backbone),
        }
    definition["identity_sha256"] = _metric_identity_sha256(definition)
    return definition


def _prepare_metric_names(metric_names: Sequence[str]) -> tuple[str, ...]:
    if not metric_names:
        raise EvaluationEvidenceError("at least one metric is required")
    normalized = tuple(name.lower() for name in metric_names)
    unknown = sorted(set(normalized) - set(SUPPORTED_METRICS))
    if unknown:
        raise EvaluationEvidenceError("unsupported metrics: " + ", ".join(unknown))
    if len(set(normalized)) != len(normalized):
        raise EvaluationEvidenceError("metric names must be unique")
    return normalized


def _manifest_artifact_paths(
    manifest_path: Path,
    *,
    artifact_root: Path | None,
) -> list[tuple[str, Path]]:
    manifest, _digest = load_json_object(manifest_path, kind="run manifest")
    records: list[tuple[str, object]] = [
        ("input image", manifest.get("input_image")),
        ("restored image", manifest.get("restored_image")),
        ("final image", manifest.get("final_image")),
    ]
    steps = manifest.get("steps")
    if isinstance(steps, list):
        for index, step in enumerate(steps):
            if isinstance(step, Mapping):
                records.extend(
                    (
                        (f"step {index + 1} trusted image", step.get("trusted_before")),
                        (f"step {index + 1} candidate image", step.get("candidate")),
                    )
                )
    result: list[tuple[str, Path]] = []
    for label, record in records:
        if not isinstance(record, Mapping):
            continue
        declared = record.get("path")
        if not isinstance(declared, str) or not declared:
            continue
        path = Path(declared).expanduser()
        if not path.is_absolute():
            path = (artifact_root or manifest_path.parent) / path
        result.append((f"{manifest_path} {label}", path))
    return result


def _installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _run_sample(
    manifest_path: Path,
    reference_path: Path | None,
    *,
    artifact_root: Path | None,
    crop_border: int,
    device: str,
    metric_names: Sequence[str],
    request_definitions: Mapping[str, Mapping[str, Any]],
    learned: Mapping[str, _LearnedMetric],
    learned_errors: Mapping[str, str],
) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    manifest, manifest_hash = load_json_object(manifest_path, kind="run manifest")
    try:
        validated = validate_run_manifest(manifest_path, artifact_root=artifact_root)
    except ManifestValidationError as error:
        raise EvaluationEvidenceError(f"invalid run manifest {manifest_path}: {error}") from error
    after_validation, after_validation_hash = load_json_object(
        manifest_path,
        kind="run manifest",
    )
    if (
        validated != manifest
        or after_validation != manifest
        or after_validation_hash != manifest_hash
    ):
        raise EvaluationEvidenceError(
            f"run manifest changed while it was being validated: {manifest_path}"
        )
    run_id = require_text(manifest, "run_id", context=str(manifest_path))
    status = require_text(manifest, "status", context=str(manifest_path))
    mock_run = manifest.get("mock")
    if type(mock_run) is not bool:
        raise EvaluationEvidenceError(f"{manifest_path}.mock must be boolean")
    raw_input = manifest.get("input_image")
    raw_output = manifest.get("final_image")
    input_evidence = verify_artifact(
        raw_input,
        context=f"{manifest_path}.input_image",
        manifest_path=manifest_path,
        artifact_root=artifact_root,
    )
    output_evidence = verify_artifact(
        raw_output,
        context=f"{manifest_path}.final_image",
        manifest_path=manifest_path,
        artifact_root=artifact_root,
    )
    resolved_output = Path(output_evidence["verified_path"])
    require_reference = any(name in FULL_REFERENCE_METRICS for name in metric_names)
    resolved_reference: Path | None = None
    reference_hash: str | None = None
    if reference_path is not None:
        resolved_reference = reference_path.expanduser().resolve()
        if not resolved_reference.is_file():
            raise EvaluationEvidenceError(f"reference image is unavailable: {resolved_reference}")
        reference_hash = sha256_file(resolved_reference)
    elif require_reference:
        raise EvaluationEvidenceError("an aligned reference is required for PSNR, SSIM, or LPIPS")

    input_image, mock_input = _validate_declared_image(
        raw_input,
        input_evidence,
        role="input",
    )
    output_image, mock_output = _validate_declared_image(
        raw_output,
        output_evidence,
        role="output",
    )
    reference_image: _LoadedImage | None = None
    output_pixels: _RGBArray | None = None
    reference_pixels: _RGBArray | None = None
    if require_reference:
        if resolved_reference is None:
            raise AssertionError("reference requirement validation did not hold")
        reference_image = _load_rgb8(resolved_reference, role="reference")
        output_pixels, reference_pixels = _validate_pair(
            output_image,
            reference_image,
            crop_border=crop_border,
            require_ssim_window="ssim" in metric_names,
        )

    issues: list[str] = []
    if mock_run:
        issues.append("mock_run")
    if mock_input:
        issues.append("mock_input_artifact")
    if mock_run != mock_output:
        issues.append("mock_provenance_mismatch")
    if status not in {"succeeded", "succeeded_with_rollback"}:
        issues.append(f"run_status:{status}")
    results: list[dict[str, Any]] = []
    for name in metric_names:
        definition = dict(request_definitions[name])
        try:
            if name == "psnr":
                if output_pixels is None or reference_pixels is None:
                    raise AssertionError("PSNR reference validation did not hold")
                value: float | str = psnr_rgb(output_pixels, reference_pixels)
            elif name == "ssim":
                if output_pixels is None or reference_pixels is None:
                    raise AssertionError("SSIM reference validation did not hold")
                value = ssim_rgb(output_pixels, reference_pixels)
            else:
                definition = dict(learned[name].metadata)
                value = learned[name].score(resolved_output, resolved_reference)
            results.append({**definition, "status": "measured", "value": value})
        except (EvaluationEvidenceError, KeyError) as error:
            detail = learned_errors.get(name, str(error))
            issue = f"metric_failed:{name}:{detail}"
            issues.append(issue)
            results.append({**definition, "status": "failed", "value": None, "issue": detail})

    stable_files: list[tuple[str, Path, str]] = [
        ("manifest", manifest_path.resolve(), manifest_hash),
        ("input", Path(input_evidence["verified_path"]), input_evidence["sha256"]),
        ("output", resolved_output, output_evidence["sha256"]),
    ]
    if resolved_reference is not None and reference_hash is not None:
        stable_files.append(("reference", resolved_reference, reference_hash))
    for role, path, expected_hash in stable_files:
        observed_hash = sha256_file(path)
        if observed_hash != expected_hash:
            raise EvaluationEvidenceError(
                f"{role} changed during metric execution: "
                f"expected {expected_hash}, observed {observed_hash}"
            )

    return {
        "run_id": run_id,
        "run_status": status,
        "mock": mock_run,
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": manifest_hash,
        },
        "input_image": {
            **input_evidence,
            "image_contract": input_image.evidence,
        },
        "output_image": {
            **output_evidence,
            "image_contract": output_image.evidence,
        },
        "reference_image": (
            None
            if resolved_reference is None
            else {
                "path": str(resolved_reference),
                "sha256": reference_hash,
                "image_contract": (
                    reference_image.evidence if reference_image is not None else None
                ),
            }
        ),
        "metrics": results,
        "issues": issues,
    }


def evaluate_metric_receipt(
    manifest_paths: Sequence[Path],
    reference_paths: Sequence[Path | None] | None,
    output_path: Path,
    *,
    metric_names: Sequence[str] = ("psnr", "ssim"),
    crop_border: int = 0,
    device: str = "cpu",
    pyiqa_weights: Mapping[str, Path] | None = None,
    pyiqa_backbones: Mapping[str, Path] | None = None,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    """Evaluate one or many manifest/reference pairs and atomically write a receipt."""

    if not manifest_paths:
        raise EvaluationEvidenceError("at least one manifest/reference pair is required")
    resolved_metrics = _prepare_metric_names(metric_names)
    supplied_references = tuple(reference_paths or ())
    requires_reference = any(name in FULL_REFERENCE_METRICS for name in resolved_metrics)
    if requires_reference and len(manifest_paths) != len(supplied_references):
        raise EvaluationEvidenceError(
            "--manifest and --reference counts must match for full-reference metrics"
        )
    if (
        not requires_reference
        and supplied_references
        and (len(manifest_paths) != len(supplied_references))
    ):
        raise EvaluationEvidenceError(
            "when supplied, --reference counts must match --manifest counts"
        )
    references: tuple[Path | None, ...] = (
        supplied_references if supplied_references else tuple(None for _manifest in manifest_paths)
    )
    if crop_border < 0:
        raise EvaluationEvidenceError("crop_border must be non-negative")
    if not device.strip():
        raise EvaluationEvidenceError("device must be a non-empty PyTorch device string")
    if crop_border and any(name in PYIQA_METRICS for name in resolved_metrics):
        raise EvaluationEvidenceError(
            "crop_border must be zero when PyIQA metrics are requested; "
            "the adapter does not create unbound cropped image files"
        )
    manifest_identities = [str(path.expanduser().resolve()) for path in manifest_paths]
    if len(set(manifest_identities)) != len(manifest_identities):
        raise EvaluationEvidenceError("manifest paths must be unique within a metric receipt")
    weights = dict(pyiqa_weights or {})
    backbones = dict(pyiqa_backbones or {})
    invalid_weight_keys = sorted(set(weights) - set(PYIQA_METRICS))
    invalid_backbone_keys = sorted(set(backbones) - {"lpips"})
    if invalid_weight_keys:
        raise EvaluationEvidenceError("weights are only accepted for " + ", ".join(PYIQA_METRICS))
    if invalid_backbone_keys:
        raise EvaluationEvidenceError("only LPIPS accepts a separate PyIQA backbone")
    unrequested_weights = sorted(set(weights) - set(resolved_metrics))
    unrequested_backbones = sorted(set(backbones) - set(resolved_metrics))
    if unrequested_weights:
        raise EvaluationEvidenceError(
            "PyIQA weights supplied for unrequested metrics: " + ", ".join(unrequested_weights)
        )
    if unrequested_backbones:
        raise EvaluationEvidenceError(
            "PyIQA backbones supplied for unrequested metrics: " + ", ".join(unrequested_backbones)
        )

    protected_inputs: list[tuple[str, Path]] = [
        (f"manifest {index}", path) for index, path in enumerate(manifest_paths)
    ]
    protected_inputs.extend(
        (f"reference {index}", path) for index, path in enumerate(references) if path is not None
    )
    protected_inputs.extend((f"PyIQA {name} weight", path) for name, path in weights.items())
    protected_inputs.extend((f"PyIQA {name} backbone", path) for name, path in backbones.items())
    for manifest_path in manifest_paths:
        try:
            protected_inputs.extend(
                _manifest_artifact_paths(
                    manifest_path,
                    artifact_root=artifact_root,
                )
            )
        except EvaluationEvidenceError:
            # The sample path will produce a structured issue receipt below.
            pass
    resolved_output_path = resolved_distinct_paths(
        {"metric receipt output": output_path},
        inputs=protected_inputs,
    )["metric receipt output"]
    request_definitions = {
        name: _request_definition(
            name,
            crop_border=crop_border,
            device=device,
            weight=weights.get(name),
            backbone=backbones.get(name),
        )
        for name in resolved_metrics
    }

    learned: dict[str, _LearnedMetric] = {}
    learned_errors: dict[str, str] = {}
    for name in resolved_metrics:
        if name not in PYIQA_METRICS:
            continue
        weight = weights.get(name)
        if weight is None:
            learned_errors[name] = f"explicit local weight is required for PyIQA {name}"
            continue
        try:
            runner = _PyiqaMetricRunner(
                name,
                weight_path=weight,
                backbone_path=backbones.get(name),
                device=device,
            )
        except (EvaluationEvidenceError, OSError, ValueError) as error:
            learned_errors[name] = f"{type(error).__name__}: {error}"
            continue
        raw_requested_parameters = request_definitions[name]["parameters"]
        raw_observed_parameters = runner.metadata["parameters"]
        if not isinstance(raw_requested_parameters, dict) or not isinstance(
            raw_observed_parameters, dict
        ):
            runner.close()
            raise AssertionError("PyIQA metric parameters must be mappings")
        requested_parameters = raw_requested_parameters
        observed_parameters = raw_observed_parameters
        changed_roles = [
            role
            for role in ("weight", "backbone")
            if requested_parameters.get(role) != observed_parameters.get(role)
        ]
        if changed_roles:
            runner.close()
            learned_errors[name] = "PyIQA evidence changed before initialization: " + ", ".join(
                changed_roles
            )
            continue
        learned[name] = runner

    samples: list[dict[str, Any]] = []
    receipt_issues: list[str] = []
    try:
        for index, (manifest_path, reference_path) in enumerate(
            zip(manifest_paths, references, strict=True)
        ):
            try:
                sample = _run_sample(
                    manifest_path,
                    reference_path,
                    artifact_root=artifact_root,
                    crop_border=crop_border,
                    device=device,
                    metric_names=resolved_metrics,
                    request_definitions=request_definitions,
                    learned=learned,
                    learned_errors=learned_errors,
                )
            except (EvaluationEvidenceError, OSError, ValueError) as error:
                issue = f"sample_failed:{index}:{type(error).__name__}:{error}"
                receipt_issues.append(issue)
                failed_manifest: dict[str, Any] = {
                    "path": str(manifest_path.expanduser().resolve())
                }
                failed_run_id: str | None = None
                try:
                    failed_snapshot, failed_digest = load_json_object(
                        manifest_path.expanduser().resolve(),
                        kind="run manifest",
                    )
                    failed_manifest["sha256"] = failed_digest
                    raw_failed_run_id = failed_snapshot.get("run_id")
                    if isinstance(raw_failed_run_id, str) and raw_failed_run_id:
                        failed_run_id = raw_failed_run_id
                except (EvaluationEvidenceError, OSError, ValueError):
                    pass
                sample = {
                    "index": index,
                    "run_id": failed_run_id,
                    "manifest": failed_manifest,
                    "reference_image": (
                        {
                            "path": str(reference_path.expanduser().resolve()),
                            "sha256": (
                                sha256_file(reference_path.expanduser().resolve())
                                if reference_path.expanduser().resolve().is_file()
                                else None
                            ),
                        }
                        if reference_path is not None
                        else None
                    ),
                    "metrics": [],
                    "issues": [issue],
                }
            else:
                sample["index"] = index
                receipt_issues.extend(f"sample:{index}:{issue}" for issue in sample["issues"])
            samples.append(sample)
    finally:
        for closing_runner in learned.values():
            closing_runner.close()

    measured_count = sum(
        result.get("status") == "measured"
        for sample in samples
        for result in sample.get("metrics", [])
    )
    failed_metric_count = sum(
        result.get("status") == "failed"
        for sample in samples
        for result in sample.get("metrics", [])
    )
    requested_metric_count = len(samples) * len(resolved_metrics)
    payload: dict[str, Any] = {
        "schema_version": METRIC_RECEIPT_SCHEMA,
        "created_at": utc_now(),
        "status": "completed" if not receipt_issues else "completed_with_issues",
        "research_eligible": (not receipt_issues and measured_count == requested_metric_count),
        "contract": {
            "image_mode": "RGB",
            "bits_per_channel": 8,
            "code_value_normalization": "uint8 / 255",
            "color_management": "no conversion; ICC profile digests must match",
            "orientation": "stored pixels; EXIF orientation must be absent or identity",
            "alignment": "exact dimensions; no implicit resize",
            "crop_border": crop_border,
        },
        "metric_requests": [request_definitions[name] for name in resolved_metrics],
        "runtime": {
            "scaleguard_version": __version__,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "pillow_version": PIL.__version__,
            "pyiqa_version": _installed_version("pyiqa"),
            "torch_version": _installed_version("torch"),
            "platform": platform.platform(),
            "requested_pyiqa_device": device,
        },
        "counts": {
            "samples": len(samples),
            "samples_with_issues": sum(bool(sample["issues"]) for sample in samples),
            "metrics_requested": requested_metric_count,
            "metrics_measured": measured_count,
            "metrics_failed": failed_metric_count,
            "metrics_not_run": requested_metric_count - measured_count - failed_metric_count,
        },
        "samples": samples,
        "issues": receipt_issues,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    write_json_atomic(resolved_output_path, payload)
    return payload


def _receipt_score_equal(observed: object, replayed: object) -> bool:
    if observed == "infinity" or replayed == "infinity":
        return observed == replayed == "infinity"
    if (
        isinstance(observed, bool)
        or not isinstance(observed, (int, float))
        or isinstance(replayed, bool)
        or not isinstance(replayed, (int, float))
    ):
        return False
    observed_float = float(observed)
    replayed_float = float(replayed)
    return (
        math.isfinite(observed_float)
        and math.isfinite(replayed_float)
        and math.isclose(
            observed_float,
            replayed_float,
            rel_tol=1e-9,
            abs_tol=1e-10,
        )
    )


def _receipt_result_identity(raw_result: Mapping[str, Any]) -> str:
    definition = dict(raw_result)
    for field in ("identity_sha256", "status", "value", "issue"):
        definition.pop(field, None)
    return canonical_sha256(definition)


def _receipt_file_path(
    raw: object,
    *,
    context: str,
    required: bool,
) -> tuple[Path | None, bool]:
    if raw is None:
        if required:
            raise EvaluationEvidenceError(f"{context} is required")
        return None, False
    if not isinstance(raw, Mapping):
        raise EvaluationEvidenceError(f"{context} must be an object or null")
    if set(raw) != {"path", "available", "sha256", "size_bytes"}:
        raise EvaluationEvidenceError(f"{context} has unexpected or missing fields")
    raw_path = raw.get("path")
    raw_digest = raw.get("sha256")
    raw_size = raw.get("size_bytes")
    if not isinstance(raw_path, str) or not raw_path:
        raise EvaluationEvidenceError(f"{context}.path must be non-empty")
    path = Path(raw_path)
    if not path.is_absolute():
        raise EvaluationEvidenceError(f"{context}.path must be absolute")
    if (
        not isinstance(raw_digest, str)
        or len(raw_digest) != 64
        or any(character not in "0123456789abcdef" for character in raw_digest)
    ):
        raise EvaluationEvidenceError(f"{context}.sha256 must be a lowercase SHA256")
    if type(raw_size) is not int or raw_size < 0:
        raise EvaluationEvidenceError(f"{context}.size_bytes must be non-negative")
    if raw.get("available") is not True:
        raise EvaluationEvidenceError(f"{context}.available must be true")
    resolved = path.resolve()
    if not resolved.is_file():
        return resolved, False
    payload, digest = load_regular_file_snapshot(resolved, context)
    if digest != raw_digest or len(payload) != raw_size:
        raise EvaluationEvidenceError(f"{context} identity changed after receipt creation")
    return resolved, True


def _validate_receipt_request(
    raw: object,
    *,
    crop_border: int,
    requested_device: str,
    context: str,
) -> tuple[str, dict[str, Any], Path | None, Path | None, bool]:
    if not isinstance(raw, dict):
        raise EvaluationEvidenceError(f"{context} must be an object")
    name = require_text(raw, "name", context=context).lower()
    if name not in SUPPORTED_METRICS:
        raise EvaluationEvidenceError(f"{context}.name is unsupported: {name!r}")
    identity = raw.get("identity_sha256")
    if identity != _metric_identity_sha256(raw):
        raise EvaluationEvidenceError(f"{context}.identity_sha256 is invalid")
    if name in {"psnr", "ssim"}:
        expected = _request_definition(
            name,
            crop_border=crop_border,
            device=requested_device,
            weight=None,
            backbone=None,
        )
        if raw != expected:
            raise EvaluationEvidenceError(f"{context} definition differs from locked code")
        return name, raw, None, None, True

    expected_base = _metric_definition(name, crop_border=0, device=requested_device)
    if set(raw) != {*expected_base, "identity_sha256"}:
        raise EvaluationEvidenceError(f"{context} has unexpected or missing fields")
    for field in (
        "name",
        "backend",
        "backend_version",
        "implementation",
        "device",
        "direction",
        "preprocessing",
    ):
        if raw.get(field) != expected_base[field]:
            raise EvaluationEvidenceError(f"{context}.{field} differs from locked code")
    dependencies = raw.get("dependency_versions")
    if not isinstance(dependencies, dict) or set(dependencies) != {
        "pillow",
        "pyiqa",
        "torch",
    }:
        raise EvaluationEvidenceError(f"{context}.dependency_versions is invalid")
    if dependencies.get("pyiqa") != PYIQA_VERSION or not isinstance(
        dependencies.get("pillow"), str
    ):
        raise EvaluationEvidenceError(f"{context}.dependency_versions is unsupported")
    torch_version = dependencies.get("torch")
    if torch_version is not None and not isinstance(torch_version, str):
        raise EvaluationEvidenceError(f"{context}.dependency_versions.torch is invalid")
    parameters = raw.get("parameters")
    if not isinstance(parameters, dict):
        raise EvaluationEvidenceError(f"{context}.parameters must be an object")
    if set(parameters) != {"offline", "implicit_downloads", "profile", "weight", "backbone"}:
        raise EvaluationEvidenceError(f"{context}.parameters has unexpected fields")
    for field in ("offline", "implicit_downloads", "profile"):
        if parameters.get(field) != expected_base["parameters"][field]:
            raise EvaluationEvidenceError(f"{context}.parameters.{field} differs from locked code")
    weight, weight_available = _receipt_file_path(
        parameters.get("weight"),
        context=f"{context}.parameters.weight",
        required=True,
    )
    backbone, backbone_available = _receipt_file_path(
        parameters.get("backbone"),
        context=f"{context}.parameters.backbone",
        required=name == "lpips",
    )
    if name != "lpips" and parameters.get("backbone") is not None:
        raise EvaluationEvidenceError(f"{context} declares an unsupported backbone")
    return name, raw, weight, backbone, weight_available and (name != "lpips" or backbone_available)


def _verified_manifest_for_receipt_sample(
    raw_sample: Mapping[str, Any],
    *,
    artifact_root: Path | None,
    context: str,
) -> tuple[Path, str, str]:
    raw_manifest = raw_sample.get("manifest")
    if not isinstance(raw_manifest, Mapping):
        raise EvaluationEvidenceError(f"{context}.manifest must be an object")
    manifest_path_text = require_text(raw_manifest, "path", context=f"{context}.manifest")
    manifest_path = Path(manifest_path_text)
    if not manifest_path.is_absolute():
        raise EvaluationEvidenceError(f"{context}.manifest.path must be absolute")
    manifest_path = manifest_path.resolve()
    expected_digest = require_text(raw_manifest, "sha256", context=f"{context}.manifest")
    snapshot, observed_digest = load_json_object(manifest_path, kind="run manifest")
    if observed_digest != expected_digest:
        raise EvaluationEvidenceError(f"{context}.manifest SHA256 changed")
    try:
        validated = validate_run_manifest(manifest_path, artifact_root=artifact_root)
    except ManifestValidationError as error:
        raise EvaluationEvidenceError(f"{context}.manifest no longer validates: {error}") from error
    if validated != snapshot:
        raise EvaluationEvidenceError(f"{context}.manifest changed during validation")
    run_id = require_text(snapshot, "run_id", context=str(manifest_path))
    if raw_sample.get("run_id") != run_id:
        raise EvaluationEvidenceError(f"{context}.run_id differs from the manifest")
    for role, manifest_field, receipt_field in (
        ("input", "input_image", "input_image"),
        ("output", "final_image", "output_image"),
    ):
        receipt_artifact = raw_sample.get(receipt_field)
        if not isinstance(receipt_artifact, Mapping):
            raise EvaluationEvidenceError(f"{context}.{receipt_field} must be an object")
        evidence = verify_artifact(
            snapshot.get(manifest_field),
            context=f"{context}.{manifest_field}",
            manifest_path=manifest_path,
            artifact_root=artifact_root,
        )
        if receipt_artifact.get("sha256") != evidence["sha256"]:
            raise EvaluationEvidenceError(f"{context}.{role} SHA256 differs from manifest")
        receipt_path = receipt_artifact.get("verified_path")
        if (
            not isinstance(receipt_path, str)
            or Path(receipt_path).resolve() != Path(str(evidence["verified_path"])).resolve()
        ):
            raise EvaluationEvidenceError(f"{context}.{role} path differs from manifest")
    return manifest_path, observed_digest, run_id


def verify_metric_receipt(
    receipt_path: Path,
    *,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    """Replay a v2 metric receipt against its immutable source evidence.

    Structural, identity, or score drift raises ``EvaluationEvidenceError``. A
    learned metric whose locked local files or runtime are unavailable remains
    auditable, but is explicitly excluded from research aggregation.
    """

    path = receipt_path.expanduser().resolve()
    receipt, file_digest = load_json_object(path, kind="metric receipt")
    if set(receipt) != {
        "schema_version",
        "created_at",
        "status",
        "research_eligible",
        "contract",
        "metric_requests",
        "runtime",
        "counts",
        "samples",
        "issues",
        "receipt_sha256",
    }:
        raise EvaluationEvidenceError("metric receipt has unexpected or missing fields")
    if receipt.get("schema_version") != METRIC_RECEIPT_SCHEMA:
        raise EvaluationEvidenceError(
            f"unsupported metric receipt schema: {receipt.get('schema_version')!r}"
        )
    self_digest = receipt.get("receipt_sha256")
    if (
        not isinstance(self_digest, str)
        or len(self_digest) != 64
        or any(character not in "0123456789abcdef" for character in self_digest)
    ):
        raise EvaluationEvidenceError("metric receipt self digest is invalid")
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256", None)
    if canonical_sha256(unsigned) != self_digest:
        raise EvaluationEvidenceError("metric receipt self digest does not match its payload")

    contract = receipt.get("contract")
    if not isinstance(contract, dict):
        raise EvaluationEvidenceError("metric receipt contract must be an object")
    crop_border = contract.get("crop_border")
    if type(crop_border) is not int or crop_border < 0:
        raise EvaluationEvidenceError("metric receipt crop_border must be non-negative")
    expected_contract = {
        "image_mode": "RGB",
        "bits_per_channel": 8,
        "code_value_normalization": "uint8 / 255",
        "color_management": "no conversion; ICC profile digests must match",
        "orientation": "stored pixels; EXIF orientation must be absent or identity",
        "alignment": "exact dimensions; no implicit resize",
        "crop_border": crop_border,
    }
    if contract != expected_contract:
        raise EvaluationEvidenceError("metric receipt image contract differs from locked code")
    runtime = receipt.get("runtime")
    requested_device = (
        runtime.get("requested_pyiqa_device") if isinstance(runtime, Mapping) else None
    )
    if not isinstance(requested_device, str) or not requested_device.strip():
        raise EvaluationEvidenceError("metric receipt requested device is invalid")

    raw_requests = receipt.get("metric_requests")
    if not isinstance(raw_requests, list) or not raw_requests:
        raise EvaluationEvidenceError("metric receipt metric_requests must be non-empty")
    requests: dict[str, dict[str, Any]] = {}
    learned_paths: dict[str, tuple[Path | None, Path | None, bool]] = {}
    protected_paths: set[Path] = {path}
    for index, raw_request in enumerate(raw_requests):
        name, definition, weight, backbone, available = _validate_receipt_request(
            raw_request,
            crop_border=crop_border,
            requested_device=requested_device,
            context=f"metric receipt.metric_requests[{index}]",
        )
        if name in requests:
            raise EvaluationEvidenceError(f"metric receipt duplicates metric {name!r}")
        requests[name] = definition
        learned_paths[name] = (weight, backbone, available)
        if weight is not None:
            protected_paths.add(weight)
        if backbone is not None:
            protected_paths.add(backbone)

    learned: dict[str, _LearnedMetric] = {}
    replay_issues: list[str] = []
    for name, (weight, backbone, available) in learned_paths.items():
        if name not in PYIQA_METRICS:
            continue
        if not available or weight is None:
            replay_issues.append(f"learned_metric_source_unavailable:{name}")
            continue
        try:
            learned[name] = _PyiqaMetricRunner(
                name,
                weight_path=weight,
                backbone_path=backbone,
                device=requested_device,
            )
        except (EvaluationEvidenceError, OSError, ValueError) as error:
            replay_issues.append(f"learned_metric_replay_unavailable:{name}:{type(error).__name__}")

    raw_samples = receipt.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        for runner in learned.values():
            runner.close()
        raise EvaluationEvidenceError("metric receipt samples must be non-empty")
    verified_samples: list[dict[str, Any]] = []
    measured_count = 0
    failed_count = 0
    reconstructed_issues: list[str] = []
    try:
        for index, raw_sample in enumerate(raw_samples):
            context = f"metric receipt.samples[{index}]"
            if not isinstance(raw_sample, dict):
                raise EvaluationEvidenceError(f"{context} must be an object")
            if set(raw_sample) != {
                "index",
                "run_id",
                "run_status",
                "mock",
                "manifest",
                "input_image",
                "output_image",
                "reference_image",
                "metrics",
                "issues",
            }:
                raise EvaluationEvidenceError(f"{context} has unexpected or missing fields")
            if raw_sample.get("index") != index:
                raise EvaluationEvidenceError(f"{context}.index is not canonical")
            manifest_path, manifest_digest, run_id = _verified_manifest_for_receipt_sample(
                raw_sample,
                artifact_root=artifact_root,
                context=context,
            )
            protected_paths.add(manifest_path)
            raw_reference = raw_sample.get("reference_image")
            reference_path: Path | None = None
            reference_sha256: str | None = None
            if raw_reference is not None:
                if not isinstance(raw_reference, Mapping):
                    raise EvaluationEvidenceError(f"{context}.reference_image is invalid")
                reference_text = require_text(
                    raw_reference,
                    "path",
                    context=f"{context}.reference_image",
                )
                reference_path = Path(reference_text)
                if not reference_path.is_absolute() or not reference_path.resolve().is_file():
                    raise EvaluationEvidenceError(
                        f"{context}.reference_image is unavailable or non-absolute"
                    )
                reference_path = reference_path.resolve()
                protected_paths.add(reference_path)
                reference_sha256 = sha256_file(reference_path)
                if raw_reference.get("sha256") != reference_sha256:
                    raise EvaluationEvidenceError(f"{context}.reference_image SHA256 changed")

            replay_names = tuple(
                name for name in requests if name not in PYIQA_METRICS or name in learned
            )
            replay = _run_sample(
                manifest_path,
                reference_path,
                artifact_root=artifact_root,
                crop_border=crop_border,
                device=requested_device,
                metric_names=replay_names,
                request_definitions={name: requests[name] for name in replay_names},
                learned=learned,
                learned_errors={},
            )
            raw_sample_issues = raw_sample.get("issues")
            nonreplayable_metric_issues = tuple(
                f"metric_failed:{name}:"
                for name in requests
                if name in PYIQA_METRICS and name not in learned
            )
            comparable_sample_issues = (
                [
                    issue
                    for issue in raw_sample_issues
                    if not issue.startswith(nonreplayable_metric_issues)
                ]
                if isinstance(raw_sample_issues, list)
                and all(isinstance(issue, str) for issue in raw_sample_issues)
                else None
            )
            if (
                not isinstance(raw_sample_issues, list)
                or any(not isinstance(issue, str) for issue in raw_sample_issues)
                or comparable_sample_issues != replay.get("issues")
            ):
                raise EvaluationEvidenceError(f"{context}.issues differ on replay")
            reconstructed_issues.extend(f"sample:{index}:{issue}" for issue in raw_sample_issues)
            raw_results = raw_sample.get("metrics")
            if not isinstance(raw_results, list):
                raise EvaluationEvidenceError(f"{context}.metrics must be a list")
            result_by_name: dict[str, Mapping[str, Any]] = {}
            for result_index, raw_result in enumerate(raw_results):
                if not isinstance(raw_result, Mapping):
                    raise EvaluationEvidenceError(
                        f"{context}.metrics[{result_index}] must be an object"
                    )
                name = require_text(
                    raw_result,
                    "name",
                    context=f"{context}.metrics[{result_index}]",
                )
                if name not in requests or name in result_by_name:
                    raise EvaluationEvidenceError(
                        f"{context} contains an unknown or duplicate metric {name!r}"
                    )
                if raw_result.get("identity_sha256") != _receipt_result_identity(raw_result):
                    raise EvaluationEvidenceError(f"{context}.{name} result identity is invalid")
                result_by_name[name] = raw_result
            replay_by_name = {str(result["name"]): result for result in replay.get("metrics", [])}
            for field in (
                "run_id",
                "run_status",
                "mock",
                "manifest",
                "input_image",
                "output_image",
            ):
                if raw_sample.get(field) != replay.get(field):
                    raise EvaluationEvidenceError(f"{context}.{field} differs on replay")
            for artifact_field in ("input_image", "output_image"):
                replayed_artifact = replay.get(artifact_field)
                if not isinstance(replayed_artifact, Mapping):
                    raise AssertionError("replayed artifacts must be mappings")
                protected_paths.add(Path(str(replayed_artifact["verified_path"])).resolve())
            expected_reference = replay.get("reference_image")
            if raw_sample.get("reference_image") != expected_reference:
                if reference_path is None:
                    raise EvaluationEvidenceError(f"{context}.reference_image differs on replay")
                reference_loaded = _load_rgb8(reference_path, role="reference")
                full_reference_expected = {
                    "path": str(reference_path),
                    "sha256": sha256_file(reference_path),
                    "image_contract": reference_loaded.evidence,
                }
                if raw_sample.get("reference_image") != full_reference_expected:
                    raise EvaluationEvidenceError(f"{context}.reference_image differs on replay")
            verified_metrics: dict[str, dict[str, Any]] = {}
            for name, definition in requests.items():
                recorded = result_by_name.get(name)
                if recorded is None:
                    verified_metrics[name] = {
                        "status": "missing",
                        "value": None,
                        "direction": definition["direction"],
                        "identity_sha256": definition["identity_sha256"],
                    }
                    continue
                status = recorded.get("status")
                if status == "failed":
                    failed_count += 1
                    verified_metrics[name] = {
                        "status": "failed",
                        "value": None,
                        "direction": recorded.get("direction"),
                        "identity_sha256": recorded.get("identity_sha256"),
                    }
                    continue
                if status != "measured":
                    raise EvaluationEvidenceError(f"{context}.{name}.status is invalid")
                measured_count += 1
                replayable = name not in PYIQA_METRICS or name in learned
                if not replayable:
                    verified_metrics[name] = {
                        "status": "unverified",
                        "value": None,
                        "reported_value": recorded.get("value"),
                        "direction": recorded.get("direction"),
                        "identity_sha256": recorded.get("identity_sha256"),
                    }
                    continue
                replayed = replay_by_name.get(name)
                if replayed is None or replayed.get("status") != "measured":
                    raise EvaluationEvidenceError(f"{context}.{name} could not be replayed")
                if recorded.get("identity_sha256") != replayed.get("identity_sha256"):
                    raise EvaluationEvidenceError(f"{context}.{name} definition drifted")
                if not _receipt_score_equal(recorded.get("value"), replayed.get("value")):
                    raise EvaluationEvidenceError(f"{context}.{name} score differs on replay")
                verified_metrics[name] = {
                    "status": "measured",
                    "value": recorded.get("value"),
                    "direction": recorded.get("direction"),
                    "identity_sha256": recorded.get("identity_sha256"),
                }
            verified_samples.append(
                {
                    "manifest_path": str(manifest_path),
                    "manifest_sha256": manifest_digest,
                    "run_id": run_id,
                    "reference_sha256": reference_sha256,
                    "metrics": verified_metrics,
                }
            )
    finally:
        for runner in learned.values():
            runner.close()

    expected_requested = len(raw_samples) * len(requests)
    counts = receipt.get("counts")
    if not isinstance(counts, dict) or counts != {
        "samples": len(raw_samples),
        "samples_with_issues": sum(
            bool(sample.get("issues")) for sample in raw_samples if isinstance(sample, Mapping)
        ),
        "metrics_requested": expected_requested,
        "metrics_measured": measured_count,
        "metrics_failed": failed_count,
        "metrics_not_run": expected_requested - measured_count - failed_count,
    }:
        raise EvaluationEvidenceError("metric receipt counts are inconsistent")
    declared_research = receipt.get("research_eligible")
    if type(declared_research) is not bool:
        raise EvaluationEvidenceError("metric receipt research_eligible must be boolean")
    raw_receipt_issues = receipt.get("issues")
    if raw_receipt_issues != reconstructed_issues:
        raise EvaluationEvidenceError("metric receipt issues differ from replayed evidence")
    expected_status = "completed" if not reconstructed_issues else "completed_with_issues"
    if receipt.get("status") != expected_status:
        raise EvaluationEvidenceError("metric receipt status is inconsistent")
    declared_research_expected = not reconstructed_issues and measured_count == expected_requested
    if declared_research != declared_research_expected:
        raise EvaluationEvidenceError("metric receipt research_eligible is inconsistent")
    source_replay_complete = (
        not replay_issues
        and not reconstructed_issues
        and all(
            metric["status"] == "measured"
            for sample in verified_samples
            for metric in sample["metrics"].values()
        )
    )
    final_receipt_payload, final_file_digest = load_regular_file_snapshot(
        path,
        "metric receipt",
    )
    if final_file_digest != file_digest:
        raise EvaluationEvidenceError("metric receipt changed during source replay")
    return {
        "path": str(path),
        "size_bytes": len(final_receipt_payload),
        "sha256": file_digest,
        "receipt_sha256": self_digest,
        "verified": source_replay_complete,
        "research_eligible": bool(declared_research and source_replay_complete),
        "issues": replay_issues,
        "protected_paths": [str(source) for source in sorted(protected_paths)],
        "metric_definitions": {
            name: {
                "identity_sha256": definition["identity_sha256"],
                "direction": definition["direction"],
                "reference_required": definition["preprocessing"]["reference_required"],
            }
            for name, definition in requests.items()
        },
        "samples": verified_samples,
    }
