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
from scaleguard.evaluation.evidence import (
    EvaluationEvidenceError,
    canonical_sha256,
    load_json_object,
    require_text,
    sha256_file,
    verify_artifact,
    write_json_atomic,
)

METRIC_RECEIPT_SCHEMA = "scaleguard.metric-receipt/v1"
SUPPORTED_METRICS = ("psnr", "ssim", "lpips", "musiq", "clipiqa")
PYIQA_METRICS = ("lpips", "musiq", "clipiqa")
PYIQA_VERSION = "0.1.16"
_CLIPIQA_RN50_SHA256 = "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762"
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

    def score(self, output_path: Path, reference_path: Path) -> float:
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
    }
    with (
        mock.patch.dict(os.environ, environment, clear=False),
        mock.patch.object(socket.socket, "connect", _blocked_network),
        mock.patch.object(socket, "create_connection", _blocked_network),
    ):
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
        if metric_name == "clipiqa" and weight_hash != _CLIPIQA_RN50_SHA256:
            raise EvaluationEvidenceError(
                "CLIPIQA weight must be the pinned OpenAI RN50 checkpoint "
                f"{_CLIPIQA_RN50_SHA256}, observed {weight_hash}"
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
            "device": actual_device,
            "requested_device": device,
            "direction": "lower_is_better" if lower_better else "higher_is_better",
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

    def _verify_source_files(self) -> None:
        for role, path, expected_hash in self._source_files:
            observed_hash = sha256_file(path)
            if observed_hash != expected_hash:
                raise EvaluationEvidenceError(
                    f"PyIQA {role} changed during execution: "
                    f"expected {expected_hash}, observed {observed_hash}"
                )

    def score(self, output_path: Path, reference_path: Path) -> float:
        try:
            self._verify_source_files()
            with _offline_environment(self._torch_home):
                if self._metric_name == "lpips":
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
            "device": "cpu",
            "direction": "higher_is_better",
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
            "device": "cpu",
            "direction": "higher_is_better",
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
        "device": device,
        "direction": "lower_is_better" if name == "lpips" else "higher_is_better",
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
            evidence["sha256"] = sha256_file(resolved)
            evidence["size_bytes"] = resolved.stat().st_size
        except (EvaluationEvidenceError, OSError) as error:
            evidence["error"] = f"{type(error).__name__}: {error}"
    return evidence


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


def _installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _run_sample(
    manifest_path: Path,
    reference_path: Path,
    *,
    artifact_root: Path | None,
    crop_border: int,
    device: str,
    metric_names: Sequence[str],
    request_definitions: Mapping[str, Mapping[str, Any]],
    learned: Mapping[str, _LearnedMetric],
    learned_errors: Mapping[str, str],
) -> dict[str, Any]:
    manifest, manifest_hash = load_json_object(manifest_path, kind="run manifest")
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
    resolved_reference = reference_path.expanduser().resolve()
    if not resolved_reference.is_file():
        raise EvaluationEvidenceError(f"reference image is unavailable: {resolved_reference}")
    reference_hash = sha256_file(resolved_reference)

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
                value: float | str = psnr_rgb(output_pixels, reference_pixels)
            elif name == "ssim":
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

    stable_files = (
        ("manifest", manifest_path.resolve(), manifest_hash),
        ("input", Path(input_evidence["verified_path"]), input_evidence["sha256"]),
        ("output", resolved_output, output_evidence["sha256"]),
        ("reference", resolved_reference, reference_hash),
    )
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
        "reference_image": {
            "path": str(resolved_reference),
            "sha256": reference_hash,
            "image_contract": reference_image.evidence,
        },
        "metrics": results,
        "issues": issues,
    }


def evaluate_metric_receipt(
    manifest_paths: Sequence[Path],
    reference_paths: Sequence[Path],
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
    if len(manifest_paths) != len(reference_paths):
        raise EvaluationEvidenceError("--manifest and --reference counts must match")
    resolved_metrics = _prepare_metric_names(metric_names)
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
            zip(manifest_paths, reference_paths, strict=True)
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
                sample = {
                    "index": index,
                    "manifest": {"path": str(manifest_path.resolve())},
                    "reference_image": {"path": str(reference_path.resolve())},
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
    write_json_atomic(output_path, payload)
    return payload
