"""Calibrate gate thresholds from human labels and immutable run evidence."""

from __future__ import annotations

import csv
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from scaleguard.config import PipelineConfig, load_config
from scaleguard.errors import ScaleGuardError
from scaleguard.evaluation.evidence import (
    EvaluationEvidenceError,
    canonical_sha256,
    load_json_object,
    optional_finite_number,
    require_finite_number,
    require_text,
    sha256_file,
    verify_artifact,
)
from scaleguard.imaging.forward_models import build_forward_model

RECEIPT_SCHEMA = "scaleguard.calibration-receipt/v1"


@dataclass(frozen=True, slots=True)
class CalibrationParameters:
    """Deterministic settings for the acceptable-sample envelope."""

    minimum_acceptable_samples: int = 20
    quality_lower_quantile: float = 0.05
    error_upper_quantile: float = 0.95
    bootstrap_samples: int = 2000
    bootstrap_confidence: float = 0.95
    bootstrap_seed: int = 20250727
    include_measurement: bool = False

    def validate(self) -> None:
        if self.minimum_acceptable_samples < 1:
            raise EvaluationEvidenceError("minimum_acceptable_samples must be positive")
        if not 0.0 <= self.quality_lower_quantile <= 1.0:
            raise EvaluationEvidenceError("quality_lower_quantile must be between 0 and 1")
        if not 0.0 <= self.error_upper_quantile <= 1.0:
            raise EvaluationEvidenceError("error_upper_quantile must be between 0 and 1")
        if self.bootstrap_samples < 1:
            raise EvaluationEvidenceError("bootstrap_samples must be positive")
        if not 0.0 < self.bootstrap_confidence < 1.0:
            raise EvaluationEvidenceError("bootstrap_confidence must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class _Label:
    run_id: str
    step_index: int
    acceptable: bool


@dataclass(frozen=True, slots=True)
class _GateSample:
    run_id: str
    step_index: int
    acceptable: bool
    mock: bool
    quality_backend: str
    quality_gain: float
    scale_nrmse: float
    scale_edge_mae: float
    measurement_nrmse: float | None
    measurement_model: str | None


def _parse_acceptable(value: str, *, row_number: int) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true"}:
        return True
    if normalized in {"0", "false"}:
        return False
    raise EvaluationEvidenceError(
        f"labels row {row_number}: acceptable must be one of 0, 1, false, true"
    )


def _load_labels(path: Path) -> tuple[dict[tuple[str, int], _Label], str]:
    labels_hash = sha256_file(path)
    try:
        handle = path.open("r", encoding="utf-8-sig", newline="")
    except OSError as error:
        raise EvaluationEvidenceError(f"cannot read labels CSV {path}: {error}") from error
    with handle:
        reader = csv.DictReader(handle)
        required = ("run_id", "step_index", "acceptable")
        fields = tuple(reader.fieldnames or ())
        if fields != required:
            raise EvaluationEvidenceError(
                "labels CSV header must be exactly: " + ",".join(required)
            )
        labels: dict[tuple[str, int], _Label] = {}
        for row_number, row in enumerate(reader, start=2):
            run_id = (row.get("run_id") or "").strip()
            if not run_id:
                raise EvaluationEvidenceError(f"labels row {row_number}: run_id is empty")
            try:
                step_index = int((row.get("step_index") or "").strip())
            except ValueError as error:
                raise EvaluationEvidenceError(
                    f"labels row {row_number}: step_index must be an integer"
                ) from error
            if step_index < 1:
                raise EvaluationEvidenceError(
                    f"labels row {row_number}: step_index must be positive"
                )
            key = (run_id, step_index)
            if key in labels:
                raise EvaluationEvidenceError(
                    f"duplicate label for run_id={run_id!r}, step_index={step_index}"
                )
            labels[key] = _Label(
                run_id=run_id,
                step_index=step_index,
                acceptable=_parse_acceptable(row.get("acceptable") or "", row_number=row_number),
            )
    if not labels:
        raise EvaluationEvidenceError("labels CSV contains no data rows")
    return labels, labels_hash


def _manifest_samples(
    path: Path,
    *,
    artifact_root: Path | None,
) -> tuple[str, list[_GateSample], dict[str, Any], set[tuple[str, int]]]:
    manifest, manifest_hash = load_json_object(path, kind="run manifest")
    run_id = require_text(manifest, "run_id", context=str(path))
    mock = manifest.get("mock")
    if type(mock) is not bool:
        raise EvaluationEvidenceError(f"{path}.mock must be boolean")
    raw_steps = manifest.get("steps")
    if not isinstance(raw_steps, list):
        raise EvaluationEvidenceError(f"{path}.steps must be a list")

    samples: list[_GateSample] = []
    metric_keys: set[tuple[str, int]] = set()
    seen_indices: set[int] = set()
    for position, raw_step in enumerate(raw_steps):
        context = f"{path}.steps[{position}]"
        if not isinstance(raw_step, dict):
            raise EvaluationEvidenceError(f"{context} must be an object")
        index_value = raw_step.get("index")
        if type(index_value) is not int or index_value < 1:
            raise EvaluationEvidenceError(f"{context}.index must be a positive integer")
        if index_value in seen_indices:
            raise EvaluationEvidenceError(f"{path} contains duplicate step index {index_value}")
        seen_indices.add(index_value)
        raw_metrics = raw_step.get("metrics")
        if raw_metrics is None:
            continue
        if not isinstance(raw_metrics, dict):
            raise EvaluationEvidenceError(f"{context}.metrics must be an object or null")
        verify_artifact(
            raw_step.get("trusted_before"),
            context=f"{context}.trusted_before",
            manifest_path=path,
            artifact_root=artifact_root,
        )
        verify_artifact(
            raw_step.get("candidate"),
            context=f"{context}.candidate",
            manifest_path=path,
            artifact_root=artifact_root,
        )
        quality_backend = require_text(
            raw_metrics,
            "quality_backend",
            context=f"{context}.metrics",
        )
        measurement_nrmse = optional_finite_number(
            raw_metrics,
            "measurement_nrmse",
            context=f"{context}.metrics",
        )
        measurement_model_value = raw_metrics.get("measurement_model")
        if measurement_model_value is not None and not isinstance(measurement_model_value, str):
            raise EvaluationEvidenceError(
                f"{context}.metrics.measurement_model must be a string or null"
            )
        key = (run_id, index_value)
        metric_keys.add(key)
        samples.append(
            _GateSample(
                run_id=run_id,
                step_index=index_value,
                acceptable=False,
                mock=mock,
                quality_backend=quality_backend,
                quality_gain=require_finite_number(
                    raw_metrics,
                    "quality_gain",
                    context=f"{context}.metrics",
                ),
                scale_nrmse=require_finite_number(
                    raw_metrics,
                    "scale_nrmse",
                    context=f"{context}.metrics",
                ),
                scale_edge_mae=require_finite_number(
                    raw_metrics,
                    "scale_edge_mae",
                    context=f"{context}.metrics",
                ),
                measurement_nrmse=measurement_nrmse,
                measurement_model=measurement_model_value,
            )
        )
    evidence = {
        "path": str(path),
        "sha256": manifest_hash,
        "run_id": run_id,
        "mock": mock,
        "metric_step_count": len(samples),
    }
    return run_id, samples, evidence, metric_keys


def _with_labels(
    samples: Sequence[_GateSample],
    labels: Mapping[tuple[str, int], _Label],
) -> list[_GateSample]:
    return [
        _GateSample(
            run_id=sample.run_id,
            step_index=sample.step_index,
            acceptable=labels[(sample.run_id, sample.step_index)].acceptable,
            mock=sample.mock,
            quality_backend=sample.quality_backend,
            quality_gain=sample.quality_gain,
            scale_nrmse=sample.scale_nrmse,
            scale_edge_mae=sample.scale_edge_mae,
            measurement_nrmse=sample.measurement_nrmse,
            measurement_model=sample.measurement_model,
        )
        for sample in samples
    ]


def _quantile_estimate(
    values: npt.NDArray[np.float64],
    *,
    quantile: float,
    rng: np.random.Generator,
    bootstrap_samples: int,
    confidence: float,
) -> dict[str, Any]:
    estimate = float(np.quantile(values, quantile, method="linear"))
    bootstrapped = np.empty(bootstrap_samples, dtype=np.float64)
    batch_size = min(128, bootstrap_samples)
    for start in range(0, bootstrap_samples, batch_size):
        stop = min(start + batch_size, bootstrap_samples)
        indexes = rng.integers(
            0,
            values.size,
            size=(stop - start, values.size),
            endpoint=False,
        )
        bootstrapped[start:stop] = np.quantile(
            values[indexes],
            quantile,
            axis=1,
            method="linear",
        )
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(
        bootstrapped,
        [alpha, 1.0 - alpha],
        method="linear",
    )
    return {
        "value": estimate,
        "bootstrap_ci": {
            "lower": float(lower),
            "upper": float(upper),
            "confidence": confidence,
        },
        "quantile": quantile,
    }


def _thresholds(
    samples: Sequence[_GateSample],
    parameters: CalibrationParameters,
) -> dict[str, dict[str, Any]]:
    if not samples:
        return {}
    rng = np.random.default_rng(parameters.bootstrap_seed)
    definitions: list[tuple[str, npt.NDArray[np.float64], float, str]] = [
        (
            "min_quality_gain",
            np.asarray([sample.quality_gain for sample in samples], dtype=np.float64),
            parameters.quality_lower_quantile,
            "minimum",
        ),
        (
            "max_scale_nrmse",
            np.asarray([sample.scale_nrmse for sample in samples], dtype=np.float64),
            parameters.error_upper_quantile,
            "maximum",
        ),
        (
            "max_scale_edge_mae",
            np.asarray([sample.scale_edge_mae for sample in samples], dtype=np.float64),
            parameters.error_upper_quantile,
            "maximum",
        ),
    ]
    if parameters.include_measurement:
        measurement_values = np.asarray(
            [
                sample.measurement_nrmse
                for sample in samples
                if sample.measurement_nrmse is not None
            ],
            dtype=np.float64,
        )
        if measurement_values.size:
            definitions.append(
                (
                    "max_measurement_nrmse",
                    measurement_values,
                    parameters.error_upper_quantile,
                    "maximum",
                )
            )
    result: dict[str, dict[str, Any]] = {}
    for name, values, quantile, direction in definitions:
        estimate = _quantile_estimate(
            values,
            quantile=quantile,
            rng=rng,
            bootstrap_samples=parameters.bootstrap_samples,
            confidence=parameters.bootstrap_confidence,
        )
        result[name] = {"gate_direction": direction, **estimate}
    return result


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def calibrate_from_manifests(
    manifest_paths: Sequence[Path],
    labels_path: Path,
    output_path: Path,
    *,
    parameters: CalibrationParameters | None = None,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    """Create a deterministic receipt from fully labeled, hash-verified steps."""

    parameters = parameters or CalibrationParameters()
    parameters.validate()
    if not manifest_paths:
        raise EvaluationEvidenceError("at least one run manifest is required")
    labels, labels_hash = _load_labels(labels_path)
    all_samples: list[_GateSample] = []
    manifest_evidence: list[dict[str, Any]] = []
    all_metric_keys: set[tuple[str, int]] = set()
    seen_run_ids: set[str] = set()
    for path in manifest_paths:
        run_id, samples, evidence, metric_keys = _manifest_samples(
            path,
            artifact_root=artifact_root,
        )
        if run_id in seen_run_ids:
            raise EvaluationEvidenceError(f"duplicate manifest run_id: {run_id}")
        seen_run_ids.add(run_id)
        all_samples.extend(samples)
        manifest_evidence.append(evidence)
        all_metric_keys.update(metric_keys)

    label_keys = set(labels)
    missing_labels = sorted(all_metric_keys - label_keys)
    unknown_labels = sorted(label_keys - all_metric_keys)
    if missing_labels:
        formatted = ", ".join(f"{run_id}:{index}" for run_id, index in missing_labels)
        raise EvaluationEvidenceError(f"metric-bearing steps are missing labels: {formatted}")
    if unknown_labels:
        formatted = ", ".join(f"{run_id}:{index}" for run_id, index in unknown_labels)
        raise EvaluationEvidenceError(f"labels do not match a metric-bearing step: {formatted}")
    labeled = sorted(
        _with_labels(all_samples, labels),
        key=lambda sample: (sample.run_id, sample.step_index),
    )
    real_samples = [sample for sample in labeled if not sample.mock]
    acceptable_real = [sample for sample in real_samples if sample.acceptable]
    quality_backends = sorted({sample.quality_backend for sample in real_samples})
    if len(quality_backends) > 1:
        raise EvaluationEvidenceError(
            "real labeled samples mix quality backends: " + ", ".join(quality_backends)
        )

    issues: list[str] = []
    if not acceptable_real:
        issues.append("no_acceptable_real_samples")
    if len(acceptable_real) < parameters.minimum_acceptable_samples:
        issues.append(
            "acceptable_real_samples_below_minimum:"
            f"{len(acceptable_real)}<{parameters.minimum_acceptable_samples}"
        )
    measurement_models: list[str] = []
    if parameters.include_measurement:
        missing_measurement = [
            f"{sample.run_id}:{sample.step_index}"
            for sample in acceptable_real
            if sample.measurement_nrmse is None or sample.measurement_model is None
        ]
        if missing_measurement:
            issues.append("missing_measurement_metrics:" + ",".join(missing_measurement))
        measurement_models = sorted(
            {
                sample.measurement_model
                for sample in acceptable_real
                if sample.measurement_model is not None
            }
        )
        if len(measurement_models) > 1:
            issues.append("mixed_measurement_models:" + ",".join(measurement_models))

    measurement_estimation_samples: list[_GateSample] = []
    if parameters.include_measurement:
        measurement_estimation_samples = [
            sample
            for sample in acceptable_real
            if sample.measurement_nrmse is not None and sample.measurement_model is not None
        ]
        if len(measurement_estimation_samples) < parameters.minimum_acceptable_samples:
            issues.append(
                "measurement_estimation_samples_below_minimum:"
                f"{len(measurement_estimation_samples)}"
                f"<{parameters.minimum_acceptable_samples}"
            )
    status = "calibrated" if not issues else "insufficient_data"
    thresholds = _thresholds(acceptable_real, parameters)
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "status": status,
        "inputs": {
            "labels": {"path": str(labels_path), "sha256": labels_hash},
            "manifests": sorted(manifest_evidence, key=lambda item: item["run_id"]),
        },
        "sample_counts": {
            "labels": len(labels),
            "matched_metric_steps": len(labeled),
            "acceptable": sum(sample.acceptable for sample in labeled),
            "acceptable_real": len(acceptable_real),
            "unacceptable_real": sum(
                not sample.acceptable and not sample.mock for sample in labeled
            ),
            "mock_excluded": sum(sample.mock for sample in labeled),
            "estimation_samples": len(acceptable_real),
            "measurement_estimation_samples": len(measurement_estimation_samples),
        },
        "metric_backend": {
            "quality": quality_backends[0] if len(quality_backends) == 1 else None,
            "quality_is_proxy": quality_backends == ["gradient_proxy_v1"],
            "measurement": (measurement_models[0] if len(measurement_models) == 1 else None),
        },
        "algorithm": {
            "name": "acceptable-sample-quantile-envelope",
            "quantile_method": "linear",
            "quality_lower_quantile": parameters.quality_lower_quantile,
            "error_upper_quantile": parameters.error_upper_quantile,
            "bootstrap_samples": parameters.bootstrap_samples,
            "bootstrap_confidence": parameters.bootstrap_confidence,
            "bootstrap_seed": parameters.bootstrap_seed,
            "minimum_acceptable_samples": parameters.minimum_acceptable_samples,
            "include_measurement": parameters.include_measurement,
            "numpy_version": np.__version__,
        },
        "thresholds": thresholds,
        "issues": issues,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    _write_json_atomic(output_path, receipt)
    return receipt


def _calibration_config_values(
    config: PipelineConfig | Mapping[str, Any],
    reasons: list[str],
) -> dict[str, Any] | None:
    if isinstance(config, PipelineConfig):
        return {
            "quality_backend": config.metrics.quality_backend,
            "quality_metric": config.metrics.quality_metric,
            "measurement_enabled": config.metrics.measurement_enabled,
            "measurement_model": config.metrics.measurement_model,
            "measurement_parameters": config.metrics.measurement_parameters,
            "min_quality_gain": config.metrics.min_quality_gain,
            "max_scale_nrmse": config.metrics.max_scale_nrmse,
            "max_scale_edge_mae": config.metrics.max_scale_edge_mae,
            "max_measurement_nrmse": config.metrics.max_measurement_nrmse,
        }
    metrics = config.get("metrics")
    if not isinstance(metrics, Mapping):
        reasons.append("config_metrics_missing")
        return None
    text_fields = ("quality_backend", "quality_metric", "measurement_model")
    numeric_fields = (
        "min_quality_gain",
        "max_scale_nrmse",
        "max_scale_edge_mae",
        "max_measurement_nrmse",
    )
    invalid = [
        name
        for name in text_fields
        if not isinstance(metrics.get(name), str) or not metrics.get(name)
    ]
    invalid.extend(
        name
        for name in numeric_fields
        if isinstance(metrics.get(name), bool)
        or not isinstance(metrics.get(name), (int, float))
        or not math.isfinite(float(metrics[name]))
    )
    if type(metrics.get("measurement_enabled")) is not bool:
        invalid.append("measurement_enabled")
    if not isinstance(metrics.get("measurement_parameters"), dict):
        invalid.append("measurement_parameters")
    if invalid:
        reasons.extend(f"config_metric_invalid:{name}" for name in sorted(set(invalid)))
        return None
    return {name: metrics[name] for name in (*text_fields, *numeric_fields)} | {
        "measurement_enabled": metrics["measurement_enabled"],
        "measurement_parameters": metrics["measurement_parameters"],
    }


def _expected_quality_backend(config: Mapping[str, Any]) -> str:
    if config["quality_backend"] == "gradient_proxy":
        return "gradient_proxy_v1"
    return f"pyiqa:{config['quality_metric']}"


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_receipt_structure(receipt: Mapping[str, Any], reasons: list[str]) -> None:
    issues = receipt.get("issues")
    if issues != []:
        reasons.append("receipt_has_issues")
    algorithm = receipt.get("algorithm")
    counts = receipt.get("sample_counts")
    if not isinstance(algorithm, dict) or not isinstance(counts, dict):
        reasons.append("calibration_evidence_missing")
    else:
        minimum = algorithm.get("minimum_acceptable_samples")
        acceptable = counts.get("acceptable_real")
        if (
            type(minimum) is not int
            or minimum < 1
            or type(acceptable) is not int
            or acceptable < minimum
        ):
            reasons.append("acceptable_sample_count_is_insufficient")

    inputs = receipt.get("inputs")
    if isinstance(inputs, dict):
        labels = inputs.get("labels")
        manifests = inputs.get("manifests")
        evidence_valid = (
            isinstance(labels, dict)
            and _is_sha256(labels.get("sha256"))
            and isinstance(manifests, list)
            and bool(manifests)
            and all(isinstance(item, dict) and _is_sha256(item.get("sha256")) for item in manifests)
        )
    else:
        evidence_valid = False
    if not evidence_valid:
        reasons.append("input_evidence_missing")


def _validate_threshold_entry(name: str, entry: Mapping[str, Any], reasons: list[str]) -> None:
    interval = entry.get("bootstrap_ci")
    if not isinstance(interval, dict):
        reasons.append(f"bootstrap_ci_missing:{name}")
        return
    lower = interval.get("lower")
    upper = interval.get("upper")
    confidence = interval.get("confidence")
    if (
        isinstance(lower, bool)
        or not isinstance(lower, (int, float))
        or not math.isfinite(float(lower))
        or isinstance(upper, bool)
        or not isinstance(upper, (int, float))
        or not math.isfinite(float(upper))
        or isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
    ):
        reasons.append(f"bootstrap_ci_invalid:{name}")
        return
    if float(lower) > float(upper) or not 0.0 < float(confidence) < 1.0:
        reasons.append(f"bootstrap_ci_invalid:{name}")


def verify_calibration_document(
    receipt: Mapping[str, Any],
    config: PipelineConfig | Mapping[str, Any],
) -> tuple[bool, list[str]]:
    """Verify one already-snapshotted receipt against exact metric settings."""

    reasons: list[str] = []
    loaded_config = _calibration_config_values(config, reasons)
    if receipt.get("schema_version") != RECEIPT_SCHEMA:
        reasons.append("unsupported_schema")
    declared_digest = receipt.get("receipt_sha256")
    receipt_body = dict(receipt)
    receipt_body.pop("receipt_sha256", None)
    if not isinstance(declared_digest, str) or declared_digest != canonical_sha256(receipt_body):
        reasons.append("receipt_sha256_mismatch")
    if receipt.get("status") != "calibrated":
        reasons.append("status_is_not_calibrated")
    _validate_receipt_structure(receipt, reasons)

    backend = receipt.get("metric_backend")
    if not isinstance(backend, dict):
        reasons.append("metric_backend_missing")
    elif loaded_config is None:
        pass
    elif backend.get("quality") != _expected_quality_backend(loaded_config):
        reasons.append("quality_backend_mismatch")
    elif backend.get("quality_is_proxy") is not (
        loaded_config["quality_backend"] == "gradient_proxy"
    ):
        reasons.append("quality_backend_proxy_flag_mismatch")

    raw_thresholds = receipt.get("thresholds")
    if not isinstance(raw_thresholds, dict):
        reasons.append("thresholds_missing")
        return False, reasons
    if loaded_config is None:
        return False, reasons
    expected = {
        "min_quality_gain": loaded_config["min_quality_gain"],
        "max_scale_nrmse": loaded_config["max_scale_nrmse"],
        "max_scale_edge_mae": loaded_config["max_scale_edge_mae"],
    }
    if loaded_config["measurement_enabled"]:
        expected["max_measurement_nrmse"] = loaded_config["max_measurement_nrmse"]
        try:
            expected_measurement_backend = build_forward_model(
                loaded_config["measurement_model"],
                loaded_config["measurement_parameters"],
            ).name
        except ScaleGuardError:
            reasons.append("measurement_config_invalid")
            expected_measurement_backend = None
        if not isinstance(backend, dict) or backend.get("measurement") != (
            expected_measurement_backend
        ):
            reasons.append("measurement_backend_mismatch")
    unexpected_thresholds = sorted(set(raw_thresholds) - set(expected))
    reasons.extend(f"unexpected_threshold:{name}" for name in unexpected_thresholds)

    for name, configured_value in expected.items():
        entry = raw_thresholds.get(name)
        if not isinstance(entry, dict):
            reasons.append(f"threshold_missing:{name}")
            continue
        value = entry.get("value")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not math.isclose(
                float(value),
                float(configured_value),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            reasons.append(f"threshold_mismatch:{name}")
        _validate_threshold_entry(name, entry, reasons)
    return not reasons, reasons


def verify_calibration_receipt(
    receipt_path: Path,
    config: PipelineConfig | Path,
) -> tuple[bool, list[str]]:
    """Verify receipt integrity, status, backend, and exact configured thresholds."""

    receipt, _ = load_json_object(receipt_path, kind="calibration receipt")
    loaded_config = load_config(config) if isinstance(config, Path) else config
    return verify_calibration_document(receipt, loaded_config)
