"""Calibrate gate thresholds from human labels and immutable run evidence."""

from __future__ import annotations

import csv
import importlib.metadata
import io
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from scaleguard import __version__
from scaleguard.config import PipelineConfig, load_config
from scaleguard.errors import ScaleGuardError
from scaleguard.evaluation.evidence import (
    EvaluationEvidenceError,
    canonical_sha256,
    load_json_object,
    optional_finite_number,
    require_finite_number,
    require_text,
    resolved_distinct_paths,
    verify_artifact,
    write_json_atomic,
)
from scaleguard.imaging.forward_models import build_forward_model
from scaleguard.manifest import ManifestValidationError, validate_run_manifest
from scaleguard.provenance import load_regular_file_snapshot

RECEIPT_SCHEMA = "scaleguard.calibration-receipt/v2"


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
        if type(self.bootstrap_seed) is not int or not 0 <= self.bootstrap_seed <= 2**63 - 1:
            raise EvaluationEvidenceError("bootstrap_seed must be an integer between 0 and 2^63-1")
        if not 0.0 < self.bootstrap_confidence < 1.0:
            raise EvaluationEvidenceError("bootstrap_confidence must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class _Label:
    run_id: str
    step_index: int
    acceptable: bool


@dataclass(frozen=True, slots=True)
class _GateSample:
    cluster_id: str
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
    try:
        payload, labels_hash = load_regular_file_snapshot(path, "calibration labels")
        text = payload.decode("utf-8-sig")
    except (OSError, ScaleGuardError, UnicodeDecodeError) as error:
        raise EvaluationEvidenceError(f"cannot read labels CSV {path}: {error}") from error
    reader = csv.DictReader(io.StringIO(text, newline=""))
    required = ("run_id", "step_index", "acceptable")
    fields = tuple(reader.fieldnames or ())
    if fields != required:
        raise EvaluationEvidenceError("labels CSV header must be exactly: " + ",".join(required))
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
            raise EvaluationEvidenceError(f"labels row {row_number}: step_index must be positive")
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


def _installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _manifest_project_root(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    artifact_root: Path | None,
) -> Path:
    provenance = manifest.get("provenance")
    declared = provenance.get("project_root") if isinstance(provenance, Mapping) else None
    if isinstance(declared, str) and declared:
        return Path(declared).expanduser().resolve()
    if artifact_root is not None:
        return artifact_root.expanduser().resolve()
    return manifest_path.resolve().parent


def _metric_identity(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    artifact_root: Path | None,
) -> tuple[dict[str, Any], str, str | None]:
    config = manifest.get("config")
    metrics = config.get("metrics") if isinstance(config, Mapping) else None
    if not isinstance(metrics, Mapping):
        raise EvaluationEvidenceError(f"{manifest_path}.config.metrics must be an object")
    backend = require_text(metrics, "quality_backend", context=f"{manifest_path}.config.metrics")
    metric = require_text(metrics, "quality_metric", context=f"{manifest_path}.config.metrics")
    device = require_text(metrics, "quality_device", context=f"{manifest_path}.config.metrics")
    if backend == "gradient_proxy":
        expected_backend = "gradient_proxy_v1"
        weight: dict[str, Any] | None = None
    elif backend == "pyiqa":
        expected_backend = f"pyiqa:{metric}"
        configured_weight = metrics.get("quality_model_path")
        if not isinstance(configured_weight, str) or not configured_weight:
            raise EvaluationEvidenceError(
                f"{manifest_path}.config.metrics.quality_model_path is required for PyIQA"
            )
        raw_weight = Path(configured_weight).expanduser()
        root = _manifest_project_root(
            manifest,
            manifest_path=manifest_path,
            artifact_root=artifact_root,
        )
        resolved_weight = (raw_weight if raw_weight.is_absolute() else root / raw_weight).resolve()
        try:
            weight_payload, weight_digest = load_regular_file_snapshot(
                resolved_weight,
                "quality model weight",
            )
        except (OSError, ScaleGuardError) as error:
            raise EvaluationEvidenceError(
                f"cannot bind quality model weight {resolved_weight}: {error}"
            ) from error
        weight = {
            "configured_path": configured_weight,
            "path": str(resolved_weight),
            "sha256": weight_digest,
            "size_bytes": len(weight_payload),
        }
    else:
        raise EvaluationEvidenceError(f"unsupported calibration quality backend: {backend}")

    quality = {
        "backend": backend,
        "recorded_backend": expected_backend,
        "metric": metric,
        "device": device,
        "implementation": "scaleguard.metrics.quality",
        "scaleguard_version": __version__,
        "pyiqa_version": _installed_version("pyiqa") if backend == "pyiqa" else None,
        "weight": weight,
        "preprocessing": {
            "image_mode": "RGB",
            "baseline_resize": "Pillow.BICUBIC",
            "direction": "higher_is_better",
            "implicit_downloads": False,
        },
    }

    measurement_enabled = metrics.get("measurement_enabled")
    if type(measurement_enabled) is not bool:
        raise EvaluationEvidenceError(
            f"{manifest_path}.config.metrics.measurement_enabled must be boolean"
        )
    measurement: dict[str, Any] | None = None
    expected_measurement: str | None = None
    if measurement_enabled:
        selector = require_text(
            metrics,
            "measurement_model",
            context=f"{manifest_path}.config.metrics",
        )
        parameters = metrics.get("measurement_parameters")
        if not isinstance(parameters, dict):
            raise EvaluationEvidenceError(
                f"{manifest_path}.config.metrics.measurement_parameters must be an object"
            )
        try:
            forward_model = build_forward_model(selector, parameters)
        except ScaleGuardError as error:
            raise EvaluationEvidenceError(
                f"invalid measurement model in {manifest_path}: {error}"
            ) from error
        expected_measurement = forward_model.name
        measurement = forward_model.identity
    return {"quality": quality, "measurement": measurement}, expected_backend, expected_measurement


def _declared_artifact_paths(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    artifact_root: Path | None,
) -> list[tuple[str, Path]]:
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
    paths: list[tuple[str, Path]] = []
    for label, record in records:
        if not isinstance(record, Mapping):
            continue
        declared = record.get("path")
        if not isinstance(declared, str) or not declared:
            continue
        candidate = Path(declared).expanduser()
        if not candidate.is_absolute():
            candidate = (artifact_root or manifest_path.parent) / candidate
        paths.append((label, candidate))
    return paths


def _manifest_samples(
    path: Path,
    *,
    artifact_root: Path | None,
) -> tuple[
    str,
    list[_GateSample],
    dict[str, Any],
    set[tuple[str, int]],
    dict[str, Any],
    list[tuple[str, Path]],
]:
    resolved_path = path.expanduser().resolve()
    manifest, manifest_hash = load_json_object(resolved_path, kind="run manifest")
    try:
        validated = validate_run_manifest(resolved_path, artifact_root=artifact_root)
    except ManifestValidationError as error:
        raise EvaluationEvidenceError(f"invalid run manifest {resolved_path}: {error}") from error
    after, after_hash = load_json_object(resolved_path, kind="run manifest")
    if validated != manifest or after != manifest or after_hash != manifest_hash:
        raise EvaluationEvidenceError(
            f"run manifest changed while it was being validated: {resolved_path}"
        )
    run_id = require_text(manifest, "run_id", context=str(path))
    mock = manifest.get("mock")
    if type(mock) is not bool:
        raise EvaluationEvidenceError(f"{path}.mock must be boolean")
    if mock:
        raise EvaluationEvidenceError(f"mock run is not eligible for calibration: {path}")
    status = require_text(manifest, "status", context=str(path))
    if status not in {"succeeded", "succeeded_with_rollback"}:
        raise EvaluationEvidenceError(
            f"run status is not eligible for calibration: {path}: {status}"
        )
    raw_input = manifest.get("input_image")
    input_evidence = verify_artifact(
        raw_input,
        context=f"{path}.input_image",
        manifest_path=resolved_path,
        artifact_root=artifact_root,
    )
    input_sha256 = input_evidence["sha256"]
    cluster_id = input_sha256
    identity, expected_quality_backend, expected_measurement_model = _metric_identity(
        manifest,
        manifest_path=resolved_path,
        artifact_root=artifact_root,
    )
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
            manifest_path=resolved_path,
            artifact_root=artifact_root,
        )
        verify_artifact(
            raw_step.get("candidate"),
            context=f"{context}.candidate",
            manifest_path=resolved_path,
            artifact_root=artifact_root,
        )
        quality_backend = require_text(
            raw_metrics,
            "quality_backend",
            context=f"{context}.metrics",
        )
        if quality_backend != expected_quality_backend:
            raise EvaluationEvidenceError(
                f"{context}.metrics.quality_backend disagrees with the configured metric"
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
        if measurement_model_value != expected_measurement_model:
            raise EvaluationEvidenceError(
                f"{context}.metrics.measurement_model disagrees with the configured operator"
            )
        key = (run_id, index_value)
        metric_keys.add(key)
        samples.append(
            _GateSample(
                cluster_id=cluster_id,
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
        "path": str(resolved_path),
        "sha256": manifest_hash,
        "run_id": run_id,
        "input_sha256": input_sha256,
        "cluster_id": cluster_id,
        "metric_step_count": len(samples),
    }
    return (
        run_id,
        samples,
        evidence,
        metric_keys,
        identity,
        _declared_artifact_paths(
            manifest,
            manifest_path=resolved_path,
            artifact_root=artifact_root,
        ),
    )


def _with_labels(
    samples: Sequence[_GateSample],
    labels: Mapping[tuple[str, int], _Label],
) -> list[_GateSample]:
    return [
        _GateSample(
            cluster_id=sample.cluster_id,
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
    clusters: Sequence[npt.NDArray[np.float64]],
    *,
    quantile: float,
    rng: np.random.Generator,
    bootstrap_samples: int,
    confidence: float,
) -> dict[str, Any]:
    if not clusters:
        raise EvaluationEvidenceError("cluster bootstrap requires at least one cluster")
    values = np.concatenate(clusters)
    estimate = float(np.quantile(values, quantile, method="linear"))
    bootstrapped = np.empty(bootstrap_samples, dtype=np.float64)
    cluster_count = len(clusters)
    for index in range(bootstrap_samples):
        selected = rng.integers(0, cluster_count, size=cluster_count, endpoint=False)
        resample = np.concatenate([clusters[int(cluster)] for cluster in selected])
        bootstrapped[index] = np.quantile(
            resample,
            quantile,
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
        "bootstrap_unit": "input_sha256_cluster",
        "cluster_count": cluster_count,
    }


def _thresholds(
    samples: Sequence[_GateSample],
    parameters: CalibrationParameters,
) -> dict[str, dict[str, Any]]:
    if not samples:
        return {}
    rng = np.random.default_rng(parameters.bootstrap_seed)
    grouped: dict[str, list[_GateSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.cluster_id, []).append(sample)

    definitions: list[tuple[str, str, float, str]] = [
        (
            "min_quality_gain",
            "quality_gain",
            parameters.quality_lower_quantile,
            "minimum",
        ),
        (
            "max_scale_nrmse",
            "scale_nrmse",
            parameters.error_upper_quantile,
            "maximum",
        ),
        (
            "max_scale_edge_mae",
            "scale_edge_mae",
            parameters.error_upper_quantile,
            "maximum",
        ),
    ]
    if parameters.include_measurement:
        if any(sample.measurement_nrmse is not None for sample in samples):
            definitions.append(
                (
                    "max_measurement_nrmse",
                    "measurement_nrmse",
                    parameters.error_upper_quantile,
                    "maximum",
                )
            )
    result: dict[str, dict[str, Any]] = {}
    for name, attribute, quantile, direction in definitions:
        clusters = []
        for cluster_samples in grouped.values():
            values = [getattr(sample, attribute) for sample in cluster_samples]
            retained = [value for value in values if value is not None]
            if retained:
                clusters.append(np.asarray(retained, dtype=np.float64))
        estimate = _quantile_estimate(
            clusters,
            quantile=quantile,
            rng=rng,
            bootstrap_samples=parameters.bootstrap_samples,
            confidence=parameters.bootstrap_confidence,
        )
        result[name] = {"gate_direction": direction, **estimate}
    return result


def _build_calibration_document(
    manifest_paths: Sequence[Path],
    labels_path: Path,
    *,
    parameters: CalibrationParameters,
    artifact_root: Path | None = None,
) -> tuple[dict[str, Any], list[tuple[str, Path]]]:
    parameters.validate()
    if not manifest_paths:
        raise EvaluationEvidenceError("at least one run manifest is required")
    resolved_labels = labels_path.expanduser().resolve()
    labels, labels_hash = _load_labels(resolved_labels)
    all_samples: list[_GateSample] = []
    manifest_evidence: list[dict[str, Any]] = []
    all_metric_keys: set[tuple[str, int]] = set()
    seen_run_ids: set[str] = set()
    metric_identities: list[dict[str, Any]] = []
    protected_inputs: list[tuple[str, Path]] = [("labels", resolved_labels)]
    for path in manifest_paths:
        (
            run_id,
            samples,
            evidence,
            metric_keys,
            metric_identity,
            artifact_paths,
        ) = _manifest_samples(
            path,
            artifact_root=artifact_root,
        )
        if run_id in seen_run_ids:
            raise EvaluationEvidenceError(f"duplicate manifest run_id: {run_id}")
        seen_run_ids.add(run_id)
        all_samples.extend(samples)
        manifest_evidence.append(evidence)
        all_metric_keys.update(metric_keys)
        metric_identities.append(metric_identity)
        protected_inputs.append((f"manifest {run_id}", Path(evidence["path"])))
        protected_inputs.extend(artifact_paths)

    identity_digests = {canonical_sha256(identity) for identity in metric_identities}
    if len(identity_digests) != 1:
        raise EvaluationEvidenceError(
            "calibration manifests use different metric or observation-model identities"
        )
    metric_identity = metric_identities[0]
    quality_identity = metric_identity["quality"]
    weight = quality_identity.get("weight")
    if isinstance(weight, Mapping) and isinstance(weight.get("path"), str):
        protected_inputs.append(("quality model weight", Path(weight["path"])))

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
    acceptable_real = [sample for sample in labeled if sample.acceptable]
    quality_backends = sorted({sample.quality_backend for sample in labeled})
    if len(quality_backends) > 1:
        raise EvaluationEvidenceError(
            "real labeled samples mix quality backends: " + ", ".join(quality_backends)
        )

    issues: list[str] = []
    if not acceptable_real:
        issues.append("no_acceptable_real_samples")
    acceptable_clusters = {sample.cluster_id for sample in acceptable_real}
    if len(acceptable_clusters) < parameters.minimum_acceptable_samples:
        issues.append(
            "acceptable_input_clusters_below_minimum:"
            f"{len(acceptable_clusters)}<{parameters.minimum_acceptable_samples}"
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
    measurement_clusters: set[str] = set()
    if parameters.include_measurement:
        measurement_estimation_samples = [
            sample
            for sample in acceptable_real
            if sample.measurement_nrmse is not None and sample.measurement_model is not None
        ]
        measurement_clusters = {sample.cluster_id for sample in measurement_estimation_samples}
        if len(measurement_clusters) < parameters.minimum_acceptable_samples:
            issues.append(
                "measurement_input_clusters_below_minimum:"
                f"{len(measurement_clusters)}"
                f"<{parameters.minimum_acceptable_samples}"
            )
    status = "calibrated" if not issues else "insufficient_data"
    thresholds = _thresholds(acceptable_real, parameters) if acceptable_real else {}
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "status": status,
        "inputs": {
            "labels": {"path": str(resolved_labels), "sha256": labels_hash},
            "manifests": sorted(manifest_evidence, key=lambda item: item["run_id"]),
            "artifact_root": (
                str(artifact_root.expanduser().resolve()) if artifact_root is not None else None
            ),
        },
        "sample_counts": {
            "labels": len(labels),
            "matched_metric_steps": len(labeled),
            "acceptable": sum(sample.acceptable for sample in labeled),
            "acceptable_real": len(acceptable_real),
            "unacceptable_real": sum(not sample.acceptable for sample in labeled),
            "mock_excluded": 0,
            "estimation_samples": len(acceptable_real),
            "measurement_estimation_samples": len(measurement_estimation_samples),
            "independent_input_clusters": len({sample.cluster_id for sample in labeled}),
            "acceptable_input_clusters": len(acceptable_clusters),
            "measurement_input_clusters": len(measurement_clusters),
        },
        "metric_backend": {
            "quality": quality_backends[0] if len(quality_backends) == 1 else None,
            "quality_is_proxy": quality_backends == ["gradient_proxy_v1"],
            "measurement": (measurement_models[0] if len(measurement_models) == 1 else None),
        },
        "metric_identity": metric_identity,
        "algorithm": {
            "name": "acceptable-sample-cluster-quantile-envelope",
            "quantile_method": "linear",
            "bootstrap_unit": "input_sha256_cluster",
            "quality_lower_quantile": parameters.quality_lower_quantile,
            "error_upper_quantile": parameters.error_upper_quantile,
            "bootstrap_samples": parameters.bootstrap_samples,
            "bootstrap_confidence": parameters.bootstrap_confidence,
            "bootstrap_seed": parameters.bootstrap_seed,
            "minimum_acceptable_clusters": parameters.minimum_acceptable_samples,
            "include_measurement": parameters.include_measurement,
            "numpy_version": np.__version__,
        },
        "thresholds": thresholds,
        "issues": issues,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)

    # Fail if any source changed after it contributed to the receipt.
    _labels_after, labels_hash_after = _load_labels(resolved_labels)
    if labels_hash_after != labels_hash:
        raise EvaluationEvidenceError("calibration labels changed during calibration")
    for evidence in manifest_evidence:
        _document, observed_hash = load_json_object(
            Path(evidence["path"]),
            kind="run manifest",
        )
        if observed_hash != evidence["sha256"]:
            raise EvaluationEvidenceError(
                f"run manifest changed during calibration: {evidence['path']}"
            )
    return receipt, protected_inputs


def calibrate_from_manifests(
    manifest_paths: Sequence[Path],
    labels_path: Path,
    output_path: Path,
    *,
    parameters: CalibrationParameters | None = None,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    """Create a source-replayable receipt from fully validated run evidence."""

    receipt, protected_inputs = _build_calibration_document(
        manifest_paths,
        labels_path,
        parameters=parameters or CalibrationParameters(),
        artifact_root=artifact_root,
    )
    resolved_output = resolved_distinct_paths(
        {"calibration output": output_path},
        inputs=protected_inputs,
    )["calibration output"]
    write_json_atomic(resolved_output, receipt)
    return receipt


def _calibration_config_values(
    config: PipelineConfig | Mapping[str, Any],
    reasons: list[str],
) -> dict[str, Any] | None:
    if isinstance(config, PipelineConfig):
        return {
            "quality_backend": config.metrics.quality_backend,
            "quality_metric": config.metrics.quality_metric,
            "quality_device": config.metrics.quality_device,
            "quality_model_path": (
                str(config.metrics.quality_model_path)
                if config.metrics.quality_model_path is not None
                else None
            ),
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
    text_fields = (
        "quality_backend",
        "quality_metric",
        "quality_device",
        "measurement_model",
    )
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
    model_path = metrics.get("quality_model_path")
    if model_path is not None and (not isinstance(model_path, str) or not model_path):
        invalid.append("quality_model_path")
    if invalid:
        reasons.extend(f"config_metric_invalid:{name}" for name in sorted(set(invalid)))
        return None
    return {name: metrics[name] for name in (*text_fields, *numeric_fields)} | {
        "measurement_enabled": metrics["measurement_enabled"],
        "measurement_parameters": metrics["measurement_parameters"],
        "quality_model_path": model_path,
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
        minimum = algorithm.get("minimum_acceptable_clusters")
        acceptable = counts.get("acceptable_input_clusters")
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
            and isinstance(labels.get("path"), str)
            and Path(labels["path"]).is_absolute()
            and isinstance(manifests, list)
            and bool(manifests)
            and all(
                isinstance(item, dict)
                and _is_sha256(item.get("sha256"))
                and isinstance(item.get("path"), str)
                and Path(item["path"]).is_absolute()
                for item in manifests
            )
        )
    else:
        evidence_valid = False
    if not evidence_valid:
        reasons.append("input_evidence_missing")
    if not isinstance(receipt.get("metric_identity"), dict):
        reasons.append("metric_identity_missing")


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


def _expected_metric_identity(
    config: Mapping[str, Any],
    recorded: Mapping[str, Any],
    reasons: list[str],
) -> dict[str, Any] | None:
    backend = config["quality_backend"]
    configured_weight = config.get("quality_model_path")
    weight: dict[str, Any] | None = None
    recorded_quality = recorded.get("quality")
    if not isinstance(recorded_quality, Mapping):
        reasons.append("quality_identity_missing")
        return None
    if backend == "pyiqa":
        recorded_weight = recorded_quality.get("weight")
        if not isinstance(recorded_weight, Mapping):
            reasons.append("quality_weight_identity_missing")
            return None
        if recorded_weight.get("configured_path") != configured_weight:
            reasons.append("quality_weight_configured_path_mismatch")
        weight_path = recorded_weight.get("path")
        if not isinstance(weight_path, str) or not Path(weight_path).is_absolute():
            reasons.append("quality_weight_path_invalid")
            return None
        try:
            payload, digest = load_regular_file_snapshot(
                Path(weight_path),
                "quality model weight",
            )
        except (OSError, ScaleGuardError):
            reasons.append("quality_weight_unreadable")
            return None
        weight = {
            "configured_path": configured_weight,
            "path": str(Path(weight_path).resolve()),
            "sha256": digest,
            "size_bytes": len(payload),
        }
    quality = {
        "backend": backend,
        "recorded_backend": _expected_quality_backend(config),
        "metric": config["quality_metric"],
        "device": config["quality_device"],
        "implementation": "scaleguard.metrics.quality",
        "scaleguard_version": __version__,
        "pyiqa_version": _installed_version("pyiqa") if backend == "pyiqa" else None,
        "weight": weight,
        "preprocessing": {
            "image_mode": "RGB",
            "baseline_resize": "Pillow.BICUBIC",
            "direction": "higher_is_better",
            "implicit_downloads": False,
        },
    }
    measurement: dict[str, Any] | None = None
    if config["measurement_enabled"]:
        try:
            measurement = build_forward_model(
                config["measurement_model"],
                config["measurement_parameters"],
            ).identity
        except ScaleGuardError:
            reasons.append("measurement_config_invalid")
            return None
    return {"quality": quality, "measurement": measurement}


def _parameters_from_receipt(
    receipt: Mapping[str, Any],
    reasons: list[str],
) -> CalibrationParameters | None:
    algorithm = receipt.get("algorithm")
    if not isinstance(algorithm, Mapping):
        reasons.append("calibration_algorithm_missing")
        return None
    expected_constants = {
        "name": "acceptable-sample-cluster-quantile-envelope",
        "quantile_method": "linear",
        "bootstrap_unit": "input_sha256_cluster",
        "numpy_version": np.__version__,
    }
    for key, expected in expected_constants.items():
        if algorithm.get(key) != expected:
            reasons.append(f"calibration_algorithm_mismatch:{key}")
    fields = {
        "minimum_acceptable_samples": algorithm.get("minimum_acceptable_clusters"),
        "quality_lower_quantile": algorithm.get("quality_lower_quantile"),
        "error_upper_quantile": algorithm.get("error_upper_quantile"),
        "bootstrap_samples": algorithm.get("bootstrap_samples"),
        "bootstrap_confidence": algorithm.get("bootstrap_confidence"),
        "bootstrap_seed": algorithm.get("bootstrap_seed"),
        "include_measurement": algorithm.get("include_measurement"),
    }
    if (
        type(fields["minimum_acceptable_samples"]) is not int
        or type(fields["bootstrap_samples"]) is not int
        or type(fields["bootstrap_seed"]) is not int
        or type(fields["include_measurement"]) is not bool
        or isinstance(fields["quality_lower_quantile"], bool)
        or not isinstance(fields["quality_lower_quantile"], (int, float))
        or isinstance(fields["error_upper_quantile"], bool)
        or not isinstance(fields["error_upper_quantile"], (int, float))
        or isinstance(fields["bootstrap_confidence"], bool)
        or not isinstance(fields["bootstrap_confidence"], (int, float))
    ):
        reasons.append("calibration_algorithm_parameters_invalid")
        return None
    parameters = CalibrationParameters(
        minimum_acceptable_samples=fields["minimum_acceptable_samples"],
        quality_lower_quantile=float(fields["quality_lower_quantile"]),
        error_upper_quantile=float(fields["error_upper_quantile"]),
        bootstrap_samples=fields["bootstrap_samples"],
        bootstrap_confidence=float(fields["bootstrap_confidence"]),
        bootstrap_seed=fields["bootstrap_seed"],
        include_measurement=fields["include_measurement"],
    )
    try:
        parameters.validate()
    except EvaluationEvidenceError:
        reasons.append("calibration_algorithm_parameters_invalid")
        return None
    return parameters


def _recompute_receipt(
    receipt: Mapping[str, Any],
    parameters: CalibrationParameters,
    reasons: list[str],
) -> None:
    inputs = receipt.get("inputs")
    if not isinstance(inputs, Mapping):
        return
    labels = inputs.get("labels")
    manifests = inputs.get("manifests")
    if not isinstance(labels, Mapping) or not isinstance(manifests, list):
        return
    labels_path = labels.get("path")
    manifest_paths: list[str] = []
    for item in manifests:
        if isinstance(item, Mapping) and isinstance(item.get("path"), str):
            manifest_paths.append(item["path"])
    if (
        not isinstance(labels_path, str)
        or not manifest_paths
        or len(manifest_paths) != len(manifests)
    ):
        reasons.append("source_paths_invalid")
        return
    raw_artifact_root = inputs.get("artifact_root")
    if raw_artifact_root is not None and not isinstance(raw_artifact_root, str):
        reasons.append("artifact_root_invalid")
        return
    artifact_root = Path(raw_artifact_root) if isinstance(raw_artifact_root, str) else None
    try:
        recomputed, _protected = _build_calibration_document(
            [Path(path) for path in manifest_paths],
            Path(labels_path),
            parameters=parameters,
            artifact_root=artifact_root,
        )
    except (OSError, ScaleGuardError, ValueError) as error:
        reasons.append(f"source_recompute_failed:{type(error).__name__}:{error}")
        return
    if recomputed != dict(receipt):
        reasons.append("source_recompute_mismatch")


def verify_calibration_document(
    receipt: Mapping[str, Any],
    config: PipelineConfig | Mapping[str, Any],
) -> tuple[bool, list[str]]:
    """Replay receipt sources and verify exact metric and threshold identity."""

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
    parameters = _parameters_from_receipt(receipt, reasons)

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

    recorded_identity = receipt.get("metric_identity")
    if loaded_config is not None and isinstance(recorded_identity, Mapping):
        expected_identity = _expected_metric_identity(
            loaded_config,
            recorded_identity,
            reasons,
        )
        if expected_identity is not None and expected_identity != recorded_identity:
            reasons.append("metric_identity_mismatch")

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
    if receipt.get("schema_version") == RECEIPT_SCHEMA and parameters is not None:
        _recompute_receipt(receipt, parameters, reasons)
    return not reasons, reasons


def verify_calibration_receipt(
    receipt_path: Path,
    config: PipelineConfig | Path,
) -> tuple[bool, list[str]]:
    """Verify receipt integrity, status, backend, and exact configured thresholds."""

    receipt, _ = load_json_object(receipt_path, kind="calibration receipt")
    loaded_config = load_config(config) if isinstance(config, Path) else config
    return verify_calibration_document(receipt, loaded_config)
