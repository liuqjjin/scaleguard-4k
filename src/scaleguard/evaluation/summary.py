"""Paired evidence tables for the four declared ablation groups."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

from scaleguard.config import EXPERIMENT_GROUPS as EXPERIMENT_GROUPS
from scaleguard.errors import ScaleGuardError
from scaleguard.evaluation.calibration import verify_calibration_document
from scaleguard.evaluation.evidence import (
    EvaluationEvidenceError,
    canonical_sha256,
    load_json_object,
    optional_finite_number,
    require_text,
    resolved_distinct_paths,
    verify_artifact,
    write_bytes_atomic,
    write_json_atomic,
)
from scaleguard.evaluation.metrics import verify_metric_receipt
from scaleguard.experiments import (
    ExperimentProtocolError,
    manifest_experiment_issues,
    validate_ablation_suite_receipt,
)
from scaleguard.manifest import ManifestValidationError, validate_run_manifest
from scaleguard.provenance import load_regular_file_snapshot
from scaleguard.strict_json import loads_object

SUMMARY_SCHEMA = "scaleguard.paired-summary/v2"
_GROUP_PREFIX = {
    "A-only": "a_only",
    "B-only": "b_only",
    "AB-fixed": "ab_fixed",
    "ScaleGuard": "scaleguard",
}
_METRIC_NAMES = (
    "quality_gain",
    "scale_nrmse",
    "scale_edge_mae",
    "measurement_nrmse",
)
_EXTERNAL_METRIC_NAMES = ("psnr", "ssim", "lpips", "musiq", "clipiqa")
_SYSTEM_METRIC_NAMES = (
    "success_rate",
    "stop_rate",
    "rollback_rate",
    "wall_time_seconds",
    "coz_initialization_seconds",
    "coz_first_step_seconds",
    "coz_steady_step_seconds",
    "peak_vram_mib",
)
_BOOTSTRAP_SAMPLES = 2000
_BOOTSTRAP_CONFIDENCE = 0.95
_BOOTSTRAP_SEED = 20260807
_PAIR_PROVENANCE_FIELDS = (
    "bootstrap_receipt_sha256",
    "materialization_marker_sha256",
    "source_weights_receipt_sha256",
    "weights_root",
    "project_commit",
    "project_root",
    "runtime_execution_binding",
    "runtime_execution_binding_sha256",
    "quality_backend_is_proxy",
    "quality_thresholds_calibrated",
    "quality_calibration_receipt",
    "quality_calibration_receipt_size_bytes",
    "quality_calibration_receipt_sha256",
)


def _normalized_pair_config(manifest: Mapping[str, Any]) -> dict[str, Any] | None:
    raw = manifest.get("config")
    if not isinstance(raw, dict):
        return None
    config = cast(dict[str, Any], json.loads(json.dumps(raw, allow_nan=False)))
    runtime = config.get("runtime")
    if isinstance(runtime, dict):
        runtime.pop("run_root", None)
        runtime.pop("experiment_group", None)
    fourkagent = config.get("fourkagent")
    if isinstance(fourkagent, dict):
        fourkagent.pop("mode", None)
    controller = config.get("controller")
    if isinstance(controller, dict):
        controller.pop("target_factor", None)
        controller.pop("max_coz_steps", None)
        controller.pop("acceptance_policy", None)
    return config


def _pairing_fingerprint(
    manifest: Mapping[str, Any],
    *,
    context: str,
) -> tuple[str | None, list[str]]:
    issues: list[str] = []
    config = _normalized_pair_config(manifest)
    if config is None:
        issues.append("pairing_config_missing")
    provenance = manifest.get("provenance")
    stable_provenance: dict[str, Any] = {}
    for field in _PAIR_PROVENANCE_FIELDS:
        value = provenance.get(field) if isinstance(provenance, dict) else None
        stable_provenance[field] = value
        if value is None or value == "":
            issues.append(f"stable_runtime_provenance_missing:{field}")
    if config is None or issues:
        return None, issues
    fingerprint = canonical_sha256(
        {
            "config": config,
            "runtime_provenance": stable_provenance,
        }
    )
    if len(fingerprint) != 64:
        raise EvaluationEvidenceError(f"{context} produced an invalid pairing fingerprint")
    return fingerprint, issues


def _quality_calibration_evidence(
    config: Mapping[str, Any],
    provenance: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    issues: list[str] = []
    metrics = config.get("metrics")
    project_root = provenance.get("project_root") if provenance is not None else None
    configured_path = metrics.get("calibration_receipt") if isinstance(metrics, dict) else None
    recorded_path = (
        provenance.get("quality_calibration_receipt") if provenance is not None else None
    )
    recorded_size = (
        provenance.get("quality_calibration_receipt_size_bytes") if provenance is not None else None
    )
    recorded_sha256 = (
        provenance.get("quality_calibration_receipt_sha256") if provenance is not None else None
    )
    if not isinstance(configured_path, str) or not configured_path:
        issues.append("quality_calibration_receipt_not_configured")
    if not isinstance(recorded_path, str) or not recorded_path:
        issues.append("quality_calibration_receipt_path_missing")
    if type(recorded_size) is not int or recorded_size < 0:
        issues.append("quality_calibration_receipt_size_invalid")
    if (
        not isinstance(recorded_sha256, str)
        or len(recorded_sha256) != 64
        or any(character not in "0123456789abcdef" for character in recorded_sha256)
    ):
        issues.append("quality_calibration_receipt_sha256_invalid")

    receipt_path: Path | None = None
    if isinstance(recorded_path, str) and recorded_path:
        candidate = Path(recorded_path)
        if not candidate.is_absolute():
            issues.append("quality_calibration_receipt_path_not_absolute")
        else:
            receipt_path = candidate.resolve()
    if isinstance(configured_path, str) and configured_path and receipt_path is not None:
        configured_candidate = Path(configured_path).expanduser()
        compare_paths = True
        if not configured_candidate.is_absolute():
            if isinstance(project_root, str) and project_root:
                configured_candidate = Path(project_root) / configured_candidate
            else:
                issues.append("quality_calibration_project_root_missing")
                compare_paths = False
        if compare_paths and configured_candidate.resolve() != receipt_path:
            issues.append("quality_calibration_receipt_path_mismatch")

    verification_reasons: list[str] = []
    observed_size: int | None = None
    observed_sha256: str | None = None
    calibration_input_sha256s: list[str] = []
    if receipt_path is not None:
        try:
            payload, observed_sha256 = load_regular_file_snapshot(
                receipt_path,
                "quality calibration receipt",
            )
            observed_size = len(payload)
            document = loads_object(payload)
            valid, verification_reasons = verify_calibration_document(document, config)
            if not valid:
                issues.extend(
                    f"quality_calibration_receipt_invalid:{reason}"
                    for reason in verification_reasons
                )
            else:
                inputs = document.get("inputs")
                manifests = inputs.get("manifests") if isinstance(inputs, Mapping) else None
                if not isinstance(manifests, list):
                    raise EvaluationEvidenceError(
                        "verified calibration receipt has no manifest input evidence"
                    )
                observed_inputs = [
                    item.get("input_sha256") if isinstance(item, Mapping) else None
                    for item in manifests
                ]
                if any(
                    not isinstance(digest, str)
                    or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)
                    for digest in observed_inputs
                ):
                    raise EvaluationEvidenceError(
                        "verified calibration receipt has invalid input identities"
                    )
                calibration_input_sha256s = sorted(set(cast(list[str], observed_inputs)))
        except (OSError, ValueError, ScaleGuardError) as error:
            verification_reasons = [f"unreadable:{type(error).__name__}"]
            issues.append("quality_calibration_receipt_unreadable")
    if observed_size is not None and recorded_size != observed_size:
        issues.append("quality_calibration_receipt_size_mismatch")
    if observed_sha256 is not None and recorded_sha256 != observed_sha256:
        issues.append("quality_calibration_receipt_sha256_mismatch")

    return (
        {
            "verified": not issues,
            "path": str(receipt_path) if receipt_path is not None else recorded_path,
            "size_bytes": observed_size,
            "sha256": observed_sha256,
            "input_sha256s": calibration_input_sha256s,
            "verification_reasons": verification_reasons,
        },
        issues,
    )


def _wall_time_seconds(manifest: Mapping[str, Any]) -> float | None:
    started = manifest.get("started_at")
    finished = manifest.get("finished_at")
    if not isinstance(started, str) or not isinstance(finished, str):
        return None
    try:
        start_time = datetime.fromisoformat(started.replace("Z", "+00:00"))
        finish_time = datetime.fromisoformat(finished.replace("Z", "+00:00"))
        duration = (finish_time - start_time).total_seconds()
    except (TypeError, ValueError):
        return None
    return duration if math.isfinite(duration) and duration >= 0.0 else None


def _peak_vram_mib(manifest: Mapping[str, Any]) -> float | None:
    processes: list[Any] = [
        manifest.get("restoration_process"),
        manifest.get("scale_session_process"),
    ]
    steps = manifest.get("steps")
    if isinstance(steps, list):
        processes.extend(step.get("process") for step in steps if isinstance(step, Mapping))
    peaks: list[int] = []
    for process in processes:
        raw_peaks = process.get("peak_vram_mib") if isinstance(process, Mapping) else None
        if isinstance(raw_peaks, Mapping):
            peaks.extend(value for value in raw_peaks.values() if type(value) is int and value >= 0)
    return float(max(peaks)) if peaks else None


def _coz_timing_metrics(manifest: Mapping[str, Any]) -> dict[str, float | None]:
    steps = manifest.get("steps")
    if not isinstance(steps, list):
        steps = []
    step_durations: list[float] = []
    initialization: float | None = None
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        metadata = step.get("worker_metadata")
        if not isinstance(metadata, Mapping):
            continue
        backend = metadata.get("backend")
        if backend not in {"chain_of_zoom_subprocess", "chain_of_zoom_persistent"}:
            continue
        duration = metadata.get("duration_seconds")
        if (
            not isinstance(duration, bool)
            and isinstance(duration, (int, float))
            and math.isfinite(float(duration))
            and float(duration) >= 0.0
        ):
            step_durations.append(float(duration))
        initialization_value = metadata.get("initialization_duration_seconds")
        if (
            initialization is None
            and not isinstance(initialization_value, bool)
            and isinstance(initialization_value, (int, float))
            and math.isfinite(float(initialization_value))
            and float(initialization_value) >= 0.0
        ):
            initialization = float(initialization_value)
    return {
        "coz_initialization_seconds": initialization,
        "coz_first_step_seconds": step_durations[0] if step_durations else None,
        "coz_steady_step_seconds": (
            float(np.mean(step_durations[1:])) if len(step_durations) > 1 else None
        ),
    }


def _system_metrics(manifest: Mapping[str, Any], status: str) -> dict[str, float | None]:
    steps = manifest.get("steps")
    last_decision: str | None = None
    if isinstance(steps, list) and steps and isinstance(steps[-1], Mapping):
        raw_decision = steps[-1].get("decision")
        last_decision = raw_decision if isinstance(raw_decision, str) else None
    events = manifest.get("events")
    final_gate_rollback = isinstance(events, list) and any(
        isinstance(event, Mapping) and event.get("event") == "final_gate_rollback"
        for event in events
    )
    rolled_back = (
        status == "succeeded_with_rollback" or last_decision == "rollback" or final_gate_rollback
    )
    return {
        "success_rate": float(status in {"succeeded", "succeeded_with_rollback"}),
        "stop_rate": None if last_decision is None else float(last_decision == "stop"),
        "rollback_rate": float(rolled_back),
        "wall_time_seconds": _wall_time_seconds(manifest),
        **_coz_timing_metrics(manifest),
        "peak_vram_mib": _peak_vram_mib(manifest),
    }


def _bootstrap_cluster_mean(
    cluster_values: Mapping[str, Sequence[float]],
    *,
    identity: str,
) -> dict[str, Any]:
    means = np.asarray(
        [float(np.mean(values)) for _cluster, values in sorted(cluster_values.items())],
        dtype=np.float64,
    )
    if not len(means):
        return {
            "mean": None,
            "median": None,
            "bootstrap_ci": {
                "lower": None,
                "upper": None,
                "confidence": _BOOTSTRAP_CONFIDENCE,
                "status": "no_data",
            },
            "independent_clusters": 0,
        }
    estimate = float(np.mean(means))
    median = float(np.median(means))
    if len(means) < 2:
        interval = {
            "lower": None,
            "upper": None,
            "confidence": _BOOTSTRAP_CONFIDENCE,
            "status": "insufficient_clusters",
        }
    else:
        identity_seed = int(hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16], 16)
        rng = np.random.default_rng(_BOOTSTRAP_SEED ^ identity_seed)
        indices = rng.integers(
            0,
            len(means),
            size=(_BOOTSTRAP_SAMPLES, len(means)),
            endpoint=False,
        )
        estimates = np.mean(means[indices], axis=1)
        alpha = (1.0 - _BOOTSTRAP_CONFIDENCE) / 2.0
        lower, upper = np.quantile(estimates, [alpha, 1.0 - alpha], method="linear")
        interval = {
            "lower": float(lower),
            "upper": float(upper),
            "confidence": _BOOTSTRAP_CONFIDENCE,
            "status": "estimated",
        }
    return {
        "mean": estimate,
        "median": median,
        "bootstrap_ci": interval,
        "independent_clusters": len(means),
    }


def _paired_effects(pairs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    effects: list[dict[str, Any]] = []
    directions = {
        "quality_gain": "higher_is_better",
        "scale_nrmse": "lower_is_better",
        "scale_edge_mae": "lower_is_better",
        "measurement_nrmse": "lower_is_better",
    }
    trusted_count = sum(pair["research_eligible"] is True for pair in pairs)
    for comparator in EXPERIMENT_GROUPS:
        if comparator == "ScaleGuard":
            continue
        for metric in _METRIC_NAMES:
            raw_by_cluster: dict[str, list[float]] = {}
            improvement_by_cluster: dict[str, list[float]] = {}
            observed_pairs = 0
            for pair in pairs:
                if pair["research_eligible"] is not True:
                    continue
                runs = pair["runs"]
                baseline = runs.get(comparator)
                scaleguard = runs.get("ScaleGuard")
                if not isinstance(baseline, Mapping) or not isinstance(scaleguard, Mapping):
                    continue
                baseline_metrics = baseline.get("metrics")
                scaleguard_metrics = scaleguard.get("metrics")
                if not isinstance(baseline_metrics, Mapping) or not isinstance(
                    scaleguard_metrics, Mapping
                ):
                    continue
                baseline_value = baseline_metrics.get(metric)
                scaleguard_value = scaleguard_metrics.get(metric)
                if (
                    isinstance(baseline_value, bool)
                    or not isinstance(baseline_value, (int, float))
                    or not math.isfinite(float(baseline_value))
                    or isinstance(scaleguard_value, bool)
                    or not isinstance(scaleguard_value, (int, float))
                    or not math.isfinite(float(scaleguard_value))
                ):
                    continue
                cluster = str(scaleguard["input_sha256"])
                raw_delta = float(scaleguard_value) - float(baseline_value)
                improvement = raw_delta if directions[metric] == "higher_is_better" else -raw_delta
                raw_by_cluster.setdefault(cluster, []).append(raw_delta)
                improvement_by_cluster.setdefault(cluster, []).append(improvement)
                observed_pairs += 1
            raw = _bootstrap_cluster_mean(
                raw_by_cluster,
                identity=f"paired:{comparator}:{metric}:raw",
            )
            improvement_summary = _bootstrap_cluster_mean(
                improvement_by_cluster,
                identity=f"paired:{comparator}:{metric}:improvement",
            )
            cluster_means = np.asarray(
                [
                    float(np.mean(values))
                    for _cluster, values in sorted(improvement_by_cluster.items())
                ],
                dtype=np.float64,
            )
            standardized: float | None = None
            if len(cluster_means) >= 2:
                standard_deviation = float(np.std(cluster_means, ddof=1))
                if standard_deviation > 0.0:
                    standardized = float(np.mean(cluster_means) / standard_deviation)
            effects.append(
                {
                    "comparison": f"ScaleGuard - {comparator}",
                    "metric": metric,
                    "direction": directions[metric],
                    "population": "research_eligible_complete_pairs",
                    "counts": {
                        "all_pairs": len(pairs),
                        "trusted_complete_pairs": trusted_count,
                        "observed_pairs": observed_pairs,
                        "missing_or_excluded_pairs": len(pairs) - observed_pairs,
                        "missing_or_excluded_rate": (
                            (len(pairs) - observed_pairs) / len(pairs) if pairs else None
                        ),
                    },
                    "raw_delta": raw,
                    "improvement_oriented_delta": improvement_summary,
                    "paired_standardized_effect_dz": standardized,
                }
            )
    return effects


def _systems_aggregates(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for group in EXPERIMENT_GROUPS:
        records = [
            pair["runs"].get(group) for pair in pairs if isinstance(pair.get("runs"), Mapping)
        ]
        real_records = [
            record
            for record in records
            if isinstance(record, Mapping) and record.get("mock") is False
        ]
        metrics: dict[str, Any] = {}
        for metric in _SYSTEM_METRIC_NAMES:
            by_cluster: dict[str, list[float]] = {}
            for record in real_records:
                systems = record.get("systems")
                value = systems.get(metric) if isinstance(systems, Mapping) else None
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    continue
                by_cluster.setdefault(str(record["input_sha256"]), []).append(float(value))
            aggregate = _bootstrap_cluster_mean(
                by_cluster,
                identity=f"systems:{group}:{metric}",
            )
            observed = sum(len(values) for values in by_cluster.values())
            metrics[metric] = {
                "population": "full_manifest_validated_non_mock_runs",
                "counts": {
                    "runs": len(real_records),
                    "observed": observed,
                    "missing": len(real_records) - observed,
                    "missing_rate": (
                        (len(real_records) - observed) / len(real_records) if real_records else None
                    ),
                },
                **aggregate,
            }
        result[group] = metrics
    return result


def _host_gpu_aggregates(
    pairs: Sequence[Mapping[str, Any]],
    suite_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    source = suite_evidence.get("system_evidence")
    by_sample = source.get("by_sample") if isinstance(source, Mapping) else None
    if not isinstance(by_sample, Mapping):
        by_sample = {}
    pairs_by_id = {str(pair.get("pair_id")): pair for pair in pairs if isinstance(pair, Mapping)}
    by_group: dict[str, Any] = {}
    for group in EXPERIMENT_GROUPS:
        duration_by_cluster: dict[str, list[float]] = {}
        interval_by_cluster: dict[str, list[float]] = {}
        gpu_values: dict[str, dict[str, Any]] = {}
        observed_runs = 0
        for sample_id, group_records in by_sample.items():
            if not isinstance(sample_id, str) or not isinstance(group_records, Mapping):
                continue
            system = group_records.get(group)
            pair = pairs_by_id.get(sample_id)
            if not isinstance(system, Mapping) or not isinstance(pair, Mapping):
                continue
            runs = pair.get("runs")
            run = runs.get(group) if isinstance(runs, Mapping) else None
            if not isinstance(run, Mapping) or run.get("mock") is not False:
                continue
            cluster = str(run.get("input_sha256"))
            duration = system.get("duration_seconds")
            sampling = system.get("gpu_sampling")
            if (
                isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or not math.isfinite(float(duration))
                or not isinstance(sampling, Mapping)
            ):
                continue
            interval = sampling.get("sample_interval_seconds")
            peaks = sampling.get("peak_by_physical_index")
            if (
                isinstance(interval, bool)
                or not isinstance(interval, (int, float))
                or not math.isfinite(float(interval))
                or not isinstance(peaks, Mapping)
            ):
                continue
            duration_by_cluster.setdefault(cluster, []).append(float(duration))
            interval_by_cluster.setdefault(cluster, []).append(float(interval))
            observed_runs += 1
            for raw_peak in peaks.values():
                if not isinstance(raw_peak, Mapping):
                    continue
                uuid_sha256 = raw_peak.get("uuid_sha256")
                if not isinstance(uuid_sha256, str):
                    continue
                identity = {
                    "uuid_sha256": uuid_sha256,
                    "name": raw_peak.get("name"),
                    "memory_total_mib": raw_peak.get("memory_total_mib"),
                }
                entry = gpu_values.setdefault(
                    uuid_sha256,
                    {
                        "identity": identity,
                        "physical_indices": set(),
                        "logical_indices": set(),
                        "peak_memory_used_mib": {},
                        "peak_utilization_percent": {},
                    },
                )
                if entry["identity"] != identity:
                    raise EvaluationEvidenceError(
                        "validated GPU UUID has inconsistent hardware identity"
                    )
                entry["physical_indices"].add(str(raw_peak.get("physical_index")))
                entry["logical_indices"].add(raw_peak.get("logical_index"))
                for metric in ("peak_memory_used_mib", "peak_utilization_percent"):
                    value = raw_peak.get(metric)
                    if type(value) is int and value >= 0:
                        entry[metric].setdefault(cluster, []).append(float(value))

        per_gpu: dict[str, Any] = {}
        for uuid_sha256, entry in sorted(gpu_values.items()):
            per_gpu[uuid_sha256] = {
                **entry["identity"],
                "physical_indices": sorted(entry["physical_indices"]),
                "logical_indices": sorted(entry["logical_indices"]),
                "peak_memory_used_mib": _bootstrap_cluster_mean(
                    entry["peak_memory_used_mib"],
                    identity=f"host-gpu:{group}:{uuid_sha256}:memory",
                ),
                "peak_utilization_percent": _bootstrap_cluster_mean(
                    entry["peak_utilization_percent"],
                    identity=f"host-gpu:{group}:{uuid_sha256}:utilization",
                ),
            }
        by_group[group] = {
            "counts": {
                "validated_non_mock_runs": sum(
                    isinstance(pair.get("runs"), Mapping)
                    and isinstance(pair["runs"].get(group), Mapping)
                    and pair["runs"][group].get("mock") is False
                    for pair in pairs
                ),
                "observed_runs": observed_runs,
            },
            "wrapper_duration_seconds": _bootstrap_cluster_mean(
                duration_by_cluster,
                identity=f"host-gpu:{group}:wrapper-duration",
            ),
            "sample_interval_seconds": _bootstrap_cluster_mean(
                interval_by_cluster,
                identity=f"host-gpu:{group}:sample-interval",
            ),
            "by_gpu_uuid_sha256": per_gpu,
        }
    return {
        "verified": suite_evidence.get("verified") is True and bool(by_sample),
        "source": "independently_replayed_wrapper_execution_and_gpu_samples",
        "attribution_scope": "physical_gpu_host_level_not_process_attributed",
        "by_group": by_group,
    }


def _run_record(
    path: Path,
    *,
    group: str,
    artifact_root: Path | None,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    path = path.resolve()
    snapshot, manifest_hash = load_json_object(path, kind="run manifest")
    try:
        manifest = validate_run_manifest(path, artifact_root=artifact_root)
    except ManifestValidationError as error:
        raise EvaluationEvidenceError(f"invalid run manifest {path}: {error}") from error
    if manifest != snapshot:
        raise EvaluationEvidenceError(f"run manifest changed while it was being validated: {path}")
    run_id = require_text(manifest, "run_id", context=str(path))
    status = require_text(manifest, "status", context=str(path))
    mock = manifest.get("mock")
    if type(mock) is not bool:
        raise EvaluationEvidenceError(f"{path}.mock must be boolean")
    input_evidence = verify_artifact(
        manifest.get("input_image"),
        context=f"{path}.input_image",
        manifest_path=path,
        artifact_root=artifact_root,
    )
    config = manifest.get("config")
    runtime = config.get("runtime") if isinstance(config, dict) else None
    embedded_group = runtime.get("experiment_group") if isinstance(runtime, dict) else None
    sample_id = runtime.get("experiment_sample_id") if isinstance(runtime, dict) else None
    if embedded_group != group:
        raise EvaluationEvidenceError(
            f"{path} embeds experiment group {embedded_group!r}, expected {group!r}"
        )
    if not isinstance(sample_id, str) or not sample_id:
        raise EvaluationEvidenceError(f"{path} has no experiment sample id")
    pair_id = sample_id

    issues: list[str] = []
    final_image = manifest.get("final_image")
    final_sha256: str | None = None
    protected_paths = [str(path), str(Path(str(input_evidence["verified_path"])).resolve())]
    if final_image is not None:
        final_evidence = verify_artifact(
            final_image,
            context=f"{path}.final_image",
            manifest_path=path,
            artifact_root=artifact_root,
        )
        final_sha256 = final_evidence["sha256"]
        protected_paths.append(str(Path(str(final_evidence["verified_path"])).resolve()))
    else:
        issues.append("missing_final_image")

    metric_config = config.get("metrics") if isinstance(config, dict) else None
    measurement_enabled = (
        metric_config.get("measurement_enabled") if isinstance(metric_config, dict) else None
    )
    if type(measurement_enabled) is not bool:
        issues.append("measurement_configuration_invalid")

    raw_final_metrics = manifest.get("final_metrics")
    raw_metrics: Any = None
    if isinstance(raw_final_metrics, dict):
        raw_metrics = raw_final_metrics.get("metrics")
    metrics: dict[str, float | None] = {}
    if isinstance(raw_metrics, dict):
        for name in _METRIC_NAMES:
            metrics[name] = optional_finite_number(
                raw_metrics,
                name,
                context=f"{path}.final_metrics.metrics",
            )
        required_metrics = _METRIC_NAMES if measurement_enabled is True else _METRIC_NAMES[:3]
        for name in required_metrics:
            if metrics[name] is None:
                issues.append(f"missing_final_metric:{name}")
    else:
        metrics = dict.fromkeys(_METRIC_NAMES)
        issues.append("missing_final_metrics")
    if mock:
        issues.append("mock")
    if status not in {"succeeded", "succeeded_with_rollback"}:
        issues.append(f"run_status:{status}")
    provenance = manifest.get("provenance")
    if (
        not isinstance(provenance, dict)
        or provenance.get("runtime_evidence_verified") is not True
        or provenance.get("runtime_profile_bound") is not True
    ):
        issues.append("runtime_evidence_unverified")
    if not isinstance(provenance, dict) or provenance.get("quality_backend_is_proxy") is not False:
        issues.append("quality_backend_proxy_or_unverified")
    if (
        not isinstance(provenance, dict)
        or provenance.get("quality_thresholds_calibrated") is not True
    ):
        issues.append("quality_thresholds_uncalibrated")
    calibration_evidence, calibration_issues = _quality_calibration_evidence(
        config if isinstance(config, dict) else {},
        provenance if isinstance(provenance, dict) else None,
    )
    calibration_path = calibration_evidence.get("path")
    if isinstance(calibration_path, str) and Path(calibration_path).is_absolute():
        protected_paths.append(str(Path(calibration_path).resolve()))
    issues.extend(calibration_issues)
    calibration_inputs = calibration_evidence.get("input_sha256s")
    if isinstance(calibration_inputs, list) and input_evidence["sha256"] in calibration_inputs:
        issues.append("calibration_evaluation_input_overlap")
    pairing_fingerprint, pairing_issues = _pairing_fingerprint(
        manifest,
        context=str(path),
    )
    issues.extend(pairing_issues)
    issues.extend(manifest_experiment_issues(manifest, group))

    record = {
        "run_id": run_id,
        "manifest_path": str(path),
        "manifest_sha256": manifest_hash,
        "status": status,
        "mock": mock,
        "experiment_sample_id": sample_id,
        "input_sha256": input_evidence["sha256"],
        "final_image_sha256": final_sha256,
        "pairing_fingerprint_sha256": pairing_fingerprint,
        "quality_calibration": calibration_evidence,
        "metrics": metrics,
        "systems": _system_metrics(manifest, status),
        "issues": issues,
    }
    evidence = {
        "group": group,
        "path": str(path),
        "sha256": manifest_hash,
        "run_id": run_id,
        "pair_id": pair_id,
        "input_sha256": input_evidence["sha256"],
        "protected_paths": sorted(set(protected_paths)),
    }
    return pair_id, record, evidence


def _flatten_pair(
    pair: Mapping[str, Any],
    external_metric_names: Sequence[str],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "pair_id": pair["pair_id"],
        "complete": pair["complete"],
        "research_eligible": pair["research_eligible"],
        "issues": ";".join(pair["issues"]),
    }
    runs = pair["runs"]
    for group in EXPERIMENT_GROUPS:
        prefix = _GROUP_PREFIX[group]
        record = runs[group]
        row[f"{prefix}_run_id"] = record["run_id"] if record else ""
        row[f"{prefix}_manifest_sha256"] = record["manifest_sha256"] if record else ""
        row[f"{prefix}_status"] = record["status"] if record else "missing"
        row[f"{prefix}_mock"] = record["mock"] if record else ""
        row[f"{prefix}_final_image_sha256"] = record["final_image_sha256"] if record else ""
        for metric in _METRIC_NAMES:
            value = record["metrics"][metric] if record else None
            row[f"{prefix}_{metric}"] = "" if value is None else value
        for metric in external_metric_names:
            external = record.get("external_metrics", {}).get(metric) if record else None
            row[f"{prefix}_{metric}_status"] = (
                external.get("status") if isinstance(external, Mapping) else "missing"
            )
            value = external.get("value") if isinstance(external, Mapping) else None
            row[f"{prefix}_{metric}"] = "" if value is None else value
    return row


def _csv_bytes(
    pairs: Sequence[Mapping[str, Any]],
    external_metric_names: Sequence[str],
) -> bytes:
    rows = [_flatten_pair(pair, external_metric_names) for pair in pairs]
    fieldnames = ["pair_id", "complete", "research_eligible", "issues"]
    for group in EXPERIMENT_GROUPS:
        prefix = _GROUP_PREFIX[group]
        fieldnames.extend(
            [
                f"{prefix}_run_id",
                f"{prefix}_manifest_sha256",
                f"{prefix}_status",
                f"{prefix}_mock",
                f"{prefix}_final_image_sha256",
                *(f"{prefix}_{metric}" for metric in _METRIC_NAMES),
                *(
                    field
                    for metric in external_metric_names
                    for field in (f"{prefix}_{metric}_status", f"{prefix}_{metric}")
                ),
            ]
        )
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode("utf-8")


def _lower_sha256(value: Any, *, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise EvaluationEvidenceError(f"{context} must be a lowercase SHA256 digest")
    return value


def _suite_receipt_evidence(
    suite_receipt: Path | None,
    manifests: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if suite_receipt is None:
        return {
            "provided": False,
            "verified": False,
            "path": None,
            "size_bytes": None,
            "sha256": None,
            "project_commit": None,
            "hardware": None,
            "system_evidence": None,
            "issues": ["suite_receipt_missing"],
        }

    try:
        validated = validate_ablation_suite_receipt(suite_receipt)
    except ExperimentProtocolError as error:
        raise EvaluationEvidenceError(
            f"invalid ablation suite receipt {suite_receipt}: {error}"
        ) from error
    context = "validated ablation suite receipt"
    path_text = require_text(validated, "path", context=context)
    receipt_path = Path(path_text)
    requested_path = suite_receipt.resolve()
    if not receipt_path.is_absolute() or receipt_path.resolve() != requested_path:
        raise EvaluationEvidenceError(
            "validated ablation suite receipt path differs from the requested receipt"
        )
    size = validated.get("size_bytes")
    if type(size) is not int or size < 0:
        raise EvaluationEvidenceError(
            "validated ablation suite receipt size_bytes must be non-negative"
        )
    receipt_sha256 = _lower_sha256(
        validated.get("sha256"),
        context=f"{context}.sha256",
    )
    project_commit = require_text(validated, "project_commit", context=context)
    raw_jobs = validated.get("jobs")
    if not isinstance(raw_jobs, list):
        raise EvaluationEvidenceError(f"{context}.jobs must be a list")

    suite_bindings: dict[tuple[str, str], tuple[str, str]] = {}
    hardware_by_sample: dict[str, dict[str, set[str]]] = {}
    systems_by_sample: dict[str, dict[str, dict[str, Any]]] = {}
    groups_by_sample: dict[str, set[str]] = {}
    for index, raw_job in enumerate(raw_jobs):
        job_context = f"{context}.jobs[{index}]"
        if not isinstance(raw_job, dict):
            raise EvaluationEvidenceError(f"{job_context} must be an object")
        sample_id = require_text(raw_job, "sample_id", context=job_context)
        group = require_text(raw_job, "group", context=job_context)
        if group not in EXPERIMENT_GROUPS:
            raise EvaluationEvidenceError(f"{job_context}.group is undeclared: {group!r}")
        manifest = raw_job.get("manifest")
        if not isinstance(manifest, dict):
            raise EvaluationEvidenceError(f"{job_context}.manifest must be an object")
        manifest_path_text = require_text(
            manifest,
            "path",
            context=f"{job_context}.manifest",
        )
        manifest_path = Path(manifest_path_text)
        if not manifest_path.is_absolute():
            raise EvaluationEvidenceError(f"{job_context}.manifest.path must be absolute")
        manifest_sha256 = _lower_sha256(
            manifest.get("sha256"),
            context=f"{job_context}.manifest.sha256",
        )
        key = (sample_id, group)
        if key in suite_bindings:
            raise EvaluationEvidenceError(
                f"validated ablation suite receipt duplicates {sample_id}, {group}"
            )
        suite_bindings[key] = (str(manifest_path.resolve()), manifest_sha256)
        groups_by_sample.setdefault(sample_id, set()).add(group)

        hardware = raw_job.get("hardware")
        if not isinstance(hardware, dict):
            raise EvaluationEvidenceError(f"{job_context}.hardware must be an object")
        identity = _lower_sha256(
            hardware.get("identity_sha256"),
            context=f"{job_context}.hardware.identity_sha256",
        )
        hardware_class = _lower_sha256(
            hardware.get("class_sha256"),
            context=f"{job_context}.hardware.class_sha256",
        )
        sample_hardware = hardware_by_sample.setdefault(
            sample_id,
            {"identity_sha256": set(), "class_sha256": set()},
        )
        sample_hardware["identity_sha256"].add(identity)
        sample_hardware["class_sha256"].add(hardware_class)
        system_evidence = raw_job.get("system_evidence")
        if system_evidence is not None:
            if not isinstance(system_evidence, dict):
                raise EvaluationEvidenceError(f"{job_context}.system_evidence must be an object")
            systems_by_sample.setdefault(sample_id, {})[group] = json.loads(
                json.dumps(system_evidence, allow_nan=False)
            )

    expected_bindings = {
        (str(item["pair_id"]), str(item["group"])): (
            str(Path(str(item["path"])).resolve()),
            str(item["sha256"]),
        )
        for item in manifests
    }
    issues: list[str] = []
    if suite_bindings != expected_bindings:
        issues.append("suite_receipt_manifest_set_mismatch")
    expected_groups = set(EXPERIMENT_GROUPS)
    hardware_summary: dict[str, dict[str, str]] = {}
    for sample_id, digests in sorted(hardware_by_sample.items()):
        if (
            groups_by_sample.get(sample_id) != expected_groups
            or len(digests["identity_sha256"]) != 1
            or len(digests["class_sha256"]) != 1
        ):
            issues.append(f"suite_receipt_hardware_pairing_mismatch:{sample_id}")
            continue
        hardware_summary[sample_id] = {
            "identity_sha256": next(iter(digests["identity_sha256"])),
            "class_sha256": next(iter(digests["class_sha256"])),
        }

    return {
        "provided": True,
        "verified": not issues,
        "path": str(receipt_path.resolve()),
        "size_bytes": size,
        "sha256": receipt_sha256,
        "project_commit": project_commit,
        "hardware": {"by_sample": hardware_summary},
        "system_evidence": {"by_sample": systems_by_sample},
        "issues": issues,
    }


def _bind_metric_receipts(
    receipt_paths: Sequence[Path],
    records: Mapping[str, Mapping[str, dict[str, Any]]],
    *,
    artifact_root: Path | None,
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, int],
    list[Path],
]:
    expected_by_path: dict[str, tuple[str, str, str, dict[str, Any]]] = {}
    for group_records in records.values():
        for group, record in group_records.items():
            manifest_path = str(Path(str(record["manifest_path"])).resolve())
            if manifest_path in expected_by_path:
                raise EvaluationEvidenceError(
                    f"manifest path is reused across paired runs: {manifest_path}"
                )
            expected_by_path[manifest_path] = (
                str(record["manifest_sha256"]),
                str(record["run_id"]),
                group,
                record,
            )

    resolved_receipts: set[str] = set()
    receipt_evidence: list[dict[str, Any]] = []
    definitions: dict[str, dict[str, Any]] = {}
    result_identities: dict[str, str] = {}
    bound_metrics_by_manifest: dict[str, set[str]] = {}
    protected_paths: set[Path] = set()
    for receipt_index, receipt_path in enumerate(receipt_paths):
        resolved_receipt = str(receipt_path.expanduser().resolve())
        if resolved_receipt in resolved_receipts:
            raise EvaluationEvidenceError(f"duplicate metric receipt path: {resolved_receipt}")
        resolved_receipts.add(resolved_receipt)
        verified = verify_metric_receipt(receipt_path, artifact_root=artifact_root)
        receipt_evidence.append(
            {
                key: value
                for key, value in verified.items()
                if key not in {"samples", "metric_definitions", "protected_paths"}
            }
        )
        raw_protected_paths = verified.get("protected_paths")
        if not isinstance(raw_protected_paths, list) or any(
            not isinstance(source, str) or not Path(source).is_absolute()
            for source in raw_protected_paths
        ):
            raise AssertionError("verified metric receipt protected paths are invalid")
        protected_paths.update(Path(source).resolve() for source in raw_protected_paths)
        raw_definitions = verified["metric_definitions"]
        if not isinstance(raw_definitions, Mapping):
            raise AssertionError("verified metric definitions must be a mapping")
        current_definition_names = set(raw_definitions)
        for name, raw_definition in raw_definitions.items():
            if name not in _EXTERNAL_METRIC_NAMES or not isinstance(raw_definition, dict):
                raise EvaluationEvidenceError(
                    f"metric receipt {receipt_index} returned an invalid definition"
                )
            previous = definitions.get(name)
            if previous is not None and previous != raw_definition:
                raise EvaluationEvidenceError(
                    f"external metric definition conflicts across receipts: {name}"
                )
            definitions[name] = dict(raw_definition)

        raw_samples = verified["samples"]
        if not isinstance(raw_samples, list):
            raise AssertionError("verified metric receipt samples must be a list")
        for sample_index, sample in enumerate(raw_samples):
            if not isinstance(sample, Mapping):
                raise AssertionError("verified metric receipt sample must be a mapping")
            manifest_path = str(Path(str(sample["manifest_path"])).resolve())
            expected = expected_by_path.get(manifest_path)
            if expected is None:
                raise EvaluationEvidenceError(
                    f"metric receipt sample does not map to a supplied manifest: {manifest_path}"
                )
            expected_sha256, expected_run_id, group, record = expected
            if (
                sample.get("manifest_sha256") != expected_sha256
                or sample.get("run_id") != expected_run_id
            ):
                raise EvaluationEvidenceError(
                    f"metric receipt sample identity drift for manifest: {manifest_path}"
                )
            raw_metrics = sample.get("metrics")
            if not isinstance(raw_metrics, Mapping):
                raise AssertionError("verified sample metrics must be a mapping")
            if set(raw_metrics) != current_definition_names:
                raise EvaluationEvidenceError(
                    f"metric receipt sample {sample_index} does not match its definitions"
                )
            already_bound = bound_metrics_by_manifest.setdefault(manifest_path, set())
            overlap = already_bound.intersection(raw_metrics)
            if overlap:
                raise EvaluationEvidenceError(
                    "multiple metric receipt samples bind the same metric for "
                    f"{manifest_path}: {', '.join(sorted(overlap))}"
                )
            already_bound.update(raw_metrics)
            observed_metrics = record.setdefault("external_metrics", {})
            if not isinstance(observed_metrics, dict):
                raise AssertionError("external metrics must be a dictionary")
            references = record.setdefault("metric_reference_sha256_by_metric", {})
            receipts = record.setdefault("metric_receipts_by_metric", {})
            if not isinstance(references, dict) or not isinstance(receipts, dict):
                raise AssertionError("metric evidence maps must be dictionaries")
            for name, raw_metric in raw_metrics.items():
                if name not in current_definition_names or not isinstance(raw_metric, Mapping):
                    raise EvaluationEvidenceError(
                        f"metric receipt sample {sample_index} has invalid metric {name!r}"
                    )
                metric = dict(raw_metric)
                identity = metric.get("identity_sha256")
                if metric.get("status") == "measured":
                    if not isinstance(identity, str):
                        raise EvaluationEvidenceError(
                            f"measured external metric has no identity: {name}"
                        )
                    previous_identity = result_identities.get(name)
                    if previous_identity is not None and previous_identity != identity:
                        raise EvaluationEvidenceError(
                            f"measured external metric definition conflicts: {name}"
                        )
                    result_identities[name] = identity
                if group == "A-only" and definitions[name]["reference_required"] is True:
                    metric = {
                        "status": "not_applicable",
                        "value": None,
                        "direction": definitions[name]["direction"],
                        "identity_sha256": definitions[name]["identity_sha256"],
                        "reason": "native_resolution_output_not_comparable_to_4x_reference",
                    }
                observed_metrics[name] = metric
                references[name] = (
                    sample.get("reference_sha256")
                    if definitions[name]["reference_required"] is True
                    else None
                )
                receipts[name] = {
                    "path": verified["path"],
                    "sha256": verified["sha256"],
                    "receipt_sha256": verified["receipt_sha256"],
                    "research_eligible": verified["research_eligible"],
                }

    for name, identity in result_identities.items():
        definitions[name]["measured_identity_sha256"] = identity

    status_counts = {
        "measured": 0,
        "missing": 0,
        "not_applicable": 0,
        "unverified": 0,
        "failed": 0,
    }
    for group_records in records.values():
        for group, record in group_records.items():
            observed = record.setdefault("external_metrics", {})
            if not isinstance(observed, dict):
                raise AssertionError("external metrics must be a dictionary")
            for name, definition in definitions.items():
                if name not in observed:
                    if group == "A-only" and definition["reference_required"] is True:
                        observed[name] = {
                            "status": "not_applicable",
                            "value": None,
                            "direction": definition["direction"],
                            "identity_sha256": definition["identity_sha256"],
                            "reason": "native_resolution_output_not_comparable_to_4x_reference",
                        }
                    else:
                        observed[name] = {
                            "status": "missing",
                            "value": None,
                            "direction": definition["direction"],
                            "identity_sha256": definition["identity_sha256"],
                            "reason": "no_verified_metric_receipt_sample",
                        }
                status = observed[name].get("status")
                if status not in status_counts:
                    raise EvaluationEvidenceError(
                        f"external metric {name} has unsupported status {status!r}"
                    )
                status_counts[status] += 1
    return receipt_evidence, definitions, status_counts, sorted(protected_paths)


def _external_metric_effects(
    pairs: Sequence[Mapping[str, Any]],
    definitions: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    effects: list[dict[str, Any]] = []
    trusted_count = sum(pair["research_eligible"] is True for pair in pairs)
    for comparator in EXPERIMENT_GROUPS:
        if comparator == "ScaleGuard":
            continue
        for metric_name in _EXTERNAL_METRIC_NAMES:
            definition = definitions.get(metric_name)
            if definition is None:
                continue
            raw_by_cluster: dict[str, list[float]] = {}
            improvement_by_cluster: dict[str, list[float]] = {}
            excluded_statuses: dict[str, int] = {}
            observed_pairs = 0
            for pair in pairs:
                if pair["research_eligible"] is not True:
                    excluded_statuses["pair_not_research_eligible"] = (
                        excluded_statuses.get("pair_not_research_eligible", 0) + 1
                    )
                    continue
                runs = pair["runs"]
                baseline = runs.get(comparator)
                treatment = runs.get("ScaleGuard")
                if not isinstance(baseline, Mapping) or not isinstance(treatment, Mapping):
                    excluded_statuses["missing_run"] = excluded_statuses.get("missing_run", 0) + 1
                    continue
                baseline_metrics = baseline.get("external_metrics")
                treatment_metrics = treatment.get("external_metrics")
                baseline_metric = (
                    baseline_metrics.get(metric_name)
                    if isinstance(baseline_metrics, Mapping)
                    else None
                )
                treatment_metric = (
                    treatment_metrics.get(metric_name)
                    if isinstance(treatment_metrics, Mapping)
                    else None
                )
                baseline_status = (
                    baseline_metric.get("status")
                    if isinstance(baseline_metric, Mapping)
                    else "missing"
                )
                treatment_status = (
                    treatment_metric.get("status")
                    if isinstance(treatment_metric, Mapping)
                    else "missing"
                )
                if baseline_status != "measured" or treatment_status != "measured":
                    key = f"baseline:{baseline_status}|scaleguard:{treatment_status}"
                    excluded_statuses[key] = excluded_statuses.get(key, 0) + 1
                    continue
                if not isinstance(baseline_metric, Mapping) or not isinstance(
                    treatment_metric, Mapping
                ):
                    raise AssertionError("measured external metrics must be mappings")
                if definition["reference_required"] is True:
                    baseline_references = baseline.get("metric_reference_sha256_by_metric")
                    treatment_references = treatment.get("metric_reference_sha256_by_metric")
                    baseline_reference = (
                        baseline_references.get(metric_name)
                        if isinstance(baseline_references, Mapping)
                        else None
                    )
                    treatment_reference = (
                        treatment_references.get(metric_name)
                        if isinstance(treatment_references, Mapping)
                        else None
                    )
                    if (
                        not isinstance(baseline_reference, str)
                        or baseline_reference != treatment_reference
                    ):
                        excluded_statuses["reference_mismatch"] = (
                            excluded_statuses.get("reference_mismatch", 0) + 1
                        )
                        continue
                baseline_value = baseline_metric.get("value")
                treatment_value = treatment_metric.get("value")
                if (
                    isinstance(baseline_value, bool)
                    or not isinstance(baseline_value, (int, float))
                    or not math.isfinite(float(baseline_value))
                    or isinstance(treatment_value, bool)
                    or not isinstance(treatment_value, (int, float))
                    or not math.isfinite(float(treatment_value))
                ):
                    excluded_statuses["non_finite"] = excluded_statuses.get("non_finite", 0) + 1
                    continue
                cluster = str(treatment["input_sha256"])
                raw_delta = float(treatment_value) - float(baseline_value)
                improvement = (
                    raw_delta if definition["direction"] == "higher_is_better" else -raw_delta
                )
                raw_by_cluster.setdefault(cluster, []).append(raw_delta)
                improvement_by_cluster.setdefault(cluster, []).append(improvement)
                observed_pairs += 1
            raw = _bootstrap_cluster_mean(
                raw_by_cluster,
                identity=f"external:{comparator}:{metric_name}:raw",
            )
            improvement_summary = _bootstrap_cluster_mean(
                improvement_by_cluster,
                identity=f"external:{comparator}:{metric_name}:improvement",
            )
            cluster_means = np.asarray(
                [
                    float(np.mean(values))
                    for _cluster, values in sorted(improvement_by_cluster.items())
                ],
                dtype=np.float64,
            )
            standardized: float | None = None
            if len(cluster_means) >= 2:
                standard_deviation = float(np.std(cluster_means, ddof=1))
                if standard_deviation > 0.0:
                    standardized = float(np.mean(cluster_means) / standard_deviation)
            effects.append(
                {
                    "comparison": f"ScaleGuard - {comparator}",
                    "metric": metric_name,
                    "direction": definition["direction"],
                    "definition_identity_sha256": definition.get(
                        "measured_identity_sha256",
                        definition["identity_sha256"],
                    ),
                    "population": "research_eligible_complete_pairs_with_verified_scores",
                    "counts": {
                        "all_pairs": len(pairs),
                        "trusted_complete_pairs": trusted_count,
                        "observed_pairs": observed_pairs,
                        "missing_not_applicable_or_excluded_pairs": len(pairs) - observed_pairs,
                        "excluded_by_status": dict(sorted(excluded_statuses.items())),
                    },
                    "raw_delta": raw,
                    "improvement_oriented_delta": improvement_summary,
                    "paired_standardized_effect_dz": standardized,
                }
            )
    return effects


def summarize_paired_manifests(
    manifests_by_group: Mapping[str, Sequence[Path]],
    output_csv: Path,
    output_json: Path,
    *,
    artifact_root: Path | None = None,
    suite_receipt: Path | None = None,
    metric_receipts: Sequence[Path] = (),
) -> dict[str, Any]:
    """Write paired rows without imputing absent groups or metrics."""

    protected_inputs = [
        (f"{group} manifest {index}", path)
        for group in EXPERIMENT_GROUPS
        for index, path in enumerate(manifests_by_group.get(group, ()))
    ]
    if suite_receipt is not None:
        protected_inputs.append(("suite receipt", suite_receipt))
    protected_inputs.extend(
        (f"metric receipt {index}", path) for index, path in enumerate(metric_receipts)
    )
    resolved_outputs = resolved_distinct_paths(
        {"summary CSV": output_csv, "summary JSON": output_json},
        inputs=protected_inputs,
    )
    resolved_csv = resolved_outputs["summary CSV"]
    resolved_json = resolved_outputs["summary JSON"]

    unknown_groups = sorted(set(manifests_by_group) - set(EXPERIMENT_GROUPS))
    if unknown_groups:
        raise EvaluationEvidenceError("unknown experiment groups: " + ", ".join(unknown_groups))
    records: dict[str, dict[str, dict[str, Any]]] = {}
    input_evidence: list[dict[str, Any]] = []
    manifest_count = 0
    for group in EXPERIMENT_GROUPS:
        for path in manifests_by_group.get(group, ()):
            manifest_count += 1
            pair_id, record, evidence = _run_record(
                path,
                group=group,
                artifact_root=artifact_root,
            )
            group_records = records.setdefault(pair_id, {})
            observed_inputs = {existing["input_sha256"] for existing in group_records.values()}
            if observed_inputs and record["input_sha256"] not in observed_inputs:
                raise EvaluationEvidenceError(
                    f"sample {pair_id} does not use identical input bytes across groups"
                )
            if group in group_records:
                previous = group_records[group]["run_id"]
                raise EvaluationEvidenceError(
                    f"duplicate pair/group for {pair_id}, {group}: {previous}, {record['run_id']}"
                )
            group_records[group] = record
            input_evidence.append(evidence)
    if manifest_count == 0:
        raise EvaluationEvidenceError("at least one experiment manifest is required")
    suite_evidence = _suite_receipt_evidence(suite_receipt, input_evidence)
    (
        metric_receipt_evidence,
        external_definitions,
        external_status_counts,
        metric_protected_paths,
    ) = _bind_metric_receipts(
        metric_receipts,
        records,
        artifact_root=artifact_root,
    )
    additional_protected_inputs = [
        (f"manifest evidence {index}", Path(source))
        for index, evidence in enumerate(input_evidence)
        for source in evidence["protected_paths"]
    ]
    additional_protected_inputs.extend(
        (f"metric source evidence {index}", source)
        for index, source in enumerate(metric_protected_paths)
    )
    resolved_distinct_paths(
        {"summary CSV": resolved_csv, "summary JSON": resolved_json},
        inputs=additional_protected_inputs,
    )

    pairs: list[dict[str, Any]] = []
    for pair_id in sorted(records):
        group_records = records[pair_id]
        issues: list[str] = []
        runs: dict[str, dict[str, Any] | None] = {}
        for group in EXPERIMENT_GROUPS:
            paired_record = group_records.get(group)
            runs[group] = paired_record
            if paired_record is None:
                issues.append(f"missing_group:{group}")
            else:
                issues.extend(f"{group}:{issue}" for issue in paired_record["issues"])
        complete = all(runs[group] is not None for group in EXPERIMENT_GROUPS)
        fingerprints = {
            record["pairing_fingerprint_sha256"]
            for record in group_records.values()
            if record["pairing_fingerprint_sha256"] is not None
        }
        if complete and len(fingerprints) != 1:
            issues.append("paired_runtime_or_config_mismatch")
        if not suite_evidence["verified"]:
            issues.append("suite_receipt_unverified")
        research_eligible = complete and not issues
        pairs.append(
            {
                "pair_id": pair_id,
                "complete": complete,
                "research_eligible": research_eligible,
                "issues": issues,
                "runs": runs,
            }
        )

    external_metric_names = tuple(
        name for name in _EXTERNAL_METRIC_NAMES if name in external_definitions
    )
    csv_payload = _csv_bytes(pairs, external_metric_names)
    payload: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA,
        "groups": list(EXPERIMENT_GROUPS),
        "suite_receipt": suite_evidence,
        "metric_receipts": metric_receipt_evidence,
        "external_metric_definitions": {
            name: external_definitions[name] for name in external_metric_names
        },
        "inputs": sorted(
            input_evidence,
            key=lambda item: (item["pair_id"], item["group"], item["run_id"]),
        ),
        "counts": {
            "manifests": manifest_count,
            "pairs": len(pairs),
            "complete_pairs": sum(pair["complete"] for pair in pairs),
            "research_eligible_pairs": sum(pair["research_eligible"] for pair in pairs),
            "mock_pairs": sum(
                any(record is not None and record["mock"] for record in pair["runs"].values())
                for pair in pairs
            ),
        },
        "external_metric_counts": {
            "receipts": len(metric_receipt_evidence),
            "definitions": len(external_metric_names),
            **external_status_counts,
        },
        "aggregate_protocol": {
            "paired_comparator": "ScaleGuard",
            "bootstrap_unit": "input_sha256_cluster",
            "cluster_aggregation": "mean_across_runs_then_equal_weight_across_inputs",
            "bootstrap_samples": _BOOTSTRAP_SAMPLES,
            "bootstrap_confidence": _BOOTSTRAP_CONFIDENCE,
            "bootstrap_seed": _BOOTSTRAP_SEED,
            "minimum_clusters_for_interval": 2,
            "standardized_effect": "paired Cohen dz over input-cluster means",
        },
        "paired_effects": _paired_effects(pairs),
        "external_metric_effects": _external_metric_effects(
            pairs,
            external_definitions,
        ),
        "systems_by_group": _systems_aggregates(pairs),
        "host_gpu_systems": _host_gpu_aggregates(pairs, suite_evidence),
        "csv_output": {
            "path": str(resolved_csv),
            "size_bytes": len(csv_payload),
            "sha256": hashlib.sha256(csv_payload).hexdigest(),
        },
        "pairs": pairs,
    }
    payload["summary_sha256"] = canonical_sha256(payload)
    # The JSON is the commit marker for the CSV bytes. Consumers must verify
    # csv_output before treating the pair as one committed summary.
    write_bytes_atomic(resolved_csv, csv_payload)
    write_json_atomic(resolved_json, payload)
    return payload
