"""Paired evidence tables for the four declared ablation groups."""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from scaleguard.config import EXPERIMENT_GROUPS as EXPERIMENT_GROUPS
from scaleguard.errors import ScaleGuardError
from scaleguard.evaluation.calibration import verify_calibration_document
from scaleguard.evaluation.evidence import (
    EvaluationEvidenceError,
    canonical_sha256,
    load_json_object,
    optional_finite_number,
    require_text,
    verify_artifact,
)
from scaleguard.experiments import (
    ExperimentProtocolError,
    manifest_experiment_issues,
    validate_ablation_suite_receipt,
)
from scaleguard.manifest import ManifestValidationError, validate_run_manifest
from scaleguard.provenance import load_regular_file_snapshot
from scaleguard.strict_json import loads_object

SUMMARY_SCHEMA = "scaleguard.paired-summary/v1"
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
            "verification_reasons": verification_reasons,
        },
        issues,
    )


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
    if final_image is not None:
        final_evidence = verify_artifact(
            final_image,
            context=f"{path}.final_image",
            manifest_path=path,
            artifact_root=artifact_root,
        )
        final_sha256 = final_evidence["sha256"]
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
    issues.extend(calibration_issues)
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
        "issues": issues,
    }
    evidence = {
        "group": group,
        "path": str(path),
        "sha256": manifest_hash,
        "run_id": run_id,
        "pair_id": pair_id,
        "input_sha256": input_evidence["sha256"],
    }
    return pair_id, record, evidence


def _flatten_pair(pair: Mapping[str, Any]) -> dict[str, Any]:
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
    return row


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_csv_atomic(path: Path, pairs: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    rows = [_flatten_pair(pair) for pair in pairs]
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
            ]
        )
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


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
        "issues": issues,
    }


def summarize_paired_manifests(
    manifests_by_group: Mapping[str, Sequence[Path]],
    output_csv: Path,
    output_json: Path,
    *,
    artifact_root: Path | None = None,
    suite_receipt: Path | None = None,
) -> dict[str, Any]:
    """Write paired rows without imputing absent groups or metrics."""

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

    payload: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA,
        "groups": list(EXPERIMENT_GROUPS),
        "suite_receipt": suite_evidence,
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
        "pairs": pairs,
    }
    payload["summary_sha256"] = canonical_sha256(payload)
    _write_csv_atomic(output_csv, pairs)
    _write_json_atomic(output_json, payload)
    return payload
