"""Paired evidence tables for the four declared ablation groups."""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scaleguard.evaluation.evidence import (
    EvaluationEvidenceError,
    canonical_sha256,
    load_json_object,
    optional_finite_number,
    require_text,
    verify_artifact,
)

SUMMARY_SCHEMA = "scaleguard.paired-summary/v1"
EXPERIMENT_GROUPS = ("A-only", "B-only", "AB-fixed", "ScaleGuard")
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


def _run_record(
    path: Path,
    *,
    group: str,
    artifact_root: Path | None,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    manifest, manifest_hash = load_json_object(path, kind="run manifest")
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
    pair_id = input_evidence["sha256"]

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
        for name in _METRIC_NAMES[:3]:
            if metrics[name] is None:
                issues.append(f"missing_final_metric:{name}")
    else:
        metrics = dict.fromkeys(_METRIC_NAMES)
        issues.append("missing_final_metrics")
    if mock:
        issues.append("mock")
    if status not in {"succeeded", "succeeded_with_rollback"}:
        issues.append(f"run_status:{status}")

    record = {
        "run_id": run_id,
        "manifest_path": str(path),
        "manifest_sha256": manifest_hash,
        "status": status,
        "mock": mock,
        "input_sha256": pair_id,
        "final_image_sha256": final_sha256,
        "metrics": metrics,
        "issues": issues,
    }
    evidence = {
        "group": group,
        "path": str(path),
        "sha256": manifest_hash,
        "run_id": run_id,
        "pair_id": pair_id,
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


def summarize_paired_manifests(
    manifests_by_group: Mapping[str, Sequence[Path]],
    output_csv: Path,
    output_json: Path,
    *,
    artifact_root: Path | None = None,
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
            if group in group_records:
                previous = group_records[group]["run_id"]
                raise EvaluationEvidenceError(
                    f"duplicate pair/group for {pair_id}, {group}: {previous}, {record['run_id']}"
                )
            group_records[group] = record
            input_evidence.append(evidence)
    if manifest_count == 0:
        raise EvaluationEvidenceError("at least one experiment manifest is required")

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
