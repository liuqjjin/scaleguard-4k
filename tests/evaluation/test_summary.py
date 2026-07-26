from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scaleguard.evaluation.evidence import EvaluationEvidenceError, canonical_sha256
from scaleguard.evaluation.summary import EXPERIMENT_GROUPS, summarize_paired_manifests

from ._fixtures import write_summary_manifest


def _images(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source.bin"
    final = tmp_path / "final.bin"
    source.write_bytes(b"shared-input")
    final.write_bytes(b"real-output")
    return source, final


def test_summary_writes_complete_paired_csv_and_json_without_aggregation(
    tmp_path: Path,
) -> None:
    source, final = _images(tmp_path)
    groups: dict[str, list[Path]] = {}
    for index, group in enumerate(EXPERIMENT_GROUPS):
        manifest = write_summary_manifest(
            tmp_path / f"{index}.json",
            run_id=f"run-{index}",
            source=source,
            final=final,
            metrics={
                "quality_gain": 0.1 + index,
                "scale_nrmse": 0.2 + index,
                "scale_edge_mae": 0.3 + index,
                "measurement_nrmse": None,
            },
        )
        groups[group] = [manifest]

    payload = summarize_paired_manifests(
        groups,
        tmp_path / "paired.csv",
        tmp_path / "paired.json",
    )

    assert payload["counts"] == {
        "manifests": 4,
        "pairs": 1,
        "complete_pairs": 1,
        "research_eligible_pairs": 1,
        "mock_pairs": 0,
    }
    pair = payload["pairs"][0]
    assert pair["complete"] is True
    assert pair["research_eligible"] is True
    assert pair["runs"]["AB-fixed"]["metrics"]["quality_gain"] == pytest.approx(2.1)
    body = dict(payload)
    digest = body.pop("summary_sha256")
    assert digest == canonical_sha256(body)
    assert json.loads((tmp_path / "paired.json").read_text(encoding="utf-8")) == payload
    with (tmp_path / "paired.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["complete"] == "True"
    assert rows[0]["scaleguard_quality_gain"] == "3.1"


def test_missing_groups_and_mock_runs_are_explicitly_ineligible(tmp_path: Path) -> None:
    source, final = _images(tmp_path)
    manifest = write_summary_manifest(
        tmp_path / "mock.json",
        run_id="mock-run",
        source=source,
        final=final,
        mock=True,
    )

    payload = summarize_paired_manifests(
        {"A-only": [manifest]},
        tmp_path / "paired.csv",
        tmp_path / "paired.json",
    )

    pair = payload["pairs"][0]
    assert pair["complete"] is False
    assert pair["research_eligible"] is False
    assert "A-only:mock" in pair["issues"]
    assert "missing_group:B-only" in pair["issues"]
    assert payload["counts"]["mock_pairs"] == 1
    with (tmp_path / "paired.csv").open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["b_only_status"] == "missing"
    assert row["b_only_quality_gain"] == ""


def test_summary_rejects_duplicate_pair_group_unknown_group_and_hash_mismatch(
    tmp_path: Path,
) -> None:
    source, final = _images(tmp_path)
    first = write_summary_manifest(
        tmp_path / "first.json",
        run_id="first",
        source=source,
        final=final,
    )
    second = write_summary_manifest(
        tmp_path / "second.json",
        run_id="second",
        source=source,
        final=final,
    )
    with pytest.raises(EvaluationEvidenceError, match="duplicate pair/group"):
        summarize_paired_manifests(
            {"A-only": [first, second]},
            tmp_path / "out.csv",
            tmp_path / "out.json",
        )
    with pytest.raises(EvaluationEvidenceError, match="unknown experiment groups"):
        summarize_paired_manifests(
            {"other": [first]},
            tmp_path / "out.csv",
            tmp_path / "out.json",
        )

    source.write_bytes(b"tampered")
    with pytest.raises(EvaluationEvidenceError, match="SHA256 mismatch"):
        summarize_paired_manifests(
            {"A-only": [first]},
            tmp_path / "out.csv",
            tmp_path / "out.json",
        )


def test_summary_requires_at_least_one_manifest(tmp_path: Path) -> None:
    with pytest.raises(EvaluationEvidenceError, match="at least one"):
        summarize_paired_manifests({}, tmp_path / "out.csv", tmp_path / "out.json")


def test_summary_rejects_duplicate_manifest_keys(tmp_path: Path) -> None:
    source, final = _images(tmp_path)
    manifest = write_summary_manifest(
        tmp_path / "manifest.json",
        run_id="run",
        source=source,
        final=final,
    )
    original = manifest.read_text(encoding="utf-8").lstrip()
    manifest.write_text('{"run_id":"forged",' + original[1:], encoding="utf-8")

    with pytest.raises(EvaluationEvidenceError, match="duplicate JSON object key 'run_id'"):
        summarize_paired_manifests(
            {"A-only": [manifest]},
            tmp_path / "out.csv",
            tmp_path / "out.json",
        )


def test_failed_run_with_missing_final_evidence_is_preserved_as_issues(
    tmp_path: Path,
) -> None:
    source, final = _images(tmp_path)
    manifest = write_summary_manifest(
        tmp_path / "failed.json",
        run_id="failed",
        source=source,
        final=final,
        status="failed",
    )
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    raw["final_image"] = None
    raw["final_metrics"] = None
    manifest.write_text(json.dumps(raw), encoding="utf-8")

    payload = summarize_paired_manifests(
        {"ScaleGuard": [manifest]},
        tmp_path / "out.csv",
        tmp_path / "out.json",
    )

    issues = payload["pairs"][0]["issues"]
    assert "ScaleGuard:missing_final_image" in issues
    assert "ScaleGuard:missing_final_metrics" in issues
    assert "ScaleGuard:run_status:failed" in issues


def test_present_metrics_with_missing_required_value_are_not_research_eligible(
    tmp_path: Path,
) -> None:
    source, final = _images(tmp_path)
    groups: dict[str, list[Path]] = {}
    for index, group in enumerate(EXPERIMENT_GROUPS):
        metrics = {
            "quality_gain": None if group == "B-only" else 0.1,
            "scale_nrmse": 0.2,
            "scale_edge_mae": 0.3,
            "measurement_nrmse": None,
        }
        groups[group] = [
            write_summary_manifest(
                tmp_path / f"{index}.json",
                run_id=f"run-{index}",
                source=source,
                final=final,
                metrics=metrics,
            )
        ]

    payload = summarize_paired_manifests(
        groups,
        tmp_path / "out.csv",
        tmp_path / "out.json",
    )

    pair = payload["pairs"][0]
    assert pair["complete"] is True
    assert pair["research_eligible"] is False
    assert "B-only:missing_final_metric:quality_gain" in pair["issues"]
