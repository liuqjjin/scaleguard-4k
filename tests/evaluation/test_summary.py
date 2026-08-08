from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from scaleguard.evaluation import metrics as metrics_module
from scaleguard.evaluation import summary as summary_module
from scaleguard.evaluation.evidence import EvaluationEvidenceError, canonical_sha256
from scaleguard.evaluation.metrics import evaluate_metric_receipt
from scaleguard.evaluation.summary import EXPERIMENT_GROUPS, summarize_paired_manifests
from scaleguard.experiments import ExperimentProtocolError
from scaleguard.manifest import ManifestValidationError

from ._fixtures import write_summary_manifest


@pytest.fixture(autouse=True)
def _accept_minimal_summary_fixtures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        summary_module,
        "validate_run_manifest",
        lambda path, **_kwargs: json.loads(path.read_text(encoding="utf-8")),
    )

    def verify_calibration(
        receipt: dict[str, object],
        config: dict[str, object],
    ) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        metrics = config.get("metrics")
        thresholds = receipt.get("thresholds")
        if not isinstance(metrics, dict) or not isinstance(thresholds, dict):
            return False, ["calibration_evidence_missing"]
        names = ["min_quality_gain", "max_scale_nrmse", "max_scale_edge_mae"]
        if metrics.get("measurement_enabled") is True:
            names.append("max_measurement_nrmse")
        for name in names:
            entry = thresholds.get(name)
            if not isinstance(entry, dict) or entry.get("value") != metrics.get(name):
                reasons.append(f"threshold_mismatch:{name}")
        return not reasons, reasons

    monkeypatch.setattr(summary_module, "verify_calibration_document", verify_calibration)


def _images(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source.bin"
    final = tmp_path / "final.bin"
    source.write_bytes(b"shared-input")
    final.write_bytes(b"real-output")
    return source, final


def _validated_suite(
    tmp_path: Path,
    groups: dict[str, list[Path]],
) -> tuple[Path, dict[str, object]]:
    receipt = tmp_path / "suite-receipt.json"
    receipt.write_text('{"status":"passed"}\n', encoding="utf-8")
    jobs = []
    for group, paths in groups.items():
        for path in paths:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            jobs.append(
                {
                    "sample_id": manifest["config"]["runtime"]["experiment_sample_id"],
                    "group": group,
                    "manifest": {
                        "path": str(path.resolve()),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    },
                    "hardware": {
                        "identity_sha256": "1" * 64,
                        "class_sha256": "2" * 64,
                    },
                }
            )
    return receipt, {
        "path": str(receipt.resolve()),
        "size_bytes": receipt.stat().st_size,
        "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
        "project_commit": "d" * 40,
        "jobs": jobs,
    }


def _accept_suite(
    monkeypatch: pytest.MonkeyPatch,
    receipt: Path,
    validated: dict[str, object],
) -> None:
    def validate(path: Path) -> dict[str, object]:
        assert path == receipt
        return validated

    monkeypatch.setattr(summary_module, "validate_ablation_suite_receipt", validate)


def _complete_groups(tmp_path: Path) -> dict[str, list[Path]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source, final = _images(tmp_path)
    return {
        group: [
            write_summary_manifest(
                tmp_path / f"{index}.json",
                run_id=f"run-{index}",
                source=source,
                final=final,
                group=group,
            )
        ]
        for index, group in enumerate(EXPERIMENT_GROUPS)
    }


def _resign_receipt(receipt_path: Path, receipt: dict[str, object]) -> None:
    receipt.pop("receipt_sha256", None)
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def test_summary_writes_complete_paired_csv_and_json_without_aggregation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
                "measurement_nrmse": 0.4 + index,
            },
            group=group,
        )
        groups[group] = [manifest]
    receipt, validated = _validated_suite(tmp_path, groups)
    _accept_suite(monkeypatch, receipt, validated)

    payload = summarize_paired_manifests(
        groups,
        tmp_path / "paired.csv",
        tmp_path / "paired.json",
        suite_receipt=receipt,
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
    assert payload["suite_receipt"] == {
        "provided": True,
        "verified": True,
        "path": str(receipt.resolve()),
        "size_bytes": receipt.stat().st_size,
        "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
        "project_commit": "d" * 40,
        "hardware": {
            "by_sample": {
                "sample-1": {
                    "identity_sha256": "1" * 64,
                    "class_sha256": "2" * 64,
                }
            }
        },
        "system_evidence": {"by_sample": {}},
        "issues": [],
    }
    assert payload["host_gpu_systems"]["verified"] is False
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
    csv_bytes = (tmp_path / "paired.csv").read_bytes()
    assert payload["csv_output"] == {
        "path": str((tmp_path / "paired.csv").resolve()),
        "size_bytes": len(csv_bytes),
        "sha256": hashlib.sha256(csv_bytes).hexdigest(),
    }


def test_summary_separates_worker_timing_from_replayed_host_gpu_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    groups = _complete_groups(tmp_path)
    receipt, validated = _validated_suite(tmp_path, groups)
    uuid_digests = ("a" * 64, "b" * 64)
    for job in validated["jobs"]:
        job["system_evidence"] = {
            "duration_seconds": 42,
            "gpu_sampling": {
                "attribution_scope": "physical_gpu_host_level_not_process_attributed",
                "sample_count": 4,
                "sample_interval_seconds": 1.0,
                "peak_by_physical_index": {
                    "0": {
                        "physical_index": "0",
                        "logical_index": 0,
                        "uuid_sha256": uuid_digests[0],
                        "name": "NVIDIA GeForce RTX 4090",
                        "memory_total_mib": 24564,
                        "peak_memory_used_mib": 2048,
                        "peak_utilization_percent": 50,
                    },
                    "1": {
                        "physical_index": "1",
                        "logical_index": 1,
                        "uuid_sha256": uuid_digests[1],
                        "name": "NVIDIA GeForce RTX 4090",
                        "memory_total_mib": 24564,
                        "peak_memory_used_mib": 3072,
                        "peak_utilization_percent": 60,
                    },
                },
            },
        }
    _accept_suite(monkeypatch, receipt, validated)

    payload = summarize_paired_manifests(
        groups,
        tmp_path / "paired.csv",
        tmp_path / "paired.json",
        suite_receipt=receipt,
    )

    host = payload["host_gpu_systems"]
    assert host["verified"] is True
    assert host["attribution_scope"] == "physical_gpu_host_level_not_process_attributed"
    scaleguard = host["by_group"]["ScaleGuard"]
    assert scaleguard["counts"] == {
        "validated_non_mock_runs": 1,
        "observed_runs": 1,
    }
    assert scaleguard["wrapper_duration_seconds"]["mean"] == 42.0
    assert scaleguard["sample_interval_seconds"]["mean"] == 1.0
    assert (
        scaleguard["by_gpu_uuid_sha256"][uuid_digests[0]]["peak_memory_used_mib"]["mean"] == 2048.0
    )
    assert (
        scaleguard["by_gpu_uuid_sha256"][uuid_digests[1]]["peak_utilization_percent"]["mean"]
        == 60.0
    )

    timing = summary_module._coz_timing_metrics(
        {
            "steps": [
                {
                    "worker_metadata": {
                        "backend": "chain_of_zoom_persistent",
                        "duration_seconds": 11.0,
                        "initialization_duration_seconds": 30.0,
                    }
                },
                {
                    "worker_metadata": {
                        "backend": "chain_of_zoom_persistent",
                        "duration_seconds": 9.0,
                    }
                },
            ]
        }
    )
    assert timing == {
        "coz_initialization_seconds": 30.0,
        "coz_first_step_seconds": 11.0,
        "coz_steady_step_seconds": 9.0,
    }
    assert summary_module._coz_timing_metrics(
        {
            "steps": [
                {
                    "worker_metadata": {
                        "backend": "chain_of_zoom_subprocess",
                        "duration_seconds": 12.5,
                    }
                }
            ]
        }
    ) == {
        "coz_initialization_seconds": None,
        "coz_first_step_seconds": 12.5,
        "coz_steady_step_seconds": None,
    }


def test_summary_source_replays_external_metrics_and_marks_a_only_not_applicable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.png"
    reference = tmp_path / "reference.png"
    Image.new("RGB", (16, 16), (10, 10, 10)).save(source)
    Image.new("RGB", (16, 16), (100, 100, 100)).save(reference)
    groups: dict[str, list[Path]] = {group: [] for group in EXPERIMENT_GROUPS}
    manifests: list[Path] = []
    references: list[Path] = []
    for index, group in enumerate(EXPERIMENT_GROUPS):
        final = tmp_path / f"final-{index}.png"
        Image.new("RGB", (16, 16), (40 + index * 10,) * 3).save(final)
        manifest = write_summary_manifest(
            tmp_path / f"manifest-{index}.json",
            run_id=f"external-{index}",
            source=source,
            final=final,
            group=group,
        )
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        raw["input_image"].update({"width": 16, "height": 16})
        raw["final_image"].update({"width": 16, "height": 16})
        manifest.write_text(json.dumps(raw), encoding="utf-8")
        groups[group].append(manifest)
        manifests.append(manifest)
        references.append(reference)

    monkeypatch.setattr(
        metrics_module,
        "validate_run_manifest",
        lambda path, **_kwargs: json.loads(path.read_text(encoding="utf-8")),
    )
    metric_receipt = tmp_path / "metrics.json"
    evaluate_metric_receipt(
        manifests,
        references,
        metric_receipt,
        metric_names=("psnr", "ssim"),
    )
    suite_receipt, validated = _validated_suite(tmp_path, groups)
    _accept_suite(monkeypatch, suite_receipt, validated)

    payload = summarize_paired_manifests(
        groups,
        tmp_path / "paired.csv",
        tmp_path / "paired.json",
        suite_receipt=suite_receipt,
        metric_receipts=[metric_receipt],
    )

    pair = payload["pairs"][0]
    assert pair["runs"]["A-only"]["external_metrics"]["psnr"] == {
        "status": "not_applicable",
        "value": None,
        "direction": "higher_is_better",
        "identity_sha256": payload["external_metric_definitions"]["psnr"]["identity_sha256"],
        "reason": "native_resolution_output_not_comparable_to_4x_reference",
    }
    assert pair["runs"]["B-only"]["external_metrics"]["psnr"]["status"] == "measured"
    a_only_effect = next(
        effect
        for effect in payload["external_metric_effects"]
        if effect["comparison"] == "ScaleGuard - A-only" and effect["metric"] == "psnr"
    )
    assert a_only_effect["counts"]["observed_pairs"] == 0
    assert a_only_effect["counts"]["excluded_by_status"] == {
        "baseline:not_applicable|scaleguard:measured": 1
    }
    b_only_effect = next(
        effect
        for effect in payload["external_metric_effects"]
        if effect["comparison"] == "ScaleGuard - B-only" and effect["metric"] == "psnr"
    )
    assert b_only_effect["counts"]["observed_pairs"] == 1
    assert b_only_effect["improvement_oriented_delta"]["independent_clusters"] == 1
    assert payload["external_metric_counts"] == {
        "receipts": 1,
        "definitions": 2,
        "measured": 6,
        "missing": 0,
        "not_applicable": 2,
        "unverified": 0,
        "failed": 0,
    }


def test_summary_rejects_output_alias_and_output_input_overwrite(tmp_path: Path) -> None:
    source, final = _images(tmp_path)
    manifest = write_summary_manifest(
        tmp_path / "manifest.json",
        run_id="run",
        source=source,
        final=final,
    )

    with pytest.raises(EvaluationEvidenceError, match="resolve to the same path"):
        summarize_paired_manifests(
            {"A-only": [manifest]},
            tmp_path / "same.out",
            tmp_path / "same.out",
        )
    with pytest.raises(EvaluationEvidenceError, match="would overwrite"):
        summarize_paired_manifests(
            {"A-only": [manifest]},
            manifest,
            tmp_path / "summary.json",
        )
    with pytest.raises(EvaluationEvidenceError, match="would overwrite manifest evidence"):
        summarize_paired_manifests(
            {"A-only": [manifest]},
            final,
            tmp_path / "summary.json",
        )

    metric_receipt = tmp_path / "metric-receipt.json"
    metric_receipt.write_text("{}\n", encoding="utf-8")
    with pytest.raises(EvaluationEvidenceError, match="would overwrite metric receipt"):
        summarize_paired_manifests(
            {"A-only": [manifest]},
            tmp_path / "summary.csv",
            metric_receipt,
            metric_receipts=[metric_receipt],
        )


def test_summary_rejects_duplicate_metric_binding_and_manifest_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, final = _images(tmp_path)
    manifest = write_summary_manifest(
        tmp_path / "manifest.json",
        run_id="run",
        source=source,
        final=final,
    )
    manifest_digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    receipts = [tmp_path / "metrics-1.json", tmp_path / "metrics-2.json"]
    for receipt in receipts:
        receipt.write_text("{}\n", encoding="utf-8")

    def verification(path: Path, **_kwargs: object) -> dict[str, object]:
        return {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": "a" * 64,
            "receipt_sha256": "b" * 64,
            "verified": True,
            "research_eligible": True,
            "issues": [],
            "protected_paths": [str(path.resolve())],
            "metric_definitions": {
                "musiq": {
                    "identity_sha256": "c" * 64,
                    "direction": "higher_is_better",
                    "reference_required": False,
                }
            },
            "samples": [
                {
                    "manifest_path": str(manifest.resolve()),
                    "manifest_sha256": manifest_digest,
                    "run_id": "run",
                    "metrics": {
                        "musiq": {
                            "status": "measured",
                            "value": 50.0,
                            "direction": "higher_is_better",
                            "identity_sha256": "d" * 64,
                        }
                    },
                }
            ],
        }

    monkeypatch.setattr(summary_module, "verify_metric_receipt", verification)
    with pytest.raises(EvaluationEvidenceError, match="multiple metric receipt samples"):
        summarize_paired_manifests(
            {"A-only": [manifest]},
            tmp_path / "duplicate.csv",
            tmp_path / "duplicate.json",
            metric_receipts=receipts,
        )

    def drifted(path: Path, **kwargs: object) -> dict[str, object]:
        payload = verification(path, **kwargs)
        samples = payload["samples"]
        assert isinstance(samples, list)
        samples[0]["manifest_sha256"] = "f" * 64
        return payload

    monkeypatch.setattr(summary_module, "verify_metric_receipt", drifted)
    with pytest.raises(EvaluationEvidenceError, match="identity drift"):
        summarize_paired_manifests(
            {"A-only": [manifest]},
            tmp_path / "drift.csv",
            tmp_path / "drift.json",
            metric_receipts=[receipts[0]],
        )


def test_summary_rejects_external_metric_definition_conflicts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, final = _images(tmp_path)
    manifests = {
        group: write_summary_manifest(
            tmp_path / f"{group}.json",
            run_id=f"run-{group}",
            source=source,
            final=final,
            group=group,
        )
        for group in ("A-only", "B-only")
    }
    receipts = [tmp_path / "metrics-1.json", tmp_path / "metrics-2.json"]
    for receipt in receipts:
        receipt.write_text("{}\n", encoding="utf-8")

    def verification(path: Path, **_kwargs: object) -> dict[str, object]:
        index = receipts.index(path)
        group = ("A-only", "B-only")[index]
        manifest = manifests[group]
        return {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": "a" * 64,
            "receipt_sha256": "b" * 64,
            "verified": True,
            "research_eligible": True,
            "issues": [],
            "protected_paths": [str(path.resolve())],
            "metric_definitions": {
                "musiq": {
                    "identity_sha256": ("c" if index == 0 else "e") * 64,
                    "direction": "higher_is_better",
                    "reference_required": False,
                }
            },
            "samples": [
                {
                    "manifest_path": str(manifest.resolve()),
                    "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                    "run_id": f"run-{group}",
                    "metrics": {},
                }
            ],
        }

    monkeypatch.setattr(summary_module, "verify_metric_receipt", verification)
    with pytest.raises(EvaluationEvidenceError, match="definition conflicts"):
        summarize_paired_manifests(
            {group: [manifest] for group, manifest in manifests.items()},
            tmp_path / "out.csv",
            tmp_path / "out.json",
            metric_receipts=receipts,
        )


def test_summary_reports_cluster_bootstrapped_paired_effects_and_systems(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    groups: dict[str, list[Path]] = {group: [] for group in EXPERIMENT_GROUPS}
    quality_values = {
        "sample-1": {"A-only": 1.0, "B-only": 1.5, "AB-fixed": 2.0, "ScaleGuard": 3.0},
        "sample-2": {"A-only": 2.0, "B-only": 3.0, "AB-fixed": 4.0, "ScaleGuard": 6.0},
    }
    for sample_index, sample_id in enumerate(("sample-1", "sample-2"), start=1):
        source = tmp_path / f"source-{sample_index}.bin"
        final = tmp_path / f"final-{sample_index}.bin"
        source.write_bytes(f"input-{sample_index}".encode())
        final.write_bytes(f"output-{sample_index}".encode())
        for group_index, group in enumerate(EXPERIMENT_GROUPS):
            groups[group].append(
                write_summary_manifest(
                    tmp_path / f"{sample_id}-{group_index}.json",
                    run_id=f"{sample_id}-{group_index}",
                    source=source,
                    final=final,
                    sample_id=sample_id,
                    group=group,
                    metrics={
                        "quality_gain": quality_values[sample_id][group],
                        "scale_nrmse": 0.2,
                        "scale_edge_mae": 0.3,
                        "measurement_nrmse": 0.4,
                    },
                )
            )
    receipt, validated = _validated_suite(tmp_path, groups)
    _accept_suite(monkeypatch, receipt, validated)

    payload = summarize_paired_manifests(
        groups,
        tmp_path / "paired.csv",
        tmp_path / "paired.json",
        suite_receipt=receipt,
    )

    effect = next(
        item
        for item in payload["paired_effects"]
        if item["comparison"] == "ScaleGuard - A-only" and item["metric"] == "quality_gain"
    )
    assert effect["counts"] == {
        "all_pairs": 2,
        "trusted_complete_pairs": 2,
        "observed_pairs": 2,
        "missing_or_excluded_pairs": 0,
        "missing_or_excluded_rate": 0.0,
    }
    assert effect["improvement_oriented_delta"]["mean"] == pytest.approx(3.0)
    assert effect["improvement_oriented_delta"]["independent_clusters"] == 2
    assert effect["improvement_oriented_delta"]["bootstrap_ci"]["status"] == "estimated"
    assert effect["paired_standardized_effect_dz"] == pytest.approx(3 / 2**0.5)

    systems = payload["systems_by_group"]["ScaleGuard"]["success_rate"]
    assert systems["counts"]["observed"] == 2
    assert systems["mean"] == pytest.approx(1.0)
    assert systems["bootstrap_ci"]["status"] == "estimated"


@pytest.mark.parametrize(
    ("field", "value", "issue"),
    [
        (
            "quality_backend_is_proxy",
            True,
            "ScaleGuard:quality_backend_proxy_or_unverified",
        ),
        (
            "quality_thresholds_calibrated",
            False,
            "ScaleGuard:quality_thresholds_uncalibrated",
        ),
    ],
)
def test_verified_suite_still_requires_calibrated_non_proxy_quality(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: bool,
    issue: str,
) -> None:
    source, final = _images(tmp_path)
    groups = {
        group: [
            write_summary_manifest(
                tmp_path / f"{index}.json",
                run_id=f"run-{index}",
                source=source,
                final=final,
                group=group,
            )
        ]
        for index, group in enumerate(EXPERIMENT_GROUPS)
    }
    scaleguard_manifest = groups["ScaleGuard"][0]
    payload = json.loads(scaleguard_manifest.read_text(encoding="utf-8"))
    payload["provenance"][field] = value
    scaleguard_manifest.write_text(json.dumps(payload), encoding="utf-8")
    receipt, validated = _validated_suite(tmp_path, groups)
    _accept_suite(monkeypatch, receipt, validated)

    summary = summarize_paired_manifests(
        groups,
        tmp_path / "paired.csv",
        tmp_path / "paired.json",
        suite_receipt=receipt,
    )

    assert summary["counts"]["research_eligible_pairs"] == 0
    assert summary["pairs"][0]["research_eligible"] is False
    assert issue in summary["pairs"][0]["issues"]


def test_summary_revalidates_calibration_receipt_semantics_not_self_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    groups = _complete_groups(tmp_path)
    receipt_path = tmp_path / "quality-calibration.json"
    calibration = json.loads(receipt_path.read_text(encoding="utf-8"))
    calibration["thresholds"]["max_scale_nrmse"]["value"] = 999.0
    _resign_receipt(receipt_path, calibration)
    for paths in groups.values():
        manifest_path = paths[0]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["provenance"]["quality_calibration_receipt_size_bytes"] = (
            receipt_path.stat().st_size
        )
        manifest["provenance"]["quality_calibration_receipt_sha256"] = hashlib.sha256(
            receipt_path.read_bytes()
        ).hexdigest()
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    suite_receipt, validated = _validated_suite(tmp_path, groups)
    _accept_suite(monkeypatch, suite_receipt, validated)

    summary = summarize_paired_manifests(
        groups,
        tmp_path / "paired.csv",
        tmp_path / "paired.json",
        suite_receipt=suite_receipt,
    )

    assert summary["counts"]["research_eligible_pairs"] == 0
    assert any(
        "quality_calibration_receipt_invalid:threshold_mismatch:max_scale_nrmse" in issue
        for issue in summary["pairs"][0]["issues"]
    )


def test_measurement_metric_is_required_exactly_when_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enabled_groups = _complete_groups(tmp_path / "enabled")
    for paths in enabled_groups.values():
        manifest_path = paths[0]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["final_metrics"]["metrics"]["measurement_nrmse"] = None
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    enabled_receipt, enabled_validated = _validated_suite(
        tmp_path / "enabled",
        enabled_groups,
    )
    _accept_suite(monkeypatch, enabled_receipt, enabled_validated)
    enabled = summarize_paired_manifests(
        enabled_groups,
        tmp_path / "enabled.csv",
        tmp_path / "enabled.json",
        suite_receipt=enabled_receipt,
    )
    assert enabled["counts"]["research_eligible_pairs"] == 0
    assert "ScaleGuard:missing_final_metric:measurement_nrmse" in enabled["pairs"][0]["issues"]

    disabled_root = tmp_path / "disabled"
    disabled_groups = _complete_groups(disabled_root)
    calibration_path = disabled_root / "quality-calibration.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration["thresholds"].pop("max_measurement_nrmse")
    calibration["metric_backend"]["measurement"] = None
    _resign_receipt(calibration_path, calibration)
    calibration_digest = hashlib.sha256(calibration_path.read_bytes()).hexdigest()
    for paths in disabled_groups.values():
        manifest_path = paths[0]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["config"]["metrics"]["measurement_enabled"] = False
        manifest["final_metrics"]["metrics"]["measurement_nrmse"] = None
        manifest["provenance"]["quality_calibration_receipt_size_bytes"] = (
            calibration_path.stat().st_size
        )
        manifest["provenance"]["quality_calibration_receipt_sha256"] = calibration_digest
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    disabled_receipt, disabled_validated = _validated_suite(
        disabled_root,
        disabled_groups,
    )
    _accept_suite(monkeypatch, disabled_receipt, disabled_validated)
    disabled = summarize_paired_manifests(
        disabled_groups,
        tmp_path / "disabled.csv",
        tmp_path / "disabled.json",
        suite_receipt=disabled_receipt,
    )
    assert disabled["counts"]["research_eligible_pairs"] == 1


def test_complete_pair_without_suite_receipt_is_not_research_eligible(
    tmp_path: Path,
) -> None:
    source, final = _images(tmp_path)
    groups = {
        group: [
            write_summary_manifest(
                tmp_path / f"{index}.json",
                run_id=f"run-{index}",
                source=source,
                final=final,
                group=group,
            )
        ]
        for index, group in enumerate(EXPERIMENT_GROUPS)
    }

    payload = summarize_paired_manifests(
        groups,
        tmp_path / "paired.csv",
        tmp_path / "paired.json",
    )

    pair = payload["pairs"][0]
    assert pair["complete"] is True
    assert pair["research_eligible"] is False
    assert "suite_receipt_unverified" in pair["issues"]
    assert payload["counts"]["research_eligible_pairs"] == 0


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
    assert "suite_receipt_unverified" in pair["issues"]
    assert payload["suite_receipt"]["provided"] is False
    assert payload["suite_receipt"]["verified"] is False
    assert payload["counts"]["mock_pairs"] == 1
    with (tmp_path / "paired.csv").open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["b_only_status"] == "missing"
    assert row["b_only_quality_gain"] == ""


def test_mock_pair_count_is_exact_across_multiple_pairs(tmp_path: Path) -> None:
    source, final = _images(tmp_path)
    mock_manifest = write_summary_manifest(
        tmp_path / "mock-pair.json",
        run_id="mock-pair",
        source=source,
        final=final,
        mock=True,
        sample_id="sample-mock",
    )
    real_manifest = write_summary_manifest(
        tmp_path / "real-pair.json",
        run_id="real-pair",
        source=source,
        final=final,
        sample_id="sample-real",
    )

    payload = summarize_paired_manifests(
        {"A-only": [mock_manifest, real_manifest]},
        tmp_path / "paired.csv",
        tmp_path / "paired.json",
    )

    assert payload["counts"]["pairs"] == 2
    assert payload["counts"]["mock_pairs"] == 1


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
        group="ScaleGuard",
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
                group=group,
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


def test_summary_rejects_embedded_group_spoofing_and_cross_input_sample_reuse(
    tmp_path: Path,
) -> None:
    source, final = _images(tmp_path)
    a_only = write_summary_manifest(
        tmp_path / "a.json",
        run_id="a",
        source=source,
        final=final,
        group="A-only",
        sample_id="paired-sample",
    )
    with pytest.raises(EvaluationEvidenceError, match="embeds experiment group"):
        summarize_paired_manifests(
            {"B-only": [a_only]},
            tmp_path / "out.csv",
            tmp_path / "out.json",
        )

    other_source = tmp_path / "other.bin"
    other_source.write_bytes(b"different-input")
    b_only = write_summary_manifest(
        tmp_path / "b.json",
        run_id="b",
        source=other_source,
        final=final,
        group="B-only",
        sample_id="paired-sample",
    )
    with pytest.raises(EvaluationEvidenceError, match="identical input bytes"):
        summarize_paired_manifests(
            {"A-only": [a_only], "B-only": [b_only]},
            tmp_path / "out.csv",
            tmp_path / "out.json",
        )


def test_summary_requires_full_manifest_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, final = _images(tmp_path)
    manifest = write_summary_manifest(
        tmp_path / "invalid.json",
        run_id="invalid",
        source=source,
        final=final,
    )

    def reject(*_args: object, **_kwargs: object) -> None:
        raise ManifestValidationError("forged runtime evidence")

    monkeypatch.setattr(summary_module, "validate_run_manifest", reject)
    with pytest.raises(EvaluationEvidenceError, match="forged runtime evidence"):
        summarize_paired_manifests(
            {"A-only": [manifest]},
            tmp_path / "out.csv",
            tmp_path / "out.json",
        )


def test_summary_marks_cross_group_config_or_runtime_mismatch_ineligible(
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
            group=group,
        )
        groups[group] = [manifest]
    raw = json.loads(groups["B-only"][0].read_text(encoding="utf-8"))
    raw["config"]["coz"]["tile_overlap"] = 32
    groups["B-only"][0].write_text(json.dumps(raw), encoding="utf-8")

    payload = summarize_paired_manifests(
        groups,
        tmp_path / "out.csv",
        tmp_path / "out.json",
    )

    assert payload["pairs"][0]["research_eligible"] is False
    assert "paired_runtime_or_config_mismatch" in payload["pairs"][0]["issues"]


def test_suite_receipt_manifest_binding_must_match_every_input_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, final = _images(tmp_path)
    groups = {
        group: [
            write_summary_manifest(
                tmp_path / f"{index}.json",
                run_id=f"run-{index}",
                source=source,
                final=final,
                group=group,
            )
        ]
        for index, group in enumerate(EXPERIMENT_GROUPS)
    }
    receipt, validated = _validated_suite(tmp_path, groups)
    jobs = validated["jobs"]
    assert isinstance(jobs, list)
    jobs[0]["manifest"]["sha256"] = "f" * 64
    _accept_suite(monkeypatch, receipt, validated)

    payload = summarize_paired_manifests(
        groups,
        tmp_path / "out.csv",
        tmp_path / "out.json",
        suite_receipt=receipt,
    )

    pair = payload["pairs"][0]
    assert pair["research_eligible"] is False
    assert "suite_receipt_unverified" in pair["issues"]
    assert payload["suite_receipt"]["verified"] is False
    assert payload["suite_receipt"]["issues"] == ["suite_receipt_manifest_set_mismatch"]


def test_invalid_suite_receipt_is_reported_as_evaluation_evidence_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, final = _images(tmp_path)
    manifest = write_summary_manifest(
        tmp_path / "manifest.json",
        run_id="run",
        source=source,
        final=final,
    )
    receipt = tmp_path / "suite-receipt.json"
    receipt.write_text("{}\n", encoding="utf-8")

    def reject(_path: Path) -> dict[str, object]:
        raise ExperimentProtocolError("suite receipt self digest is invalid")

    monkeypatch.setattr(summary_module, "validate_ablation_suite_receipt", reject)

    with pytest.raises(
        EvaluationEvidenceError,
        match=r"invalid ablation suite receipt.*self digest is invalid",
    ):
        summarize_paired_manifests(
            {"A-only": [manifest]},
            tmp_path / "out.csv",
            tmp_path / "out.json",
            suite_receipt=receipt,
        )


def test_summary_reuses_public_manifest_experiment_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, final = _images(tmp_path)
    manifest = write_summary_manifest(
        tmp_path / "manifest.json",
        run_id="run",
        source=source,
        final=final,
    )
    calls: list[tuple[dict[str, object], str]] = []

    def strict_contract(
        payload: dict[str, object],
        group_id: str,
    ) -> list[str]:
        calls.append((payload, group_id))
        return ["strict_contract_failure"]

    monkeypatch.setattr(summary_module, "manifest_experiment_issues", strict_contract)

    payload = summarize_paired_manifests(
        {"A-only": [manifest]},
        tmp_path / "out.csv",
        tmp_path / "out.json",
    )

    assert calls
    assert calls[0][1] == "A-only"
    assert "A-only:strict_contract_failure" in payload["pairs"][0]["issues"]


def test_summary_requires_group_specific_persistent_coz_evidence(
    tmp_path: Path,
) -> None:
    source, final = _images(tmp_path)
    manifest = write_summary_manifest(
        tmp_path / "b.json",
        run_id="b",
        source=source,
        final=final,
        group="B-only",
    )
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    raw["steps"] = []
    raw["scale_session_process"] = None
    manifest.write_text(json.dumps(raw), encoding="utf-8")

    payload = summarize_paired_manifests(
        {"B-only": [manifest]},
        tmp_path / "out.csv",
        tmp_path / "out.json",
    )

    issues = payload["pairs"][0]["issues"]
    assert "B-only:manifest_coz_step_count:0" in issues
    assert payload["pairs"][0]["research_eligible"] is False


def test_summary_calls_the_real_manifest_validator_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, final = _images(tmp_path)
    manifest = write_summary_manifest(
        tmp_path / "structurally-invalid.json",
        run_id="invalid",
        source=source,
        final=final,
    )
    from scaleguard.manifest import validate_run_manifest as real_validator

    monkeypatch.setattr(summary_module, "validate_run_manifest", real_validator)
    with pytest.raises(EvaluationEvidenceError, match="invalid run manifest"):
        summarize_paired_manifests(
            {"A-only": [manifest]},
            tmp_path / "out.csv",
            tmp_path / "out.json",
        )


def test_summary_excludes_pairs_scored_against_different_references(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A paired delta is only meaningful when both groups used the same reference."""

    source = tmp_path / "source.png"
    Image.new("RGB", (16, 16), (10, 10, 10)).save(source)
    shared_reference = tmp_path / "reference-shared.png"
    other_reference = tmp_path / "reference-other.png"
    Image.new("RGB", (16, 16), (100, 100, 100)).save(shared_reference)
    Image.new("RGB", (16, 16), (200, 200, 200)).save(other_reference)

    groups: dict[str, list[Path]] = {group: [] for group in EXPERIMENT_GROUPS}
    manifests: list[Path] = []
    references: list[Path] = []
    for index, group in enumerate(EXPERIMENT_GROUPS):
        final = tmp_path / f"final-{index}.png"
        Image.new("RGB", (16, 16), (40 + index * 10,) * 3).save(final)
        manifest = write_summary_manifest(
            tmp_path / f"manifest-{index}.json",
            run_id=f"reference-{index}",
            source=source,
            final=final,
            group=group,
        )
        raw = json.loads(manifest.read_text(encoding="utf-8"))
        raw["input_image"].update({"width": 16, "height": 16})
        raw["final_image"].update({"width": 16, "height": 16})
        manifest.write_text(json.dumps(raw), encoding="utf-8")
        groups[group].append(manifest)
        manifests.append(manifest)
        # B-only is scored against a different reference than ScaleGuard.
        references.append(other_reference if group == "B-only" else shared_reference)

    monkeypatch.setattr(
        metrics_module,
        "validate_run_manifest",
        lambda path, **_kwargs: json.loads(path.read_text(encoding="utf-8")),
    )
    metric_receipt = tmp_path / "metrics.json"
    evaluate_metric_receipt(
        manifests,
        references,
        metric_receipt,
        metric_names=("psnr",),
    )
    suite_receipt, validated = _validated_suite(tmp_path, groups)
    _accept_suite(monkeypatch, suite_receipt, validated)

    payload = summarize_paired_manifests(
        groups,
        tmp_path / "paired.csv",
        tmp_path / "paired.json",
        suite_receipt=suite_receipt,
        metric_receipts=[metric_receipt],
    )

    b_only_effect = next(
        effect
        for effect in payload["external_metric_effects"]
        if effect["comparison"] == "ScaleGuard - B-only" and effect["metric"] == "psnr"
    )
    assert b_only_effect["counts"]["observed_pairs"] == 0
    assert b_only_effect["counts"]["excluded_by_status"] == {"reference_mismatch": 1}

    # AB-fixed shares ScaleGuard's reference and is still compared.
    ab_effect = next(
        effect
        for effect in payload["external_metric_effects"]
        if effect["comparison"] == "ScaleGuard - AB-fixed" and effect["metric"] == "psnr"
    )
    assert ab_effect["counts"]["observed_pairs"] == 1
