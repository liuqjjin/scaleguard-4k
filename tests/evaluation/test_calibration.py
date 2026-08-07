from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scaleguard.config import load_config
from scaleguard.evaluation import calibration as calibration_module
from scaleguard.evaluation.calibration import (
    CalibrationParameters,
    calibrate_from_manifests,
    verify_calibration_receipt,
)
from scaleguard.evaluation.evidence import EvaluationEvidenceError, canonical_sha256
from scaleguard.manifest import ManifestValidationError

from ._fixtures import write_calibration_manifest


@pytest.fixture(autouse=True)
def _accept_minimal_calibration_fixtures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        calibration_module,
        "validate_run_manifest",
        lambda path, **_kwargs: json.loads(path.read_text(encoding="utf-8")),
    )

    def identity(
        manifest: dict[str, Any],
        **_kwargs: object,
    ) -> tuple[dict[str, Any], str, str | None]:
        steps = manifest.get("steps")
        metrics = steps[0]["metrics"] if isinstance(steps, list) and steps else {}
        return (
            {"quality": {"backend": "fixture"}, "measurement": None},
            str(metrics.get("quality_backend")),
            metrics.get("measurement_model"),
        )

    monkeypatch.setattr(calibration_module, "_metric_identity", identity)
    monkeypatch.setattr(
        calibration_module,
        "_expected_metric_identity",
        lambda _config, recorded, _reasons: dict(recorded),
    )


def _evidence_files(tmp_path: Path) -> tuple[Path, Path]:
    trusted = tmp_path / "trusted.bin"
    candidate = tmp_path / "candidate.bin"
    trusted.write_bytes(b"trusted-state")
    candidate.write_bytes(b"candidate-state")
    return trusted, candidate


def _labels(path: Path, rows: list[tuple[str, int, str]]) -> Path:
    path.write_text(
        "run_id,step_index,acceptable\n"
        + "".join(
            f"{run_id},{step_index},{acceptable}\n" for run_id, step_index, acceptable in rows
        ),
        encoding="utf-8",
    )
    return path


def _parameters(**overrides) -> CalibrationParameters:
    values = {
        "minimum_acceptable_samples": 2,
        "quality_lower_quantile": 0.25,
        "error_upper_quantile": 0.75,
        "bootstrap_samples": 40,
        "bootstrap_confidence": 0.8,
        "bootstrap_seed": 7,
        "include_measurement": False,
    }
    values.update(overrides)
    return CalibrationParameters(**values)


def test_calibration_receipt_is_deterministic_and_hashes_all_inputs(tmp_path: Path) -> None:
    trusted, candidate = _evidence_files(tmp_path)
    first = write_calibration_manifest(
        tmp_path / "first.json",
        run_id="real-1",
        trusted=trusted,
        candidate=candidate,
        values=[(-0.1, 0.1, 0.2), (0.3, 0.5, 0.6)],
    )
    second = write_calibration_manifest(
        tmp_path / "second.json",
        run_id="real-2",
        trusted=trusted,
        candidate=candidate,
        values=[(0.9, 0.9, 1.0)],
    )
    labels = _labels(
        tmp_path / "labels.csv",
        [("real-1", 1, "true"), ("real-1", 2, "1"), ("real-2", 1, "true")],
    )
    one = calibrate_from_manifests(
        [second, first],
        labels,
        tmp_path / "receipt-one.json",
        parameters=_parameters(),
    )
    two = calibrate_from_manifests(
        [first, second],
        labels,
        tmp_path / "receipt-two.json",
        parameters=_parameters(),
    )

    assert one == two
    assert one["status"] == "calibrated"
    assert one["sample_counts"] == {
        "labels": 3,
        "matched_metric_steps": 3,
        "acceptable": 3,
        "acceptable_real": 3,
        "unacceptable_real": 0,
        "mock_excluded": 0,
        "estimation_samples": 3,
        "measurement_estimation_samples": 0,
        "independent_input_clusters": 2,
        "acceptable_input_clusters": 2,
        "measurement_input_clusters": 0,
    }
    assert one["thresholds"]["min_quality_gain"]["value"] == pytest.approx(0.1)
    assert one["thresholds"]["max_scale_nrmse"]["value"] == pytest.approx(0.7)
    assert one["thresholds"]["max_scale_edge_mae"]["value"] == pytest.approx(0.8)
    assert one["algorithm"]["bootstrap_seed"] == 7
    assert one["algorithm"]["numpy_version"] == np.__version__
    assert all(item["sha256"] for item in one["inputs"]["manifests"])
    body = dict(one)
    digest = body.pop("receipt_sha256")
    assert digest == canonical_sha256(body)


def test_repeated_runs_of_one_input_do_not_satisfy_the_independent_sample_minimum(
    tmp_path: Path,
) -> None:
    trusted, candidate = _evidence_files(tmp_path)
    shared_input = tmp_path / "shared-input.bin"
    shared_input.write_bytes(b"one independent image")
    manifests = [
        write_calibration_manifest(
            tmp_path / f"seed-{seed}.json",
            run_id=f"seed-{seed}",
            trusted=trusted,
            candidate=candidate,
            values=[(0.1 + seed, 0.2, 0.3)],
            input_image=shared_input,
        )
        for seed in (1, 2)
    ]
    labels = _labels(
        tmp_path / "labels.csv",
        [("seed-1", 1, "true"), ("seed-2", 1, "true")],
    )

    receipt = calibrate_from_manifests(
        manifests,
        labels,
        tmp_path / "receipt.json",
        parameters=_parameters(minimum_acceptable_samples=2),
    )

    assert receipt["status"] == "insufficient_data"
    assert receipt["sample_counts"]["acceptable_input_clusters"] == 1
    assert "acceptable_input_clusters_below_minimum:1<2" in receipt["issues"]


def test_mock_run_is_rejected_from_calibration(
    tmp_path: Path,
) -> None:
    trusted, candidate = _evidence_files(tmp_path)
    real = write_calibration_manifest(
        tmp_path / "real.json",
        run_id="real",
        trusted=trusted,
        candidate=candidate,
        values=[(0.1, 0.1, 0.1)],
    )
    mock = write_calibration_manifest(
        tmp_path / "mock.json",
        run_id="mock",
        trusted=trusted,
        candidate=candidate,
        values=[(99.0, 99.0, 99.0)],
        mock=True,
    )
    labels = _labels(
        tmp_path / "labels.csv",
        [("real", 1, "true"), ("mock", 1, "true")],
    )

    with pytest.raises(EvaluationEvidenceError, match="mock run is not eligible"):
        calibrate_from_manifests(
            [real, mock],
            labels,
            tmp_path / "receipt.json",
            parameters=_parameters(minimum_acceptable_samples=2),
        )


def test_no_acceptable_samples_produces_an_explicit_insufficient_receipt(
    tmp_path: Path,
) -> None:
    trusted, candidate = _evidence_files(tmp_path)
    manifest = write_calibration_manifest(
        tmp_path / "manifest.json",
        run_id="rejected",
        trusted=trusted,
        candidate=candidate,
        values=[(0.1, 0.2, 0.3)],
    )
    labels = _labels(tmp_path / "labels.csv", [("rejected", 1, "false")])

    receipt = calibrate_from_manifests(
        [manifest],
        labels,
        tmp_path / "receipt.json",
        parameters=_parameters(minimum_acceptable_samples=1),
    )

    assert receipt["status"] == "insufficient_data"
    assert receipt["thresholds"] == {}
    assert "no_acceptable_real_samples" in receipt["issues"]
    assert "acceptable_input_clusters_below_minimum:0<1" in receipt["issues"]


@pytest.mark.parametrize(
    ("labels_rows", "message"),
    [
        ([("run", 1, "true"), ("run", 1, "false")], "duplicate label"),
        ([], "contains no data rows"),
        ([("unknown", 1, "true")], "missing labels"),
    ],
)
def test_label_coverage_and_duplicates_are_hard_errors(
    tmp_path: Path,
    labels_rows: list[tuple[str, int, str]],
    message: str,
) -> None:
    trusted, candidate = _evidence_files(tmp_path)
    manifest = write_calibration_manifest(
        tmp_path / "manifest.json",
        run_id="run",
        trusted=trusted,
        candidate=candidate,
        values=[(0.1, 0.2, 0.3)],
    )
    labels = _labels(tmp_path / "labels.csv", labels_rows)

    with pytest.raises(EvaluationEvidenceError, match=message):
        calibrate_from_manifests([manifest], labels, tmp_path / "receipt.json")


def test_calibration_rejects_duplicate_manifest_keys(tmp_path: Path) -> None:
    trusted, candidate = _evidence_files(tmp_path)
    manifest = write_calibration_manifest(
        tmp_path / "manifest.json",
        run_id="run",
        trusted=trusted,
        candidate=candidate,
        values=[(0.1, 0.2, 0.3)],
    )
    original = manifest.read_text(encoding="utf-8").lstrip()
    manifest.write_text('{"run_id":"forged",' + original[1:], encoding="utf-8")
    labels = _labels(tmp_path / "labels.csv", [("run", 1, "true")])

    with pytest.raises(EvaluationEvidenceError, match="duplicate JSON object key 'run_id'"):
        calibrate_from_manifests([manifest], labels, tmp_path / "receipt.json")


@pytest.mark.parametrize(
    "header",
    [
        "run_id,step_index,acceptable,acceptable",
        "run_id,acceptable,step_index",
        "run_id,step_index,acceptable,notes",
    ],
)
def test_labels_reject_ambiguous_or_noncanonical_headers(
    tmp_path: Path,
    header: str,
) -> None:
    trusted, candidate = _evidence_files(tmp_path)
    manifest = write_calibration_manifest(
        tmp_path / "manifest.json",
        run_id="run",
        trusted=trusted,
        candidate=candidate,
        values=[(0.1, 0.2, 0.3)],
    )
    labels = tmp_path / "labels.csv"
    labels.write_text(f"{header}\nrun,1,true,false\n", encoding="utf-8")

    with pytest.raises(EvaluationEvidenceError, match="header must be exactly"):
        calibrate_from_manifests([manifest], labels, tmp_path / "receipt.json")


def test_unknown_label_and_duplicate_manifest_run_are_rejected(tmp_path: Path) -> None:
    trusted, candidate = _evidence_files(tmp_path)
    first = write_calibration_manifest(
        tmp_path / "first.json",
        run_id="run",
        trusted=trusted,
        candidate=candidate,
        values=[(0.1, 0.2, 0.3)],
    )
    other = write_calibration_manifest(
        tmp_path / "other.json",
        run_id="other",
        trusted=trusted,
        candidate=candidate,
        values=[(0.2, 0.3, 0.4)],
    )
    labels = _labels(
        tmp_path / "labels.csv",
        [("run", 1, "true"), ("other", 1, "true"), ("ghost", 1, "true")],
    )
    with pytest.raises(EvaluationEvidenceError, match="do not match"):
        calibrate_from_manifests([first, other], labels, tmp_path / "receipt.json")

    duplicate = write_calibration_manifest(
        tmp_path / "duplicate.json",
        run_id="run",
        trusted=trusted,
        candidate=candidate,
        values=[(0.2, 0.3, 0.4)],
    )
    exact_labels = _labels(tmp_path / "exact.csv", [("run", 1, "true")])
    with pytest.raises(EvaluationEvidenceError, match="duplicate manifest run_id"):
        calibrate_from_manifests([first, duplicate], exact_labels, tmp_path / "receipt.json")


def test_artifact_hash_mismatch_and_mixed_backends_are_rejected(tmp_path: Path) -> None:
    trusted, candidate = _evidence_files(tmp_path)
    first = write_calibration_manifest(
        tmp_path / "first.json",
        run_id="first",
        trusted=trusted,
        candidate=candidate,
        values=[(0.1, 0.2, 0.3)],
    )
    candidate.write_bytes(b"tampered")
    labels = _labels(tmp_path / "labels.csv", [("first", 1, "true")])
    with pytest.raises(EvaluationEvidenceError, match="SHA256 mismatch"):
        calibrate_from_manifests([first], labels, tmp_path / "receipt.json")

    trusted, candidate = _evidence_files(tmp_path)
    first = write_calibration_manifest(
        tmp_path / "first.json",
        run_id="first",
        trusted=trusted,
        candidate=candidate,
        values=[(0.1, 0.2, 0.3)],
    )
    second = write_calibration_manifest(
        tmp_path / "second.json",
        run_id="second",
        trusted=trusted,
        candidate=candidate,
        values=[(0.2, 0.3, 0.4)],
        quality_backend="pyiqa:musiq",
    )
    labels = _labels(
        tmp_path / "labels.csv",
        [("first", 1, "true"), ("second", 1, "true")],
    )
    with pytest.raises(EvaluationEvidenceError, match="mix quality backends"):
        calibrate_from_manifests([first, second], labels, tmp_path / "receipt.json")


def test_measurement_calibration_requires_complete_consistent_measurements(
    tmp_path: Path,
) -> None:
    trusted, candidate = _evidence_files(tmp_path)
    measured = write_calibration_manifest(
        tmp_path / "measured.json",
        run_id="measured",
        trusted=trusted,
        candidate=candidate,
        values=[(0.1, 0.2, 0.3)],
        measurement=(0.4, "resize"),
    )
    missing = write_calibration_manifest(
        tmp_path / "missing.json",
        run_id="missing",
        trusted=trusted,
        candidate=candidate,
        values=[(0.2, 0.3, 0.4)],
    )
    labels = _labels(
        tmp_path / "labels.csv",
        [("measured", 1, "true"), ("missing", 1, "true")],
    )

    receipt = calibrate_from_manifests(
        [measured, missing],
        labels,
        tmp_path / "receipt.json",
        parameters=_parameters(include_measurement=True),
    )

    assert receipt["status"] == "insufficient_data"
    assert receipt["sample_counts"]["estimation_samples"] == 2
    assert receipt["sample_counts"]["measurement_estimation_samples"] == 1
    assert receipt["thresholds"]["max_measurement_nrmse"]["value"] == pytest.approx(0.4)
    assert any(issue.startswith("missing_measurement_metrics") for issue in receipt["issues"])


def test_verifier_requires_integrity_calibrated_status_backend_and_exact_config(
    tmp_path: Path,
) -> None:
    trusted, candidate = _evidence_files(tmp_path)
    manifest = write_calibration_manifest(
        tmp_path / "manifest.json",
        run_id="run",
        trusted=trusted,
        candidate=candidate,
        values=[(0.1, 0.2, 0.3), (0.5, 0.6, 0.7)],
    )
    labels = _labels(
        tmp_path / "labels.csv",
        [("run", 1, "true"), ("run", 2, "true")],
    )
    receipt_path = tmp_path / "receipt.json"
    receipt = calibrate_from_manifests(
        [manifest],
        labels,
        receipt_path,
        parameters=_parameters(minimum_acceptable_samples=1),
    )
    thresholds = receipt["thresholds"]
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
metrics:
  quality_backend: gradient_proxy
  min_quality_gain: {thresholds["min_quality_gain"]["value"]}
  max_scale_nrmse: {thresholds["max_scale_nrmse"]["value"]}
  max_scale_edge_mae: {thresholds["max_scale_edge_mae"]["value"]}
""",
        encoding="utf-8",
    )

    assert verify_calibration_receipt(receipt_path, load_config(config_path)) == (True, [])

    config_path.write_text(
        """
metrics:
  quality_backend: gradient_proxy
  min_quality_gain: -999
""",
        encoding="utf-8",
    )
    valid, reasons = verify_calibration_receipt(receipt_path, config_path)
    assert not valid
    assert "threshold_mismatch:min_quality_gain" in reasons

    tampered = json.loads(receipt_path.read_text(encoding="utf-8"))
    tampered["status"] = "insufficient_data"
    receipt_path.write_text(json.dumps(tampered), encoding="utf-8")
    valid, reasons = verify_calibration_receipt(receipt_path, config_path)
    assert not valid
    assert "status_is_not_calibrated" in reasons
    assert "receipt_sha256_mismatch" in reasons


def test_verifier_rejects_structurally_incomplete_self_consistent_receipts(
    tmp_path: Path,
) -> None:
    trusted, candidate = _evidence_files(tmp_path)
    manifest = write_calibration_manifest(
        tmp_path / "manifest.json",
        run_id="run",
        trusted=trusted,
        candidate=candidate,
        values=[(0.1, 0.2, 0.3)],
    )
    labels = _labels(tmp_path / "labels.csv", [("run", 1, "true")])
    receipt_path = tmp_path / "receipt.json"
    baseline = calibrate_from_manifests(
        [manifest],
        labels,
        receipt_path,
        parameters=_parameters(minimum_acceptable_samples=1),
    )
    thresholds = baseline["thresholds"]
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
metrics:
  min_quality_gain: {thresholds["min_quality_gain"]["value"]}
  max_scale_nrmse: {thresholds["max_scale_nrmse"]["value"]}
  max_scale_edge_mae: {thresholds["max_scale_edge_mae"]["value"]}
""",
        encoding="utf-8",
    )

    def verify_mutation(mutated: dict[str, Any]) -> list[str]:
        mutated.pop("receipt_sha256", None)
        mutated["receipt_sha256"] = canonical_sha256(mutated)
        receipt_path.write_text(json.dumps(mutated), encoding="utf-8")
        return verify_calibration_receipt(receipt_path, config)[1]

    mutated = deepcopy(baseline)
    mutated["issues"] = ["manually-added"]
    assert "receipt_has_issues" in verify_mutation(mutated)

    mutated = deepcopy(baseline)
    mutated.pop("algorithm")
    assert "calibration_evidence_missing" in verify_mutation(mutated)

    mutated = deepcopy(baseline)
    mutated["inputs"] = {}
    assert "input_evidence_missing" in verify_mutation(mutated)

    mutated = deepcopy(baseline)
    mutated["metric_backend"] = None
    assert "metric_backend_missing" in verify_mutation(mutated)

    mutated = deepcopy(baseline)
    mutated["thresholds"]["min_quality_gain"].pop("bootstrap_ci")
    assert "bootstrap_ci_missing:min_quality_gain" in verify_mutation(mutated)

    mutated = deepcopy(baseline)
    mutated["thresholds"]["max_scale_nrmse"]["bootstrap_ci"]["confidence"] = 2.0
    assert "bootstrap_ci_invalid:max_scale_nrmse" in verify_mutation(mutated)

    mutated = deepcopy(baseline)
    mutated["thresholds"]["invented"] = {
        "value": 0.0,
        "bootstrap_ci": {"lower": 0.0, "upper": 0.0, "confidence": 0.95},
    }
    assert "unexpected_threshold:invented" in verify_mutation(mutated)

    mutated = deepcopy(baseline)
    mutated["thresholds"] = None
    assert "thresholds_missing" in verify_mutation(mutated)


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("run_id,acceptable\nrun,true\n", "header must be exactly"),
        ("run_id,step_index,acceptable\n,1,true\n", "run_id is empty"),
        ("run_id,step_index,acceptable\nrun,nope,true\n", "must be an integer"),
        ("run_id,step_index,acceptable\nrun,0,true\n", "must be positive"),
        ("run_id,step_index,acceptable\nrun,1,yes\n", "must be one of"),
    ],
)
def test_malformed_label_rows_are_rejected(
    tmp_path: Path,
    contents: str,
    message: str,
) -> None:
    labels = tmp_path / "labels.csv"
    labels.write_text(contents, encoding="utf-8")
    with pytest.raises(EvaluationEvidenceError, match=message):
        calibrate_from_manifests(
            [tmp_path / "unused.json"],
            labels,
            tmp_path / "receipt.json",
        )


def test_relative_artifacts_use_the_explicit_artifact_root(tmp_path: Path) -> None:
    root = tmp_path / "artifact-root"
    root.mkdir()
    trusted = root / "trusted.bin"
    candidate = root / "candidate.bin"
    trusted.write_bytes(b"trusted")
    candidate.write_bytes(b"candidate")
    manifest = write_calibration_manifest(
        tmp_path / "manifest.json",
        run_id="run",
        trusted=trusted,
        candidate=candidate,
        values=[(0.1, 0.2, 0.3)],
    )
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    raw["steps"][0]["trusted_before"]["path"] = "trusted.bin"
    raw["steps"][0]["candidate"]["path"] = "candidate.bin"
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    labels = _labels(tmp_path / "labels.csv", [("run", 1, "true")])

    receipt = calibrate_from_manifests(
        [manifest],
        labels,
        tmp_path / "receipt.json",
        parameters=_parameters(minimum_acceptable_samples=1),
        artifact_root=root,
    )

    assert receipt["status"] == "calibrated"


def test_calibration_requires_full_manifests_and_protects_all_source_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trusted, candidate = _evidence_files(tmp_path)
    manifest = write_calibration_manifest(
        tmp_path / "manifest.json",
        run_id="run",
        trusted=trusted,
        candidate=candidate,
        values=[(0.1, 0.2, 0.3)],
    )
    labels = _labels(tmp_path / "labels.csv", [("run", 1, "true")])

    with pytest.raises(EvaluationEvidenceError, match="would overwrite labels"):
        calibrate_from_manifests(
            [manifest],
            labels,
            labels,
            parameters=_parameters(minimum_acceptable_samples=1),
        )

    def reject(*_args: object, **_kwargs: object) -> None:
        raise ManifestValidationError("forged runtime evidence")

    monkeypatch.setattr(calibration_module, "validate_run_manifest", reject)
    with pytest.raises(EvaluationEvidenceError, match="forged runtime evidence"):
        calibrate_from_manifests(
            [manifest],
            labels,
            tmp_path / "receipt.json",
            parameters=_parameters(minimum_acceptable_samples=1),
        )


def test_verifier_replays_sources_and_rejects_legacy_schema(tmp_path: Path) -> None:
    trusted, candidate = _evidence_files(tmp_path)
    manifest = write_calibration_manifest(
        tmp_path / "manifest.json",
        run_id="run",
        trusted=trusted,
        candidate=candidate,
        values=[(0.1, 0.2, 0.3)],
    )
    labels = _labels(tmp_path / "labels.csv", [("run", 1, "true")])
    receipt_path = tmp_path / "receipt.json"
    receipt = calibrate_from_manifests(
        [manifest],
        labels,
        receipt_path,
        parameters=_parameters(minimum_acceptable_samples=1),
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "metrics:\n"
        "  quality_backend: gradient_proxy\n"
        f"  min_quality_gain: {receipt['thresholds']['min_quality_gain']['value']}\n"
        f"  max_scale_nrmse: {receipt['thresholds']['max_scale_nrmse']['value']}\n"
        f"  max_scale_edge_mae: {receipt['thresholds']['max_scale_edge_mae']['value']}\n",
        encoding="utf-8",
    )

    labels.write_text("run_id,step_index,acceptable\nrun,1,false\n", encoding="utf-8")
    valid, reasons = verify_calibration_receipt(receipt_path, config_path)
    assert not valid
    assert any(reason.startswith("source_recompute_") for reason in reasons)

    legacy = deepcopy(receipt)
    legacy["schema_version"] = "scaleguard.calibration-receipt/v1"
    legacy.pop("receipt_sha256")
    legacy["receipt_sha256"] = canonical_sha256(legacy)
    receipt_path.write_text(json.dumps(legacy), encoding="utf-8")
    valid, reasons = verify_calibration_receipt(receipt_path, config_path)
    assert not valid
    assert "unsupported_schema" in reasons


@pytest.mark.parametrize(
    "parameters",
    [
        CalibrationParameters(minimum_acceptable_samples=0),
        CalibrationParameters(quality_lower_quantile=-0.1),
        CalibrationParameters(error_upper_quantile=1.1),
        CalibrationParameters(bootstrap_samples=0),
        CalibrationParameters(bootstrap_seed=-1),
        CalibrationParameters(bootstrap_seed=True),
        CalibrationParameters(bootstrap_seed=2**63),
        CalibrationParameters(bootstrap_confidence=1.0),
    ],
)
def test_invalid_calibration_parameters_are_rejected(
    tmp_path: Path,
    parameters: CalibrationParameters,
) -> None:
    labels = _labels(tmp_path / "labels.csv", [("run", 1, "true")])
    with pytest.raises(EvaluationEvidenceError):
        calibrate_from_manifests(
            [tmp_path / "unused.json"],
            labels,
            tmp_path / "receipt.json",
            parameters=parameters,
        )
