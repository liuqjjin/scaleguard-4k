from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import scaleguard.manifest as manifest_validation
from scaleguard.backends.fake import FakeRestorationBackend, FakeScaleBackend
from scaleguard.config import ControllerConfig, MetricConfig, PipelineConfig, RuntimeConfig
from scaleguard.controller.trusted_scale import TrustedScaleController
from scaleguard.images import inspect_image
from scaleguard.manifest import ManifestValidationError, validate_run_manifest


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _generated_manifest(
    tmp_path: Path,
    make_image: Callable[..., Path],
    *,
    target_factor: int = 4,
    measurement_enabled: bool = False,
    calibration_receipt: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    config = PipelineConfig(
        runtime=RuntimeConfig(run_root=tmp_path / "runs"),
        metrics=MetricConfig(
            min_quality_gain=-10.0,
            max_scale_nrmse=10.0,
            max_scale_edge_mae=10.0,
            max_measurement_nrmse=10.0,
            measurement_enabled=measurement_enabled,
            calibration_receipt=calibration_receipt,
        ),
        controller=ControllerConfig(
            target_factor=target_factor,
            max_coz_steps=2,
            color_strategy="none",
        ),
    )
    source = make_image(tmp_path / "source.png", size=(8, 5))
    TrustedScaleController(
        config,
        FakeRestorationBackend(),
        FakeScaleBackend(),
        project_root=tmp_path,
    ).run(source, tmp_path / "output.png", run_id="contract-run")
    path = config.runtime.run_root / "contract-run" / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return path, payload


def _replace_artifact(
    record: dict[str, Any],
    replacement: Path,
    make_image: Callable[..., Path],
    *,
    size: tuple[int, int],
) -> None:
    make_image(replacement, size=size, color=(219, 17, 91))
    observed = inspect_image(
        replacement,
        mock=bool(record["mock"]),
        stage=str(record["stage"]),
    )
    record.update(
        path=str(observed.path),
        sha256=observed.sha256,
        width=observed.width,
        height=observed.height,
        media_type=observed.media_type,
    )


def _metrics(*, measurement: bool = False) -> dict[str, Any]:
    return {
        "quality_baseline": 0.2,
        "quality_candidate": 0.5,
        "quality_gain": 0.3,
        "quality_backend": "audited",
        "scale_nrmse": 0.1,
        "scale_edge_mae": 0.1,
        "measurement_nrmse": 0.1 if measurement else None,
        "measurement_model": "resize_lanczos" if measurement else None,
    }


def test_scalar_contract_helpers_reject_ambiguous_types_and_timezones() -> None:
    with pytest.raises(ManifestValidationError, match="must be an object"):
        manifest_validation._object([], "record")
    with pytest.raises(ManifestValidationError, match="non-empty string"):
        manifest_validation._text({"name": ""}, "name", "record")
    with pytest.raises(ManifestValidationError, match="must be boolean"):
        manifest_validation._boolean({"enabled": 1}, "enabled", "record")
    with pytest.raises(ManifestValidationError, match="must be an integer"):
        manifest_validation._integer({"count": True}, "count", "record")
    with pytest.raises(ManifestValidationError, match="must be finite"):
        manifest_validation._number({"score": float("inf")}, "score", "record")

    assert manifest_validation._timestamp(None, "optional", optional=True) is None
    with pytest.raises(ManifestValidationError, match="ISO-8601"):
        manifest_validation._timestamp(None, "required")
    with pytest.raises(ManifestValidationError, match="not an ISO-8601"):
        manifest_validation._timestamp("not-a-time", "invalid")
    with pytest.raises(ManifestValidationError, match="must include a UTC offset"):
        manifest_validation._timestamp("2026-07-27T08:00:00+08:00", "non-utc")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda metrics: metrics.update(measurement_nrmse=None),
            "measurement_nrmse must be numeric",
        ),
        (
            lambda metrics: metrics.update(measurement_model="resize_bilinear"),
            "measurement_model disagrees",
        ),
        (
            lambda metrics: metrics.update(quality_gain=0.4),
            "quality_gain disagrees",
        ),
    ],
)
def test_measurement_metric_record_rejects_incomplete_or_derived_claims(
    mutate: Callable[[dict[str, Any]], Any],
    message: str,
) -> None:
    metrics = _metrics(measurement=True)
    mutate(metrics)

    with pytest.raises(ManifestValidationError, match=message):
        manifest_validation._metric_record(
            metrics,
            "metrics",
            measurement_enabled=True,
            measurement_model="resize_lanczos",
        )


def test_measurement_metric_record_is_conditional_and_gate_enforced() -> None:
    disabled = _metrics()
    assert (
        manifest_validation._metric_record(
            disabled,
            "metrics",
            measurement_enabled=False,
            measurement_model="resize_lanczos",
        )
        is disabled
    )
    disabled["measurement_model"] = "resize_lanczos"
    with pytest.raises(ManifestValidationError, match="measurement consistency is disabled"):
        manifest_validation._metric_record(
            disabled,
            "metrics",
            measurement_enabled=False,
            measurement_model="resize_lanczos",
        )

    thresholds = {
        "min_quality_gain": 0.0,
        "max_scale_nrmse": 0.2,
        "max_scale_edge_mae": 0.2,
        "max_measurement_nrmse": 0.2,
    }
    metrics = _metrics(measurement=True)
    assert manifest_validation._passes_gates(metrics, thresholds, require_quality=True)
    for field in ("scale_nrmse", "scale_edge_mae", "measurement_nrmse"):
        rejected = dict(metrics)
        rejected[field] = 0.3
        assert not manifest_validation._passes_gates(
            rejected,
            thresholds,
            require_quality=True,
        )
    low_quality = dict(metrics, quality_gain=-0.1)
    assert not manifest_validation._passes_gates(
        low_quality,
        thresholds,
        require_quality=True,
    )
    assert manifest_validation._passes_gates(
        low_quality,
        thresholds,
        require_quality=False,
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"argv": []}, "argv must be a non-empty string list"),
        ({"duration_seconds": -0.1}, "duration_seconds must be non-negative"),
        ({"peak_vram_mib": {"0": -1}}, "must map device strings"),
    ],
)
def test_process_evidence_rejects_non_replayable_records(
    mutation: dict[str, Any],
    message: str,
) -> None:
    process: dict[str, Any] = {
        "argv": ["worker"],
        "cwd": "/work",
        "returncode": 0,
        "duration_seconds": 0.1,
        "stdout_path": "/logs/stdout",
        "stderr_path": "/logs/stderr",
        "peak_vram_mib": {"0": 1024},
    }
    process.update(mutation)
    with pytest.raises(ManifestValidationError, match=message):
        manifest_validation._process(process, "process")


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("acceptance", "invalid acceptance policy"),
        ("partial_experiment_id", "experiment identifiers are incomplete"),
        ("unknown_group", "experiment group is not declared"),
        ("unsafe_sample", "experiment sample id is unsafe"),
        ("group_semantics", "violates its fixed experiment semantics"),
        ("fixed_without_group", "fixed acceptance is reserved"),
        ("identity_without_b", "identity restoration is reserved"),
        ("measurement_model", "invalid measurement model"),
    ],
)
def test_manifest_rejects_config_semantic_drift(
    tmp_path: Path,
    make_image: Callable[..., Path],
    case: str,
    message: str,
) -> None:
    path, payload = _generated_manifest(tmp_path, make_image)
    if case == "acceptance":
        payload["config"]["controller"]["acceptance_policy"] = "automatic"
    elif case == "partial_experiment_id":
        payload["config"]["runtime"]["experiment_group"] = "ScaleGuard"
    elif case == "unknown_group":
        payload["config"]["runtime"].update(
            experiment_group="third-runtime",
            experiment_sample_id="sample",
        )
    elif case == "unsafe_sample":
        payload["config"]["runtime"].update(
            experiment_group="ScaleGuard",
            experiment_sample_id="../sample",
        )
    elif case == "group_semantics":
        payload["config"]["runtime"].update(
            experiment_group="ScaleGuard",
            experiment_sample_id="sample",
        )
    elif case == "fixed_without_group":
        payload["config"]["controller"]["acceptance_policy"] = "fixed"
    elif case == "identity_without_b":
        payload["config"]["fourkagent"]["mode"] = "identity"
        payload["mock"] = False
    elif case == "measurement_model":
        payload["config"]["metrics"]["measurement_model"] = "unknown"
    _write_manifest(path, payload)

    with pytest.raises(ManifestValidationError, match=message):
        validate_run_manifest(path)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("hash_format", "lowercase SHA-256"),
        ("dimensions", "dimensions must be positive"),
        ("media_type", "artifact mismatch"),
        ("missing_file", "input image does not exist"),
    ],
)
def test_manifest_rejects_invalid_artifact_identity_before_state_claims(
    tmp_path: Path,
    make_image: Callable[..., Path],
    case: str,
    message: str,
) -> None:
    path, payload = _generated_manifest(tmp_path, make_image)
    artifact = payload["input_image"]
    if case == "hash_format":
        artifact["sha256"] = "A" * 64
    elif case == "dimensions":
        artifact["width"] = 0
    elif case == "media_type":
        artifact["media_type"] = "image/jpeg"
    elif case == "missing_file":
        artifact["path"] = str(tmp_path / "missing.png")
    _write_manifest(path, payload)

    with pytest.raises(ManifestValidationError, match=message):
        validate_run_manifest(path)


def test_manifest_rejects_restoration_bridge_dimension_drift(
    tmp_path: Path,
    make_image: Callable[..., Path],
) -> None:
    path, payload = _generated_manifest(tmp_path, make_image)
    _replace_artifact(
        payload["restored_image"],
        tmp_path / "wrong-restored.png",
        make_image,
        size=(16, 10),
    )
    _write_manifest(path, payload)

    with pytest.raises(ManifestValidationError, match="declared bridge factor"):
        validate_run_manifest(path)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("terminal_then_step", "appears after a terminal decision"),
        ("chain", "does not continue the accepted artifact chain"),
        ("trusted_scale", "dimensions disagree with input_scale"),
        ("candidate_dimensions", "candidate dimensions are not exactly 4x"),
        ("decision", "not a valid Decision"),
        ("accepted_without_metrics", "accepted without candidate metrics"),
        ("metrics_without_candidate", "metrics without a candidate"),
        ("continue_rejected", "continue decision must be accepted"),
        ("rollback_accepted", "rollback decision cannot be accepted"),
        ("contrary_rejection", "rejected metrics contrary"),
        ("timestamps", "invalid timestamps"),
    ],
)
def test_manifest_rejects_scale_step_state_machine_contradictions(
    tmp_path: Path,
    make_image: Callable[..., Path],
    case: str,
    message: str,
) -> None:
    target = 16 if case in {"terminal_then_step", "chain"} else 4
    path, payload = _generated_manifest(tmp_path, make_image, target_factor=target)
    step = payload["steps"][0]
    if case == "terminal_then_step":
        step["decision"] = "stop"
    elif case == "chain":
        payload["steps"][1]["trusted_before"] = payload["restored_image"]
    elif case == "trusted_scale":
        step["input_scale"] = 2
        step["candidate_scale"] = 8
    elif case == "candidate_dimensions":
        _replace_artifact(
            step["candidate"],
            tmp_path / "wrong-candidate.png",
            make_image,
            size=(16, 10),
        )
    elif case == "decision":
        step["decision"] = "promote"
    elif case == "accepted_without_metrics":
        step["metrics"] = None
    elif case == "metrics_without_candidate":
        step["candidate"] = None
        step["accepted"] = False
        step["decision"] = "stop"
    elif case == "continue_rejected":
        step["accepted"] = False
        step["decision"] = "continue"
    elif case == "rollback_accepted":
        step["decision"] = "rollback"
    elif case == "contrary_rejection":
        step["accepted"] = False
        step["decision"] = "stop"
    elif case == "timestamps":
        step["finished_at"] = "2000-01-01T00:00:00Z"
    _write_manifest(path, payload)

    with pytest.raises(ManifestValidationError, match=message):
        validate_run_manifest(path)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("policy", "acceptance policy disagrees"),
        ("selected_scale", "selected_scale disagrees"),
        ("gate", "gate result disagrees"),
        ("policy_result", "policy result disagrees"),
        ("events", "events must be a list"),
        ("event_record", r"events\[0\] must be an object"),
        ("error_record", "manifest.error.type"),
    ],
)
def test_manifest_rejects_final_gate_and_event_claim_drift(
    tmp_path: Path,
    make_image: Callable[..., Path],
    case: str,
    message: str,
) -> None:
    path, payload = _generated_manifest(tmp_path, make_image)
    final = payload["final_metrics"]
    if case == "policy":
        final["acceptance_policy"] = "fixed"
    elif case == "selected_scale":
        final["selected_scale"] = 1
    elif case == "gate":
        final["gate_passed"] = False
    elif case == "policy_result":
        final["accepted_by_policy"] = False
    elif case == "events":
        payload["events"] = {}
    elif case == "event_record":
        payload["events"] = ["not-an-event"]
    elif case == "error_record":
        payload["error"] = {}
    _write_manifest(path, payload)

    with pytest.raises(ManifestValidationError, match=message):
        validate_run_manifest(path)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("successful_missing_artifact", "requires finished_at, restored_image, and final_image"),
        ("successful_error", "successful manifest cannot contain an error"),
        ("target_plan", "complete accepted scale plan"),
        ("rollback_target", "succeeded_with_rollback status requires target_reached=false"),
        ("failed", "failed manifest requires finished_at and error"),
        ("running", "running manifest cannot have finished_at or error"),
        ("aggregate_completion", "belongs in a separate aggregate evidence receipt"),
        ("component_mock", "mock manifest cannot exceed"),
    ],
)
def test_manifest_rejects_status_and_completion_contradictions(
    tmp_path: Path,
    make_image: Callable[..., Path],
    case: str,
    message: str,
) -> None:
    path, payload = _generated_manifest(tmp_path, make_image)
    if case == "successful_missing_artifact":
        payload["restored_image"] = None
        payload["steps"] = []
        payload["achieved_factor"] = 1
        payload["target_reached"] = False
        payload["final_metrics"]["selected_scale"] = 1
    elif case == "successful_error":
        payload["error"] = {"type": "Injected", "message": "must not coexist"}
    elif case == "target_plan":
        payload["steps"] = []
    elif case == "rollback_target":
        payload["status"] = "succeeded_with_rollback"
    elif case == "failed":
        payload["status"] = "failed"
        payload["finished_at"] = None
    elif case == "running":
        payload["status"] = "running"
    elif case == "aggregate_completion":
        payload["completion_level"] = "RESEARCH_EVALUATED"
        payload["mock"] = False
        payload["config"]["fourkagent"]["mode"] = "command"
        payload["config"]["coz"]["mode"] = "command"
        for artifact in (
            payload["input_image"],
            payload["restored_image"],
            payload["final_image"],
            payload["steps"][0]["trusted_before"],
            payload["steps"][0]["candidate"],
        ):
            artifact["mock"] = False
    elif case == "component_mock":
        payload["completion_level"] = "COMPONENT_REPRODUCED"
    _write_manifest(path, payload)

    with pytest.raises(ManifestValidationError, match=message):
        validate_run_manifest(path)


def test_calibration_receipt_bytes_are_disclosed_in_run_provenance(
    tmp_path: Path,
    make_image: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = tmp_path / "calibration.json"
    receipt_payload = b'{"schema_version":"fixture"}\n'
    receipt.write_bytes(receipt_payload)
    monkeypatch.setattr(
        "scaleguard.controller.trusted_scale.verify_calibration_document",
        lambda _document, _config: (True, []),
    )

    path, payload = _generated_manifest(
        tmp_path,
        make_image,
        calibration_receipt=Path("calibration.json"),
    )

    assert payload["provenance"]["quality_thresholds_calibrated"] is True
    assert payload["provenance"]["quality_calibration_reasons"] == []
    assert payload["provenance"]["quality_calibration_receipt"] == str(receipt.resolve())
    assert payload["provenance"]["quality_calibration_receipt_size_bytes"] == len(receipt_payload)
    assert (
        payload["provenance"]["quality_calibration_receipt_sha256"]
        == hashlib.sha256(receipt_payload).hexdigest()
    )
    assert validate_run_manifest(path)["run_id"] == "contract-run"


def test_unreadable_calibration_receipt_is_explicitly_untrusted(
    tmp_path: Path,
    make_image: Callable[..., Path],
) -> None:
    missing = Path("missing-calibration.json")
    _path, payload = _generated_manifest(
        tmp_path,
        make_image,
        calibration_receipt=missing,
    )

    assert payload["provenance"]["quality_thresholds_calibrated"] is False
    assert payload["provenance"]["quality_calibration_receipt"] == str(
        (tmp_path / missing).resolve()
    )
    assert payload["provenance"]["quality_calibration_receipt_size_bytes"] is None
    assert payload["provenance"]["quality_calibration_receipt_sha256"] is None
    assert payload["provenance"]["quality_calibration_reasons"] == [
        "calibration_receipt_unreadable:RuntimePreflightError"
    ]
