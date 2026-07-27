from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scaleguard.backends.fake import FakeRestorationBackend, FakeScaleBackend
from scaleguard.config import ControllerConfig, MetricConfig, PipelineConfig, RuntimeConfig
from scaleguard.controller.trusted_scale import TrustedScaleController
from scaleguard.manifest import ManifestValidationError, validate_run_manifest
from scaleguard.provenance import RuntimePreflightError


def _manifest(
    tmp_path: Path,
    make_image: Callable[..., Path],
    *,
    target_factor: int = 4,
    measurement_enabled: bool = False,
) -> Path:
    config = PipelineConfig(
        runtime=RuntimeConfig(run_root=tmp_path / "runs"),
        metrics=MetricConfig(
            min_quality_gain=-10.0,
            max_scale_nrmse=10.0,
            max_scale_edge_mae=10.0,
            measurement_enabled=measurement_enabled,
        ),
        controller=ControllerConfig(
            target_factor=target_factor,
            color_strategy="none",
        ),
    )
    source = make_image(tmp_path / "source.png", size=(8, 5))
    TrustedScaleController(
        config,
        FakeRestorationBackend(),
        FakeScaleBackend(),
    ).run(source, tmp_path / "output.png", run_id="validated-run")
    return config.runtime.run_root / "validated-run" / "manifest.json"


def _rewrite(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _process_evidence(tmp_path: Path) -> dict[str, object]:
    return {
        "argv": ["audited-worker"],
        "cwd": str(tmp_path),
        "returncode": 0,
        "duration_seconds": 0.1,
        "stdout_path": str(tmp_path / "stdout.log"),
        "stderr_path": str(tmp_path / "stderr.log"),
        "peak_vram_mib": {"0": 1024},
    }


def _runtime_manifest(
    tmp_path: Path,
    make_image: Callable[..., Path],
    *,
    backend: str = "chain_of_zoom_persistent",
    completion_level: str = "AB_INTEGRATED",
) -> tuple[Path, dict[str, Any]]:
    path = _manifest(tmp_path, make_image)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["mock"] = False
    payload["completion_level"] = completion_level
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
    payload["restoration_metadata"].update(
        {
            "backend": "4kagent_upstream",
            "mock": False,
        }
    )
    payload["restoration_process"] = _process_evidence(tmp_path)
    payload["scale_session_process"] = (
        _process_evidence(tmp_path) if backend == "chain_of_zoom_persistent" else None
    )
    step = payload["steps"][0]
    step["process"] = _process_evidence(tmp_path) if backend == "chain_of_zoom_subprocess" else None
    step["worker_metadata"] = {
        "backend": backend,
        "candidate_sha256": step["candidate"]["sha256"],
    }
    payload["provenance"].update(
        {
            "runtime_evidence_verified": True,
            "runtime_profile_bound": True,
            "runtime_preflight_receipt": str(tmp_path / "runtime-preflight.json"),
            "runtime_preflight_sha256": "a" * 64,
            "bootstrap_receipt_sha256": "b" * 64,
            "runtime_environment_receipt_sha256": {
                "scaleguard": "1" * 64,
                "4kagent": "2" * 64,
                "depictqa": "3" * 64,
                "coz": "4" * 64,
            },
            "materialization_receipt_sha256": "5" * 64,
            "materialization_marker_sha256": "6" * 64,
            "source_weights_receipt_sha256": "7" * 64,
            "weights_root": str(tmp_path / "weights"),
            "project_commit": "8" * 40,
            "project_root": str(tmp_path),
            "runtime_config_path": str(tmp_path / "runtime.yaml"),
            "runtime_config_sha256": "c" * 64,
            "runtime_stage_started_at": "2026-07-27T00:00:00+00:00",
            "runtime_execution_binding": {"schema_version": 1},
            "runtime_execution_binding_sha256": "d" * 64,
        }
    )
    _rewrite(path, payload)
    return path, payload


def _mock_runtime_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    config: dict[str, Any],
    provenance: dict[str, Any],
    *,
    digest: str = "a" * 64,
    config_digest: str = "c" * 64,
    current_config_digest: str | None = None,
) -> None:
    fields = {
        "runtime_evidence_verified",
        "runtime_profile_bound",
        "runtime_preflight_receipt",
        "runtime_preflight_sha256",
        "bootstrap_receipt_sha256",
        "runtime_environment_receipt_sha256",
        "materialization_receipt_sha256",
        "materialization_marker_sha256",
        "source_weights_receipt_sha256",
        "weights_root",
        "project_commit",
        "project_root",
        "runtime_config_path",
        "runtime_config_sha256",
        "runtime_stage_started_at",
        "runtime_execution_binding",
        "runtime_execution_binding_sha256",
    }
    validated = {field: provenance[field] for field in fields}
    validated["runtime_preflight_sha256"] = digest
    validated["runtime_config_sha256"] = config_digest
    monkeypatch.setattr(
        "scaleguard.provenance.validate_runtime_preflight",
        lambda *_args, **_kwargs: validated,
    )
    monkeypatch.setattr(
        "scaleguard.provenance.load_regular_file_snapshot",
        lambda *_args, **_kwargs: (
            b"fixture",
            config_digest if current_config_digest is None else current_config_digest,
        ),
    )
    monkeypatch.setattr(
        "scaleguard.config.parse_config",
        lambda _document, **_kwargs: SimpleNamespace(as_dict=lambda: config),
    )
    monkeypatch.setattr(
        "scaleguard.provenance.bind_runtime_config",
        lambda parsed, **_kwargs: parsed,
    )


def test_validator_reloads_every_artifact_and_accepts_a_consistent_run(
    tmp_path: Path,
    make_image: Callable[..., Path],
) -> None:
    path = _manifest(tmp_path, make_image)

    manifest = validate_run_manifest(path)

    assert manifest["run_id"] == "validated-run"
    assert manifest["achieved_factor"] == 4
    assert manifest["target_reached"] is True


def test_validator_rejects_continue_at_the_final_planned_scale(
    tmp_path: Path,
    make_image: Callable[..., Path],
) -> None:
    path = _manifest(tmp_path, make_image)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["steps"][0]["decision"] = "continue"
    _rewrite(path, payload)

    with pytest.raises(ManifestValidationError, match="final planned scale decision"):
        validate_run_manifest(path)


def test_validator_binds_measurement_records_to_the_configured_forward_model(
    tmp_path: Path,
    make_image: Callable[..., Path],
) -> None:
    path = _manifest(tmp_path, make_image, measurement_enabled=True)
    validate_run_manifest(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["steps"][0]["metrics"]["measurement_nrmse"] = None
    payload["steps"][0]["metrics"]["measurement_model"] = None
    _rewrite(path, payload)
    with pytest.raises(ManifestValidationError, match="measurement_nrmse must be numeric"):
        validate_run_manifest(path)

    path = _manifest(tmp_path / "wrong-model", make_image, measurement_enabled=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["final_metrics"]["metrics"]["measurement_model"] = "forged"
    _rewrite(path, payload)
    with pytest.raises(ManifestValidationError, match="disagrees with config"):
        validate_run_manifest(path)


def test_validator_rejects_measurement_evidence_when_disabled(
    tmp_path: Path,
    make_image: Callable[..., Path],
) -> None:
    path = _manifest(tmp_path, make_image)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["final_metrics"]["metrics"]["measurement_nrmse"] = 0.01
    payload["final_metrics"]["metrics"]["measurement_model"] = "resize"
    _rewrite(path, payload)

    with pytest.raises(ManifestValidationError, match="measurement consistency is disabled"):
        validate_run_manifest(path)


def test_validator_accepts_rfc3339_z_timestamps_at_every_manifest_level(
    tmp_path: Path,
    make_image: Callable[..., Path],
) -> None:
    path = _manifest(tmp_path, make_image)
    payload = json.loads(path.read_text(encoding="utf-8"))

    def as_z(value: str) -> str:
        if value.endswith("Z"):
            return value
        assert value.endswith("+00:00")
        return value[:-6] + "Z"

    payload["started_at"] = as_z(payload["started_at"])
    payload["finished_at"] = as_z(payload["finished_at"])
    for step in payload["steps"]:
        step["started_at"] = as_z(step["started_at"])
        step["finished_at"] = as_z(step["finished_at"])
    for event in payload["events"]:
        event["at"] = as_z(event["at"])
    _rewrite(path, payload)

    manifest = validate_run_manifest(path)

    assert manifest["started_at"].endswith("Z")
    assert manifest["finished_at"].endswith("Z")
    assert all(step["started_at"].endswith("Z") for step in manifest["steps"])
    assert all(event["at"].endswith("Z") for event in manifest["events"])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sha256", "0" * 64, "artifact mismatch"),
        ("width", 999, "artifact mismatch"),
        ("mock", False, "disagrees with run mock"),
    ],
)
def test_validator_rejects_tampered_final_artifact_evidence(
    tmp_path: Path,
    make_image: Callable[..., Path],
    field: str,
    value: object,
    message: str,
) -> None:
    path = _manifest(tmp_path, make_image)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["final_image"][field] = value
    _rewrite(path, payload)

    with pytest.raises(ManifestValidationError, match=message):
        validate_run_manifest(path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"target_reached": False}, "target_reached disagrees"),
        ({"achieved_factor": 2}, "target_reached disagrees"),
        ({"status": "unknown"}, "not a valid RunStatus"),
        ({"completion_level": "AB_INTEGRATED"}, "mock manifest cannot exceed"),
    ],
)
def test_validator_rejects_state_and_completion_contradictions(
    tmp_path: Path,
    make_image: Callable[..., Path],
    mutation: dict[str, object],
    message: str,
) -> None:
    path = _manifest(tmp_path, make_image)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(mutation)
    _rewrite(path, payload)

    with pytest.raises(ManifestValidationError, match=message):
        validate_run_manifest(path)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing_field", "missing fields"),
        ("schema", "schema_version"),
        ("unsafe_run_id", "safe path component"),
        ("time_order", "precedes started_at"),
        ("backend_mock", "mock disagrees"),
        ("unsupported_factor", "requested_factor is unsupported"),
        ("controller_factor", "disagrees with config.controller"),
        ("achieved_factor", "achieved_factor is invalid"),
        ("steps_type", "steps must be a list"),
        ("too_many_steps", "exceeds the 1-step"),
        ("step_index", "index must equal 1"),
        ("scale_transition", "invalid 4x scale transition"),
    ],
)
def test_validator_rejects_malformed_core_contracts(
    tmp_path: Path,
    make_image: Callable[..., Path],
    case: str,
    message: str,
) -> None:
    path = _manifest(tmp_path, make_image)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if case == "missing_field":
        del payload["events"]
    elif case == "schema":
        payload["schema_version"] = 2
    elif case == "unsafe_run_id":
        payload["run_id"] = "../escaped"
    elif case == "time_order":
        payload["finished_at"] = "2000-01-01T00:00:00Z"
    elif case == "backend_mock":
        payload["config"]["fourkagent"]["mode"] = "command"
        payload["config"]["coz"]["mode"] = "command"
    elif case == "unsupported_factor":
        payload["requested_factor"] = 3
        payload["config"]["controller"]["target_factor"] = 3
    elif case == "controller_factor":
        payload["config"]["controller"]["target_factor"] = 8
    elif case == "achieved_factor":
        payload["achieved_factor"] = 3
    elif case == "steps_type":
        payload["steps"] = {}
    elif case == "too_many_steps":
        payload["steps"].append(dict(payload["steps"][0]))
    elif case == "step_index":
        payload["steps"][0]["index"] = 2
    elif case == "scale_transition":
        payload["steps"][0]["candidate_scale"] = 5
    else:
        raise AssertionError(f"unknown mutation case: {case}")
    _rewrite(path, payload)

    with pytest.raises(ManifestValidationError, match=message):
        validate_run_manifest(path)


def test_validator_rejects_nonstandard_json_numbers(
    tmp_path: Path,
    make_image: Callable[..., Path],
) -> None:
    path = _manifest(tmp_path, make_image)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["final_metrics"]["metrics"]["quality_gain"] = float("nan")
    _rewrite(path, payload)

    with pytest.raises(ManifestValidationError, match="non-standard JSON constant NaN"):
        validate_run_manifest(path)


def test_validator_rejects_duplicate_keys_before_state_validation(
    tmp_path: Path,
    make_image: Callable[..., Path],
) -> None:
    path = _manifest(tmp_path, make_image)
    original = path.read_text(encoding="utf-8").lstrip()
    path.write_text(
        '{"run_id":"forged",' + original[1:],
        encoding="utf-8",
    )

    with pytest.raises(ManifestValidationError, match="duplicate JSON object key 'run_id'"):
        validate_run_manifest(path)


def test_validator_rejects_forged_gate_pass_for_out_of_bounds_metrics(
    tmp_path: Path,
    make_image: Callable[..., Path],
) -> None:
    path = _manifest(tmp_path, make_image)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["steps"][0]["metrics"]["scale_nrmse"] = 999.0
    payload["final_metrics"]["metrics"]["scale_nrmse"] = 999.0
    _rewrite(path, payload)

    with pytest.raises(ManifestValidationError, match=r"accepted metrics.*fail configured gates"):
        validate_run_manifest(path)


def test_validator_detects_post_run_artifact_mutation(
    tmp_path: Path,
    make_image: Callable[..., Path],
) -> None:
    path = _manifest(tmp_path, make_image)
    payload = json.loads(path.read_text(encoding="utf-8"))
    final_path = Path(payload["final_image"]["path"])
    make_image(final_path, size=(32, 20), color=(255, 0, 0))

    with pytest.raises(ManifestValidationError, match="sha256"):
        validate_run_manifest(path)


@pytest.mark.parametrize(
    ("backend", "completion_level"),
    [
        ("chain_of_zoom_subprocess", "COMPONENT_REPRODUCED"),
        ("chain_of_zoom_persistent", "AB_INTEGRATED"),
    ],
)
def test_validator_accepts_verified_runtime_completion_evidence(
    tmp_path: Path,
    make_image: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    completion_level: str,
) -> None:
    path, payload = _runtime_manifest(
        tmp_path,
        make_image,
        backend=backend,
        completion_level=completion_level,
    )
    _mock_runtime_dependencies(
        monkeypatch,
        payload["config"],
        payload["provenance"],
    )

    manifest = validate_run_manifest(path)

    assert manifest["completion_level"] == completion_level


def test_validator_rejects_unverified_runtime_completion(
    tmp_path: Path,
    make_image: Callable[..., Path],
) -> None:
    path, payload = _runtime_manifest(tmp_path, make_image)
    payload["provenance"]["runtime_evidence_verified"] = False
    _rewrite(path, payload)

    with pytest.raises(
        ManifestValidationError,
        match="requires verified and profile-bound runtime provenance",
    ):
        validate_run_manifest(path)


@pytest.mark.parametrize(
    "field",
    [
        "runtime_preflight_receipt",
        "project_root",
        "runtime_config_path",
        "runtime_config_sha256",
    ],
)
def test_validator_rejects_incomplete_runtime_provenance(
    tmp_path: Path,
    make_image: Callable[..., Path],
    field: str,
) -> None:
    path, payload = _runtime_manifest(tmp_path, make_image)
    payload["provenance"][field] = None
    _rewrite(path, payload)

    with pytest.raises(ManifestValidationError, match="runtime preflight provenance is incomplete"):
        validate_run_manifest(path)


def test_validator_wraps_runtime_preflight_validation_errors(
    tmp_path: Path,
    make_image: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _payload = _runtime_manifest(tmp_path, make_image)

    def fail_preflight(*_args: object, **_kwargs: object) -> dict[str, Any]:
        raise RuntimePreflightError("receipt chain is invalid")

    monkeypatch.setattr(
        "scaleguard.provenance.validate_runtime_preflight",
        fail_preflight,
    )

    with pytest.raises(ManifestValidationError, match="runtime preflight is invalid"):
        validate_run_manifest(path)


def test_validator_rejects_runtime_preflight_digest_mismatch(
    tmp_path: Path,
    make_image: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, payload = _runtime_manifest(tmp_path, make_image)
    _mock_runtime_dependencies(
        monkeypatch,
        payload["config"],
        payload["provenance"],
        digest="b" * 64,
    )

    with pytest.raises(ManifestValidationError, match="runtime provenance disagrees"):
        validate_run_manifest(path)


def test_validator_rejects_runtime_config_digest_mismatch(
    tmp_path: Path,
    make_image: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, payload = _runtime_manifest(tmp_path, make_image)
    _mock_runtime_dependencies(
        monkeypatch,
        payload["config"],
        payload["provenance"],
        current_config_digest="d" * 64,
    )

    with pytest.raises(ManifestValidationError, match="config digest disagrees"):
        validate_run_manifest(path)


def test_validator_rejects_runtime_config_mismatch(
    tmp_path: Path,
    make_image: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, payload = _runtime_manifest(tmp_path, make_image)
    _mock_runtime_dependencies(
        monkeypatch,
        {"controller": {"target_factor": 8}},
        payload["provenance"],
    )

    with pytest.raises(ManifestValidationError, match="differs from its preflighted config"):
        validate_run_manifest(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("backend", "fake_4kagent"),
        ("process", None),
        ("returncode", 1),
    ],
)
def test_validator_rejects_missing_4kagent_process_evidence(
    tmp_path: Path,
    make_image: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    path, payload = _runtime_manifest(tmp_path, make_image)
    if field == "backend":
        payload["restoration_metadata"]["backend"] = value
    elif field == "process":
        payload["restoration_process"] = value
    else:
        payload["restoration_process"]["returncode"] = value
    _rewrite(path, payload)
    _mock_runtime_dependencies(
        monkeypatch,
        payload["config"],
        payload["provenance"],
    )

    with pytest.raises(ManifestValidationError, match="requires 4KAgent upstream process evidence"):
        validate_run_manifest(path)


def test_validator_rejects_unknown_coz_worker_evidence(
    tmp_path: Path,
    make_image: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, payload = _runtime_manifest(tmp_path, make_image)
    payload["steps"][0]["worker_metadata"]["backend"] = "third_runtime"
    _rewrite(path, payload)
    _mock_runtime_dependencies(
        monkeypatch,
        payload["config"],
        payload["provenance"],
    )

    with pytest.raises(ManifestValidationError, match="requires Chain-of-Zoom worker evidence"):
        validate_run_manifest(path)


@pytest.mark.parametrize(
    ("backend", "process_field", "message"),
    [
        (
            "chain_of_zoom_subprocess",
            "process",
            "one-shot CoZ steps require successful process evidence",
        ),
        (
            "chain_of_zoom_persistent",
            "scale_session_process",
            "persistent CoZ requires successful session process evidence",
        ),
    ],
)
def test_validator_rejects_missing_coz_process_evidence(
    tmp_path: Path,
    make_image: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    process_field: str,
    message: str,
) -> None:
    path, payload = _runtime_manifest(tmp_path, make_image, backend=backend)
    if process_field == "process":
        payload["steps"][0]["process"] = None
    else:
        payload[process_field] = None
    _rewrite(path, payload)
    _mock_runtime_dependencies(
        monkeypatch,
        payload["config"],
        payload["provenance"],
    )

    with pytest.raises(ManifestValidationError, match=message):
        validate_run_manifest(path)


def test_validator_rejects_coz_candidate_hash_mismatch(
    tmp_path: Path,
    make_image: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, payload = _runtime_manifest(tmp_path, make_image)
    payload["steps"][0]["worker_metadata"]["candidate_sha256"] = "0" * 64
    _rewrite(path, payload)
    _mock_runtime_dependencies(
        monkeypatch,
        payload["config"],
        payload["provenance"],
    )

    with pytest.raises(ManifestValidationError, match="candidate hash disagrees"):
        validate_run_manifest(path)
