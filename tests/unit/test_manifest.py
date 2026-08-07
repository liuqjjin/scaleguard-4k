from __future__ import annotations

import hashlib
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
        project_root=tmp_path,
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


def _scheduler_evidence() -> dict[str, object]:
    return {
        "provider": "dashscope",
        "api_style": "openai-compatible-chat-completions",
        "region": "cn-beijing",
        "endpoint_host_sha256": hashlib.sha256(b"dashscope.aliyuncs.com").hexdigest(),
        "requested_model": "qwen3.7-flash-2026-07-15",
        "request_parameters": {
            "max_completion_tokens": 1024,
            "temperature": 0.0,
            "response_format": "json_object",
            "enable_thinking": False,
            "connect_timeout_seconds": 10.0,
            "read_timeout_seconds": 120.0,
            "max_transport_retries": 4,
        },
        "attempts": [],
    }


def _managed_depictqa_evidence(tmp_path: Path) -> dict[str, object]:
    return {
        "managed": True,
        "argv": ["python", "serve_depictqa.py"],
        "cwd": str(tmp_path),
        "host": "127.0.0.1",
        "port": 5001,
        "returncode": -15,
        "duration_seconds": 0.1,
        "stdout_path": str(tmp_path / "depictqa.stdout.log"),
        "stderr_path": str(tmp_path / "depictqa.stderr.log"),
    }


def _completed_scheduler_attempt(request_id: str) -> dict[str, object]:
    return {
        "outcome": "completed",
        "status_code": 200,
        "request_id": request_id,
        "response_model": "qwen3.7-flash-2026-07-15",
        "finish_reason": "stop",
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "total_tokens": 12,
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
    payload["config"]["fourkagent"]["command"] = ["fixture-restoration"]
    payload["config"]["fourkagent"]["depictqa_command"] = ["python", "serve_depictqa.py"]
    payload["config"]["fourkagent"]["depictqa_cwd"] = str(tmp_path)
    payload["config"]["coz"]["mode"] = "command"
    payload["config"]["coz"]["command"] = ["fixture-scale"]
    payload["config"]["controller"]["accept_unvalidated_quality_proxy"] = True
    for artifact in (
        payload["input_image"],
        payload["restored_image"],
        payload["final_image"],
        payload["steps"][0]["trusted_before"],
        payload["steps"][0]["candidate"],
    ):
        artifact["mock"] = False
    payload["restoration_metadata"] = {
        "backend": "4kagent_upstream",
        "bridge_factor": 1,
        "execution_path": {"subtasks": [], "tools": []},
        "terminal_generative_sr": False,
        "depictqa_service": _managed_depictqa_evidence(tmp_path),
        "remote_scheduler": _scheduler_evidence(),
    }
    payload["restoration_process"] = _process_evidence(tmp_path)
    payload["scale_session_process"] = (
        _process_evidence(tmp_path) if backend == "chain_of_zoom_persistent" else None
    )
    step = payload["steps"][0]
    step["process"] = _process_evidence(tmp_path) if backend == "chain_of_zoom_subprocess" else None
    step["worker_metadata"] = {
        "source_size": [step["trusted_before"]["width"], step["trusted_before"]["height"]],
        "output_size": [step["candidate"]["width"], step["candidate"]["height"]],
        "backend": backend,
        "persistent": backend == "chain_of_zoom_persistent",
        "step_index": 1,
        "seed": payload["config"]["coz"]["seed"],
        "root_sha256": step["trusted_before"]["sha256"],
        "input_sha256": step["trusted_before"]["sha256"],
        "candidate_sha256": step["candidate"]["sha256"],
        "prompts": ["restore fine image detail"],
        "requested_precision": payload["config"]["coz"]["mixed_precision"],
        "actual_precision": {"transformer": "torch.float32", "vae": "torch.float32"},
        "component_placement": {
            name: {"device": "cuda:0", "dtype": "torch.float32"}
            for name in (
                "text_encoder_1",
                "text_encoder_2",
                "text_encoder_3",
                "transformer",
                "vae",
                "vlm_first_parameter",
            )
        },
        "semantic_anchor": str(tmp_path / "semantic-anchor.png"),
        "gpu_inventory": [
            {
                "logical_index": str(index),
                "uuid": f"GPU-{index}",
                "name": "fixture-gpu",
                "memory_total_mib": "24564",
            }
            for index in range(2)
        ],
        "mock": False,
        "duration_seconds": 0.2,
        "peak_torch_allocated_mib": {"0": 1024, "1": 2048},
    }
    if backend == "chain_of_zoom_persistent":
        step["worker_metadata"]["initialization_duration_seconds"] = 1.5
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


def _persistent_runtime_manifest(
    tmp_path: Path,
    make_image: Callable[..., Path],
) -> tuple[Path, dict[str, Any]]:
    path, payload = _runtime_manifest(
        tmp_path,
        make_image,
        backend="chain_of_zoom_persistent",
        completion_level="AB_INTEGRATED",
    )
    payload["config"]["fourkagent"].update(
        {
            "mode": "upstream",
            "command": [],
            "checkout": "third_party/checkouts/4KAgent",
            "depictqa_command": ["python", "serve_depictqa.py"],
            "depictqa_cwd": "third_party/checkouts/DepictQA",
            "perception_model_path": "weights/fourkagent-qwen",
            "toolbox_root": "weights/fourkagent-toolbox",
            "hps_root": "weights/fourkagent-hps",
            "quality_model_path": "weights/fourkagent-musiq.pth",
        }
    )
    payload["config"]["coz"].update(
        {
            "mode": "persistent",
            "command": [],
            "checkout": "third_party/checkouts/Chain-of-Zoom",
            "model_path": "weights/coz-sd3",
            "qwen_model_path": "weights/coz-qwen",
            "sr_lora_path": "weights/coz-sr-lora",
            "vae_path": "weights/coz-vae",
            "vlm_lora_path": "weights/coz-vlm-lora",
        }
    )
    payload["completion_level"] = "STATIC_READY"
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


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("scheduler_extra", "unexpected fields"),
        ("scheduler_secret", "unexpected fields"),
        ("scheduler_identity", "scheduler provider"),
        ("scheduler_sequence", "does not end in a completed attempt"),
        ("terminal_sr", "terminal_generative_sr"),
        ("bridge_factor", "bridge_factor"),
        ("depictqa_schema", "DepictQA evidence"),
    ],
)
def test_manifest_replays_fourkagent_parent_metadata_contract(
    tmp_path: Path,
    make_image: Callable[..., Path],
    case: str,
    message: str,
) -> None:
    path, payload = _runtime_manifest(tmp_path, make_image)
    metadata = payload["restoration_metadata"]
    scheduler = metadata["remote_scheduler"]
    if case == "scheduler_extra":
        scheduler["raw_prompt"] = "not persisted"
    elif case == "scheduler_secret":
        scheduler["api_key"] = "not persisted"
    elif case == "scheduler_identity":
        scheduler["provider"] = "forged"
    elif case == "scheduler_sequence":
        scheduler["attempts"] = [
            _completed_scheduler_attempt("request-1"),
            {"outcome": "transport_error", "error_type": "Timeout"},
        ]
    elif case == "terminal_sr":
        metadata["terminal_generative_sr"] = True
    elif case == "bridge_factor":
        metadata["bridge_factor"] = 2
    else:
        metadata["depictqa_service"]["stopped"] = True
    _rewrite(path, payload)

    with pytest.raises(ManifestValidationError, match=message):
        validate_run_manifest(path)


def test_manifest_allows_zero_or_multiple_completed_scheduler_attempts(
    tmp_path: Path,
    make_image: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for suffix, attempts in (
        ("empty", []),
        (
            "multiple",
            [
                _completed_scheduler_attempt("request-1"),
                _completed_scheduler_attempt("request-2"),
            ],
        ),
    ):
        path, payload = _runtime_manifest(tmp_path / suffix, make_image)
        payload["restoration_metadata"]["remote_scheduler"]["attempts"] = attempts
        _rewrite(path, payload)
        _mock_runtime_dependencies(
            monkeypatch,
            payload["config"],
            payload["provenance"],
        )
        assert validate_run_manifest(path)["status"] == "succeeded"


@pytest.mark.parametrize(
    "field",
    [
        "step_index",
        "seed",
        "input_sha256",
        "candidate_sha256",
        "requested_precision",
        "mock",
        "persistent",
        "raw_prompt",
    ],
)
def test_manifest_replays_coz_parent_metadata_contract(
    tmp_path: Path,
    make_image: Callable[..., Path],
    field: str,
) -> None:
    path, payload = _runtime_manifest(tmp_path, make_image)
    metadata = payload["steps"][0]["worker_metadata"]
    replacements: dict[str, object] = {
        "step_index": 2,
        "seed": 999,
        "input_sha256": "0" * 64,
        "candidate_sha256": "0" * 64,
        "requested_precision": "fp16",
        "mock": True,
        "persistent": False,
        "raw_prompt": "not part of the worker contract",
    }
    metadata[field] = replacements[field]
    _rewrite(path, payload)

    with pytest.raises(ManifestValidationError, match="CoZ parent contract"):
        validate_run_manifest(path)


@pytest.mark.parametrize("status", ["running", "failed"])
@pytest.mark.parametrize(
    "phase",
    ["before_restoration", "after_restoration", "after_persistent_candidate"],
)
def test_validator_accepts_audited_runtime_intermediate_and_failure_states(
    tmp_path: Path,
    make_image: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    status: str,
) -> None:
    path, payload = _persistent_runtime_manifest(tmp_path, make_image)
    payload.update(
        status=status,
        completion_level="STATIC_READY",
        finished_at=None if status == "running" else payload["finished_at"],
        achieved_factor=None,
        target_reached=False,
        final_image=None,
        final_metrics={},
        events=[],
        error=(
            None
            if status == "running"
            else {"type": "SyntheticFailure", "message": f"failed {phase}"}
        ),
    )
    if phase == "before_restoration":
        payload.update(
            restored_image=None,
            restoration_metadata={},
            restoration_process=None,
            scale_session_process=None,
            steps=[],
        )
    elif phase == "after_restoration":
        payload.update(scale_session_process=None, steps=[])
    else:
        payload["scale_session_process"] = (
            None if status == "running" else {**_process_evidence(tmp_path), "returncode": 9}
        )
    _rewrite(path, payload)
    _mock_runtime_dependencies(
        monkeypatch,
        payload["config"],
        payload["provenance"],
    )

    manifest = validate_run_manifest(path)

    assert manifest["status"] == status


@pytest.mark.parametrize(
    ("process_field", "message"),
    [
        ("restoration_process", "requires 4KAgent upstream process evidence"),
        ("scale_session_process", "persistent CoZ requires successful session process evidence"),
    ],
)
def test_successful_audited_runtime_rejects_failed_process_evidence(
    tmp_path: Path,
    make_image: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
    process_field: str,
    message: str,
) -> None:
    path, payload = _persistent_runtime_manifest(tmp_path, make_image)
    payload["completion_level"] = "AB_INTEGRATED"
    payload[process_field]["returncode"] = 9
    _rewrite(path, payload)
    _mock_runtime_dependencies(
        monkeypatch,
        payload["config"],
        payload["provenance"],
    )

    with pytest.raises(ManifestValidationError, match=message):
        validate_run_manifest(path)


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


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("root_sha256", "e" * 64, "root_sha256 disagrees with the session root"),
        ("source_size", [1, 1], "source_size disagrees with the input artifact"),
    ],
)
def test_validator_binds_coz_worker_metadata_to_the_artifact_chain(
    tmp_path: Path,
    make_image: Callable[..., Path],
    field: str,
    value: object,
    message: str,
) -> None:
    path, payload = _runtime_manifest(tmp_path, make_image)
    payload["steps"][0]["worker_metadata"][field] = value
    if field == "source_size":
        payload["steps"][0]["worker_metadata"]["output_size"] = [4, 4]
    _rewrite(path, payload)

    with pytest.raises(ManifestValidationError, match=message):
        validate_run_manifest(path)


@pytest.mark.parametrize(
    "value",
    [None, True, -0.1, 10**400],
)
def test_validator_rejects_tampered_coz_step_duration(
    tmp_path: Path,
    make_image: Callable[..., Path],
    value: object,
) -> None:
    path, payload = _runtime_manifest(tmp_path, make_image)
    payload["steps"][0]["worker_metadata"]["duration_seconds"] = value
    _rewrite(path, payload)

    with pytest.raises(
        ManifestValidationError,
        match=r"worker_metadata\.duration_seconds must be (numeric|finite|non-negative)",
    ):
        validate_run_manifest(path)


@pytest.mark.parametrize(
    "value",
    [None, True, -0.1, 10**400],
)
def test_validator_rejects_tampered_persistent_initialization_duration(
    tmp_path: Path,
    make_image: Callable[..., Path],
    value: object,
) -> None:
    path, payload = _runtime_manifest(tmp_path, make_image)
    payload["steps"][0]["worker_metadata"]["initialization_duration_seconds"] = value
    _rewrite(path, payload)

    with pytest.raises(
        ManifestValidationError,
        match=(
            r"worker_metadata\.initialization_duration_seconds must be "
            r"(numeric|finite|non-negative)"
        ),
    ):
        validate_run_manifest(path)


def test_validator_rejects_initialization_duration_from_one_shot_coz(
    tmp_path: Path,
    make_image: Callable[..., Path],
) -> None:
    path, payload = _runtime_manifest(
        tmp_path,
        make_image,
        backend="chain_of_zoom_subprocess",
        completion_level="COMPONENT_REPRODUCED",
    )
    payload["steps"][0]["worker_metadata"]["initialization_duration_seconds"] = 1.5
    _rewrite(path, payload)

    with pytest.raises(ManifestValidationError, match="only valid for the first persistent"):
        validate_run_manifest(path)


def test_validator_rejects_initialization_duration_after_first_persistent_step(
    tmp_path: Path,
    make_image: Callable[..., Path],
) -> None:
    path = _manifest(tmp_path, make_image, target_factor=16)
    payload = json.loads(path.read_text(encoding="utf-8"))
    _template_path, template_payload = _runtime_manifest(tmp_path / "template", make_image)
    template_metadata = template_payload["steps"][0]["worker_metadata"]
    for index, step in enumerate(payload["steps"], start=1):
        metadata = json.loads(json.dumps(template_metadata))
        metadata.update(
            source_size=[step["trusted_before"]["width"], step["trusted_before"]["height"]],
            output_size=[step["candidate"]["width"], step["candidate"]["height"]],
            seed=payload["config"]["coz"]["seed"] + index - 1,
            step_index=index,
            root_sha256=payload["restored_image"]["sha256"],
            input_sha256=step["trusted_before"]["sha256"],
            candidate_sha256=step["candidate"]["sha256"],
        )
        metadata.pop("initialization_duration_seconds", None)
        if index == 1:
            metadata["initialization_duration_seconds"] = 1.5
        step["worker_metadata"] = metadata
    payload["steps"][1]["worker_metadata"]["initialization_duration_seconds"] = 0.5
    _rewrite(path, payload)

    with pytest.raises(ManifestValidationError, match="only valid for the first persistent"):
        validate_run_manifest(path)


@pytest.mark.parametrize(
    "peaks",
    [
        None,
        [],
        {},
        {"0": 1024},
        {"0": 1024, "2": 2048},
        {"0": True, "1": 2048},
        {"0": -1, "1": 2048},
    ],
)
def test_validator_rejects_tampered_coz_allocator_peaks(
    tmp_path: Path,
    make_image: Callable[..., Path],
    peaks: object,
) -> None:
    path, payload = _runtime_manifest(tmp_path, make_image)
    payload["steps"][0]["worker_metadata"]["peak_torch_allocated_mib"] = peaks
    _rewrite(path, payload)

    with pytest.raises(ManifestValidationError, match="peak_torch_allocated_mib"):
        validate_run_manifest(path)


def test_validator_rejects_incomplete_coz_metadata_without_allocator_peaks(
    tmp_path: Path,
    make_image: Callable[..., Path],
) -> None:
    path, payload = _runtime_manifest(tmp_path, make_image)
    payload["steps"][0]["worker_metadata"].pop("peak_torch_allocated_mib")
    _rewrite(path, payload)
    with pytest.raises(ManifestValidationError, match="unexpected or missing fields"):
        validate_run_manifest(path)


def test_validator_does_not_apply_coz_metadata_contract_to_custom_backends(
    tmp_path: Path,
    make_image: Callable[..., Path],
) -> None:
    path = _manifest(tmp_path, make_image)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["steps"][0]["worker_metadata"].update(
        {
            "duration_seconds": True,
            "initialization_duration_seconds": -1,
            "peak_torch_allocated_mib": {"unknown": False},
        }
    )
    _rewrite(path, payload)

    assert validate_run_manifest(path)["status"] == "succeeded"
