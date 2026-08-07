from __future__ import annotations

import hashlib
from typing import Any

import pytest

from scaleguard.config import FourKAgentConfig
from scaleguard.worker_contracts import (
    WorkerContractError,
    validate_coz_worker_metadata,
    validate_scheduler_evidence,
)


def _scheduler_evidence() -> dict[str, Any]:
    config = FourKAgentConfig()
    return {
        "provider": config.llm_provider,
        "api_style": "openai-compatible-chat-completions",
        "region": config.llm_region,
        "endpoint_host_sha256": hashlib.sha256(b"dashscope.aliyuncs.com").hexdigest(),
        "requested_model": config.llm_model,
        "request_parameters": {
            "max_completion_tokens": config.llm_max_completion_tokens,
            "temperature": config.llm_temperature,
            "response_format": "json_object",
            "enable_thinking": False,
            "connect_timeout_seconds": config.llm_connect_timeout_seconds,
            "read_timeout_seconds": config.llm_read_timeout_seconds,
            "max_transport_retries": config.llm_max_transport_retries,
        },
        "attempts": [
            {
                "outcome": "completed",
                "status_code": 200,
                "request_id": "request-1",
                "response_model": config.llm_model,
                "finish_reason": "stop",
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
            }
        ],
    }


def _coz_metadata() -> dict[str, Any]:
    placements = {
        name: {"device": "cuda:0", "dtype": "torch.float32"}
        for name in (
            "text_encoder_1",
            "text_encoder_2",
            "text_encoder_3",
            "transformer",
            "vae",
            "vlm_first_parameter",
        )
    }
    return {
        "source_size": [4, 3],
        "output_size": [16, 12],
        "seed": 0,
        "step_index": 1,
        "root_sha256": "a" * 64,
        "input_sha256": "a" * 64,
        "candidate_sha256": "b" * 64,
        "prompts": ["recover faithful detail"],
        "duration_seconds": 0.1,
        "initialization_duration_seconds": 1.0,
        "peak_torch_allocated_mib": {"0": 1024, "1": 2048},
        "requested_precision": "fp32",
        "actual_precision": {"transformer": "torch.float32", "vae": "torch.float32"},
        "component_placement": placements,
        "semantic_anchor": "/private/session/semantic_anchor.png",
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
        "backend": "chain_of_zoom_persistent",
        "persistent": True,
    }


def _validate_coz(
    metadata: dict[str, Any],
    *,
    initialization: str = "required",
) -> None:
    validate_coz_worker_metadata(
        metadata,
        step_index=1,
        seed=0,
        input_sha256="a" * 64,
        candidate_sha256="b" * 64,
        requested_precision="fp32",
        mock=False,
        backend="chain_of_zoom_persistent",
        persistent=True,
        visible_device_count=2,
        require_duration=True,
        initialization=initialization,
        exact_fields=True,
        expected_source_size=(4, 3),
        expected_output_size=(16, 12),
        expected_root_sha256="a" * 64,
    )


def test_scheduler_contract_rejects_inconsistent_token_totals() -> None:
    evidence = _scheduler_evidence()
    attempts = evidence["attempts"]
    assert isinstance(attempts, list)
    attempts[0]["total_tokens"] = 13

    with pytest.raises(WorkerContractError, match="token counts are inconsistent"):
        validate_scheduler_evidence(evidence, config=FourKAgentConfig())


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("identity", "does not match the configured worker contract"),
        ("parameters", "request parameters do not match configuration"),
        ("attempt_type", "attempt evidence must be a list"),
        ("retry_budget", "exceeds its transport retry budget"),
        ("terminal", "contains a terminal scheduler attempt"),
        ("unfinished", "does not end in a completed attempt"),
    ],
)
def test_scheduler_contract_rejects_invalid_evidence(case: str, message: str) -> None:
    evidence = _scheduler_evidence()
    if case == "identity":
        evidence["provider"] = "openai"
    elif case == "parameters":
        evidence["request_parameters"] = []
    elif case == "attempt_type":
        evidence["attempts"] = {}
    elif case == "retry_budget":
        evidence["attempts"] = [
            {"outcome": "transport_error", "error_type": "Timeout"}
            for _ in range(FourKAgentConfig().llm_max_transport_retries + 1)
        ]
    elif case == "terminal":
        evidence["attempts"] = [
            {
                "outcome": "terminal_http_error",
                "status_code": 400,
                "request_id": "request-1",
            }
        ]
    else:
        evidence["attempts"] = [{"outcome": "transport_error", "error_type": "Timeout"}]

    with pytest.raises(WorkerContractError, match=message):
        validate_scheduler_evidence(evidence, config=FourKAgentConfig())


def test_scheduler_contract_accepts_a_bounded_protocol_retry() -> None:
    evidence = _scheduler_evidence()
    evidence["attempts"].insert(
        0,
        {
            "outcome": "protocol_error",
            "status_code": 200,
            "request_id": "request-invalid",
        },
    )

    assert validate_scheduler_evidence(evidence, config=FourKAgentConfig()) == evidence


def test_scheduler_contract_rejects_excessive_protocol_retries() -> None:
    evidence = _scheduler_evidence()
    evidence["attempts"] = [
        {
            "outcome": "protocol_error",
            "status_code": 200,
            "request_id": f"request-invalid-{index}",
        }
        for index in range(FourKAgentConfig().llm_max_structure_retries + 1)
    ] + evidence["attempts"]

    with pytest.raises(WorkerContractError, match="exceeds its structure retry budget"):
        validate_scheduler_evidence(evidence, config=FourKAgentConfig())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_size", [0, 3], "source_size is invalid"),
        ("root_sha256", "not-a-digest", "root_sha256 is invalid"),
        ("gpu_inventory", [], "gpu_inventory is invalid"),
        (
            "gpu_inventory",
            [
                {
                    "logical_index": str(index),
                    "uuid": "GPU-duplicate",
                    "name": "fixture-gpu",
                    "memory_total_mib": "24564",
                }
                for index in range(2)
            ],
            "does not match the visible devices",
        ),
    ],
)
def test_exact_coz_contract_rejects_malformed_structural_evidence(
    field: str,
    value: object,
    message: str,
) -> None:
    metadata = _coz_metadata()
    metadata[field] = value

    with pytest.raises(WorkerContractError, match=message):
        _validate_coz(metadata)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing_field", "unexpected or missing fields"),
        ("output_size", "output_size is not exactly 4x"),
        ("prompts", "prompts must be a string list"),
        ("actual_precision", "actual_precision is invalid"),
        ("placement_fields", "component_placement is invalid"),
        ("placement_value", "component_placement is invalid"),
        ("semantic_anchor", "semantic_anchor must be a non-empty string"),
        ("inventory_value", "gpu_inventory is invalid"),
        ("duration", "duration_seconds must be numeric"),
        ("initialization", "only valid for the first persistent CoZ step"),
        ("peaks", "must map exactly the visible logical devices"),
    ],
)
def test_exact_coz_contract_rejects_invalid_nested_evidence(
    case: str,
    message: str,
) -> None:
    metadata = _coz_metadata()
    initialization = "required"
    if case == "missing_field":
        metadata.pop("prompts")
    elif case == "output_size":
        metadata["output_size"] = [15, 12]
    elif case == "prompts":
        metadata["prompts"] = [1]
    elif case == "actual_precision":
        metadata["actual_precision"] = {"transformer": "torch.float32"}
    elif case == "placement_fields":
        metadata["component_placement"].pop("vae")
    elif case == "placement_value":
        metadata["component_placement"]["vae"] = {"device": "cuda:1"}
    elif case == "semantic_anchor":
        metadata["semantic_anchor"] = ""
    elif case == "inventory_value":
        metadata["gpu_inventory"][0]["memory_total_mib"] = 24564
    elif case == "duration":
        metadata["duration_seconds"] = True
    elif case == "initialization":
        initialization = "forbidden"
    else:
        metadata["peak_torch_allocated_mib"] = {"0": 1024}

    with pytest.raises(WorkerContractError, match=message):
        _validate_coz(metadata, initialization=initialization)


@pytest.mark.parametrize(
    ("source_size", "output_size", "root_sha256", "message"),
    [
        ([1, 1], [4, 4], "a" * 64, "source_size disagrees"),
        ([4, 3], [16, 12], "c" * 64, "root_sha256 disagrees"),
    ],
)
def test_exact_coz_contract_binds_artifact_dimensions_and_session_root(
    source_size: list[int],
    output_size: list[int],
    root_sha256: str,
    message: str,
) -> None:
    metadata = _coz_metadata()
    metadata["source_size"] = source_size
    metadata["output_size"] = output_size
    metadata["root_sha256"] = root_sha256

    with pytest.raises(WorkerContractError, match=message):
        _validate_coz(metadata)
