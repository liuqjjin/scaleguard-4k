"""Pure validators for evidence returned across worker process boundaries."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from typing import Any, TypeGuard
from urllib.parse import urlsplit

from scaleguard.config import FourKAgentConfig
from scaleguard.runtime.process import redact_argv


class WorkerContractError(ValueError):
    """Raised when persisted worker evidence violates its parent-side contract."""


def _non_negative_int(value: object) -> TypeGuard[int]:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _non_negative_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkerContractError(f"{field} must be numeric")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise WorkerContractError(f"{field} must be finite") from error
    if not math.isfinite(result):
        raise WorkerContractError(f"{field} must be finite")
    if result < 0:
        raise WorkerContractError(f"{field} must be non-negative")
    return result


def _non_empty_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise WorkerContractError(f"{field} must be a non-empty string")
    return value


def _exact_scalar(observed: object, expected: object, *, field: str) -> None:
    if type(observed) is not type(expected) or observed != expected:
        raise WorkerContractError(
            f"{field} does not match the configured worker contract: "
            f"expected {expected!r}, observed {observed!r}"
        )


def _validate_scheduler_attempt(
    attempt: object,
    *,
    requested_model: str,
) -> str:
    if not isinstance(attempt, Mapping):
        raise WorkerContractError("4KAgent remote scheduler attempt must be an object")
    outcome = attempt.get("outcome")
    request_id = attempt.get("request_id")
    if request_id is not None and (not isinstance(request_id, str) or not request_id):
        raise WorkerContractError("4KAgent remote scheduler request id must be non-empty or null")

    if outcome == "transport_error":
        if set(attempt) != {"outcome", "error_type"}:
            raise WorkerContractError("4KAgent scheduler transport evidence has unexpected fields")
        _non_empty_text(
            attempt.get("error_type"),
            field="4KAgent scheduler transport error type",
        )
        return outcome

    status_code = attempt.get("status_code")
    if isinstance(status_code, bool) or not isinstance(status_code, int):
        raise WorkerContractError("4KAgent scheduler status code must be an integer")
    if outcome == "retryable_http_error":
        if set(attempt) != {"outcome", "status_code", "request_id"}:
            raise WorkerContractError("4KAgent scheduler retry evidence has unexpected fields")
        if status_code != 429 and not 500 <= status_code <= 599:
            raise WorkerContractError("4KAgent scheduler retry evidence has a terminal status")
        return outcome
    if outcome == "terminal_http_error":
        if set(attempt) != {"outcome", "status_code", "request_id"}:
            raise WorkerContractError("4KAgent scheduler terminal evidence has unexpected fields")
        if 200 <= status_code <= 299 or status_code == 429 or 500 <= status_code <= 599:
            raise WorkerContractError("4KAgent scheduler terminal evidence has a retryable status")
        return outcome
    if outcome == "protocol_error":
        if set(attempt) != {"outcome", "status_code", "request_id"}:
            raise WorkerContractError("4KAgent scheduler protocol evidence has unexpected fields")
        if not 200 <= status_code <= 299:
            raise WorkerContractError(
                "4KAgent scheduler protocol evidence has a non-success status"
            )
        return outcome
    if outcome != "completed":
        raise WorkerContractError("4KAgent remote scheduler attempt has an unknown outcome")

    expected_fields = {
        "outcome",
        "status_code",
        "request_id",
        "response_model",
        "finish_reason",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    }
    if set(attempt) != expected_fields:
        raise WorkerContractError("4KAgent completed scheduler evidence has unexpected fields")
    if not 200 <= status_code <= 299:
        raise WorkerContractError("4KAgent completed scheduler evidence has a non-success status")
    if request_id is None:
        raise WorkerContractError("4KAgent completed scheduler evidence is missing its request id")
    if attempt.get("response_model") != requested_model or attempt.get("finish_reason") != "stop":
        raise WorkerContractError(
            "4KAgent completed scheduler evidence violates the model contract"
        )
    prompt_tokens = attempt.get("prompt_tokens")
    completion_tokens = attempt.get("completion_tokens")
    total_tokens = attempt.get("total_tokens")
    if (
        not _non_negative_int(prompt_tokens)
        or not _non_negative_int(completion_tokens)
        or not _non_negative_int(total_tokens)
    ):
        raise WorkerContractError("4KAgent completed scheduler token counts are invalid")
    if total_tokens != prompt_tokens + completion_tokens:
        raise WorkerContractError("4KAgent completed scheduler token counts are inconsistent")
    return outcome


def validate_scheduler_evidence(
    evidence: object,
    *,
    config: FourKAgentConfig,
) -> dict[str, Any]:
    """Validate redacted scheduler identity, request settings, and attempt order.

    An empty attempt list is valid when the upstream agent does not need a scheduler
    call. Multiple completed attempts are also valid: structure retries and later
    rescheduling calls each produce their own completed transport attempt.
    """

    if not isinstance(evidence, Mapping):
        raise WorkerContractError("4KAgent evidence is missing remote scheduler provenance")
    endpoint_host = urlsplit(config.llm_base_url).hostname
    if endpoint_host is None:
        raise WorkerContractError("configured scheduler endpoint has no host")
    expected_identity = {
        "provider": config.llm_provider,
        "api_style": "openai-compatible-chat-completions",
        "region": config.llm_region,
        "endpoint_host_sha256": hashlib.sha256(endpoint_host.encode("utf-8")).hexdigest(),
        "requested_model": config.llm_model,
    }
    expected_fields = set(expected_identity) | {"request_parameters", "attempts"}
    if set(evidence) != expected_fields:
        raise WorkerContractError("4KAgent remote scheduler provenance has unexpected fields")
    for key, identity_expected in expected_identity.items():
        _exact_scalar(
            evidence.get(key),
            identity_expected,
            field=f"4KAgent remote scheduler {key}",
        )

    parameters = evidence.get("request_parameters")
    expected_parameters: dict[str, object] = {
        "max_completion_tokens": config.llm_max_completion_tokens,
        "temperature": config.llm_temperature,
        "response_format": "json_object",
        "enable_thinking": False if config.llm_provider == "dashscope" else None,
        "connect_timeout_seconds": config.llm_connect_timeout_seconds,
        "read_timeout_seconds": config.llm_read_timeout_seconds,
        "max_transport_retries": config.llm_max_transport_retries,
    }
    if not isinstance(parameters, Mapping) or set(parameters) != set(expected_parameters):
        raise WorkerContractError("4KAgent scheduler request parameters do not match configuration")
    for key, parameter_expected in expected_parameters.items():
        try:
            _exact_scalar(
                parameters.get(key),
                parameter_expected,
                field=f"4KAgent scheduler request parameter {key}",
            )
        except WorkerContractError as error:
            raise WorkerContractError(
                "4KAgent scheduler request parameters do not match configuration"
            ) from error

    attempts = evidence.get("attempts")
    if not isinstance(attempts, list):
        raise WorkerContractError("4KAgent scheduler attempt evidence must be a list")
    consecutive_transport_retries = 0
    consecutive_structure_retries = 0
    last_outcome: str | None = None
    for attempt in attempts:
        outcome = _validate_scheduler_attempt(
            attempt,
            requested_model=config.llm_model,
        )
        if outcome == "completed":
            consecutive_transport_retries = 0
            consecutive_structure_retries = 0
        elif outcome in {"transport_error", "retryable_http_error"}:
            consecutive_transport_retries += 1
            if consecutive_transport_retries > config.llm_max_transport_retries:
                raise WorkerContractError(
                    "4KAgent scheduler evidence exceeds its transport retry budget"
                )
        elif outcome == "protocol_error":
            consecutive_transport_retries = 0
            consecutive_structure_retries += 1
            if consecutive_structure_retries > config.llm_max_structure_retries:
                raise WorkerContractError(
                    "4KAgent scheduler evidence exceeds its structure retry budget"
                )
        else:
            raise WorkerContractError(
                "successful 4KAgent evidence contains a terminal scheduler attempt"
            )
        last_outcome = outcome
    if attempts and last_outcome != "completed":
        raise WorkerContractError("4KAgent scheduler evidence does not end in a completed attempt")
    return dict(evidence)


def _validate_execution_path(value: object, *, bridge_factor: int) -> None:
    if not isinstance(value, Mapping) or set(value) != {"subtasks", "tools"}:
        raise WorkerContractError("4KAgent execution_path must contain exactly subtasks and tools")
    subtasks = value.get("subtasks")
    tools = value.get("tools")
    if not isinstance(subtasks, list) or not all(
        isinstance(item, str) and item for item in subtasks
    ):
        raise WorkerContractError("4KAgent execution_path.subtasks must be a string list")
    if not isinstance(tools, list) or not all(isinstance(item, str) and item for item in tools):
        raise WorkerContractError("4KAgent execution_path.tools must be a string list")
    forbidden = {
        "super-resolution",
        "super-resolution_16x",
        "face restoration",
        "old_photo_restoration",
    }
    if forbidden.intersection(subtasks):
        raise WorkerContractError("4KAgent execution_path contains a forbidden terminal task")
    bridge_count = subtasks.count("super-resolution_2x")
    maximum_bridges = 1 if bridge_factor == 2 else 0
    if bridge_count > maximum_bridges:
        raise WorkerContractError("4KAgent execution_path has an invalid 2x bridge count")


def _validate_depictqa_evidence(
    value: object,
    *,
    config: FourKAgentConfig,
) -> None:
    if not isinstance(value, Mapping):
        raise WorkerContractError("4KAgent DepictQA evidence must be an object")
    if config.depictqa_command:
        expected_fields = {
            "managed",
            "argv",
            "cwd",
            "host",
            "port",
            "returncode",
            "duration_seconds",
            "stdout_path",
            "stderr_path",
        }
        if set(value) != expected_fields:
            raise WorkerContractError("managed DepictQA evidence has unexpected or missing fields")
        if value.get("managed") is not True:
            raise WorkerContractError("managed DepictQA evidence must declare managed=true")
        argv = value.get("argv")
        if (
            not isinstance(argv, list)
            or not argv
            or not all(isinstance(token, str) and token for token in argv)
        ):
            raise WorkerContractError("managed DepictQA argv must be a non-empty string list")
        if list(redact_argv(argv)) != argv:
            raise WorkerContractError("managed DepictQA argv contains unredacted secret material")
        _non_empty_text(value.get("cwd"), field="managed DepictQA cwd")
        returncode = value.get("returncode")
        if isinstance(returncode, bool) or not isinstance(returncode, int):
            raise WorkerContractError("managed DepictQA returncode must be an integer")
        _non_negative_number(
            value.get("duration_seconds"),
            field="managed DepictQA duration_seconds",
        )
        _non_empty_text(
            value.get("stdout_path"),
            field="managed DepictQA stdout_path",
        )
        _non_empty_text(
            value.get("stderr_path"),
            field="managed DepictQA stderr_path",
        )
    else:
        if set(value) != {"managed", "host", "port"}:
            raise WorkerContractError("external DepictQA evidence has unexpected or missing fields")
        if value.get("managed") is not False:
            raise WorkerContractError("external DepictQA evidence must declare managed=false")
    _exact_scalar(
        value.get("host"),
        config.depictqa_host,
        field="DepictQA host",
    )
    _exact_scalar(
        value.get("port"),
        config.depictqa_port,
        field="DepictQA port",
    )


def validate_fourkagent_restoration_metadata(
    metadata: object,
    *,
    config: FourKAgentConfig,
    bridge_factor: int,
) -> dict[str, Any]:
    """Replay the metadata contract applied by the 4KAgent parent adapter."""

    if not isinstance(metadata, Mapping):
        raise WorkerContractError("4KAgent restoration metadata must be an object")
    expected_fields = {
        "backend",
        "bridge_factor",
        "execution_path",
        "terminal_generative_sr",
        "depictqa_service",
        "remote_scheduler",
    }
    if set(metadata) != expected_fields:
        raise WorkerContractError("4KAgent restoration metadata has unexpected or missing fields")
    _exact_scalar(
        metadata.get("backend"),
        "4kagent_upstream",
        field="4KAgent restoration backend",
    )
    _exact_scalar(
        metadata.get("bridge_factor"),
        bridge_factor,
        field="4KAgent restoration bridge_factor",
    )
    if metadata.get("terminal_generative_sr") is not False:
        raise WorkerContractError("4KAgent terminal_generative_sr must remain false")
    _validate_execution_path(metadata.get("execution_path"), bridge_factor=bridge_factor)
    _validate_depictqa_evidence(metadata.get("depictqa_service"), config=config)
    validate_scheduler_evidence(metadata.get("remote_scheduler"), config=config)
    return dict(metadata)


def validate_coz_worker_metadata(
    metadata: object,
    *,
    step_index: int,
    seed: int,
    input_sha256: str,
    candidate_sha256: str,
    requested_precision: str,
    mock: bool,
    backend: str | None = None,
    persistent: bool | None = None,
    visible_device_count: int | None = None,
    require_duration: bool = False,
    initialization: str = "ignore",
    exact_fields: bool = False,
    expected_source_size: tuple[int, int] | None = None,
    expected_output_size: tuple[int, int] | None = None,
    expected_root_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate CoZ worker fields that the parent adapter independently knows."""

    if not isinstance(metadata, Mapping):
        raise WorkerContractError("CoZ worker metadata must be an object")
    raw_fields = {
        "source_size",
        "output_size",
        "seed",
        "step_index",
        "root_sha256",
        "input_sha256",
        "candidate_sha256",
        "prompts",
        "duration_seconds",
        "peak_torch_allocated_mib",
        "requested_precision",
        "actual_precision",
        "component_placement",
        "semantic_anchor",
        "gpu_inventory",
        "mock",
    }
    if exact_fields:
        expected_fields = set(raw_fields)
        if backend is not None:
            expected_fields.add("backend")
        if persistent is not None:
            expected_fields.add("persistent")
        if initialization == "required":
            expected_fields.add("initialization_duration_seconds")
        observed_fields = set(metadata)
        comparison_fields = (
            observed_fields - {"initialization_duration_seconds"}
            if initialization == "forbidden"
            else observed_fields
        )
        if comparison_fields != expected_fields:
            raise WorkerContractError("CoZ worker metadata has unexpected or missing fields")
    expected: dict[str, object] = {
        "step_index": step_index,
        "seed": seed,
        "input_sha256": input_sha256,
        "candidate_sha256": candidate_sha256,
        "requested_precision": requested_precision,
        "mock": mock,
    }
    if backend is not None:
        expected["backend"] = backend
    if persistent is not None:
        expected["persistent"] = persistent
    for field, expected_value in expected.items():
        if field == "candidate_sha256" and (
            type(metadata.get(field)) is not type(expected_value)
            or metadata.get(field) != expected_value
        ):
            raise WorkerContractError(
                "CoZ worker metadata candidate hash disagrees with the candidate artifact"
            )
        _exact_scalar(
            metadata.get(field),
            expected_value,
            field=f"CoZ worker metadata {field}",
        )

    if exact_fields:
        source_size = metadata.get("source_size")
        output_size = metadata.get("output_size")
        if (
            not isinstance(source_size, list)
            or len(source_size) != 2
            or any(not _non_negative_int(item) or item == 0 for item in source_size)
        ):
            raise WorkerContractError("CoZ worker metadata source_size is invalid")
        if (
            not isinstance(output_size, list)
            or len(output_size) != 2
            or any(not _non_negative_int(item) or item == 0 for item in output_size)
            or output_size != [source_size[0] * 4, source_size[1] * 4]
        ):
            raise WorkerContractError("CoZ worker metadata output_size is not exactly 4x")
        if expected_source_size is not None and source_size != list(expected_source_size):
            raise WorkerContractError(
                "CoZ worker metadata source_size disagrees with the input artifact"
            )
        if expected_output_size is not None and output_size != list(expected_output_size):
            raise WorkerContractError(
                "CoZ worker metadata output_size disagrees with the candidate artifact"
            )
        root_sha256 = metadata.get("root_sha256")
        if (
            not isinstance(root_sha256, str)
            or len(root_sha256) != 64
            or any(character not in "0123456789abcdef" for character in root_sha256)
        ):
            raise WorkerContractError("CoZ worker metadata root_sha256 is invalid")
        if expected_root_sha256 is not None and root_sha256 != expected_root_sha256:
            raise WorkerContractError(
                "CoZ worker metadata root_sha256 disagrees with the session root"
            )
        prompts = metadata.get("prompts")
        if not isinstance(prompts, list) or not all(isinstance(prompt, str) for prompt in prompts):
            raise WorkerContractError("CoZ worker metadata prompts must be a string list")
        actual_precision = metadata.get("actual_precision")
        if (
            not isinstance(actual_precision, Mapping)
            or set(actual_precision) != {"transformer", "vae"}
            or any(not isinstance(value, str) or not value for value in actual_precision.values())
        ):
            raise WorkerContractError("CoZ worker metadata actual_precision is invalid")
        component_placement = metadata.get("component_placement")
        placement_fields = {
            "text_encoder_1",
            "text_encoder_2",
            "text_encoder_3",
            "transformer",
            "vae",
            "vlm_first_parameter",
        }
        if (
            not isinstance(component_placement, Mapping)
            or set(component_placement) != placement_fields
        ):
            raise WorkerContractError("CoZ worker metadata component_placement is invalid")
        for placement in component_placement.values():
            if (
                not isinstance(placement, Mapping)
                or set(placement) != {"device", "dtype"}
                or any(not isinstance(value, str) or not value for value in placement.values())
            ):
                raise WorkerContractError("CoZ worker metadata component_placement is invalid")
        _non_empty_text(
            metadata.get("semantic_anchor"),
            field="CoZ worker metadata semantic_anchor",
        )
        inventory = metadata.get("gpu_inventory")
        inventory_fields = {"logical_index", "uuid", "name", "memory_total_mib"}
        if (
            not isinstance(inventory, list)
            or not inventory
            or any(
                not isinstance(device, Mapping)
                or set(device) != inventory_fields
                or any(not isinstance(value, str) or not value for value in device.values())
                for device in inventory
            )
        ):
            raise WorkerContractError("CoZ worker metadata gpu_inventory is invalid")
        if visible_device_count is not None:
            logical_indices = {
                device["logical_index"] for device in inventory if isinstance(device, Mapping)
            }
            uuids = {device["uuid"] for device in inventory if isinstance(device, Mapping)}
            if (
                len(inventory) != visible_device_count
                or logical_indices != {str(index) for index in range(visible_device_count)}
                or len(uuids) != visible_device_count
            ):
                raise WorkerContractError(
                    "CoZ worker metadata gpu_inventory does not match the visible devices"
                )

    if require_duration or "duration_seconds" in metadata:
        _non_negative_number(
            metadata.get("duration_seconds"),
            field="worker_metadata.duration_seconds",
        )

    if initialization not in {"ignore", "required", "forbidden"}:
        raise ValueError(f"unknown CoZ initialization policy: {initialization}")
    initialization_field = "initialization_duration_seconds"
    if initialization == "required":
        _non_negative_number(
            metadata.get(initialization_field),
            field=f"worker_metadata.{initialization_field}",
        )
    elif initialization == "forbidden" and initialization_field in metadata:
        raise WorkerContractError(
            "CoZ worker metadata initialization_duration_seconds is only valid for "
            "the first persistent CoZ step"
        )
    elif initialization_field in metadata:
        _non_negative_number(
            metadata.get(initialization_field),
            field=f"worker_metadata.{initialization_field}",
        )

    peak_field = "peak_torch_allocated_mib"
    if peak_field in metadata:
        peaks = metadata[peak_field]
        if not isinstance(peaks, Mapping) or not peaks:
            raise WorkerContractError(
                "CoZ worker metadata peak_torch_allocated_mib must be a non-empty device mapping"
            )
        if visible_device_count is not None:
            expected_devices = {str(index) for index in range(visible_device_count)}
            if set(peaks) != expected_devices:
                expected_text = ", ".join(sorted(expected_devices))
                raise WorkerContractError(
                    "CoZ worker metadata peak_torch_allocated_mib must map exactly "
                    f"the visible logical devices ({expected_text})"
                )
        if any(
            not isinstance(device, str) or not _non_negative_int(value)
            for device, value in peaks.items()
        ):
            raise WorkerContractError(
                "CoZ worker metadata peak_torch_allocated_mib values must be non-negative integers"
            )
    return dict(metadata)
