"""Strict validation for run manifests and their referenced image bytes."""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from scaleguard.config import EXPERIMENT_GROUP_SEMANTICS
from scaleguard.contracts import CompletionLevel, Decision, RunStatus
from scaleguard.errors import ArtifactError, ConfigurationError, ScaleGuardError
from scaleguard.images import inspect_image
from scaleguard.imaging.forward_models import build_forward_model
from scaleguard.strict_json import StrictJSONError, loads

_SUPPORTED_FACTORS = {1, 2, 4, 8, 16}
_BRIDGE_FACTORS = {1: 1, 2: 2, 4: 1, 8: 2, 16: 1}


class ManifestValidationError(ScaleGuardError):
    """Raised when a manifest or one of its artifacts is inconsistent."""


def _object(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestValidationError(f"{context} must be an object")
    return value


def _text(mapping: Mapping[str, Any], key: str, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ManifestValidationError(f"{context}.{key} must be a non-empty string")
    return value


def _boolean(mapping: Mapping[str, Any], key: str, context: str) -> bool:
    value = mapping.get(key)
    if type(value) is not bool:
        raise ManifestValidationError(f"{context}.{key} must be boolean")
    return value


def _integer(mapping: Mapping[str, Any], key: str, context: str) -> int:
    value = mapping.get(key)
    if type(value) is not int:
        raise ManifestValidationError(f"{context}.{key} must be an integer")
    return value


def _number(mapping: Mapping[str, Any], key: str, context: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestValidationError(f"{context}.{key} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ManifestValidationError(f"{context}.{key} must be finite")
    return result


def _timestamp(value: object, context: str, *, optional: bool = False) -> datetime | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value:
        raise ManifestValidationError(f"{context} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError as error:
        raise ManifestValidationError(f"{context} is not an ISO-8601 timestamp") from error
    if parsed.utcoffset() != timedelta(0):
        raise ManifestValidationError(f"{context} must include a UTC offset")
    return parsed


def _artifact_path(
    declared: str,
    *,
    manifest_path: Path,
    artifact_root: Path | None,
) -> Path:
    raw = Path(declared).expanduser()
    if raw.is_absolute():
        return raw
    return (artifact_root or manifest_path.parent) / raw


def _artifact(
    value: object,
    *,
    context: str,
    manifest_path: Path,
    artifact_root: Path | None,
    expected_mock: bool | None,
) -> dict[str, Any]:
    raw = _object(value, context)
    declared_path = _text(raw, "path", context)
    declared_hash = _text(raw, "sha256", context)
    if len(declared_hash) != 64 or any(
        character not in "0123456789abcdef" for character in declared_hash
    ):
        raise ManifestValidationError(f"{context}.sha256 must be a lowercase SHA-256 digest")
    width = _integer(raw, "width", context)
    height = _integer(raw, "height", context)
    if width <= 0 or height <= 0:
        raise ManifestValidationError(f"{context} dimensions must be positive")
    media_type = _text(raw, "media_type", context)
    stage = _text(raw, "stage", context)
    mock = _boolean(raw, "mock", context)
    if expected_mock is not None and mock is not expected_mock:
        raise ManifestValidationError(
            f"{context}.mock={mock!r} disagrees with run mock={expected_mock!r}"
        )
    resolved = _artifact_path(
        declared_path,
        manifest_path=manifest_path,
        artifact_root=artifact_root,
    )
    try:
        observed = inspect_image(resolved, mock=mock, stage=stage)
    except ArtifactError as error:
        raise ManifestValidationError(str(error)) from error
    mismatches: list[str] = []
    if observed.sha256 != declared_hash:
        mismatches.append(f"sha256 expected {declared_hash}, observed {observed.sha256}")
    if observed.width != width or observed.height != height:
        mismatches.append(
            f"dimensions expected {width}x{height}, observed {observed.width}x{observed.height}"
        )
    if observed.media_type != media_type:
        mismatches.append(f"media_type expected {media_type}, observed {observed.media_type}")
    if mismatches:
        raise ManifestValidationError(f"{context} artifact mismatch: " + "; ".join(mismatches))
    return {
        **raw,
        "_resolved_path": str(observed.path),
        "_observed_width": observed.width,
        "_observed_height": observed.height,
    }


def _metric_record(
    value: object,
    context: str,
    *,
    measurement_enabled: bool,
    measurement_model: str,
) -> dict[str, Any]:
    raw = _object(value, context)
    for key in (
        "quality_baseline",
        "quality_candidate",
        "quality_gain",
        "scale_nrmse",
        "scale_edge_mae",
    ):
        _number(raw, key, context)
    _text(raw, "quality_backend", context)
    measurement = raw.get("measurement_nrmse")
    observed_measurement_model = raw.get("measurement_model")
    if measurement_enabled:
        _number(raw, "measurement_nrmse", context)
        if _text(raw, "measurement_model", context) != measurement_model:
            raise ManifestValidationError(
                f"{context}.measurement_model disagrees with config.metrics.measurement_model"
            )
    elif measurement is not None or observed_measurement_model is not None:
        raise ManifestValidationError(
            f"{context} records measurement evidence while measurement consistency is disabled"
        )
    expected_gain = _number(raw, "quality_candidate", context) - _number(
        raw, "quality_baseline", context
    )
    if not math.isclose(
        _number(raw, "quality_gain", context),
        expected_gain,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise ManifestValidationError(
            f"{context}.quality_gain disagrees with candidate minus baseline"
        )
    return raw


def _passes_gates(
    metrics: Mapping[str, Any],
    thresholds: Mapping[str, float],
    *,
    require_quality: bool,
) -> bool:
    if float(metrics["scale_nrmse"]) > thresholds["max_scale_nrmse"]:
        return False
    if float(metrics["scale_edge_mae"]) > thresholds["max_scale_edge_mae"]:
        return False
    measurement = metrics.get("measurement_nrmse")
    if measurement is not None and float(measurement) > thresholds["max_measurement_nrmse"]:
        return False
    return not (require_quality and float(metrics["quality_gain"]) < thresholds["min_quality_gain"])


def _process(value: object, context: str) -> dict[str, Any] | None:
    if value is None:
        return None
    raw = _object(value, context)
    argv = raw.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(token, str) for token in argv):
        raise ManifestValidationError(f"{context}.argv must be a non-empty string list")
    _text(raw, "cwd", context)
    _integer(raw, "returncode", context)
    if _number(raw, "duration_seconds", context) < 0:
        raise ManifestValidationError(f"{context}.duration_seconds must be non-negative")
    _text(raw, "stdout_path", context)
    _text(raw, "stderr_path", context)
    peaks = raw.get("peak_vram_mib")
    if not isinstance(peaks, dict) or any(
        not isinstance(key, str) or type(value) is not int or value < 0
        for key, value in peaks.items()
    ):
        raise ManifestValidationError(
            f"{context}.peak_vram_mib must map device strings to non-negative integers"
        )
    return raw


def _safe_run_id(run_id: str) -> bool:
    return not (
        len(run_id) > 128
        or run_id in {".", ".."}
        or "/" in run_id
        or "\\" in run_id
        or any(character in "*?[]" for character in run_id)
        or any(ord(character) < 32 or ord(character) == 127 for character in run_id)
    )


def validate_run_manifest(
    path: Path,
    *,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    """Validate schema, state invariants, scales, and every referenced image."""

    try:
        payload = loads(path.read_text(encoding="utf-8"))
    except (OSError, StrictJSONError) as error:
        raise ManifestValidationError(f"invalid manifest {path}: {error}") from error
    manifest = _object(payload, "manifest")
    required = {
        "schema_version",
        "run_id",
        "status",
        "completion_level",
        "started_at",
        "finished_at",
        "mock",
        "config",
        "provenance",
        "input_image",
        "requested_factor",
        "achieved_factor",
        "target_reached",
        "restored_image",
        "restoration_metadata",
        "restoration_process",
        "scale_session_process",
        "steps",
        "final_image",
        "final_metrics",
        "events",
        "error",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ManifestValidationError(f"manifest is missing fields: {', '.join(missing)}")
    if manifest["schema_version"] != "1.0":
        raise ManifestValidationError("manifest.schema_version must be '1.0'")
    run_id = _text(manifest, "run_id", "manifest")
    if not _safe_run_id(run_id):
        raise ManifestValidationError("manifest.run_id must be one safe path component")
    try:
        status = RunStatus(_text(manifest, "status", "manifest"))
        completion = CompletionLevel(_text(manifest, "completion_level", "manifest"))
    except ValueError as error:
        raise ManifestValidationError(str(error)) from error
    started = _timestamp(manifest["started_at"], "manifest.started_at")
    finished = _timestamp(
        manifest["finished_at"],
        "manifest.finished_at",
        optional=True,
    )
    if started is None:
        raise AssertionError("required timestamp validation did not hold")
    if finished is not None and finished < started:
        raise ManifestValidationError("manifest.finished_at precedes started_at")
    mock = _boolean(manifest, "mock", "manifest")
    config = _object(manifest["config"], "manifest.config")
    provenance = _object(manifest["provenance"], "manifest.provenance")
    metric_config = _object(config.get("metrics"), "manifest.config.metrics")
    runtime_config = _object(config.get("runtime"), "manifest.config.runtime")
    fourkagent_config = _object(
        config.get("fourkagent"),
        "manifest.config.fourkagent",
    )
    coz_config = _object(config.get("coz"), "manifest.config.coz")
    controller_config = _object(
        config.get("controller"),
        "manifest.config.controller",
    )
    acceptance_policy = _text(
        controller_config,
        "acceptance_policy",
        "manifest.config.controller",
    )
    if acceptance_policy not in {"trusted", "fixed"}:
        raise ManifestValidationError("manifest config has an invalid acceptance policy")
    experiment_group = runtime_config.get("experiment_group")
    experiment_sample_id = runtime_config.get("experiment_sample_id")
    if (experiment_group is None) != (experiment_sample_id is None):
        raise ManifestValidationError("manifest experiment identifiers are incomplete")
    if experiment_group is not None:
        if experiment_group not in EXPERIMENT_GROUP_SEMANTICS:
            raise ManifestValidationError("manifest experiment group is not declared")
        if not isinstance(experiment_sample_id, str) or not _safe_run_id(experiment_sample_id):
            raise ManifestValidationError("manifest experiment sample id is unsafe")
        expected = EXPERIMENT_GROUP_SEMANTICS[experiment_group]
        observed = (
            fourkagent_config.get("mode"),
            coz_config.get("mode"),
            controller_config.get("target_factor"),
            controller_config.get("max_coz_steps"),
            acceptance_policy,
        )
        if observed != expected:
            raise ManifestValidationError(
                f"manifest {experiment_group} config violates its fixed experiment semantics"
            )
    if acceptance_policy == "fixed" and experiment_group not in {
        "A-only",
        "B-only",
        "AB-fixed",
    }:
        raise ManifestValidationError(
            "fixed acceptance is reserved for declared component/fixed ablations"
        )
    if fourkagent_config.get("mode") == "identity" and experiment_group != "B-only":
        raise ManifestValidationError("identity restoration is reserved for the B-only manifest")
    configured_mock = fourkagent_config.get("mode") == "fake" or coz_config.get("mode") == "fake"
    if mock != configured_mock:
        raise ManifestValidationError("manifest.mock disagrees with configured backend modes")
    thresholds = {
        name: _number(metric_config, name, "manifest.config.metrics")
        for name in (
            "min_quality_gain",
            "max_scale_nrmse",
            "max_scale_edge_mae",
            "max_measurement_nrmse",
        )
    }
    measurement_enabled = _boolean(
        metric_config,
        "measurement_enabled",
        "manifest.config.metrics",
    )
    measurement_model_selector = _text(
        metric_config,
        "measurement_model",
        "manifest.config.metrics",
    )
    measurement_parameters = _object(
        metric_config.get("measurement_parameters"),
        "manifest.config.metrics.measurement_parameters",
    )
    try:
        measurement_model = build_forward_model(
            measurement_model_selector,
            measurement_parameters,
        ).name
    except ConfigurationError as error:
        raise ManifestValidationError(
            f"manifest.config.metrics has an invalid measurement model: {error}"
        ) from error
    requested = _integer(manifest, "requested_factor", "manifest")
    if requested not in _SUPPORTED_FACTORS:
        raise ManifestValidationError("manifest.requested_factor is unsupported")
    if controller_config.get("target_factor") != requested:
        raise ManifestValidationError(
            "manifest.requested_factor disagrees with config.controller.target_factor"
        )
    achieved_value = manifest["achieved_factor"]
    achieved: int | None
    if achieved_value is None:
        achieved = None
    elif type(achieved_value) is int and achieved_value in _SUPPORTED_FACTORS:
        achieved = achieved_value
    else:
        raise ManifestValidationError("manifest.achieved_factor is invalid")
    target_reached = _boolean(manifest, "target_reached", "manifest")
    if target_reached != (achieved is not None and achieved == requested):
        raise ManifestValidationError(
            "manifest.target_reached disagrees with requested_factor and achieved_factor"
        )

    input_artifact = _artifact(
        manifest["input_image"],
        context="manifest.input_image",
        manifest_path=path,
        artifact_root=artifact_root,
        expected_mock=None,
    )
    restored_artifact = None
    if manifest["restored_image"] is not None:
        restored_artifact = _artifact(
            manifest["restored_image"],
            context="manifest.restored_image",
            manifest_path=path,
            artifact_root=artifact_root,
            expected_mock=mock,
        )
        bridge = _BRIDGE_FACTORS[requested]
        if (
            restored_artifact["width"] != input_artifact["width"] * bridge
            or restored_artifact["height"] != input_artifact["height"] * bridge
        ):
            raise ManifestValidationError(
                "manifest.restored_image dimensions disagree with the declared bridge factor"
            )
    restoration_metadata = _object(
        manifest["restoration_metadata"],
        "manifest.restoration_metadata",
    )
    restoration_process = _process(
        manifest["restoration_process"],
        "manifest.restoration_process",
    )
    scale_session_process = _process(
        manifest["scale_session_process"],
        "manifest.scale_session_process",
    )

    raw_steps = manifest["steps"]
    if not isinstance(raw_steps, list):
        raise ManifestValidationError("manifest.steps must be a list")
    accepted_candidates = 0
    recorded_candidates = 0
    expected_steps = {1: 0, 2: 0, 4: 1, 8: 1, 16: 2}[requested]
    if len(raw_steps) > expected_steps:
        raise ManifestValidationError(
            f"manifest.steps exceeds the {expected_steps}-step factor policy"
        )
    chained_artifact = restored_artifact
    terminated = False
    for position, value in enumerate(raw_steps, start=1):
        context = f"manifest.steps[{position - 1}]"
        step = _object(value, context)
        if _integer(step, "index", context) != position:
            raise ManifestValidationError(f"{context}.index must equal {position}")
        input_scale = _number(step, "input_scale", context)
        candidate_scale = _number(step, "candidate_scale", context)
        if input_scale <= 0 or candidate_scale != input_scale * 4:
            raise ManifestValidationError(f"{context} has an invalid 4x scale transition")
        trusted_before = _artifact(
            step.get("trusted_before"),
            context=f"{context}.trusted_before",
            manifest_path=path,
            artifact_root=artifact_root,
            expected_mock=mock,
        )
        if terminated:
            raise ManifestValidationError(f"{context} appears after a terminal decision")
        if chained_artifact is None or (
            trusted_before["sha256"] != chained_artifact["sha256"]
            or Path(trusted_before["_resolved_path"]) != Path(chained_artifact["_resolved_path"])
        ):
            raise ManifestValidationError(
                f"{context}.trusted_before does not continue the accepted artifact chain"
            )
        if (
            trusted_before["width"] != input_artifact["width"] * input_scale
            or trusted_before["height"] != input_artifact["height"] * input_scale
        ):
            raise ManifestValidationError(
                f"{context}.trusted_before dimensions disagree with input_scale"
            )
        candidate_value = step.get("candidate")
        candidate = None
        if candidate_value is not None:
            recorded_candidates += 1
            candidate = _artifact(
                candidate_value,
                context=f"{context}.candidate",
                manifest_path=path,
                artifact_root=artifact_root,
                expected_mock=mock,
            )
            if (
                candidate["width"] != trusted_before["width"] * 4
                or candidate["height"] != trusted_before["height"] * 4
            ):
                raise ManifestValidationError(f"{context}.candidate dimensions are not exactly 4x")
        metrics_value = step.get("metrics")
        metrics = (
            _metric_record(
                metrics_value,
                f"{context}.metrics",
                measurement_enabled=measurement_enabled,
                measurement_model=measurement_model,
            )
            if metrics_value is not None
            else None
        )
        try:
            decision = Decision(_text(step, "decision", context))
        except ValueError as error:
            raise ManifestValidationError(str(error)) from error
        accepted = _boolean(step, "accepted", context)
        if accepted:
            accepted_candidates += 1
            if candidate is None:
                raise AssertionError("accepted candidate validation did not hold")
            chained_artifact = candidate
        if accepted and (candidate is None or metrics is None):
            raise ManifestValidationError(f"{context} accepted without candidate metrics")
        if candidate is None and metrics is not None:
            raise ManifestValidationError(f"{context} has metrics without a candidate")
        if decision is Decision.CONTINUE and not accepted:
            raise ManifestValidationError(f"{context} continue decision must be accepted")
        if decision is Decision.CONTINUE and position == expected_steps:
            raise ManifestValidationError(
                f"{context} final planned scale decision must stop or rollback"
            )
        if decision is Decision.ROLLBACK and accepted:
            raise ManifestValidationError(f"{context} rollback decision cannot be accepted")
        gates_passed = (
            _passes_gates(metrics, thresholds, require_quality=True)
            if metrics is not None
            else False
        )
        if accepted and not gates_passed and acceptance_policy == "trusted":
            raise ManifestValidationError(f"{context} accepted metrics that fail configured gates")
        if (
            metrics is not None
            and not accepted
            and (
                acceptance_policy == "fixed" or (gates_passed and decision is not Decision.ROLLBACK)
            )
        ):
            raise ManifestValidationError(
                f"{context} rejected metrics contrary to its acceptance policy"
            )
        reason = _text(step, "reason", context)
        if (
            acceptance_policy == "fixed"
            and metrics is not None
            and not reason.startswith("fixed ablation policy accepted")
        ):
            raise ManifestValidationError(f"{context} does not disclose fixed acceptance")
        if decision is not Decision.CONTINUE:
            terminated = True
        step_started = _timestamp(step.get("started_at"), f"{context}.started_at")
        step_finished = _timestamp(step.get("finished_at"), f"{context}.finished_at")
        if step_started is None or step_finished is None or step_finished < step_started:
            raise ManifestValidationError(f"{context} has invalid timestamps")
        _object(step.get("worker_metadata"), f"{context}.worker_metadata")
        _process(step.get("process"), f"{context}.process")

    final_artifact = None
    if manifest["final_image"] is not None:
        final_artifact = _artifact(
            manifest["final_image"],
            context="manifest.final_image",
            manifest_path=path,
            artifact_root=artifact_root,
            expected_mock=mock,
        )
    final_metrics = _object(manifest["final_metrics"], "manifest.final_metrics")
    if final_metrics:
        _boolean(final_metrics, "after_color_alignment", "manifest.final_metrics")
        gate_passed = _boolean(final_metrics, "gate_passed", "manifest.final_metrics")
        accepted_by_policy = _boolean(
            final_metrics,
            "accepted_by_policy",
            "manifest.final_metrics",
        )
        if (
            _text(
                final_metrics,
                "acceptance_policy",
                "manifest.final_metrics",
            )
            != acceptance_policy
        ):
            raise ManifestValidationError(
                "manifest.final_metrics acceptance policy disagrees with config"
            )
        _text(final_metrics, "gate_reason", "manifest.final_metrics")
        selected_scale = _number(
            final_metrics,
            "selected_scale",
            "manifest.final_metrics",
        )
        if achieved is not None and selected_scale != achieved:
            raise ManifestValidationError(
                "manifest.final_metrics.selected_scale disagrees with achieved_factor"
            )
        _text(final_metrics, "selected_state", "manifest.final_metrics")
        final_metric_record = _metric_record(
            final_metrics.get("metrics"),
            "manifest.final_metrics.metrics",
            measurement_enabled=measurement_enabled,
            measurement_model=measurement_model,
        )
        bridge = _BRIDGE_FACTORS[requested]
        observed_gate_pass = _passes_gates(
            final_metric_record,
            thresholds,
            require_quality=selected_scale > bridge,
        )
        if gate_passed != observed_gate_pass:
            raise ManifestValidationError(
                "manifest.final_metrics gate result disagrees with configured gates"
            )
        if accepted_by_policy != (observed_gate_pass or acceptance_policy == "fixed"):
            raise ManifestValidationError(
                "manifest.final_metrics policy result disagrees with configured acceptance"
            )

    events = manifest["events"]
    if not isinstance(events, list):
        raise ManifestValidationError("manifest.events must be a list")
    for index, value in enumerate(events):
        event = _object(value, f"manifest.events[{index}]")
        _text(event, "event", f"manifest.events[{index}]")
        _timestamp(event.get("at"), f"manifest.events[{index}].at")

    error_value = manifest["error"]
    if error_value is not None:
        error_record = _object(error_value, "manifest.error")
        _text(error_record, "type", "manifest.error")
        _text(error_record, "message", "manifest.error")

    successful = status in {RunStatus.SUCCEEDED, RunStatus.SUCCEEDED_WITH_ROLLBACK}
    if successful:
        if finished is None or restored_artifact is None or final_artifact is None:
            raise ManifestValidationError(
                "successful manifest requires finished_at, restored_image, and final_image"
            )
        if error_value is not None or achieved is None:
            raise ManifestValidationError(
                "successful manifest cannot contain an error or omit achieved_factor"
            )
        if (
            final_artifact["width"] != input_artifact["width"] * achieved
            or final_artifact["height"] != input_artifact["height"] * achieved
        ):
            raise ManifestValidationError(
                "manifest.final_image dimensions disagree with achieved_factor"
            )
        if not final_metrics or final_metrics.get("accepted_by_policy") is not True:
            raise ManifestValidationError("successful manifest requires a policy-accepted final")
        if target_reached and (
            len(raw_steps) != expected_steps or accepted_candidates != expected_steps
        ):
            raise ManifestValidationError(
                "target-reaching manifest does not contain the complete accepted scale plan"
            )
    if status is RunStatus.SUCCEEDED and not target_reached:
        raise ManifestValidationError("succeeded status requires target_reached=true")
    if status is RunStatus.SUCCEEDED_WITH_ROLLBACK and target_reached:
        raise ManifestValidationError(
            "succeeded_with_rollback status requires target_reached=false"
        )
    if status is RunStatus.FAILED and (finished is None or error_value is None):
        raise ManifestValidationError("failed manifest requires finished_at and error")
    if status is RunStatus.RUNNING and (finished is not None or error_value is not None):
        raise ManifestValidationError("running manifest cannot have finished_at or error")
    if mock and completion is not CompletionLevel.STATIC_READY:
        raise ManifestValidationError("mock manifest cannot exceed STATIC_READY")
    if completion in {
        CompletionLevel.SCALEGUARD_VALIDATED,
        CompletionLevel.RESEARCH_EVALUATED,
    }:
        raise ManifestValidationError(
            f"{completion.value} belongs in a separate aggregate evidence receipt, "
            "not a run manifest"
        )
    if completion is CompletionLevel.COMPONENT_REPRODUCED and (
        not successful or mock or recorded_candidates == 0
    ):
        raise ManifestValidationError(
            "COMPONENT_REPRODUCED requires a successful non-mock recorded candidate"
        )
    if completion is CompletionLevel.AB_INTEGRATED and (
        status is not RunStatus.SUCCEEDED
        or mock
        or not target_reached
        or accepted_candidates == 0
        or acceptance_policy != "trusted"
    ):
        raise ManifestValidationError(
            "AB_INTEGRATED requires a successful trusted non-mock target-reaching candidate"
        )
    runtime_completion = completion in {
        CompletionLevel.COMPONENT_REPRODUCED,
        CompletionLevel.AB_INTEGRATED,
    }
    requires_runtime_attestation = coz_config.get("mode") == "persistent" and fourkagent_config.get(
        "mode"
    ) in {"upstream", "identity"}
    if runtime_completion or requires_runtime_attestation:
        claim = completion.value if runtime_completion else "audited runtime"
        if (
            provenance.get("runtime_evidence_verified") is not True
            or provenance.get("runtime_profile_bound") is not True
        ):
            raise ManifestValidationError(
                f"{claim} requires verified and profile-bound runtime provenance"
            )
        preflight_path = provenance.get("runtime_preflight_receipt")
        project_root = provenance.get("project_root")
        config_path = provenance.get("runtime_config_path")
        config_digest = provenance.get("runtime_config_sha256")
        if not isinstance(preflight_path, str) or not preflight_path:
            raise ManifestValidationError(f"{claim} runtime preflight provenance is incomplete")
        if not isinstance(project_root, str) or not project_root:
            raise ManifestValidationError(f"{claim} runtime preflight provenance is incomplete")
        if not isinstance(config_path, str) or not config_path:
            raise ManifestValidationError(f"{claim} runtime preflight provenance is incomplete")
        if not isinstance(config_digest, str) or not config_digest:
            raise ManifestValidationError(f"{claim} runtime preflight provenance is incomplete")
        from scaleguard.config import parse_config
        from scaleguard.provenance import (
            RuntimePreflightError,
            bind_runtime_config,
            load_regular_file_snapshot,
            validate_runtime_preflight,
        )

        try:
            validated_provenance = validate_runtime_preflight(
                Path(preflight_path),
                config_path=Path(config_path),
                project_root=Path(project_root),
            )
            config_payload, current_config_digest = load_regular_file_snapshot(
                Path(config_path),
                "runtime config",
            )
            parsed_config = parse_config(config_payload, source=Path(config_path))
            binding = validated_provenance.get("runtime_execution_binding")
            if not isinstance(binding, dict):
                raise RuntimePreflightError(
                    "runtime preflight did not reconstruct an execution binding"
                )
            current_config = bind_runtime_config(
                parsed_config,
                project_root=Path(project_root),
                binding=binding,
            )
        except (ConfigurationError, RuntimePreflightError) as error:
            raise ManifestValidationError(
                f"{claim} runtime preflight is invalid: {error}"
            ) from error
        mismatched_provenance = [
            field
            for field, expected in validated_provenance.items()
            if provenance.get(field) != expected
        ]
        if mismatched_provenance:
            raise ManifestValidationError(
                f"{claim} runtime provenance disagrees with current evidence: "
                + ", ".join(sorted(mismatched_provenance))
            )
        if (
            validated_provenance.get("runtime_config_sha256") != config_digest
            or current_config_digest != config_digest
        ):
            raise ManifestValidationError(
                f"{claim} runtime config digest disagrees with provenance"
            )
        if current_config.as_dict() != config:
            raise ManifestValidationError(
                f"{claim} manifest config differs from its preflighted config"
            )
        if experiment_group == "B-only":
            if (
                restoration_metadata.get("backend") != "scaleguard_identity_observation"
                or restoration_metadata.get("algorithmic_restoration") is not False
                or restoration_process is not None
            ):
                raise ManifestValidationError(
                    "B-only requires the non-algorithmic identity observation"
                )
        elif (
            restoration_metadata.get("backend") != "4kagent_upstream"
            or restoration_process is None
            or restoration_process.get("returncode") != 0
        ):
            raise ManifestValidationError(f"{claim} requires 4KAgent upstream process evidence")
        for index, value in enumerate(raw_steps):
            step = _object(value, f"manifest.steps[{index}]")
            candidate_value = step.get("candidate")
            if candidate_value is None:
                continue
            metadata = _object(
                step.get("worker_metadata"),
                f"manifest.steps[{index}].worker_metadata",
            )
            backend = metadata.get("backend")
            if backend not in {
                "chain_of_zoom_subprocess",
                "chain_of_zoom_persistent",
            }:
                raise ManifestValidationError(f"{claim} requires Chain-of-Zoom worker evidence")
            step_process = _process(
                step.get("process"),
                f"manifest.steps[{index}].process",
            )
            if backend == "chain_of_zoom_subprocess" and (
                step_process is None or step_process.get("returncode") != 0
            ):
                raise ManifestValidationError(
                    f"{claim} one-shot CoZ steps require successful process evidence"
                )
            if backend == "chain_of_zoom_persistent" and (
                scale_session_process is None or scale_session_process.get("returncode") != 0
            ):
                raise ManifestValidationError(
                    f"{claim} persistent CoZ requires successful session process evidence"
                )
            if metadata.get("candidate_sha256") != _object(
                candidate_value,
                f"manifest.steps[{index}].candidate",
            ).get("sha256"):
                raise ManifestValidationError(
                    f"{claim} candidate hash disagrees with CoZ worker metadata"
                )
    return manifest
