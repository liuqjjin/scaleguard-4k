"""Strict validation for run manifests and their referenced image bytes."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from collections.abc import Mapping
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from scaleguard.color import apply_adain
from scaleguard.config import EXPERIMENT_GROUP_SEMANTICS, PipelineConfig, parse_config
from scaleguard.contracts import (
    FINAL_ADAIN_ALGORITHM,
    QUALITY_IDENTITY_SCHEMA,
    CompletionLevel,
    Decision,
    RunStatus,
)
from scaleguard.errors import ArtifactError, ConfigurationError, ScaleGuardError
from scaleguard.images import inspect_image
from scaleguard.imaging.forward_models import build_forward_model
from scaleguard.strict_json import StrictJSONError, loads
from scaleguard.worker_contracts import (
    WorkerContractError,
    validate_coz_worker_metadata,
    validate_fourkagent_restoration_metadata,
)

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
    try:
        result = float(value)
    except OverflowError as error:
        raise ManifestValidationError(f"{context}.{key} must be finite") from error
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
    expected_stages: frozenset[str] | None = None,
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
    if expected_stages is not None and stage not in expected_stages:
        allowed = ", ".join(sorted(expected_stages))
        raise ManifestValidationError(
            f"{context}.stage={stage!r} is invalid for this artifact role; expected {allowed}"
        )
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
    quality_backend: str,
    quality_identity_sha256: str,
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
    if _text(raw, "quality_backend", context) != quality_backend:
        raise ManifestValidationError(
            f"{context}.quality_backend disagrees with manifest quality identity"
        )
    if _text(raw, "quality_identity_sha256", context) != quality_identity_sha256:
        raise ManifestValidationError(
            f"{context}.quality_identity_sha256 disagrees with manifest quality identity"
        )
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


def _validated_manifest_config(
    raw: dict[str, Any],
    *,
    manifest_path: Path,
) -> PipelineConfig:
    try:
        encoded = json.dumps(
            raw,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        parsed = parse_config(encoded, source=manifest_path)
    except (ConfigurationError, TypeError, ValueError) as error:
        raise ManifestValidationError(f"manifest.config is invalid: {error}") from error
    canonical = parsed.as_dict()
    if raw != canonical:
        raise ManifestValidationError(
            "manifest.config is not canonical; it must contain every validated field "
            "with the exact serialized value"
        )
    return parsed


def _sha256_file(path: Path, *, context: str) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ManifestValidationError(f"{context} cannot be read: {path}: {error}") from error
    return digest.hexdigest()


def _validate_quality_identity(
    provenance: Mapping[str, Any],
    config: PipelineConfig,
) -> tuple[str, str]:
    context = "manifest.provenance.quality_identity"
    identity = _object(provenance.get("quality_identity"), context)
    required = {
        "schema_version",
        "configured_backend",
        "metric",
        "evaluator",
        "higher_is_better",
        "device",
        "project_root",
        "model_path",
        "model_sha256",
        "is_proxy",
    }
    if set(identity) != required:
        missing = sorted(required - set(identity))
        unexpected = sorted(set(identity) - required)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append("unexpected " + ", ".join(unexpected))
        raise ManifestValidationError(f"{context} has invalid fields: {'; '.join(details)}")
    if identity.get("schema_version") != QUALITY_IDENTITY_SCHEMA:
        raise ManifestValidationError(f"{context}.schema_version is unsupported")

    configured_backend = config.metrics.quality_backend
    expected_proxy = configured_backend == "gradient_proxy"
    expected_metric = config.metrics.quality_metric if configured_backend == "pyiqa" else None
    expected_evaluator = (
        "gradient_proxy_v1" if expected_proxy else f"pyiqa:{config.metrics.quality_metric}"
    )
    expected_values = {
        "configured_backend": configured_backend,
        "metric": expected_metric,
        "evaluator": expected_evaluator,
        "higher_is_better": True,
        "device": config.metrics.quality_device,
        "is_proxy": expected_proxy,
    }
    for field, expected in expected_values.items():
        if identity.get(field) != expected:
            raise ManifestValidationError(f"{context}.{field} disagrees with manifest.config")

    declared_project_root = _text(identity, "project_root", context)
    project_root = Path(declared_project_root).expanduser().resolve()
    if declared_project_root != str(project_root):
        raise ManifestValidationError(f"{context}.project_root must be canonical")
    provenance_root_value = provenance.get("project_root")
    if not isinstance(provenance_root_value, str) or not provenance_root_value:
        raise ManifestValidationError(
            "manifest runtime preflight provenance is incomplete: project_root"
        )
    provenance_root = Path(provenance_root_value).expanduser().resolve()
    if provenance_root_value != str(provenance_root):
        raise ManifestValidationError("manifest.provenance.project_root must be canonical")
    if provenance_root != project_root:
        raise ManifestValidationError(f"{context}.project_root disagrees with provenance")

    configured_model = config.metrics.quality_model_path
    if configured_model is None:
        if identity.get("model_path") is not None or identity.get("model_sha256") is not None:
            raise ManifestValidationError(f"{context} records weights for an unweighted backend")
    else:
        model_path_value = identity.get("model_path")
        model_sha256 = identity.get("model_sha256")
        if not isinstance(model_path_value, str) or not model_path_value:
            raise ManifestValidationError(
                f"{context}.model_path must identify the configured weights"
            )
        if (
            not isinstance(model_sha256, str)
            or len(model_sha256) != 64
            or any(character not in "0123456789abcdef" for character in model_sha256)
        ):
            raise ManifestValidationError(f"{context}.model_sha256 must be a lowercase SHA-256")
        expected_path = (
            configured_model if configured_model.is_absolute() else project_root / configured_model
        ).resolve()
        observed_path = Path(model_path_value).expanduser().resolve()
        if model_path_value != str(observed_path):
            raise ManifestValidationError(f"{context}.model_path must be canonical")
        if observed_path != expected_path:
            raise ManifestValidationError(f"{context}.model_path disagrees with manifest.config")
        observed_sha256 = _sha256_file(observed_path, context=f"{context}.model_path")
        if observed_sha256 != model_sha256:
            raise ManifestValidationError(f"{context}.model_sha256 disagrees with current weights")

    if provenance.get("quality_backend") != expected_evaluator:
        raise ManifestValidationError("manifest.provenance.quality_backend disagrees with config")
    if provenance.get("quality_backend_is_proxy") is not expected_proxy:
        raise ManifestValidationError(
            "manifest.provenance.quality_backend_is_proxy disagrees with config"
        )
    identity_sha256 = hashlib.sha256(
        json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if provenance.get("quality_identity_sha256") != identity_sha256:
        raise ManifestValidationError(
            "manifest.provenance.quality_identity_sha256 disagrees with quality identity"
        )
    return expected_evaluator, identity_sha256


def _verify_adain_derivation(
    *,
    source: Mapping[str, Any],
    reference: Mapping[str, Any],
    final_artifact: Mapping[str, Any],
) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            temporary = Path(handle.name)
        apply_adain(
            Path(str(source["_resolved_path"])),
            Path(str(reference["_resolved_path"])),
            temporary,
        )
        recomputed = inspect_image(temporary, mock=False, stage="adain_recomputation")
    except (ArtifactError, OSError, ValueError) as error:
        raise ManifestValidationError(
            f"cannot deterministically recompute final AdaIN: {error}"
        ) from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    if (
        recomputed.sha256 != final_artifact["sha256"]
        or recomputed.width != final_artifact["width"]
        or recomputed.height != final_artifact["height"]
    ):
        raise ManifestValidationError(
            "manifest.final_image does not match deterministic AdaIN derivation"
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
    if completion in {
        CompletionLevel.SCALEGUARD_VALIDATED,
        CompletionLevel.RESEARCH_EVALUATED,
    }:
        raise ManifestValidationError(
            f"{completion.value} belongs in a separate aggregate evidence receipt, "
            "not a run manifest"
        )
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
    requested = _integer(manifest, "requested_factor", "manifest")
    if requested not in _SUPPORTED_FACTORS:
        raise ManifestValidationError("manifest.requested_factor is unsupported")
    if controller_config.get("target_factor") != requested:
        raise ManifestValidationError(
            "manifest.requested_factor disagrees with config.controller.target_factor"
        )
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
    parsed_config = _validated_manifest_config(config, manifest_path=path)
    quality_backend, quality_identity_sha256 = _validate_quality_identity(
        provenance,
        parsed_config,
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
        expected_mock=False,
        expected_stages=frozenset({"input"}),
    )
    bridge = _BRIDGE_FACTORS[requested]
    restored_artifact = None
    retained_states: dict[str, tuple[dict[str, Any], float]] = {}
    if manifest["restored_image"] is not None:
        restored_artifact = _artifact(
            manifest["restored_image"],
            context="manifest.restored_image",
            manifest_path=path,
            artifact_root=artifact_root,
            expected_mock=mock,
            expected_stages=frozenset({"4kagent_restoration", "identity_observation"}),
        )
        if (
            restored_artifact["width"] != input_artifact["width"] * bridge
            or restored_artifact["height"] != input_artifact["height"] * bridge
        ):
            raise ManifestValidationError(
                "manifest.restored_image dimensions disagree with the declared bridge factor"
            )
        retained_states[restored_artifact["sha256"]] = (restored_artifact, float(bridge))
    restoration_metadata = _object(
        manifest["restoration_metadata"],
        "manifest.restoration_metadata",
    )
    restoration_process = _process(
        manifest["restoration_process"],
        "manifest.restoration_process",
    )
    if restored_artifact is not None:
        if restoration_metadata.get("backend") == "4kagent_upstream":
            try:
                validate_fourkagent_restoration_metadata(
                    restoration_metadata,
                    config=parsed_config.fourkagent,
                    bridge_factor=bridge,
                )
            except WorkerContractError as error:
                raise ManifestValidationError(
                    f"manifest.restoration_metadata violates the 4KAgent parent contract: {error}"
                ) from error
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
    chained_scale = float(bridge) if restored_artifact is not None else None
    persistent_session_root_sha256: str | None = None
    terminated = False
    last_decision: Decision | None = None
    for position, value in enumerate(raw_steps, start=1):
        context = f"manifest.steps[{position - 1}]"
        step = _object(value, context)
        if _integer(step, "index", context) != position:
            raise ManifestValidationError(f"{context}.index must equal {position}")
        input_scale = _number(step, "input_scale", context)
        candidate_scale = _number(step, "candidate_scale", context)
        if (
            input_scale not in _SUPPORTED_FACTORS
            or candidate_scale not in _SUPPORTED_FACTORS
            or candidate_scale != input_scale * 4
        ):
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
            or trusted_before["stage"] != chained_artifact["stage"]
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
        if chained_scale is None or input_scale != chained_scale:
            raise ManifestValidationError(
                f"{context}.input_scale does not continue the retained scale chain"
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
            if candidate["stage"] in {
                "input",
                "4kagent_restoration",
                "identity_observation",
                "final_output",
                trusted_before["stage"],
            }:
                raise ManifestValidationError(
                    f"{context}.candidate.stage is invalid for a newly generated scale state"
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
                quality_backend=quality_backend,
                quality_identity_sha256=quality_identity_sha256,
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
            chained_scale = candidate_scale
            retained_states[candidate["sha256"]] = (candidate, candidate_scale)
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
        last_decision = decision
        step_started = _timestamp(step.get("started_at"), f"{context}.started_at")
        step_finished = _timestamp(step.get("finished_at"), f"{context}.finished_at")
        if step_started is None or step_finished is None or step_finished < step_started:
            raise ManifestValidationError(f"{context} has invalid timestamps")
        worker_metadata = _object(
            step.get("worker_metadata"),
            f"{context}.worker_metadata",
        )
        if candidate is not None:
            claimed_backend = worker_metadata.get("backend")
            if claimed_backend in {
                "chain_of_zoom_subprocess",
                "chain_of_zoom_persistent",
            }:
                expected_backend = str(claimed_backend)
                expected_persistent = expected_backend == "chain_of_zoom_persistent"
                if expected_persistent and persistent_session_root_sha256 is None:
                    persistent_session_root_sha256 = trusted_before["sha256"]
                expected_root_sha256 = (
                    persistent_session_root_sha256
                    if expected_persistent
                    else trusted_before["sha256"]
                )
                try:
                    validate_coz_worker_metadata(
                        worker_metadata,
                        step_index=position,
                        seed=parsed_config.coz.seed + position - 1,
                        input_sha256=trusted_before["sha256"],
                        candidate_sha256=candidate["sha256"],
                        requested_precision=parsed_config.coz.mixed_precision,
                        mock=False,
                        backend=expected_backend,
                        persistent=expected_persistent,
                        visible_device_count=len(parsed_config.coz.visible_devices.split(",")),
                        require_duration=True,
                        initialization=(
                            "required" if expected_persistent and position == 1 else "forbidden"
                        ),
                        exact_fields=True,
                        expected_source_size=(
                            trusted_before["width"],
                            trusted_before["height"],
                        ),
                        expected_output_size=(candidate["width"], candidate["height"]),
                        expected_root_sha256=expected_root_sha256,
                    )
                except WorkerContractError as error:
                    raise ManifestValidationError(
                        f"{context}.worker_metadata violates the CoZ parent contract: {error}"
                    ) from error
        _process(step.get("process"), f"{context}.process")

    final_artifact = None
    if manifest["final_image"] is not None:
        final_artifact = _artifact(
            manifest["final_image"],
            context="manifest.final_image",
            manifest_path=path,
            artifact_root=artifact_root,
            expected_mock=mock,
            expected_stages=frozenset({"final_output"}),
        )
    final_metrics = _object(manifest["final_metrics"], "manifest.final_metrics")
    if (final_artifact is None) != (not final_metrics):
        raise ManifestValidationError(
            "manifest.final_image and manifest.final_metrics must be recorded together"
        )
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
        if error_value is not None:
            raise ManifestValidationError("successful manifest cannot contain an error")
        if achieved is None:
            raise ManifestValidationError("successful manifest cannot omit achieved_factor")
        if target_reached and (
            len(raw_steps) != expected_steps or accepted_candidates != expected_steps
        ):
            raise ManifestValidationError(
                "target-reaching manifest does not contain the complete accepted scale plan"
            )
    if final_metrics:
        if final_artifact is None:
            raise AssertionError("final artifact pairing validation did not hold")
        after_color_alignment = _boolean(
            final_metrics,
            "after_color_alignment",
            "manifest.final_metrics",
        )
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
        if selected_scale not in _SUPPORTED_FACTORS:
            raise ManifestValidationError(
                "manifest.final_metrics.selected_scale is not a retained supported factor"
            )
        if achieved is not None and selected_scale != achieved:
            raise ManifestValidationError(
                "manifest.final_metrics.selected_scale disagrees with achieved_factor"
            )
        selected_state = _text(final_metrics, "selected_state", "manifest.final_metrics")
        final_metric_record = _metric_record(
            final_metrics.get("metrics"),
            "manifest.final_metrics.metrics",
            quality_backend=quality_backend,
            quality_identity_sha256=quality_identity_sha256,
            measurement_enabled=measurement_enabled,
            measurement_model=measurement_model,
        )
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

        derivation = _object(
            final_metrics.get("derivation"),
            "manifest.final_metrics.derivation",
        )
        kind = _text(derivation, "kind", "manifest.final_metrics.derivation")
        expected_derivation_fields = {
            "kind",
            "source_sha256",
            "source_scale",
        }
        if kind == "adain":
            expected_derivation_fields.update({"reference_sha256", "algorithm"})
        elif kind != "copy":
            raise ManifestValidationError(
                "manifest.final_metrics.derivation.kind must be copy or adain"
            )
        if set(derivation) != expected_derivation_fields:
            raise ManifestValidationError(
                "manifest.final_metrics.derivation fields disagree with its kind"
            )
        source_sha256 = _text(
            derivation,
            "source_sha256",
            "manifest.final_metrics.derivation",
        )
        retained = retained_states.get(source_sha256)
        if retained is None:
            raise ManifestValidationError(
                "manifest.final_metrics.derivation source is not a retained state"
            )
        source_artifact, retained_scale = retained
        source_scale = _number(
            derivation,
            "source_scale",
            "manifest.final_metrics.derivation",
        )
        if source_scale != retained_scale or selected_scale != source_scale:
            raise ManifestValidationError(
                "manifest.final_metrics derivation scale disagrees with retained source"
            )
        if kind == "copy":
            if after_color_alignment or selected_state not in {
                "trusted",
                "previous_trusted",
                "restored",
            }:
                raise ManifestValidationError(
                    "copy final derivation has an invalid selected state or color alignment"
                )
            if final_artifact["sha256"] != source_sha256:
                raise ManifestValidationError(
                    "manifest.final_image is not a byte-identical copy of its retained source"
                )
        else:
            if not after_color_alignment or selected_state != "adain":
                raise ManifestValidationError(
                    "AdaIN final derivation must declare the adain selected state"
                )
            if parsed_config.controller.color_strategy != "adain":
                raise ManifestValidationError(
                    "AdaIN final derivation disagrees with controller.color_strategy"
                )
            if derivation.get("algorithm") != FINAL_ADAIN_ALGORITHM:
                raise ManifestValidationError(
                    "manifest.final_metrics.derivation.algorithm is unsupported"
                )
            if (
                restored_artifact is None
                or derivation.get("reference_sha256") != (restored_artifact["sha256"])
            ):
                raise ManifestValidationError(
                    "AdaIN final derivation reference must be the restored state"
                )
            _verify_adain_derivation(
                source=source_artifact,
                reference=restored_artifact,
                final_artifact=final_artifact,
            )

    events = manifest["events"]
    if not isinstance(events, list):
        raise ManifestValidationError("manifest.events must be a list")
    for index, value in enumerate(events):
        event = _object(value, f"manifest.events[{index}]")
        _text(event, "event", f"manifest.events[{index}]")
        _timestamp(event.get("at"), f"manifest.events[{index}].at")

    if restored_artifact is None and (
        scale_session_process is not None or raw_steps or final_artifact is not None
    ):
        raise ManifestValidationError(
            "manifest cannot record post-restoration evidence before restored_image"
        )
    if status is RunStatus.RUNNING:
        if finished is not None or error_value is not None:
            raise ManifestValidationError("running manifest cannot have finished_at or error")
        if achieved is not None or target_reached or final_artifact is not None:
            raise ManifestValidationError(
                "running manifest cannot claim an achieved or final state"
            )
    elif status is RunStatus.FAILED:
        if finished is None or error_value is None:
            raise ManifestValidationError("failed manifest requires finished_at and error")
        if achieved is not None or target_reached:
            raise ManifestValidationError("failed manifest cannot claim an achieved factor")
        if completion is not CompletionLevel.STATIC_READY:
            raise ManifestValidationError("failed manifest must remain STATIC_READY")
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
        if last_decision is Decision.CONTINUE:
            raise ManifestValidationError("successful manifest has a dangling continue decision")
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
    if mock and completion is not CompletionLevel.STATIC_READY:
        raise ManifestValidationError("mock manifest cannot exceed STATIC_READY")
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
        if restored_artifact is not None:
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
            if (
                backend == "chain_of_zoom_persistent"
                and successful
                and (scale_session_process is None or scale_session_process.get("returncode") != 0)
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
