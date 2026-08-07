"""Executable, paired orchestration for the four ScaleGuard ablation groups."""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import math
import os
import re
import stat
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any

import yaml

from scaleguard.config import (
    EXPERIMENT_GROUP_SEMANTICS,
    EXPERIMENT_GROUPS,
    load_config,
)
from scaleguard.errors import ScaleGuardError
from scaleguard.manifest import validate_run_manifest
from scaleguard.provenance import load_regular_file_snapshot, require_clean_git_commit
from scaleguard.strict_json import loads_object
from scaleguard.strict_yaml import StrictYAMLError
from scaleguard.strict_yaml import loads as load_strict_yaml

PROTOCOL_SCHEMA = "1.0"
RECEIPT_SCHEMA = "scaleguard.ablation-suite/v1"
INTEGRATION_RUNNER = "scripts/autodl/run_experiment.sh"
PROTOCOL_NAME = "core-ablation"
GROUP_IDS = EXPERIMENT_GROUPS


class ExperimentProtocolError(ScaleGuardError):
    """Raised when an ablation protocol or suite boundary is unsafe."""


@dataclass(frozen=True, slots=True)
class GroupSpec:
    id: str
    description: str
    fourkagent_mode: str
    coz_mode: str
    target_factor: int
    max_coz_steps: int
    acceptance_policy: str
    comparison_resolution: str


@dataclass(frozen=True, slots=True)
class AblationProtocol:
    path: Path
    size_bytes: int
    sha256: str
    name: str
    integration_runner: str
    groups: tuple[GroupSpec, ...]
    paired_requirements: Mapping[str, bool]
    metrics: Mapping[str, tuple[str, ...]]
    notes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _InputSource:
    requested_path: str
    path: Path
    size_bytes: int
    sha256: str
    suffix: str


CommandRunner = Callable[[Sequence[str], Path], int]

_COMPARISON_RESOLUTION = {
    "A-only": "restoration_native",
    "B-only": "target_4x",
    "AB-fixed": "target_4x",
    "ScaleGuard": "target_4x",
}
_GROUP_CONTRACT = tuple(
    {
        "id": group,
        "fourkagent_mode": EXPERIMENT_GROUP_SEMANTICS[group][0],
        "coz_mode": EXPERIMENT_GROUP_SEMANTICS[group][1],
        "target_factor": EXPERIMENT_GROUP_SEMANTICS[group][2],
        "max_coz_steps": EXPERIMENT_GROUP_SEMANTICS[group][3],
        "acceptance_policy": EXPERIMENT_GROUP_SEMANTICS[group][4],
        "comparison_resolution": _COMPARISON_RESOLUTION[group],
    }
    for group in EXPERIMENT_GROUPS
)
_GROUP_KEYS = {
    "id",
    "description",
    "fourkagent_mode",
    "coz_mode",
    "target_factor",
    "max_coz_steps",
    "acceptance_policy",
    "comparison_resolution",
}
_ROOT_KEYS = {
    "schema_version",
    "name",
    "status",
    "integration_runner",
    "base_requirements",
    "groups",
    "paired_requirements",
    "metrics",
    "notes",
}
_BASE_REQUIREMENTS = {
    "controller_target_factor": 4,
    "minimum_max_coz_steps": 1,
    "fourkagent_mode": "upstream",
    "coz_mode": "persistent",
}
_PAIRED_REQUIREMENT_KEYS = {
    "same_input_snapshot",
    "same_coz_seed_per_sample",
    "same_quality_metric_revision",
    "preserve_raw_manifests",
    "fresh_preflight_per_job",
    "continue_after_failure",
    "no_metric_imputation",
}
_METRIC_KEYS = {"full_reference", "no_reference", "consistency", "systems"}
_METRIC_CONTRACT = {
    "full_reference": ("psnr", "ssim", "lpips"),
    "no_reference": ("musiq", "clipiqa"),
    "consistency": ("scale_nrmse", "scale_edge_mae", "measurement_nrmse"),
    "systems": (
        "success_rate",
        "stop_rate",
        "rollback_rate",
        "wall_time_seconds",
        "coz_initialization_seconds",
        "coz_first_step_seconds",
        "coz_steady_step_seconds",
        "peak_vram_mib",
    ),
}
_BASE_CONFIG_SECTIONS = {"runtime", "fourkagent", "coz", "metrics", "controller"}
_SAFE_SAMPLE_ID_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789._-")
_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
_ATTEMPT_FILE_RELATIVE_PATHS = {
    "execution": "execution.json",
    "run_manifest": "scaleguard-run-manifest.json",
    "model_evidence": "model-evidence.json",
    "raw_log": "experiment.log",
    "gpu_samples": "gpu-samples.csv",
    "nvidia_smi_before": "nvidia-smi-before.txt",
    "nvidia_smi_after": "nvidia-smi-after.txt",
    "gpu_inventory": "gpu-preflight/gpu_inventory.csv",
    "gpu_preflight": "gpu-preflight/gpu_check.json",
    "files_inventory": "files.json",
    "runtime_preflight": "runtime-preflight.json",
}
_PARSED_ATTEMPT_FILE_ROLES = {
    "execution",
    "gpu_samples",
    "run_manifest",
    "model_evidence",
    "gpu_inventory",
    "gpu_preflight",
    "files_inventory",
    "runtime_preflight",
}
_COMMON_PROVENANCE_FIELDS = (
    "runtime_evidence_verified",
    "runtime_profile_bound",
    "bootstrap_receipt_sha256",
    "materialization_marker_sha256",
    "source_weights_receipt_sha256",
    "weights_root",
    "project_commit",
    "project_root",
    "runtime_execution_binding",
    "runtime_execution_binding_sha256",
)
_JOB_PROVENANCE_FIELDS = (
    *_COMMON_PROVENANCE_FIELDS,
    "runtime_preflight_receipt",
    "runtime_preflight_sha256",
    "runtime_environment_receipt_sha256",
    "materialization_receipt_sha256",
    "runtime_config_path",
    "runtime_config_sha256",
    "runtime_stage_started_at",
)
_SUITE_RECEIPT_KEYS = {
    "schema_version",
    "status",
    "plan_only",
    "started_at_utc",
    "completed_at_utc",
    "project_root",
    "project_commit",
    "output_directory",
    "protocol",
    "base_config",
    "integration_runner",
    "groups",
    "seeds",
    "inputs",
    "jobs",
    "issues",
    "counts",
    "receipt_sha256",
}
_SUITE_JOB_KEYS = {
    "job_id",
    "sample_id",
    "group",
    "seed",
    "input",
    "config",
    "output_path",
    "run_root",
    "wrapper_evidence_pointer",
    "argv",
    "status",
    "started_at_utc",
    "completed_at_utc",
    "returncode",
    "project_commit_before",
    "project_commit_after",
    "manifest",
    "wrapper_attempt",
    "runtime_evidence",
    "issues",
}
_EXPERIMENT_ENVIRONMENT_NAMES = (
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "HF_HOME",
    "HUGGINGFACE_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "TORCH_HOME",
    "XDG_CACHE_HOME",
    "PIP_CACHE_DIR",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "SCALEGUARD_ARTIFACT_ROOT",
    "SCALEGUARD_CACHE_ROOT",
    "SCALEGUARD_DIAGNOSTICS_MAX_MODEL_RUNS",
    "SCALEGUARD_DIAGNOSTICS_MAX_TOTAL_FILES",
    "SCALEGUARD_DIAGNOSTICS_MAX_TOTAL_MIB",
    "SCALEGUARD_GPU_NAME_PATTERN",
    "SCALEGUARD_GPU_SAMPLE_INTERVAL_SECONDS",
    "SCALEGUARD_MIN_DISK_GIB",
    "SCALEGUARD_MIN_GPUS",
    "SCALEGUARD_MIN_GPU_MEMORY_MIB",
    "SCALEGUARD_MIN_NVIDIA_DRIVER",
    "SCALEGUARD_RUNTIME_DEPENDENCIES_LOCK",
    "SCALEGUARD_UPSTREAM_LOCK",
    "SCALEGUARD_WEIGHTS_ROOT",
    "SCALEGUARD_WEIGHT_RECEIPT",
)
_FIXED_SYSTEM_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(canonical)


def _wrapper_canonical_sha256(payload: Mapping[str, Any]) -> str:
    """Match the experiment wrapper's hardware-identity serialization exactly."""

    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(canonical)


def _require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(type(key) is str for key in value):
        raise ExperimentProtocolError(f"{context} must be a string-keyed mapping")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise ExperimentProtocolError(f"{context} has invalid keys: {'; '.join(details)}")


def _require_text(value: Any, context: str) -> str:
    if type(value) is not str or not value.strip():
        raise ExperimentProtocolError(f"{context} must be a non-empty string")
    return value


def _require_integer(value: Any, context: str) -> int:
    if type(value) is not int:
        raise ExperimentProtocolError(f"{context} must be an integer")
    return value


def _require_timestamp(value: Any, context: str) -> datetime:
    if type(value) is not str or not value:
        raise ExperimentProtocolError(f"{context} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ExperimentProtocolError(f"{context} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ExperimentProtocolError(f"{context} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _load_yaml_snapshot(path: Path, context: str) -> tuple[Path, bytes, Any]:
    try:
        resolved, payload, _size, _digest = _load_regular_snapshot(path, context)
        document = load_strict_yaml(payload)
    except StrictYAMLError as error:
        raise ExperimentProtocolError(f"invalid {context} YAML in {path}: {error}") from error
    return resolved, payload, document


def _load_regular_snapshot(
    path: Path,
    context: str,
) -> tuple[Path, bytes, int, str]:
    requested = path.expanduser()
    try:
        before = requested.resolve(strict=True)
        payload, digest = load_regular_file_snapshot(requested, context)
        after = requested.resolve(strict=True)
    except (OSError, ScaleGuardError) as error:
        raise ExperimentProtocolError(f"cannot read {context} {path}: {error}") from error
    if before != after:
        raise ExperimentProtocolError(f"{context} path changed while it was read: {path}")
    return after, payload, len(payload), digest


def _load_json_snapshot(
    path: Path,
    context: str,
) -> tuple[Path, dict[str, Any], bytes, int, str]:
    resolved, payload, size, digest = _load_regular_snapshot(path, context)
    try:
        document = loads_object(payload)
    except ValueError as error:
        raise ExperimentProtocolError(f"invalid {context} JSON in {path}: {error}") from error
    return resolved, document, payload, size, digest


def load_ablation_protocol(path: Path) -> AblationProtocol:
    """Load the executable four-group protocol with no permissive extensions."""

    resolved, payload, raw = _load_yaml_snapshot(path, "ablation protocol")
    root = _require_mapping(raw, "ablation protocol")
    _require_exact_keys(root, _ROOT_KEYS, "ablation protocol")
    if root["schema_version"] != PROTOCOL_SCHEMA:
        raise ExperimentProtocolError(
            f"ablation protocol schema_version must be {PROTOCOL_SCHEMA!r}"
        )
    name = _require_text(root["name"], "ablation protocol.name")
    if name != PROTOCOL_NAME:
        raise ExperimentProtocolError(
            f"ablation protocol.name must be the fixed protocol {PROTOCOL_NAME!r}"
        )
    if root["status"] != "executable":
        raise ExperimentProtocolError("ablation protocol.status must be 'executable'")
    if root["integration_runner"] != INTEGRATION_RUNNER:
        raise ExperimentProtocolError(
            f"ablation protocol.integration_runner must be {INTEGRATION_RUNNER!r}"
        )

    requirements = _require_mapping(
        root["base_requirements"],
        "ablation protocol.base_requirements",
    )
    _require_exact_keys(
        requirements,
        set(_BASE_REQUIREMENTS),
        "ablation protocol.base_requirements",
    )
    if requirements != _BASE_REQUIREMENTS:
        raise ExperimentProtocolError(
            "ablation protocol.base_requirements disagrees with the executable contract"
        )

    raw_groups = root["groups"]
    if not isinstance(raw_groups, list) or len(raw_groups) != len(_GROUP_CONTRACT):
        raise ExperimentProtocolError(
            "ablation protocol.groups must contain A-only, B-only, AB-fixed, and ScaleGuard"
        )
    groups: list[GroupSpec] = []
    for index, (raw_group, expected) in enumerate(zip(raw_groups, _GROUP_CONTRACT, strict=True)):
        context = f"ablation protocol.groups[{index}]"
        group = _require_mapping(raw_group, context)
        _require_exact_keys(group, _GROUP_KEYS, context)
        description = _require_text(group["description"], f"{context}.description")
        for key, expected_value in expected.items():
            if group[key] != expected_value or type(group[key]) is not type(expected_value):
                raise ExperimentProtocolError(f"{context}.{key} must be {expected_value!r}")
        groups.append(
            GroupSpec(
                id=group["id"],
                description=description,
                fourkagent_mode=group["fourkagent_mode"],
                coz_mode=group["coz_mode"],
                target_factor=group["target_factor"],
                max_coz_steps=group["max_coz_steps"],
                acceptance_policy=group["acceptance_policy"],
                comparison_resolution=group["comparison_resolution"],
            )
        )

    paired = _require_mapping(
        root["paired_requirements"],
        "ablation protocol.paired_requirements",
    )
    _require_exact_keys(
        paired,
        _PAIRED_REQUIREMENT_KEYS,
        "ablation protocol.paired_requirements",
    )
    if any(value is not True for value in paired.values()):
        raise ExperimentProtocolError(
            "every ablation protocol.paired_requirements value must be true"
        )

    raw_metrics = _require_mapping(root["metrics"], "ablation protocol.metrics")
    _require_exact_keys(raw_metrics, _METRIC_KEYS, "ablation protocol.metrics")
    metrics: dict[str, tuple[str, ...]] = {}
    for metric_family, raw_names in raw_metrics.items():
        if (
            not isinstance(raw_names, list)
            or not raw_names
            or any(type(item) is not str or not item for item in raw_names)
            or len(set(raw_names)) != len(raw_names)
        ):
            raise ExperimentProtocolError(
                f"ablation protocol.metrics.{metric_family} must be a non-empty unique string list"
            )
        metrics[metric_family] = tuple(raw_names)
    if metrics != _METRIC_CONTRACT:
        raise ExperimentProtocolError(
            "ablation protocol.metrics disagrees with the executable metric contract"
        )

    raw_notes = root["notes"]
    if (
        not isinstance(raw_notes, list)
        or not raw_notes
        or any(type(note) is not str or not note for note in raw_notes)
    ):
        raise ExperimentProtocolError("ablation protocol.notes must be a non-empty string list")
    return AblationProtocol(
        path=resolved,
        size_bytes=len(payload),
        sha256=_sha256_bytes(payload),
        name=name,
        integration_runner=INTEGRATION_RUNNER,
        groups=tuple(groups),
        paired_requirements=dict(paired),
        metrics=metrics,
        notes=tuple(raw_notes),
    )


def _load_base_config(path: Path) -> tuple[Path, int, str, dict[str, Any]]:
    resolved, payload, raw = _load_yaml_snapshot(path, "base config")
    root = _require_mapping(raw, "base config")
    _require_exact_keys(root, _BASE_CONFIG_SECTIONS, "base config")
    sections = {
        name: _require_mapping(root[name], f"base config.{name}") for name in _BASE_CONFIG_SECTIONS
    }
    controller = sections["controller"]
    target = _require_integer(
        controller.get("target_factor"),
        "base config.controller.target_factor",
    )
    maximum = _require_integer(
        controller.get("max_coz_steps"),
        "base config.controller.max_coz_steps",
    )
    if target != 4:
        raise ExperimentProtocolError(
            "base config.controller.target_factor must be 4 for paired ablations"
        )
    if maximum < 1:
        raise ExperimentProtocolError("base config.controller.max_coz_steps must be at least 1")
    if sections["fourkagent"].get("mode") != "upstream":
        raise ExperimentProtocolError("base config.fourkagent.mode must be 'upstream'")
    if sections["coz"].get("mode") != "persistent":
        raise ExperimentProtocolError("base config.coz.mode must be 'persistent'")
    try:
        load_config(resolved)
    except ScaleGuardError as error:
        raise ExperimentProtocolError(f"invalid base config {resolved}: {error}") from error
    return resolved, len(payload), _sha256_bytes(payload), root


def _hash_regular_file(path: Path, context: str) -> tuple[Path, int, str]:
    try:
        resolved = path.expanduser().resolve(strict=True)
        with resolved.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ExperimentProtocolError(f"{context} is not a regular file: {resolved}")
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            after = os.fstat(handle.fileno())
    except OSError as error:
        raise ExperimentProtocolError(f"cannot read {context} {path}: {error}") from error
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after:
        raise ExperimentProtocolError(f"{context} changed while it was hashed: {resolved}")
    return resolved, before.st_size, digest.hexdigest()


def _clean_commit(project_root: Path, context: str) -> str:
    try:
        return require_clean_git_commit(project_root)
    except ScaleGuardError as error:
        raise ExperimentProtocolError(
            f"{context} requires the frozen clean project HEAD: {error}"
        ) from error


def _verify_file_evidence(
    evidence: Mapping[str, Any],
    context: str,
) -> list[str]:
    try:
        _, size, digest = _hash_regular_file(Path(str(evidence["path"])), context)
    except (ExperimentProtocolError, KeyError) as error:
        return [f"{context}_unreadable:{error}"]
    issues = []
    if size != evidence.get("size_bytes"):
        issues.append(f"{context}_size_changed")
    if digest != evidence.get("sha256"):
        issues.append(f"{context}_sha256_changed")
    return issues


def _tree_inventory(root: Path) -> tuple[list[dict[str, Any]], str]:
    records: list[dict[str, Any]] = []
    if not root.is_dir() or root.is_symlink():
        raise ExperimentProtocolError(f"run root is missing or unsafe: {root}")
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ExperimentProtocolError(f"run artifact must not be a symlink: {path}")
        if not path.is_file():
            continue
        _, size, digest = _hash_regular_file(path, "run artifact")
        records.append(
            {
                "path": str(path.relative_to(root)),
                "size_bytes": size,
                "sha256": digest,
            }
        )
    if not records:
        raise ExperimentProtocolError(f"run root contains no artifacts: {root}")
    return records, _canonical_sha256({"files": records})


def _load_inputs(paths: Sequence[Path]) -> tuple[_InputSource, ...]:
    if not paths:
        raise ExperimentProtocolError("at least one --input is required")
    inputs: list[_InputSource] = []
    for index, requested in enumerate(paths):
        resolved, size, digest = _hash_regular_file(
            requested,
            f"input[{index}]",
        )
        suffix = resolved.suffix.lower()
        if (
            not suffix
            or len(suffix) > 16
            or any(character not in ".abcdefghijklmnopqrstuvwxyz0123456789" for character in suffix)
        ):
            suffix = ".input"
        inputs.append(
            _InputSource(
                requested_path=str(requested),
                path=resolved,
                size_bytes=size,
                sha256=digest,
                suffix=suffix,
            )
        )
    return tuple(inputs)


def _validate_seeds(
    raw_seeds: Sequence[int] | None,
    base_config: Mapping[str, Any],
) -> tuple[int, ...]:
    if raw_seeds:
        seeds = tuple(raw_seeds)
    else:
        coz = _require_mapping(base_config["coz"], "base config.coz")
        seeds = (_require_integer(coz.get("seed"), "base config.coz.seed"),)
    for seed in seeds:
        if type(seed) is not int or not 0 <= seed <= 2**63 - 1:
            raise ExperimentProtocolError("every seed must be an integer between 0 and 2^63-1")
    if len(set(seeds)) != len(seeds):
        raise ExperimentProtocolError("duplicate seeds would create duplicate sample IDs")
    return seeds


def _sample_id(input_sha256: str, seed: int) -> str:
    sample_id = f"{input_sha256[:16]}-s{seed}"
    if (
        len(sample_id) > 128
        or sample_id[0] not in "abcdefghijklmnopqrstuvwxyz0123456789"
        or any(character not in _SAFE_SAMPLE_ID_CHARS for character in sample_id)
    ):
        raise ExperimentProtocolError(f"generated an unsafe sample ID: {sample_id!r}")
    return sample_id


def _validate_sample_ids(
    inputs: Sequence[_InputSource],
    seeds: Sequence[int],
) -> dict[tuple[str, int], str]:
    by_id: dict[str, tuple[str, int]] = {}
    samples: dict[tuple[str, int], str] = {}
    for source in inputs:
        for seed in seeds:
            sample_id = _sample_id(source.sha256, seed)
            identity = (source.sha256, seed)
            if sample_id in by_id:
                raise ExperimentProtocolError(
                    "duplicate sample ID "
                    f"{sample_id!r} for {source.path} and input SHA-256 {source.sha256}"
                )
            by_id[sample_id] = identity
            samples[identity] = sample_id
    return samples


def _safe_output_directory(path: Path, project_root: Path) -> Path:
    requested = path.expanduser()
    if requested.exists() and requested.is_symlink():
        raise ExperimentProtocolError(f"output directory must not be a symlink: {requested}")
    try:
        resolved = requested.resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise ExperimentProtocolError(
            f"output directory cannot be resolved safely: {path}: {error}"
        ) from error
    dangerous = {
        Path("/").resolve(),
        Path.home().resolve(),
        Path.cwd().resolve(),
        project_root.resolve(),
    }
    if resolved in dangerous:
        raise ExperimentProtocolError(f"refusing dangerous output directory: {resolved}")
    if resolved.exists():
        if not resolved.is_dir():
            raise ExperimentProtocolError(f"output path is not a directory: {resolved}")
        try:
            if next(resolved.iterdir(), None) is not None:
                raise ExperimentProtocolError(f"output directory must be empty: {resolved}")
        except OSError as error:
            raise ExperimentProtocolError(
                f"cannot inspect output directory {resolved}: {error}"
            ) from error
    return resolved


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _copy_input_snapshot(source: _InputSource, destination: Path) -> None:
    try:
        with source.path.open("rb") as handle:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
            )
            temporary = Path(temporary_name)
            digest = hashlib.sha256()
            size = 0
            try:
                with os.fdopen(descriptor, "wb") as output:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                        size += len(chunk)
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                if size != source.size_bytes or digest.hexdigest() != source.sha256:
                    raise ExperimentProtocolError(
                        f"input changed before snapshotting: {source.path}"
                    )
                os.replace(temporary, destination)
                _fsync_directory(destination.parent)
            finally:
                temporary.unlink(missing_ok=True)
    except OSError as error:
        raise ExperimentProtocolError(f"cannot snapshot input {source.path}: {error}") from error


def _group_slug(group_id: str) -> str:
    return group_id.lower().replace(" ", "-")


def _build_group_config(
    base_config: Mapping[str, Any],
    *,
    group: GroupSpec,
    sample_id: str,
    seed: int,
    run_root: Path,
) -> dict[str, Any]:
    generated = copy.deepcopy(dict(base_config))
    runtime = _require_mapping(generated["runtime"], "generated config.runtime")
    fourkagent = _require_mapping(
        generated["fourkagent"],
        "generated config.fourkagent",
    )
    coz = _require_mapping(generated["coz"], "generated config.coz")
    controller = _require_mapping(
        generated["controller"],
        "generated config.controller",
    )
    runtime["run_root"] = str(run_root)
    runtime["experiment_group"] = group.id
    runtime["experiment_sample_id"] = sample_id
    fourkagent["mode"] = group.fourkagent_mode
    coz["mode"] = group.coz_mode
    coz["seed"] = seed
    controller["target_factor"] = group.target_factor
    controller["max_coz_steps"] = group.max_coz_steps
    controller["acceptance_policy"] = group.acceptance_policy
    return generated


def _config_bytes(config: Mapping[str, Any]) -> bytes:
    return yaml.safe_dump(
        dict(config),
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    ).encode("utf-8")


def _receipt_counts(jobs: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "total": len(jobs),
        "planned": sum(job["status"] == "planned" for job in jobs),
        "running": sum(job["status"] == "running" for job in jobs),
        "passed": sum(job["status"] == "passed" for job in jobs),
        "failed": sum(job["status"] == "failed" for job in jobs),
    }


def _write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    receipt["counts"] = _receipt_counts(receipt["jobs"])
    unsigned = copy.deepcopy(receipt)
    unsigned.pop("receipt_sha256", None)
    receipt["receipt_sha256"] = _canonical_sha256(unsigned)
    payload = (
        json.dumps(
            receipt,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _atomic_write(path, payload)


def _default_command_runner(argv: Sequence[str], cwd: Path) -> int:
    try:
        config_index = argv.index("--config") + 1
        config_path = Path(argv[config_index])
    except (ValueError, IndexError) as error:
        raise ExperimentProtocolError("fixed experiment command has no runtime config") from error
    config = load_config(config_path)
    scheduler_name = config.fourkagent.api_key_env
    runtime_root = cwd / ".runtime"
    if runtime_root.is_symlink():
        raise ExperimentProtocolError(
            f"experiment runtime root must not be a symlink: {runtime_root}"
        )
    runtime_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    runtime_identity = runtime_root.stat()
    if not stat.S_ISDIR(runtime_identity.st_mode) or runtime_identity.st_uid != os.geteuid():
        raise ExperimentProtocolError(
            f"experiment runtime root must be an owned directory: {runtime_root}"
        )
    runtime_root.chmod(0o700)
    isolated_home = Path(tempfile.mkdtemp(prefix="experiment-home-", dir=runtime_root))
    isolated_home.chmod(0o700)
    environment = {
        name: os.environ[name] for name in _EXPERIMENT_ENVIRONMENT_NAMES if name in os.environ
    }
    environment.update(
        {
            "HOME": str(isolated_home.resolve()),
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_SYSTEM": os.devnull,
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "PATH": _FIXED_SYSTEM_PATH,
            "PIP_CONFIG_FILE": os.devnull,
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TMPDIR": "/tmp",
            "UV_NO_CONFIG": "1",
        }
    )
    if config.fourkagent.mode == "upstream":
        scheduler_value = os.environ.get(scheduler_name)
        if scheduler_value:
            environment[scheduler_name] = scheduler_value
    completed = subprocess.run(
        ["/bin/bash", "-p", *argv],
        cwd=cwd,
        check=False,
        env=environment,
    )
    return completed.returncode


def _manifest_binding_issues(
    manifest: Mapping[str, Any],
    *,
    group: GroupSpec,
    sample_id: str,
    seed: int,
    project_commit: str,
) -> list[str]:
    issues: list[str] = []
    config = manifest.get("config")
    if not isinstance(config, dict):
        return ["manifest_config_missing"]
    runtime = config.get("runtime")
    fourkagent = config.get("fourkagent")
    coz = config.get("coz")
    controller = config.get("controller")
    if not isinstance(runtime, dict):
        issues.append("manifest_runtime_config_missing")
    else:
        if runtime.get("experiment_group") != group.id:
            issues.append("manifest_experiment_group_mismatch")
        if runtime.get("experiment_sample_id") != sample_id:
            issues.append("manifest_experiment_sample_id_mismatch")
    if not isinstance(fourkagent, dict) or fourkagent.get("mode") != group.fourkagent_mode:
        issues.append("manifest_fourkagent_mode_mismatch")
    if not isinstance(coz, dict) or type(coz.get("seed")) is not int or coz.get("seed") != seed:
        issues.append("manifest_coz_seed_mismatch")
    elif coz.get("mode") != group.coz_mode:
        issues.append("manifest_coz_mode_mismatch")
    if not isinstance(controller, dict):
        issues.append("manifest_controller_config_missing")
    else:
        expected_controller = {
            "target_factor": group.target_factor,
            "max_coz_steps": group.max_coz_steps,
            "acceptance_policy": group.acceptance_policy,
        }
        for name, expected in expected_controller.items():
            if controller.get(name) != expected or type(controller.get(name)) is not type(expected):
                issues.append(f"manifest_controller_{name}_mismatch")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("project_commit") != project_commit:
        issues.append("manifest_project_commit_mismatch")
    issues.extend(manifest_experiment_issues(manifest, group.id))
    return issues


def _successful_process(value: Any) -> bool:
    return isinstance(value, dict) and value.get("returncode") == 0


def _persistent_candidate(step: Any) -> bool:
    if not isinstance(step, dict):
        return False
    candidate = step.get("candidate")
    metadata = step.get("worker_metadata")
    return (
        isinstance(candidate, dict)
        and candidate.get("mock") is False
        and type(candidate.get("sha256")) is str
        and isinstance(metadata, dict)
        and metadata.get("backend") == "chain_of_zoom_persistent"
        and metadata.get("candidate_sha256") == candidate.get("sha256")
    )


def _has_final_gate_rollback(manifest: Mapping[str, Any]) -> bool:
    events = manifest.get("events")
    final_metrics = manifest.get("final_metrics")
    selected_scale = (
        final_metrics.get("selected_scale") if isinstance(final_metrics, dict) else None
    )
    return (
        isinstance(events, list)
        and any(
            isinstance(event, dict) and event.get("event") == "final_gate_rollback"
            for event in events
        )
        and isinstance(selected_scale, (int, float))
        and not isinstance(selected_scale, bool)
        and selected_scale < 4
    )


def manifest_experiment_issues(
    manifest: Mapping[str, Any],
    group_id: str,
) -> list[str]:
    """Return exact runtime-contract issues for one declared ablation group."""

    if group_id not in EXPERIMENT_GROUP_SEMANTICS:
        return ["manifest_experiment_group_undeclared"]
    issues: list[str] = []
    if manifest.get("mock") is not False:
        issues.append("manifest_experiment_mock")
    steps = manifest.get("steps")
    if not isinstance(steps, list):
        return [*issues, "manifest_experiment_steps_missing"]
    provenance = manifest.get("provenance")
    expected_restoration_backend = (
        "scaleguard_identity_observation" if group_id == "B-only" else "4kagent_upstream"
    )
    if (
        not isinstance(provenance, dict)
        or provenance.get("restoration_backend") != expected_restoration_backend
        or provenance.get("scale_backend") != "chain_of_zoom"
    ):
        issues.append("manifest_experiment_backend_unverified")
    restoration_metadata = manifest.get("restoration_metadata")
    restoration_process = manifest.get("restoration_process")
    if group_id == "B-only":
        if (
            not isinstance(restoration_metadata, dict)
            or restoration_metadata.get("backend") != "scaleguard_identity_observation"
            or restoration_metadata.get("algorithmic_restoration") is not False
            or restoration_process is not None
        ):
            issues.append("manifest_identity_observation_unverified")
    elif (
        not isinstance(restoration_metadata, dict)
        or restoration_metadata.get("backend") != "4kagent_upstream"
        or not _successful_process(restoration_process)
    ):
        issues.append("manifest_fourkagent_process_unverified")

    status = manifest.get("status")
    completion = manifest.get("completion_level")
    target_reached = manifest.get("target_reached")
    scale_session = manifest.get("scale_session_process")
    if group_id == "A-only":
        if steps:
            issues.append("manifest_unexpected_coz_step")
        if scale_session is not None:
            issues.append("manifest_unexpected_coz_session")
        if status != "succeeded" or completion != "STATIC_READY" or target_reached is not True:
            issues.append("manifest_experiment_outcome_invalid")
        return issues

    if len(steps) != 1:
        issues.append(f"manifest_coz_step_count:{len(steps)}")
        return issues
    step = steps[0]
    if not _persistent_candidate(step):
        issues.append("manifest_coz_candidate_unverified")
    if not _successful_process(scale_session):
        issues.append("manifest_coz_session_process_unverified")
    if not isinstance(step, dict):
        issues.append("manifest_experiment_outcome_invalid")
        return issues

    if group_id in {"B-only", "AB-fixed"}:
        if (
            status != "succeeded"
            or completion != "STATIC_READY"
            or target_reached is not True
            or step.get("accepted") is not True
        ):
            issues.append("manifest_experiment_outcome_invalid")
        return issues

    integrated = (
        status == "succeeded"
        and completion == "AB_INTEGRATED"
        and target_reached is True
        and step.get("accepted") is True
    )
    rejected_candidate = step.get("accepted") is False and step.get("decision") in {
        "stop",
        "rollback",
    }
    post_final_gate_rollback = (
        step.get("accepted") is True
        and step.get("decision") == "stop"
        and _has_final_gate_rollback(manifest)
    )
    rolled_back = (
        status == "succeeded_with_rollback"
        and completion == "COMPONENT_REPRODUCED"
        and target_reached is False
        and (rejected_candidate or post_final_gate_rollback)
    )
    if not integrated and not rolled_back:
        issues.append("manifest_scaleguard_outcome_invalid")
    return issues


def _inspect_manifest(
    run_root: Path,
    *,
    group: GroupSpec,
    sample_id: str,
    seed: int,
    project_commit: str,
) -> tuple[dict[str, Any] | None, list[str], dict[str, Any] | None]:
    candidates = sorted(run_root.glob("*/manifest.json"))
    if not candidates:
        return None, ["manifest_missing"], None
    if len(candidates) != 1:
        return None, [f"manifest_count:{len(candidates)}"], None
    path = candidates[0]
    try:
        before_files, before_digest = _tree_inventory(run_root)
        resolved, snapshot, _payload, size, digest = _load_json_snapshot(
            path,
            "run manifest",
        )
        manifest = validate_run_manifest(resolved)
        if manifest != snapshot:
            raise ExperimentProtocolError(
                f"run manifest changed while it was validated: {resolved}"
            )
        after_files, after_digest = _tree_inventory(run_root)
    except (ExperimentProtocolError, OSError, ScaleGuardError, ValueError) as error:
        return None, [f"manifest_invalid:{type(error).__name__}:{error}"], None
    if before_files != after_files or before_digest != after_digest:
        return (
            None,
            ["run_artifacts_changed_during_manifest_validation"],
            None,
        )
    issues = _manifest_binding_issues(
        manifest,
        group=group,
        sample_id=sample_id,
        seed=seed,
        project_commit=project_commit,
    )
    return (
        {
            "path": str(resolved),
            "size_bytes": size,
            "sha256": digest,
            "artifact_count": len(after_files),
            "artifact_inventory_sha256": after_digest,
        },
        issues,
        manifest,
    )


def _attempt_file_entry(
    raw: Any,
    *,
    role: str,
    attempt_dir: Path,
) -> tuple[dict[str, Any] | None, bytes | None, list[str]]:
    context = f"wrapper attempt files.{role}"
    try:
        entry = _require_mapping(raw, context)
        _require_exact_keys(entry, {"path", "size_bytes", "sha256"}, context)
        path_text = _require_text(entry["path"], f"{context}.path")
        size = _require_integer(entry["size_bytes"], f"{context}.size_bytes")
        digest = entry["sha256"]
        if size < 0 or type(digest) is not str or _DIGEST_PATTERN.fullmatch(digest) is None:
            raise ExperimentProtocolError(f"{context} has invalid size or SHA-256")
        declared = Path(path_text)
        if not declared.is_absolute() or declared.is_symlink():
            raise ExperimentProtocolError(f"{context}.path must be an absolute regular file")
        expected = attempt_dir / _ATTEMPT_FILE_RELATIVE_PATHS[role]
        payload: bytes | None = None
        if role in _PARSED_ATTEMPT_FILE_ROLES:
            resolved, payload, observed_size, observed_digest = _load_regular_snapshot(
                declared,
                context,
            )
        else:
            resolved, observed_size, observed_digest = _hash_regular_file(
                declared,
                context,
            )
        if resolved != expected.resolve(strict=True):
            raise ExperimentProtocolError(
                f"{context}.path is not the fixed attempt artifact {expected}"
            )
        if observed_size != size or observed_digest != digest:
            raise ExperimentProtocolError(f"{context} byte identity changed")
    except (ExperimentProtocolError, KeyError, OSError) as error:
        return None, None, [f"wrapper_attempt_file_invalid:{role}:{error}"]
    return (
        {
            "path": str(resolved),
            "size_bytes": size,
            "sha256": digest,
        },
        payload,
        [],
    )


def _validate_attempt_inventory(
    attempt_dir: Path,
    inventory_entry: Mapping[str, Any],
    inventory_payload: bytes,
) -> tuple[str | None, list[str]]:
    try:
        inventory_path = Path(str(inventory_entry["path"])).resolve(strict=True)
        inventory = loads_object(inventory_payload)
        _require_exact_keys(
            inventory,
            {"schema_version", "root", "files"},
            "wrapper attempt files inventory",
        )
        if inventory["schema_version"] != 1 or inventory["root"] != attempt_dir.name:
            raise ExperimentProtocolError(
                "wrapper attempt files inventory has the wrong schema or root"
            )
        raw_files = inventory["files"]
        if not isinstance(raw_files, list):
            raise ExperimentProtocolError("wrapper attempt files inventory.files must be a list")
        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        for index, raw_record in enumerate(raw_files):
            context = f"wrapper attempt files inventory.files[{index}]"
            record = _require_mapping(raw_record, context)
            _require_exact_keys(record, {"path", "size_bytes", "sha256"}, context)
            relative_text = _require_text(record["path"], f"{context}.path")
            relative = Path(relative_text)
            if (
                relative.is_absolute()
                or relative_text != relative.as_posix()
                or any(part in {"", ".", ".."} for part in relative.parts)
                or relative_text == "files.json"
                or relative_text in seen
            ):
                raise ExperimentProtocolError(f"{context}.path is unsafe or duplicated")
            seen.add(relative_text)
            size = _require_integer(record["size_bytes"], f"{context}.size_bytes")
            digest = record["sha256"]
            if size < 0 or type(digest) is not str or _DIGEST_PATTERN.fullmatch(digest) is None:
                raise ExperimentProtocolError(f"{context} has invalid byte identity")
            _, observed_size, observed_digest = _hash_regular_file(
                attempt_dir / relative,
                context,
            )
            if observed_size != size or observed_digest != digest:
                raise ExperimentProtocolError(f"{context} byte identity changed")
            records.append(
                {
                    "path": relative_text,
                    "size_bytes": size,
                    "sha256": digest,
                }
            )

        actual: set[str] = set()
        for path in sorted(attempt_dir.rglob("*")):
            if path.is_symlink():
                raise ExperimentProtocolError(f"wrapper attempt contains a symlink: {path}")
            if path.is_file() and path.resolve() != inventory_path:
                actual.add(path.relative_to(attempt_dir).as_posix())
        if actual != seen:
            missing = sorted(actual - seen)
            stale = sorted(seen - actual)
            raise ExperimentProtocolError(
                "wrapper attempt files inventory coverage mismatch: "
                f"unlisted={missing}, missing={stale}"
            )
    except (ExperimentProtocolError, KeyError, OSError, ValueError) as error:
        return None, [f"wrapper_attempt_inventory_invalid:{error}"]
    return _canonical_sha256({"files": records}), []


def _validate_attempt_hardware(
    hardware_raw: Any,
    *,
    gpu_preflight_payload: bytes,
    gpu_inventory_payload: bytes,
    project_commit: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        hardware = _require_mapping(hardware_raw, "wrapper attempt.hardware")
        _require_exact_keys(
            hardware,
            {
                "identity_sha256",
                "class_sha256",
                "selected_gpu_count",
                "cuda_visible_devices",
            },
            "wrapper attempt.hardware",
        )
        identity_digest = hardware["identity_sha256"]
        class_digest = hardware["class_sha256"]
        if (
            type(identity_digest) is not str
            or _DIGEST_PATTERN.fullmatch(identity_digest) is None
            or type(class_digest) is not str
            or _DIGEST_PATTERN.fullmatch(class_digest) is None
        ):
            raise ExperimentProtocolError(
                "wrapper attempt.hardware digests must be lowercase SHA-256"
            )
        selected_count = _require_integer(
            hardware["selected_gpu_count"],
            "wrapper attempt.hardware.selected_gpu_count",
        )
        cuda_visible = hardware["cuda_visible_devices"]
        if cuda_visible is not None and type(cuda_visible) is not str:
            raise ExperimentProtocolError(
                "wrapper attempt.hardware.cuda_visible_devices must be string or null"
            )
        gpu_preflight = loads_object(gpu_preflight_payload)
        if (
            gpu_preflight.get("schema_version") != 1
            or gpu_preflight.get("status") != "passed"
            or gpu_preflight.get("git_commit") != project_commit
            or gpu_preflight.get("cuda_visible_devices") != cuda_visible
        ):
            raise ExperimentProtocolError(
                "wrapper attempt GPU preflight is not passed or commit-bound"
            )
        selected = gpu_preflight.get("selected_gpus")
        if not isinstance(selected, list) or len(selected) != selected_count:
            raise ExperimentProtocolError("wrapper attempt selected GPU count is inconsistent")
        normalized: list[dict[str, Any]] = []
        required_gpu_fields = {
            "logical_index",
            "physical_index",
            "uuid",
            "name",
            "memory_total_mib",
            "driver_version",
        }
        for index, raw_gpu in enumerate(selected):
            gpu = _require_mapping(raw_gpu, f"selected_gpus[{index}]")
            _require_exact_keys(gpu, required_gpu_fields, f"selected_gpus[{index}]")
            if (
                type(gpu["logical_index"]) is not int
                or gpu["logical_index"] != index
                or type(gpu["physical_index"]) is not str
                or type(gpu["uuid"]) is not str
                or type(gpu["name"]) is not str
                or type(gpu["memory_total_mib"]) is not int
                or gpu["memory_total_mib"] <= 0
                or type(gpu["driver_version"]) is not str
            ):
                raise ExperimentProtocolError(
                    f"selected_gpus[{index}] has invalid hardware identity"
                )
            normalized.append(dict(gpu))
        requirements = gpu_preflight.get("requirements")
        minimum = requirements.get("minimum_gpu_count") if isinstance(requirements, dict) else None
        if type(minimum) is not int or selected_count < minimum:
            raise ExperimentProtocolError(
                "wrapper attempt hardware does not meet its GPU-count requirement"
            )

        inventory_text = gpu_inventory_payload.decode("utf-8", errors="strict")
        rows = list(csv.reader(io.StringIO(inventory_text, newline="")))
        inventory: dict[str, tuple[str, str, int, str]] = {}
        for row in rows:
            if len(row) != 5:
                raise ExperimentProtocolError(
                    "wrapper attempt gpu_inventory.csv has a malformed row"
                )
            physical_index, uuid, name, memory_text, driver = (value.strip() for value in row)
            if physical_index in inventory:
                raise ExperimentProtocolError(
                    "wrapper attempt gpu_inventory.csv duplicates a physical index"
                )
            try:
                memory = int(float(memory_text))
            except ValueError as error:
                raise ExperimentProtocolError(
                    "wrapper attempt gpu_inventory.csv has invalid memory"
                ) from error
            inventory[physical_index] = (uuid, name, memory, driver)
        for gpu in normalized:
            observed = inventory.get(gpu["physical_index"])
            expected = (
                gpu["uuid"],
                gpu["name"],
                gpu["memory_total_mib"],
                gpu["driver_version"],
            )
            if observed != expected:
                raise ExperimentProtocolError(
                    "wrapper attempt selected GPU identity disagrees with raw inventory"
                )

        identity_payload = {
            "cuda_visible_devices": cuda_visible,
            "selected_gpus": normalized,
        }
        class_payload = {
            "selected_gpus": [
                {
                    "logical_index": gpu["logical_index"],
                    "name": gpu["name"],
                    "memory_total_mib": gpu["memory_total_mib"],
                    "driver_version": gpu["driver_version"],
                }
                for gpu in normalized
            ]
        }
        if (
            _wrapper_canonical_sha256(identity_payload) != identity_digest
            or _wrapper_canonical_sha256(class_payload) != class_digest
        ):
            raise ExperimentProtocolError(
                "wrapper attempt hardware digest disagrees with GPU preflight"
            )
    except (csv.Error, ExperimentProtocolError, KeyError, OSError, ValueError) as error:
        return None, [f"wrapper_attempt_hardware_invalid:{error}"]
    return {
        "identity_sha256": identity_digest,
        "class_sha256": class_digest,
        "selected_gpu_count": selected_count,
        "cuda_visible_devices": cuda_visible,
        "selected_gpus": normalized,
    }, []


def _validate_attempt_execution(
    execution_payload: bytes,
    runtime_preflight_entry: Mapping[str, Any],
    runtime_preflight_payload: bytes,
    *,
    job: Mapping[str, Any],
    project_commit: str,
) -> list[str]:
    try:
        execution = loads_object(execution_payload)
        inputs = _require_mapping(
            execution.get("inputs"),
            "wrapper attempt execution.inputs",
        )
        config = _require_mapping(
            inputs.get("runtime_config"),
            "wrapper attempt execution.inputs.runtime_config",
        )
        input_image = _require_mapping(
            inputs.get("input_image"),
            "wrapper attempt execution.inputs.input_image",
        )
        model_evidence = execution.get("model_evidence")
        outputs = execution.get("outputs")
        if (
            execution.get("schema_version") != 1
            or execution.get("stage") != "experiment"
            or execution.get("status") != "passed"
            or execution.get("return_code") != 0
            or execution.get("scaleguard_command_return_code") != 0
            or execution.get("git_commit") != project_commit
            or config.get("sha256") != job["config"]["sha256"]
            or config.get("size_bytes") != job["config"]["size_bytes"]
            or input_image.get("sha256") != job["input"]["sha256"]
            or input_image.get("size_bytes") != job["input"]["size_bytes"]
            or not isinstance(model_evidence, dict)
            or model_evidence.get("complete") is not True
            or model_evidence.get("helper_completed") is not True
            or model_evidence.get("hashes_consistent") is not True
            or not isinstance(outputs, list)
            or len(outputs) != 1
        ):
            raise ExperimentProtocolError(
                "wrapper attempt execution.json is not passed or job-bound"
            )
        config_path = Path(_require_text(config.get("path"), "execution config path"))
        input_path = Path(_require_text(input_image.get("path"), "execution input image path"))
        if (
            not config_path.is_absolute()
            or config_path.is_symlink()
            or config_path.resolve(strict=True)
            != Path(str(job["config"]["path"])).resolve(strict=True)
            or not input_path.is_absolute()
            or input_path.is_symlink()
            or input_path.resolve(strict=True)
            != Path(str(job["input"]["path"])).resolve(strict=True)
        ):
            raise ExperimentProtocolError("wrapper attempt execution input paths are not job-bound")
        runtime_preflight_input = _require_mapping(
            inputs.get("runtime_preflight"),
            "wrapper attempt execution.inputs.runtime_preflight",
        )
        if (
            runtime_preflight_input.get("sha256") != runtime_preflight_entry["sha256"]
            or runtime_preflight_input.get("size_bytes") != runtime_preflight_entry["size_bytes"]
            or Path(
                _require_text(
                    runtime_preflight_input.get("path"),
                    "execution runtime preflight path",
                )
            ).resolve(strict=True)
            != Path(str(runtime_preflight_entry["path"])).resolve(strict=True)
        ):
            raise ExperimentProtocolError(
                "wrapper attempt execution runtime preflight is not pointer-bound"
            )
        output_record = _require_mapping(
            outputs[0],
            "wrapper attempt execution.outputs[0]",
        )
        _require_exact_keys(
            output_record,
            {"path", "size_bytes", "sha256"},
            "wrapper attempt execution.outputs[0]",
        )
        output_path = Path(
            _require_text(
                output_record["path"],
                "wrapper attempt execution.outputs[0].path",
            )
        )
        if not output_path.is_absolute() or output_path.is_symlink():
            raise ExperimentProtocolError("wrapper attempt execution output path is unsafe")
        resolved_output, output_size, output_sha256 = _hash_regular_file(
            output_path,
            "wrapper attempt execution output",
        )
        if (
            resolved_output != Path(str(job["output_path"])).resolve(strict=True)
            or output_record["size_bytes"] != output_size
            or output_record["sha256"] != output_sha256
        ):
            raise ExperimentProtocolError("wrapper attempt execution output is not job-bound")
        preflight = loads_object(runtime_preflight_payload)
        if (
            preflight.get("schema_version") != 2
            or preflight.get("status") != "passed"
            or preflight.get("project_commit") != project_commit
        ):
            raise ExperimentProtocolError(
                "wrapper attempt runtime preflight is not passed or commit-bound"
            )
    except (ExperimentProtocolError, KeyError, OSError, ValueError) as error:
        return [f"wrapper_attempt_execution_invalid:{error}"]
    return []


def _validate_attempt_system_evidence(
    execution_payload: bytes,
    gpu_samples_payload: bytes,
    hardware: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        execution = loads_object(execution_payload)
        duration_seconds = _require_integer(
            execution.get("duration_seconds"),
            "wrapper attempt execution.duration_seconds",
        )
        if duration_seconds < 0:
            raise ExperimentProtocolError(
                "wrapper attempt execution.duration_seconds must be non-negative"
            )
        sampling = _require_mapping(
            execution.get("gpu_sampling"),
            "wrapper attempt execution.gpu_sampling",
        )
        _require_exact_keys(
            sampling,
            {
                "sample_count",
                "sample_interval_seconds",
                "window_started_at_utc",
                "window_completed_at_utc",
                "window_duration_seconds",
                "boundary_tolerance_seconds",
                "maximum_gap_tolerance_seconds",
                "maximum_observed_gap_seconds",
                "temporal_coverage_complete",
                "minimum_gpu_count",
                "preflight_receipt_bound",
                "inventory_binding_complete",
                "workload_sampling_complete",
                "workload_observed_by_uuid",
                "workload_samples_by_uuid",
                "attribution_scope",
                "evidence_complete",
                "peak_by_physical_index",
                "raw_csv",
            },
            "wrapper attempt execution.gpu_sampling",
        )
        sample_count = _require_integer(
            sampling["sample_count"],
            "wrapper attempt execution.gpu_sampling.sample_count",
        )
        minimum_gpu_count = _require_integer(
            sampling["minimum_gpu_count"],
            "wrapper attempt execution.gpu_sampling.minimum_gpu_count",
        )
        interval = sampling["sample_interval_seconds"]
        if (
            isinstance(interval, bool)
            or not isinstance(interval, (int, float))
            or not math.isfinite(float(interval))
            or not 0.1 <= float(interval) <= 60.0
        ):
            raise ExperimentProtocolError(
                "wrapper attempt GPU sample interval is outside the audited range"
            )
        interval_seconds = float(interval)
        execution_started_text = _require_text(
            execution.get("started_at_utc"),
            "wrapper attempt execution.started_at_utc",
        )
        execution_completed_text = _require_text(
            execution.get("completed_at_utc"),
            "wrapper attempt execution.completed_at_utc",
        )
        window_started_text = _require_text(
            sampling["window_started_at_utc"],
            "wrapper attempt GPU sampling window_started_at_utc",
        )
        window_completed_text = _require_text(
            sampling["window_completed_at_utc"],
            "wrapper attempt GPU sampling window_completed_at_utc",
        )
        if not all(
            value.endswith("Z")
            for value in (
                execution_started_text,
                execution_completed_text,
                window_started_text,
                window_completed_text,
            )
        ):
            raise ExperimentProtocolError(
                "wrapper attempt GPU sampling timestamps must use canonical UTC"
            )
        execution_started = _require_timestamp(
            execution_started_text,
            "wrapper attempt execution.started_at_utc",
        )
        execution_completed = _require_timestamp(
            execution_completed_text,
            "wrapper attempt execution.completed_at_utc",
        )
        window_started = _require_timestamp(
            window_started_text,
            "wrapper attempt GPU sampling window_started_at_utc",
        )
        window_completed = _require_timestamp(
            window_completed_text,
            "wrapper attempt GPU sampling window_completed_at_utc",
        )
        window_duration = sampling["window_duration_seconds"]
        boundary_tolerance = sampling["boundary_tolerance_seconds"]
        maximum_gap_tolerance = sampling["maximum_gap_tolerance_seconds"]
        maximum_observed_gap = sampling["maximum_observed_gap_seconds"]
        temporal_numbers = {
            "window_duration_seconds": window_duration,
            "boundary_tolerance_seconds": boundary_tolerance,
            "maximum_gap_tolerance_seconds": maximum_gap_tolerance,
            "maximum_observed_gap_seconds": maximum_observed_gap,
        }
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in temporal_numbers.values()
        ):
            raise ExperimentProtocolError(
                "wrapper attempt GPU sampling temporal fields must be finite and non-negative"
            )
        expected_window_duration = (window_completed - window_started).total_seconds()
        expected_boundary_tolerance = max(5.0, interval_seconds * 1.5)
        expected_gap_tolerance = max(1.0, interval_seconds * 2.0)
        if (
            expected_window_duration < 0.0
            or execution_completed < execution_started
            or window_started < execution_started - timedelta(seconds=1)
            or window_completed > execution_completed + timedelta(seconds=1)
            or expected_window_duration > duration_seconds + 1.0
            or not math.isclose(
                float(window_duration),
                expected_window_duration,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            or not math.isclose(
                float(boundary_tolerance),
                expected_boundary_tolerance,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                float(maximum_gap_tolerance),
                expected_gap_tolerance,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ExperimentProtocolError(
                "wrapper attempt GPU sampling window is not bound to execution"
            )
        for field in (
            "preflight_receipt_bound",
            "inventory_binding_complete",
            "workload_sampling_complete",
            "temporal_coverage_complete",
            "evidence_complete",
        ):
            if sampling[field] is not True:
                raise ExperimentProtocolError(
                    f"wrapper attempt execution.gpu_sampling.{field} must be true"
                )
        if (
            sampling["attribution_scope"] != "physical_gpu_host_level_not_process_attributed"
            or sampling["raw_csv"] != "gpu-samples.csv"
        ):
            raise ExperimentProtocolError("wrapper attempt GPU sampling scope is invalid")

        selected_raw = hardware.get("selected_gpus")
        selected_count = hardware.get("selected_gpu_count")
        if (
            not isinstance(selected_raw, list)
            or type(selected_count) is not int
            or selected_count != minimum_gpu_count
            or minimum_gpu_count != 2
            or len(selected_raw) != selected_count
        ):
            raise ExperimentProtocolError(
                "wrapper attempt GPU sampling does not cover the selected topology"
            )
        selected_by_uuid: dict[str, dict[str, Any]] = {}
        selected_by_physical: dict[str, dict[str, Any]] = {}
        for raw_gpu in selected_raw:
            gpu = _require_mapping(raw_gpu, "wrapper attempt selected GPU")
            uuid = _require_text(gpu.get("uuid"), "wrapper attempt selected GPU uuid")
            physical = _require_text(
                gpu.get("physical_index"),
                "wrapper attempt selected GPU physical index",
            )
            selected_by_uuid[uuid] = gpu
            selected_by_physical[physical] = gpu
        if len(selected_by_uuid) != 2 or len(selected_by_physical) != 2:
            raise ExperimentProtocolError("wrapper attempt selected GPU identity is duplicated")

        observed_raw = _require_mapping(
            sampling["workload_observed_by_uuid"],
            "wrapper attempt workload observations",
        )
        samples_raw = _require_mapping(
            sampling["workload_samples_by_uuid"],
            "wrapper attempt workload sample counts",
        )
        if set(observed_raw) != set(selected_by_uuid) or set(samples_raw) != set(selected_by_uuid):
            raise ExperimentProtocolError(
                "wrapper attempt workload maps do not match selected GPUs"
            )
        if any(value is not True for value in observed_raw.values()) or any(
            type(value) is not int or value <= 0 for value in samples_raw.values()
        ):
            raise ExperimentProtocolError("wrapper attempt has incomplete GPU workload evidence")

        peaks_raw = _require_mapping(
            sampling["peak_by_physical_index"],
            "wrapper attempt GPU peaks",
        )
        if set(peaks_raw) != set(selected_by_physical):
            raise ExperimentProtocolError("wrapper attempt GPU peaks do not cover selected devices")

        csv_text = gpu_samples_payload.decode("utf-8", errors="strict")
        reader = csv.DictReader(io.StringIO(csv_text, newline=""))
        expected_fields = [
            "timestamp_utc",
            "sample_kind",
            "index",
            "uuid",
            "name",
            "memory_used_mib",
            "memory_total_mib",
            "utilization_gpu_percent",
        ]
        if reader.fieldnames != expected_fields:
            raise ExperimentProtocolError("wrapper attempt gpu-samples.csv header is invalid")
        derived_peaks: dict[str, dict[str, Any]] = {}
        baseline_by_uuid: dict[str, int] = {}
        derived_inventory_counts = dict.fromkeys(selected_by_uuid, 0)
        derived_workload_counts = dict.fromkeys(selected_by_uuid, 0)
        derived_workload_observed = dict.fromkeys(selected_by_uuid, False)
        sample_times_by_uuid: dict[str, list[datetime]] = {uuid: [] for uuid in selected_by_uuid}
        observed_rows = 0
        for index, row in enumerate(reader):
            if set(row) != set(expected_fields) or any(value is None for value in row.values()):
                raise ExperimentProtocolError(
                    f"wrapper attempt gpu-samples.csv row {index} is malformed"
                )
            try:
                timestamp_text = row["timestamp_utc"].strip()
                if not timestamp_text.endswith("Z"):
                    raise ValueError("GPU sample timestamp is not canonical UTC")
                timestamp = datetime.fromisoformat(timestamp_text[:-1] + "+00:00")
                memory_used = int(float(row["memory_used_mib"]))
                memory_total = int(float(row["memory_total_mib"]))
                utilization = int(float(row["utilization_gpu_percent"]))
            except (TypeError, ValueError) as error:
                raise ExperimentProtocolError(
                    f"wrapper attempt gpu-samples.csv row {index} has invalid values"
                ) from error
            uuid = row["uuid"].strip()
            physical = row["index"].strip()
            sample_kind = row["sample_kind"].strip()
            selected = selected_by_uuid.get(uuid)
            if (
                timestamp.tzinfo is None
                or selected is None
                or selected.get("physical_index") != physical
                or selected.get("name") != row["name"].strip()
                or selected.get("memory_total_mib") != memory_total
                or sample_kind not in {"inventory", "workload"}
                or memory_used < 0
                or memory_total <= 0
                or memory_used > memory_total
                or not 0 <= utilization <= 100
            ):
                raise ExperimentProtocolError(
                    f"wrapper attempt gpu-samples.csv row {index} violates GPU identity"
                )
            observed_rows += 1
            timestamps = sample_times_by_uuid[uuid]
            timestamp = timestamp.astimezone(timezone.utc)
            if timestamps and timestamp < timestamps[-1]:
                raise ExperimentProtocolError(
                    "wrapper attempt GPU sample timestamps are not monotonic"
                )
            timestamps.append(timestamp)
            peak = derived_peaks.setdefault(
                physical,
                {
                    "uuid": uuid,
                    "name": selected["name"],
                    "memory_total_mib": memory_total,
                    "peak_memory_used_mib": 0,
                    "peak_utilization_percent": 0,
                },
            )
            peak["peak_memory_used_mib"] = max(peak["peak_memory_used_mib"], memory_used)
            peak["peak_utilization_percent"] = max(peak["peak_utilization_percent"], utilization)
            if sample_kind == "inventory":
                derived_inventory_counts[uuid] += 1
                baseline_by_uuid.setdefault(uuid, memory_used)
            else:
                derived_workload_counts[uuid] += 1
                baseline = baseline_by_uuid.get(uuid)
                if baseline is not None and (utilization > 0 or memory_used > baseline + 16):
                    derived_workload_observed[uuid] = True

        derived_maximum_gap = 0.0
        for uuid, timestamps in sample_times_by_uuid.items():
            if not timestamps:
                raise ExperimentProtocolError(f"wrapper attempt GPU {uuid} has no temporal samples")
            if (
                abs((timestamps[0] - window_started).total_seconds()) > expected_boundary_tolerance
                or abs((timestamps[-1] - window_completed).total_seconds())
                > expected_boundary_tolerance
            ):
                raise ExperimentProtocolError(
                    "wrapper attempt GPU samples do not cover the sampling window"
                )
            for before, after in pairwise(timestamps):
                gap = (after - before).total_seconds()
                if gap < 0.0:
                    raise ExperimentProtocolError(
                        "wrapper attempt GPU sample timestamps are not monotonic"
                    )
                derived_maximum_gap = max(derived_maximum_gap, gap)
        if derived_maximum_gap > expected_gap_tolerance:
            raise ExperimentProtocolError(
                "wrapper attempt GPU sampling exceeds the maximum allowed gap"
            )

        if (
            observed_rows != sample_count
            or set(baseline_by_uuid) != set(selected_by_uuid)
            or any(count != 1 for count in derived_inventory_counts.values())
            or derived_workload_counts != samples_raw
            or derived_workload_observed != observed_raw
            or derived_peaks != peaks_raw
            or not math.isclose(
                float(maximum_observed_gap),
                derived_maximum_gap,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        ):
            raise ExperimentProtocolError(
                "wrapper attempt GPU summary does not replay from gpu-samples.csv"
            )

        normalized_peaks: dict[str, dict[str, Any]] = {}
        for physical, peak in sorted(derived_peaks.items()):
            selected = selected_by_physical[physical]
            normalized_peaks[physical] = {
                "physical_index": physical,
                "logical_index": selected["logical_index"],
                "uuid_sha256": hashlib.sha256(peak["uuid"].encode("utf-8")).hexdigest(),
                "name": peak["name"],
                "memory_total_mib": peak["memory_total_mib"],
                "peak_memory_used_mib": peak["peak_memory_used_mib"],
                "peak_utilization_percent": peak["peak_utilization_percent"],
            }
        return {
            "duration_seconds": duration_seconds,
            "gpu_sampling": {
                "attribution_scope": "physical_gpu_host_level_not_process_attributed",
                "sample_count": sample_count,
                "sample_interval_seconds": interval_seconds,
                "window_started_at_utc": window_started_text,
                "window_completed_at_utc": window_completed_text,
                "window_duration_seconds": expected_window_duration,
                "boundary_tolerance_seconds": expected_boundary_tolerance,
                "maximum_gap_tolerance_seconds": expected_gap_tolerance,
                "maximum_observed_gap_seconds": derived_maximum_gap,
                "temporal_coverage_complete": True,
                "peak_by_physical_index": normalized_peaks,
            },
        }, []
    except (
        csv.Error,
        ExperimentProtocolError,
        KeyError,
        OSError,
        UnicodeDecodeError,
        ValueError,
    ) as error:
        return None, [f"wrapper_attempt_system_evidence_invalid:{error}"]


def _validate_attempt_model_evidence(
    model_evidence_payload: bytes,
    runtime_preflight_entry: Mapping[str, Any],
    run_manifest_payload: bytes,
    *,
    attempt_dir: Path,
    job: Mapping[str, Any],
) -> list[str]:
    try:
        summary = loads_object(model_evidence_payload)
        copied_manifest = loads_object(run_manifest_payload)
        run_manifest_sha256 = _sha256_bytes(run_manifest_payload)
        expected_manifest = job.get("manifest")
        if not isinstance(expected_manifest, dict):
            raise ExperimentProtocolError("wrapper attempt has no validated source manifest")
        expected_source_manifest = Path(
            _require_text(
                expected_manifest.get("path"),
                "validated source manifest path",
            )
        ).resolve(strict=True)
        source_manifest = Path(
            _require_text(
                summary.get("source_manifest"),
                "wrapper attempt model evidence source_manifest",
            )
        )
        runtime_preflight = Path(
            _require_text(
                summary.get("runtime_preflight_path"),
                "wrapper attempt model evidence runtime_preflight_path",
            )
        )
        output_evidence = Path(
            _require_text(
                summary.get("output_evidence_path"),
                "wrapper attempt model evidence output_evidence_path",
            )
        )
        if (
            not source_manifest.is_absolute()
            or source_manifest.is_symlink()
            or source_manifest.resolve(strict=True) != expected_source_manifest
            or not runtime_preflight.is_absolute()
            or runtime_preflight.is_symlink()
            or runtime_preflight.resolve(strict=True)
            != Path(str(runtime_preflight_entry["path"])).resolve(strict=True)
            or not output_evidence.is_absolute()
            or output_evidence.is_symlink()
            or output_evidence.resolve(strict=True)
            != (attempt_dir / "output-evidence.png").resolve(strict=True)
        ):
            raise ExperimentProtocolError(
                "wrapper attempt model evidence paths are not attempt-bound"
            )
        _, _, output_evidence_sha256 = _hash_regular_file(
            output_evidence,
            "wrapper attempt output evidence",
        )
        expected_candidates = 0 if job["group"] == "A-only" else 1
        if (
            summary.get("status") != "passed"
            or summary.get("stage") != "experiment"
            or summary.get("mock") is not False
            or summary.get("experiment_group") != job["group"]
            or summary.get("experiment_sample_id") != job["sample_id"]
            or summary.get("successful_coz_candidates") != expected_candidates
            or summary.get("invoked_input_sha256") != job["input"]["sha256"]
            or summary.get("invoked_config_sha256") != job["config"]["sha256"]
            or summary.get("runtime_config_sha256") != job["config"]["sha256"]
            or summary.get("runtime_preflight_sha256") != runtime_preflight_entry["sha256"]
            or summary.get("manifest_sha256") != run_manifest_sha256
            or summary.get("manifest_status") != copied_manifest.get("status")
            or summary.get("completion_level") != copied_manifest.get("completion_level")
            or summary.get("output_evidence_sha256") != output_evidence_sha256
            or summary.get("final_output_sha256") != output_evidence_sha256
        ):
            raise ExperimentProtocolError(
                "wrapper attempt model evidence is incomplete or inconsistent"
            )
        expected_restoration = (
            "scaleguard_identity_observation" if job["group"] == "B-only" else "4kagent_upstream"
        )
        if (
            summary.get("restoration_backend") != expected_restoration
            or summary.get("scale_backend") != "chain_of_zoom"
        ):
            raise ExperimentProtocolError(
                "wrapper attempt model evidence has the wrong real backends"
            )
    except (ExperimentProtocolError, KeyError, OSError, ValueError) as error:
        return [f"wrapper_attempt_model_evidence_invalid:{error}"]
    return []


def _inspect_wrapper_attempt(
    pointer_path: Path,
    *,
    job: Mapping[str, Any],
    project_commit: str,
    manifest_sha256: str | None,
) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        if pointer_path.is_symlink():
            raise ExperimentProtocolError(
                f"wrapper attempt pointer must not be a symlink: {pointer_path}"
            )
        (
            resolved_pointer,
            pointer,
            _pointer_payload,
            pointer_size,
            pointer_digest,
        ) = _load_json_snapshot(
            pointer_path,
            "wrapper attempt pointer",
        )
        _require_exact_keys(
            pointer,
            {
                "schema_version",
                "status",
                "stage",
                "attempt_id",
                "attempt_dir",
                "started_at_utc",
                "completed_at_utc",
                "experiment_group",
                "experiment_sample_id",
                "files",
                "hardware",
            },
            "wrapper attempt pointer",
        )
        if pointer["schema_version"] != 1 or pointer["stage"] != "experiment":
            raise ExperimentProtocolError("wrapper attempt pointer has the wrong schema or stage")
        status_value = pointer["status"]
        if status_value not in {"running", "failed", "succeeded"}:
            raise ExperimentProtocolError("wrapper attempt pointer has an invalid status")
        started_at = _require_timestamp(
            pointer["started_at_utc"],
            "wrapper attempt pointer.started_at_utc",
        )
        completed_at_raw = pointer["completed_at_utc"]
        if status_value == "running":
            if completed_at_raw is not None:
                raise ExperimentProtocolError("running wrapper attempt pointer cannot be completed")
        else:
            completed_at = _require_timestamp(
                completed_at_raw,
                "wrapper attempt pointer.completed_at_utc",
            )
            if completed_at < started_at:
                raise ExperimentProtocolError("wrapper attempt pointer completed before it started")
        attempt_id = _require_text(
            pointer["attempt_id"],
            "wrapper attempt pointer.attempt_id",
        )
        attempt_dir_text = _require_text(
            pointer["attempt_dir"],
            "wrapper attempt pointer.attempt_dir",
        )
        attempt_dir = Path(attempt_dir_text)
        if (
            not attempt_dir.is_absolute()
            or attempt_dir.is_symlink()
            or not attempt_dir.is_dir()
            or attempt_dir.name != attempt_id
        ):
            raise ExperimentProtocolError("wrapper attempt pointer has an unsafe attempt directory")
    except (ExperimentProtocolError, OSError, ValueError) as error:
        return None, [f"wrapper_attempt_pointer_invalid:{error}"]

    evidence: dict[str, Any] = {
        "pointer": {
            "path": str(resolved_pointer),
            "size_bytes": pointer_size,
            "sha256": pointer_digest,
        },
        "status": status_value,
        "attempt_id": attempt_id,
        "attempt_dir": str(attempt_dir.resolve()),
        "started_at_utc": pointer["started_at_utc"],
        "completed_at_utc": pointer["completed_at_utc"],
        "files": {},
        "hardware": None,
        "system_evidence": None,
        "files_inventory_sha256": None,
    }
    if status_value != "succeeded":
        return evidence, [f"wrapper_attempt_status:{status_value}"]
    issues: list[str] = []
    if (
        pointer["experiment_group"] != job["group"]
        or pointer["experiment_sample_id"] != job["sample_id"]
        or type(pointer["started_at_utc"]) is not str
        or type(pointer["completed_at_utc"]) is not str
    ):
        issues.append("wrapper_attempt_experiment_binding_mismatch")
    try:
        raw_files = _require_mapping(pointer["files"], "wrapper attempt pointer.files")
        _require_exact_keys(
            raw_files,
            set(_ATTEMPT_FILE_RELATIVE_PATHS),
            "wrapper attempt pointer.files",
        )
    except ExperimentProtocolError as error:
        return evidence, [*issues, f"wrapper_attempt_files_invalid:{error}"]
    files: dict[str, dict[str, Any]] = {}
    file_payloads: dict[str, bytes] = {}
    for role in _ATTEMPT_FILE_RELATIVE_PATHS:
        entry, payload, entry_issues = _attempt_file_entry(
            raw_files[role],
            role=role,
            attempt_dir=attempt_dir,
        )
        issues.extend(entry_issues)
        if entry is not None:
            files[role] = entry
        if payload is not None:
            file_payloads[role] = payload
    evidence["files"] = files
    if set(files) != set(_ATTEMPT_FILE_RELATIVE_PATHS):
        return evidence, issues
    if set(file_payloads) != _PARSED_ATTEMPT_FILE_ROLES:
        return evidence, [*issues, "wrapper_attempt_parsed_file_snapshot_missing"]

    inventory_digest, inventory_issues = _validate_attempt_inventory(
        attempt_dir,
        files["files_inventory"],
        file_payloads["files_inventory"],
    )
    evidence["files_inventory_sha256"] = inventory_digest
    issues.extend(inventory_issues)
    hardware, hardware_issues = _validate_attempt_hardware(
        pointer["hardware"],
        gpu_preflight_payload=file_payloads["gpu_preflight"],
        gpu_inventory_payload=file_payloads["gpu_inventory"],
        project_commit=project_commit,
    )
    evidence["hardware"] = hardware
    issues.extend(hardware_issues)
    execution_issues = _validate_attempt_execution(
        file_payloads["execution"],
        files["runtime_preflight"],
        file_payloads["runtime_preflight"],
        job=job,
        project_commit=project_commit,
    )
    issues.extend(execution_issues)
    if not execution_issues and hardware is not None:
        system_evidence, system_issues = _validate_attempt_system_evidence(
            file_payloads["execution"],
            file_payloads["gpu_samples"],
            hardware,
        )
        evidence["system_evidence"] = system_evidence
        issues.extend(system_issues)
    issues.extend(
        _validate_attempt_model_evidence(
            file_payloads["model_evidence"],
            files["runtime_preflight"],
            file_payloads["run_manifest"],
            attempt_dir=attempt_dir,
            job=job,
        )
    )
    if manifest_sha256 is None:
        issues.append("wrapper_attempt_run_manifest_unbound")
    elif files["run_manifest"]["sha256"] != manifest_sha256:
        issues.append("wrapper_attempt_run_manifest_sha256_mismatch")
    return evidence, issues


def _normalized_pair_config(manifest: Mapping[str, Any]) -> dict[str, Any]:
    raw_config = manifest.get("config")
    if not isinstance(raw_config, dict):
        return {}
    config = copy.deepcopy(raw_config)
    runtime = config.get("runtime")
    if isinstance(runtime, dict):
        runtime.pop("run_root", None)
        runtime.pop("experiment_group", None)
    fourkagent = config.get("fourkagent")
    if isinstance(fourkagent, dict):
        fourkagent.pop("mode", None)
    controller = config.get("controller")
    if isinstance(controller, dict):
        controller.pop("target_factor", None)
        controller.pop("max_coz_steps", None)
        controller.pop("acceptance_policy", None)
    return config


def _pairing_fields(manifest: Mapping[str, Any]) -> dict[str, Any]:
    provenance = manifest.get("provenance")
    config = manifest.get("config")
    input_image = manifest.get("input_image")
    metrics = config.get("metrics") if isinstance(config, dict) else None
    return {
        "normalized_config": _normalized_pair_config(manifest),
        "quality_config": metrics,
        "input_image_sha256": (
            input_image.get("sha256") if isinstance(input_image, dict) else None
        ),
        "runtime_provenance": {
            field: provenance.get(field) if isinstance(provenance, dict) else None
            for field in _COMMON_PROVENANCE_FIELDS
        },
    }


def _job_runtime_evidence(manifest: Mapping[str, Any]) -> dict[str, Any] | None:
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        return None
    return {field: copy.deepcopy(provenance.get(field)) for field in _JOB_PROVENANCE_FIELDS}


def _apply_pairing_checks(
    jobs: Sequence[dict[str, Any]],
    manifests: Mapping[str, Mapping[str, Any]],
) -> None:
    jobs_by_sample: dict[str, list[dict[str, Any]]] = {}
    for job in jobs:
        jobs_by_sample.setdefault(job["sample_id"], []).append(job)
    for sample_jobs in jobs_by_sample.values():
        available = [
            (job, manifests[job["job_id"]]) for job in sample_jobs if job["job_id"] in manifests
        ]
        if len(available) < 2:
            continue
        fields_by_job = {job["job_id"]: _pairing_fields(manifest) for job, manifest in available}
        for field in (
            "normalized_config",
            "quality_config",
            "input_image_sha256",
            "runtime_provenance",
        ):
            values = {
                _canonical_sha256({"value": fields[field]}) for fields in fields_by_job.values()
            }
            if len(values) == 1:
                continue
            issue = f"sample_pairing_mismatch:{field}"
            for job in sample_jobs:
                if issue not in job["issues"]:
                    job["issues"].append(issue)
                job["status"] = "failed"
        hardware_records = [
            job["wrapper_attempt"]["hardware"]
            for job in sample_jobs
            if isinstance(job.get("wrapper_attempt"), dict)
            and isinstance(job["wrapper_attempt"].get("hardware"), dict)
        ]
        for field in ("identity_sha256", "class_sha256"):
            values = {record.get(field) for record in hardware_records}
            if len(hardware_records) < 2 or len(values) == 1:
                continue
            issue = f"sample_pairing_mismatch:hardware_{field}"
            for job in sample_jobs:
                if issue not in job["issues"]:
                    job["issues"].append(issue)
                job["status"] = "failed"


def _apply_attempt_uniqueness(jobs: Sequence[dict[str, Any]]) -> None:
    by_id: dict[str, list[dict[str, Any]]] = {}
    by_directory: dict[str, list[dict[str, Any]]] = {}
    for job in jobs:
        attempt = job.get("wrapper_attempt")
        if not isinstance(attempt, dict):
            continue
        by_id.setdefault(str(attempt.get("attempt_id")), []).append(job)
        by_directory.setdefault(str(attempt.get("attempt_dir")), []).append(job)
    for field, grouped in (("attempt_id", by_id), ("attempt_dir", by_directory)):
        for repeated in grouped.values():
            if len(repeated) < 2:
                continue
            issue = f"duplicate_wrapper_{field}"
            for job in repeated:
                if issue not in job["issues"]:
                    job["issues"].append(issue)
                job["status"] = "failed"


def _frozen_file_issues(
    frozen_files: Sequence[Mapping[str, Any]],
) -> list[str]:
    issues: list[str] = []
    labels = ("protocol", "base_config", "integration_runner")
    for label, evidence in zip(labels, frozen_files, strict=True):
        issues.extend(_verify_file_evidence(evidence, label))
    return issues


def _job_file_issues(
    job: Mapping[str, Any],
    *,
    before_launch: bool = False,
) -> list[str]:
    issues = _verify_file_evidence(job["config"], "generated_config")
    issues.extend(_verify_file_evidence(job["input"], "input_snapshot"))
    if before_launch:
        pointer = Path(str(job["wrapper_evidence_pointer"]))
        if pointer.exists() or pointer.is_symlink():
            issues.append("wrapper_evidence_pointer_preexists")
    return issues


def run_ablation_suite(
    *,
    protocol_path: Path,
    base_config_path: Path,
    inputs: Sequence[Path],
    output_directory: Path,
    project_root: Path,
    seeds: Sequence[int] | None = None,
    plan_only: bool = False,
    command_runner: CommandRunner | None = None,
) -> dict[str, Any]:
    """Plan and optionally execute every paired input, seed, and group job."""

    protocol = load_ablation_protocol(protocol_path)
    base_path, base_size, base_sha256, base_config = _load_base_config(base_config_path)
    input_sources = _load_inputs(inputs)
    resolved_seeds = _validate_seeds(seeds, base_config)
    sample_ids = _validate_sample_ids(input_sources, resolved_seeds)
    resolved_project = project_root.resolve(strict=True)
    output_root = _safe_output_directory(output_directory, resolved_project)
    runner = resolved_project / protocol.integration_runner
    if (
        protocol.integration_runner != INTEGRATION_RUNNER
        or not runner.is_file()
        or runner.is_symlink()
        or not os.access(runner, os.X_OK)
    ):
        raise ExperimentProtocolError(
            f"fixed integration runner is missing, symlinked, or not executable: {runner}"
        )
    runner_path, runner_size, runner_sha256 = _hash_regular_file(
        runner,
        "integration runner",
    )
    project_commit = _clean_commit(resolved_project, "ablation suite start")
    frozen_files = (
        {
            "path": str(protocol.path),
            "size_bytes": protocol.size_bytes,
            "sha256": protocol.sha256,
        },
        {
            "path": str(base_path),
            "size_bytes": base_size,
            "sha256": base_sha256,
        },
        {
            "path": str(runner_path),
            "size_bytes": runner_size,
            "sha256": runner_sha256,
        },
    )

    output_root.mkdir(parents=True, exist_ok=True)
    if next(output_root.iterdir(), None) is not None:
        raise ExperimentProtocolError(f"output directory must be empty: {output_root}")
    inputs_directory = output_root / "inputs"
    jobs_directory = output_root / "jobs"
    inputs_directory.mkdir()
    jobs_directory.mkdir()

    input_records: list[dict[str, Any]] = []
    snapshot_by_sha256: dict[str, Path] = {}
    for source in input_sources:
        snapshot = inputs_directory / f"{source.sha256}{source.suffix}"
        _copy_input_snapshot(source, snapshot)
        snapshot_by_sha256[source.sha256] = snapshot
        input_records.append(
            {
                "requested_path": source.requested_path,
                "source_path": str(source.path),
                "snapshot_path": str(snapshot),
                "size_bytes": source.size_bytes,
                "sha256": source.sha256,
            }
        )

    jobs: list[dict[str, Any]] = []
    group_by_id = {group.id: group for group in protocol.groups}
    for source in input_sources:
        input_snapshot = snapshot_by_sha256[source.sha256]
        for seed in resolved_seeds:
            sample_id = sample_ids[(source.sha256, seed)]
            for group in protocol.groups:
                group_directory = jobs_directory / _group_slug(group.id) / sample_id
                run_root = group_directory / "runs"
                config_path = group_directory / "config.yaml"
                output_path = group_directory / "output.png"
                evidence_pointer = group_directory / "wrapper-attempt.json"
                group_directory.mkdir(parents=True)
                generated = _build_group_config(
                    base_config,
                    group=group,
                    sample_id=sample_id,
                    seed=seed,
                    run_root=run_root,
                )
                config_payload = _config_bytes(generated)
                _atomic_write(config_path, config_payload)
                try:
                    load_config(config_path)
                except ScaleGuardError as error:
                    raise ExperimentProtocolError(
                        f"generated config failed full validation for {group.id}, "
                        f"sample {sample_id}: {error}"
                    ) from error
                argv = [
                    str(runner),
                    "--config",
                    str(config_path),
                    "--input",
                    str(input_snapshot),
                    "--output",
                    str(output_path),
                    "--evidence-output",
                    str(evidence_pointer),
                ]
                jobs.append(
                    {
                        "job_id": f"{sample_id}-{_group_slug(group.id)}",
                        "sample_id": sample_id,
                        "group": group.id,
                        "seed": seed,
                        "input": {
                            "path": str(input_snapshot),
                            "size_bytes": source.size_bytes,
                            "sha256": source.sha256,
                        },
                        "config": {
                            "path": str(config_path),
                            "size_bytes": len(config_payload),
                            "sha256": _sha256_bytes(config_payload),
                        },
                        "output_path": str(output_path),
                        "run_root": str(run_root),
                        "wrapper_evidence_pointer": str(evidence_pointer),
                        "argv": argv,
                        "status": "planned",
                        "started_at_utc": None,
                        "completed_at_utc": None,
                        "returncode": None,
                        "project_commit_before": None,
                        "project_commit_after": None,
                        "manifest": None,
                        "wrapper_attempt": None,
                        "runtime_evidence": None,
                        "issues": [],
                    }
                )

    receipt_path = output_root / "suite-receipt.json"
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "planned" if plan_only else "running",
        "plan_only": plan_only,
        "started_at_utc": _utc_now(),
        "completed_at_utc": _utc_now() if plan_only else None,
        "project_root": str(resolved_project),
        "project_commit": project_commit,
        "output_directory": str(output_root),
        "protocol": {
            "path": str(protocol.path),
            "size_bytes": protocol.size_bytes,
            "sha256": protocol.sha256,
            "name": protocol.name,
            "status": "executable",
            "integration_runner": protocol.integration_runner,
        },
        "base_config": {
            "path": str(base_path),
            "size_bytes": base_size,
            "sha256": base_sha256,
        },
        "integration_runner": {
            "path": str(runner_path),
            "size_bytes": runner_size,
            "sha256": runner_sha256,
        },
        "groups": [
            {
                "id": group.id,
                "fourkagent_mode": group.fourkagent_mode,
                "coz_mode": group.coz_mode,
                "target_factor": group.target_factor,
                "max_coz_steps": group.max_coz_steps,
                "acceptance_policy": group.acceptance_policy,
                "comparison_resolution": group.comparison_resolution,
            }
            for group in protocol.groups
        ],
        "seeds": list(resolved_seeds),
        "inputs": input_records,
        "jobs": jobs,
        "issues": [],
    }
    materialized_commit = _clean_commit(
        resolved_project,
        "ablation suite materialization",
    )
    if materialized_commit != project_commit:
        raise ExperimentProtocolError(
            "project HEAD changed while the ablation suite was materialized"
        )
    frozen_issues = _frozen_file_issues(frozen_files)
    if frozen_issues:
        raise ExperimentProtocolError(
            "suite source evidence changed while planning: " + ", ".join(frozen_issues)
        )
    _write_receipt(receipt_path, receipt)
    if plan_only:
        return receipt

    invoke = command_runner or _default_command_runner
    validated_manifests: dict[str, dict[str, Any]] = {}
    for job in jobs:
        job["status"] = "running"
        job["started_at_utc"] = _utc_now()
        _write_receipt(receipt_path, receipt)
        issues = _job_file_issues(job, before_launch=True)
        issues.extend(_frozen_file_issues(frozen_files))
        try:
            before_commit = _clean_commit(
                resolved_project,
                f"job {job['job_id']} preflight",
            )
            job["project_commit_before"] = before_commit
            if before_commit != project_commit:
                issues.append("project_commit_changed_before_job")
        except ExperimentProtocolError as error:
            issues.append(f"project_not_clean_before_job:{error}")

        returncode: int | None = None
        if not issues:
            try:
                result = invoke(tuple(job["argv"]), resolved_project)
                if type(result) is not int:
                    raise TypeError("command runner return code must be an integer")
                returncode = result
            except Exception as error:  # preserve this job and continue the suite
                issues.append(f"runner_error:{type(error).__name__}:{error}")
        job["returncode"] = returncode
        try:
            after_commit = _clean_commit(
                resolved_project,
                f"job {job['job_id']} completion",
            )
            job["project_commit_after"] = after_commit
            if after_commit != project_commit:
                issues.append("project_commit_changed_after_job")
        except ExperimentProtocolError as error:
            issues.append(f"project_not_clean_after_job:{error}")
        issues.extend(_job_file_issues(job))
        issues.extend(_frozen_file_issues(frozen_files))
        group = group_by_id[job["group"]]
        manifest, manifest_issues, validated_manifest = _inspect_manifest(
            Path(job["run_root"]),
            group=group,
            sample_id=job["sample_id"],
            seed=job["seed"],
            project_commit=project_commit,
        )
        job["manifest"] = manifest
        issues.extend(manifest_issues)
        if validated_manifest is not None:
            validated_manifests[job["job_id"]] = validated_manifest
            job["runtime_evidence"] = _job_runtime_evidence(validated_manifest)
        wrapper_attempt, attempt_issues = _inspect_wrapper_attempt(
            Path(job["wrapper_evidence_pointer"]),
            job=job,
            project_commit=project_commit,
            manifest_sha256=(str(manifest["sha256"]) if manifest is not None else None),
        )
        job["wrapper_attempt"] = wrapper_attempt
        issues.extend(attempt_issues)
        if returncode is not None and returncode != 0:
            issues.append(f"runner_returncode:{returncode}")
        job["issues"] = issues
        job["status"] = "passed" if returncode == 0 and not issues else "failed"
        job["completed_at_utc"] = _utc_now()
        _write_receipt(receipt_path, receipt)

    for job in jobs:
        revalidated_manifest: dict[str, Any] | None = None
        if job["manifest"] is not None:
            group = group_by_id[job["group"]]
            revalidated_manifest, issues, validated_manifest = _inspect_manifest(
                Path(job["run_root"]),
                group=group,
                sample_id=job["sample_id"],
                seed=job["seed"],
                project_commit=project_commit,
            )
            if revalidated_manifest != job["manifest"]:
                issues.append("post_suite_manifest_or_artifact_inventory_changed")
            for issue in issues:
                tagged = f"post_suite_revalidation:{issue}"
                if tagged not in job["issues"]:
                    job["issues"].append(tagged)
            if validated_manifest is not None:
                validated_manifests[job["job_id"]] = validated_manifest
        file_issues = _job_file_issues(job)
        for issue in file_issues:
            tagged = f"post_suite_revalidation:{issue}"
            if tagged not in job["issues"]:
                job["issues"].append(tagged)
        attempt, attempt_issues = _inspect_wrapper_attempt(
            Path(job["wrapper_evidence_pointer"]),
            job=job,
            project_commit=project_commit,
            manifest_sha256=(
                str(revalidated_manifest["sha256"])
                if revalidated_manifest is not None
                else (str(job["manifest"]["sha256"]) if isinstance(job["manifest"], dict) else None)
            ),
        )
        if attempt != job["wrapper_attempt"]:
            attempt_issues.append("wrapper_attempt_changed_after_job")
        for issue in attempt_issues:
            tagged = f"post_suite_revalidation:{issue}"
            if tagged not in job["issues"]:
                job["issues"].append(tagged)
        if job["issues"]:
            job["status"] = "failed"

    _apply_attempt_uniqueness(jobs)
    _apply_pairing_checks(jobs, validated_manifests)
    suite_issues = _frozen_file_issues(frozen_files)
    try:
        final_commit = _clean_commit(resolved_project, "ablation suite completion")
        if final_commit != project_commit:
            suite_issues.append("project_commit_changed_at_suite_completion")
    except ExperimentProtocolError as error:
        suite_issues.append(f"project_not_clean_at_suite_completion:{error}")
    receipt["issues"] = suite_issues
    receipt["status"] = (
        "passed"
        if not suite_issues and all(job["status"] == "passed" for job in jobs)
        else "completed_with_failures"
    )
    receipt["completed_at_utc"] = _utc_now()
    _write_receipt(receipt_path, receipt)
    return receipt


def _validated_suite_file(
    raw: Any,
    *,
    context: str,
) -> dict[str, Any]:
    evidence = _require_mapping(raw, context)
    _require_exact_keys(evidence, {"path", "size_bytes", "sha256"}, context)
    path = Path(_require_text(evidence["path"], f"{context}.path"))
    size = _require_integer(evidence["size_bytes"], f"{context}.size_bytes")
    digest = evidence["sha256"]
    if size < 0 or type(digest) is not str or _DIGEST_PATTERN.fullmatch(digest) is None:
        raise ExperimentProtocolError(f"{context} has an invalid byte identity")
    resolved, observed_size, observed_digest = _hash_regular_file(path, context)
    if observed_size != size or observed_digest != digest:
        raise ExperimentProtocolError(f"{context} byte identity changed")
    return {
        "path": str(resolved),
        "size_bytes": size,
        "sha256": digest,
    }


def _validated_suite_inputs(
    raw_inputs: Any,
    *,
    output_root: Path,
) -> dict[str, dict[str, Any]]:
    if not isinstance(raw_inputs, list) or not raw_inputs:
        raise ExperimentProtocolError("ablation suite receipt has no input snapshots")
    inputs: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_inputs):
        context = f"ablation suite inputs[{index}]"
        record = _require_mapping(raw, context)
        _require_exact_keys(
            record,
            {
                "requested_path",
                "source_path",
                "snapshot_path",
                "size_bytes",
                "sha256",
            },
            context,
        )
        _require_text(record["requested_path"], f"{context}.requested_path")
        _require_text(record["source_path"], f"{context}.source_path")
        snapshot = Path(_require_text(record["snapshot_path"], f"{context}.snapshot_path"))
        size = _require_integer(record["size_bytes"], f"{context}.size_bytes")
        digest = record["sha256"]
        if (
            size < 0
            or type(digest) is not str
            or _DIGEST_PATTERN.fullmatch(digest) is None
            or digest in inputs
        ):
            raise ExperimentProtocolError(f"{context} has an invalid or duplicate identity")
        resolved, observed_size, observed_digest = _hash_regular_file(snapshot, context)
        if (
            resolved.parent != output_root / "inputs"
            or not resolved.name.startswith(digest)
            or observed_size != size
            or observed_digest != digest
        ):
            raise ExperimentProtocolError(f"{context} snapshot byte identity changed")
        inputs[digest] = {
            "path": str(resolved),
            "size_bytes": size,
            "sha256": digest,
        }
    return inputs


def _suite_group_specs(
    raw_groups: Any,
    *,
    protocol: AblationProtocol,
) -> dict[str, GroupSpec]:
    if not isinstance(raw_groups, list) or raw_groups != list(_GROUP_CONTRACT):
        raise ExperimentProtocolError(
            "ablation suite receipt group semantics differ from the fixed protocol"
        )
    return {group.id: group for group in protocol.groups}


def validate_ablation_suite_receipt(path: Path) -> dict[str, Any]:
    """Independently revalidate a passed suite and every bound runtime artifact."""

    try:
        resolved = path.expanduser().resolve(strict=True)
        payload, receipt_digest = load_regular_file_snapshot(
            resolved,
            "ablation suite receipt",
        )
        receipt = loads_object(payload)
    except (OSError, ScaleGuardError, ValueError) as error:
        raise ExperimentProtocolError(
            f"cannot read ablation suite receipt {path}: {error}"
        ) from error
    _require_exact_keys(receipt, _SUITE_RECEIPT_KEYS, "ablation suite receipt")
    recorded_digest = receipt["receipt_sha256"]
    if type(recorded_digest) is not str or _DIGEST_PATTERN.fullmatch(recorded_digest) is None:
        raise ExperimentProtocolError("ablation suite receipt has an invalid self digest")
    unsigned = copy.deepcopy(receipt)
    unsigned.pop("receipt_sha256")
    if _canonical_sha256(unsigned) != recorded_digest:
        raise ExperimentProtocolError("ablation suite receipt self digest is invalid")
    if (
        receipt["schema_version"] != RECEIPT_SCHEMA
        or receipt["status"] != "passed"
        or receipt["plan_only"] is not False
        or receipt["issues"] != []
    ):
        raise ExperimentProtocolError("ablation suite receipt is not a passed real suite")
    started = _require_timestamp(
        receipt["started_at_utc"],
        "ablation suite receipt.started_at_utc",
    )
    completed = _require_timestamp(
        receipt["completed_at_utc"],
        "ablation suite receipt.completed_at_utc",
    )
    if completed < started:
        raise ExperimentProtocolError("ablation suite receipt completion precedes its start")

    project_root_text = _require_text(
        receipt["project_root"],
        "ablation suite receipt.project_root",
    )
    project_root = Path(project_root_text)
    if not project_root.is_absolute() or project_root.is_symlink() or not project_root.is_dir():
        raise ExperimentProtocolError("ablation suite receipt has an unsafe project root")
    project_root = project_root.resolve(strict=True)
    project_commit = _require_text(
        receipt["project_commit"],
        "ablation suite receipt.project_commit",
    )
    if re.fullmatch(r"[0-9a-f]{40}", project_commit) is None:
        raise ExperimentProtocolError("ablation suite receipt has an invalid project commit")
    if _clean_commit(project_root, "ablation suite receipt validation") != project_commit:
        raise ExperimentProtocolError("ablation suite receipt is bound to another project commit")

    output_root = Path(
        _require_text(
            receipt["output_directory"],
            "ablation suite receipt.output_directory",
        )
    )
    if (
        not output_root.is_absolute()
        or output_root.is_symlink()
        or not output_root.is_dir()
        or output_root.resolve(strict=True) != resolved.parent
    ):
        raise ExperimentProtocolError("ablation suite receipt has an unsafe output directory")
    output_root = output_root.resolve(strict=True)

    protocol_record = _require_mapping(
        receipt["protocol"],
        "ablation suite receipt.protocol",
    )
    _require_exact_keys(
        protocol_record,
        {
            "path",
            "size_bytes",
            "sha256",
            "name",
            "status",
            "integration_runner",
        },
        "ablation suite receipt.protocol",
    )
    if (
        protocol_record["name"] != "core-ablation"
        or protocol_record["status"] != "executable"
        or protocol_record["integration_runner"] != INTEGRATION_RUNNER
    ):
        raise ExperimentProtocolError("ablation suite receipt has an invalid protocol binding")
    protocol_file = _validated_suite_file(
        {key: protocol_record[key] for key in ("path", "size_bytes", "sha256")},
        context="ablation suite protocol",
    )
    loaded_protocol = load_ablation_protocol(Path(protocol_file["path"]))
    if loaded_protocol.sha256 != protocol_file["sha256"]:
        raise ExperimentProtocolError("ablation suite protocol no longer matches its receipt")

    base_file = _validated_suite_file(
        receipt["base_config"],
        context="ablation suite base config",
    )
    base_path, base_size, base_sha256, _base_config = _load_base_config(Path(base_file["path"]))
    if (
        base_path != Path(base_file["path"])
        or base_size != base_file["size_bytes"]
        or base_sha256 != base_file["sha256"]
    ):
        raise ExperimentProtocolError("ablation suite base config no longer matches its receipt")
    runner_file = _validated_suite_file(
        receipt["integration_runner"],
        context="ablation suite integration runner",
    )
    if Path(runner_file["path"]) != (project_root / INTEGRATION_RUNNER).resolve(strict=True):
        raise ExperimentProtocolError("ablation suite receipt uses an unexpected runner")
    groups = _suite_group_specs(receipt["groups"], protocol=loaded_protocol)

    raw_seeds = receipt["seeds"]
    if (
        not isinstance(raw_seeds, list)
        or not raw_seeds
        or any(type(seed) is not int or not 0 <= seed <= 2**63 - 1 for seed in raw_seeds)
        or len(set(raw_seeds)) != len(raw_seeds)
    ):
        raise ExperimentProtocolError("ablation suite receipt has an invalid seed set")
    seeds = tuple(raw_seeds)
    inputs = _validated_suite_inputs(receipt["inputs"], output_root=output_root)

    raw_jobs = receipt["jobs"]
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise ExperimentProtocolError("ablation suite receipt has no jobs")
    expected_job_count = len(inputs) * len(seeds) * len(EXPERIMENT_GROUPS)
    if len(raw_jobs) != expected_job_count:
        raise ExperimentProtocolError("ablation suite receipt has an incomplete job matrix")
    counts = receipt["counts"]
    expected_counts = {
        "total": expected_job_count,
        "planned": 0,
        "running": 0,
        "passed": expected_job_count,
        "failed": 0,
    }
    if counts != expected_counts:
        raise ExperimentProtocolError("ablation suite receipt counts are inconsistent")

    expected_matrix = {
        (f"{input_digest[:16]}-s{seed}", group)
        for input_digest in inputs
        for seed in seeds
        for group in EXPERIMENT_GROUPS
    }
    observed_matrix: set[tuple[str, str]] = set()
    verified_jobs: list[dict[str, Any]] = []
    validation_jobs: list[dict[str, Any]] = []
    validated_manifests: dict[str, dict[str, Any]] = {}
    for index, raw_job in enumerate(raw_jobs):
        context = f"ablation suite jobs[{index}]"
        job = _require_mapping(raw_job, context)
        _require_exact_keys(job, _SUITE_JOB_KEYS, context)
        sample_id = _require_text(job["sample_id"], f"{context}.sample_id")
        group_id = _require_text(job["group"], f"{context}.group")
        seed = _require_integer(job["seed"], f"{context}.seed")
        input_record = _require_mapping(job["input"], f"{context}.input")
        _require_exact_keys(
            input_record,
            {"path", "size_bytes", "sha256"},
            f"{context}.input",
        )
        input_digest = input_record["sha256"]
        if (
            group_id not in groups
            or seed not in seeds
            or re.fullmatch(r"[0-9a-f]{16}-s(?:0|[1-9][0-9]*)", sample_id) is None
            or type(input_digest) is not str
            or sample_id != f"{input_digest[:16]}-s{seed}"
            or (sample_id, group_id) in observed_matrix
        ):
            raise ExperimentProtocolError(f"{context} has an invalid matrix identity")
        observed_matrix.add((sample_id, group_id))
        if (
            job["job_id"] != f"{sample_id}-{_group_slug(group_id)}"
            or job["status"] != "passed"
            or job["returncode"] != 0
            or job["issues"] != []
            or job["project_commit_before"] != project_commit
            or job["project_commit_after"] != project_commit
        ):
            raise ExperimentProtocolError(f"{context} is not a clean passed job")
        job_started = _require_timestamp(job["started_at_utc"], f"{context}.started_at_utc")
        job_completed = _require_timestamp(
            job["completed_at_utc"],
            f"{context}.completed_at_utc",
        )
        if job_started < started or job_completed < job_started or job_completed > completed:
            raise ExperimentProtocolError(f"{context} has timestamps outside the suite")

        group_root = output_root / "jobs" / _group_slug(group_id) / sample_id
        expected_paths = {
            "config": group_root / "config.yaml",
            "output": group_root / "output.png",
            "run_root": group_root / "runs",
            "pointer": group_root / "wrapper-attempt.json",
        }
        if input_digest not in inputs or input_record != inputs[input_digest]:
            raise ExperimentProtocolError(f"{context} input snapshot is unbound")
        config_file = _validated_suite_file(job["config"], context=f"{context}.config")
        if Path(config_file["path"]) != expected_paths["config"].resolve(strict=True):
            raise ExperimentProtocolError(f"{context} config path is not fixed")
        if Path(str(job["run_root"])).resolve(strict=True) != expected_paths["run_root"].resolve(
            strict=True
        ):
            raise ExperimentProtocolError(f"{context} run root is not fixed")
        if Path(str(job["wrapper_evidence_pointer"])).resolve(strict=True) != expected_paths[
            "pointer"
        ].resolve(strict=True):
            raise ExperimentProtocolError(f"{context} wrapper pointer is not fixed")
        if Path(str(job["output_path"])).resolve(strict=True) != expected_paths["output"].resolve(
            strict=True
        ):
            raise ExperimentProtocolError(f"{context} output path is not fixed")
        expected_argv = [
            runner_file["path"],
            "--config",
            config_file["path"],
            "--input",
            inputs[input_digest]["path"],
            "--output",
            str(expected_paths["output"]),
            "--evidence-output",
            str(expected_paths["pointer"]),
        ]
        if job["argv"] != expected_argv:
            raise ExperimentProtocolError(f"{context} command differs from the fixed wrapper")

        manifest_record, manifest_issues, manifest = _inspect_manifest(
            expected_paths["run_root"],
            group=groups[group_id],
            sample_id=sample_id,
            seed=seed,
            project_commit=project_commit,
        )
        if manifest_issues or manifest is None or manifest_record != job["manifest"]:
            raise ExperimentProtocolError(
                f"{context} manifest revalidation failed: {', '.join(manifest_issues)}"
            )
        assert manifest_record is not None
        assert manifest is not None
        runtime_evidence = _job_runtime_evidence(manifest)
        if runtime_evidence != job["runtime_evidence"]:
            raise ExperimentProtocolError(f"{context} runtime evidence changed")
        attempt, attempt_issues = _inspect_wrapper_attempt(
            expected_paths["pointer"],
            job=job,
            project_commit=project_commit,
            manifest_sha256=str(manifest_record["sha256"]),
        )
        if attempt_issues or attempt != job["wrapper_attempt"] or attempt is None:
            raise ExperimentProtocolError(
                f"{context} wrapper attempt revalidation failed: " + ", ".join(attempt_issues)
            )
        hardware = attempt.get("hardware")
        if not isinstance(hardware, dict):
            raise ExperimentProtocolError(f"{context} has no verified hardware identity")
        verified_jobs.append(
            {
                "sample_id": sample_id,
                "group": group_id,
                "manifest": {
                    "path": manifest_record["path"],
                    "sha256": manifest_record["sha256"],
                },
                "hardware": {
                    "identity_sha256": hardware["identity_sha256"],
                    "class_sha256": hardware["class_sha256"],
                },
                "system_evidence": copy.deepcopy(attempt["system_evidence"]),
            }
        )
        validation_job = copy.deepcopy(job)
        validation_job["issues"] = []
        validation_job["status"] = "passed"
        validation_jobs.append(validation_job)
        validated_manifests[job["job_id"]] = manifest

    if observed_matrix != expected_matrix:
        raise ExperimentProtocolError("ablation suite receipt job matrix is incomplete")
    _apply_attempt_uniqueness(validation_jobs)
    _apply_pairing_checks(validation_jobs, validated_manifests)
    invalid_jobs = [
        job["job_id"] for job in validation_jobs if job["status"] != "passed" or job["issues"]
    ]
    if invalid_jobs:
        raise ExperimentProtocolError(
            "ablation suite pairing revalidation failed: " + ", ".join(invalid_jobs)
        )

    final_payload, final_digest = load_regular_file_snapshot(
        resolved,
        "ablation suite receipt",
    )
    if final_digest != receipt_digest or final_payload != payload:
        raise ExperimentProtocolError("ablation suite receipt changed during validation")
    return {
        "path": str(resolved),
        "size_bytes": len(payload),
        "sha256": receipt_digest,
        "project_commit": project_commit,
        "jobs": sorted(
            verified_jobs,
            key=lambda item: (item["sample_id"], item["group"]),
        ),
    }
