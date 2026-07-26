"""Serializable contracts shared by controllers, adapters, and reports."""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


def utc_now() -> str:
    """Return a stable ISO-8601 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class Decision(str, Enum):
    CONTINUE = "continue"
    STOP = "stop"
    ROLLBACK = "rollback"


class RunStatus(str, Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    SUCCEEDED_WITH_ROLLBACK = "succeeded_with_rollback"
    FAILED = "failed"


class CompletionLevel(str, Enum):
    STATIC_READY = "STATIC_READY"
    COMPONENT_REPRODUCED = "COMPONENT_REPRODUCED"
    AB_INTEGRATED = "AB_INTEGRATED"
    SCALEGUARD_VALIDATED = "SCALEGUARD_VALIDATED"
    RESEARCH_EVALUATED = "RESEARCH_EVALUATED"


@dataclass(frozen=True, slots=True)
class ImageArtifact:
    path: Path
    sha256: str
    width: int
    height: int
    media_type: str
    mock: bool
    stage: str


@dataclass(frozen=True, slots=True)
class ProcessEvidence:
    argv: tuple[str, ...]
    cwd: str
    returncode: int
    duration_seconds: float
    stdout_path: str
    stderr_path: str
    peak_vram_mib: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class WorkerResult:
    image: ImageArtifact
    metadata: dict[str, Any] = field(default_factory=dict)
    process: ProcessEvidence | None = None


@dataclass(frozen=True, slots=True)
class MetricRecord:
    quality_baseline: float
    quality_candidate: float
    quality_gain: float
    quality_backend: str
    scale_nrmse: float
    scale_edge_mae: float
    measurement_nrmse: float | None
    measurement_model: str | None


@dataclass(frozen=True, slots=True)
class ScaleStepRecord:
    index: int
    input_scale: float
    candidate_scale: float
    trusted_before: ImageArtifact
    candidate: ImageArtifact | None
    metrics: MetricRecord | None
    decision: Decision
    accepted: bool
    reason: str
    started_at: str
    finished_at: str
    worker_metadata: dict[str, Any] = field(default_factory=dict)
    process: ProcessEvidence | None = None


@dataclass(slots=True)
class RunManifest:
    schema_version: str
    run_id: str
    status: RunStatus
    completion_level: CompletionLevel
    started_at: str
    finished_at: str | None
    mock: bool
    config: dict[str, Any]
    provenance: dict[str, Any]
    input_image: ImageArtifact
    requested_factor: int
    achieved_factor: int | None = None
    target_reached: bool = False
    restored_image: ImageArtifact | None = None
    restoration_metadata: dict[str, Any] = field(default_factory=dict)
    restoration_process: ProcessEvidence | None = None
    scale_session_process: ProcessEvidence | None = None
    steps: list[ScaleStepRecord] = field(default_factory=list)
    final_image: ImageArtifact | None = None
    final_metrics: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    error: dict[str, str] | None = None


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            item.name: _jsonable(getattr(value, item.name)) for item in dataclasses.fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def manifest_as_dict(manifest: RunManifest) -> dict[str, Any]:
    """Convert a manifest to its stable JSON representation."""

    result = _jsonable(manifest)
    if not isinstance(result, dict):
        raise TypeError("manifest serialization did not produce an object")
    return result


class ManifestRecorder:
    """Persist every state transition with an atomic file replacement."""

    def __init__(self, path: Path, manifest: RunManifest) -> None:
        self.path = path
        self.manifest = manifest
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.write()

    def event(self, event: str, **fields: Any) -> None:
        self.manifest.events.append({"at": utc_now(), "event": event, **fields})
        self.write()

    def append_step(self, step: ScaleStepRecord) -> None:
        self.manifest.steps.append(step)
        self.write()

    def write(self) -> None:
        payload = json.dumps(
            manifest_as_dict(self.manifest),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(payload + "\n", encoding="utf-8")
        os.replace(temporary, self.path)
