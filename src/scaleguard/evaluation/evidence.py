"""Shared validation helpers for evaluation evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scaleguard.errors import ScaleGuardError
from scaleguard.provenance import RuntimePreflightError, load_regular_file_snapshot
from scaleguard.strict_json import StrictJSONError, loads


class EvaluationEvidenceError(ScaleGuardError):
    """Raised when evaluation inputs cannot support an auditable result."""


def sha256_file(path: Path) -> str:
    """Hash a file without loading it into memory."""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise EvaluationEvidenceError(f"cannot read evidence file {path}: {error}") from error
    return digest.hexdigest()


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a JSON mapping using the receipt's canonical representation."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably replace a JSON receipt without exposing a partial file."""

    encoded = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    write_bytes_atomic(path, encoded)


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    """Durably replace one file via a unique same-directory temporary file."""

    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        _fsync_directory(destination.parent)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _fsync_directory(path: Path) -> None:
    """Best-effort directory sync for durable rename metadata."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        os.fsync(descriptor)
    except OSError:
        # Some supported filesystems do not permit directory fsync. The file
        # itself has already been synced and atomically replaced.
        pass
    finally:
        if descriptor is not None:
            os.close(descriptor)


def resolved_distinct_paths(
    outputs: Mapping[str, Path],
    *,
    inputs: Sequence[tuple[str, Path]] = (),
) -> dict[str, Path]:
    """Resolve paths once and reject output/output or output/input aliases."""

    try:
        resolved_outputs = {label: path.expanduser().resolve() for label, path in outputs.items()}
        resolved_inputs = [(label, path.expanduser().resolve()) for label, path in inputs]
    except (OSError, RuntimeError) as error:
        raise EvaluationEvidenceError(f"cannot resolve evidence path safely: {error}") from error

    output_items = list(resolved_outputs.items())
    for index, (label, path) in enumerate(output_items):
        for other_label, other_path in output_items[index + 1 :]:
            if path == other_path:
                raise EvaluationEvidenceError(
                    f"{label} and {other_label} resolve to the same path: {path}"
                )
        for input_label, input_path in resolved_inputs:
            if path == input_path:
                raise EvaluationEvidenceError(f"{label} would overwrite {input_label}: {path}")
    return resolved_outputs


def load_json_object(path: Path, *, kind: str) -> tuple[dict[str, Any], str]:
    """Load a JSON object and return it with the exact input file hash."""

    try:
        payload, file_hash = load_regular_file_snapshot(path, kind)
        raw = loads(payload)
    except (RuntimePreflightError, StrictJSONError) as error:
        raise EvaluationEvidenceError(f"invalid {kind} JSON {path}: {error}") from error
    if not isinstance(raw, dict):
        raise EvaluationEvidenceError(f"{kind} must be a JSON object: {path}")
    return raw, file_hash


def require_text(mapping: Mapping[str, Any], key: str, *, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise EvaluationEvidenceError(f"{context}.{key} must be a non-empty string")
    return value


def require_finite_number(mapping: Mapping[str, Any], key: str, *, context: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationEvidenceError(f"{context}.{key} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise EvaluationEvidenceError(f"{context}.{key} must be finite")
    return result


def optional_finite_number(
    mapping: Mapping[str, Any],
    key: str,
    *,
    context: str,
) -> float | None:
    value = mapping.get(key)
    if value is None:
        return None
    return require_finite_number(mapping, key, context=context)


def _artifact_path(
    declared: str,
    *,
    manifest_path: Path,
    artifact_root: Path | None,
) -> Path:
    raw = Path(declared).expanduser()
    if raw.is_absolute():
        return raw
    if artifact_root is not None:
        return artifact_root / raw
    return manifest_path.parent / raw


def verify_artifact(
    raw: Any,
    *,
    context: str,
    manifest_path: Path,
    artifact_root: Path | None,
) -> dict[str, Any]:
    """Validate a manifest artifact against its on-disk SHA256."""

    if not isinstance(raw, dict):
        raise EvaluationEvidenceError(f"{context} must be an artifact object")
    declared_path = require_text(raw, "path", context=context)
    expected_hash = require_text(raw, "sha256", context=context).lower()
    if len(expected_hash) != 64 or any(
        character not in "0123456789abcdef" for character in expected_hash
    ):
        raise EvaluationEvidenceError(f"{context}.sha256 is not a lowercase SHA256 digest")
    resolved = _artifact_path(
        declared_path,
        manifest_path=manifest_path,
        artifact_root=artifact_root,
    )
    if not resolved.is_file():
        raise EvaluationEvidenceError(
            f"{context} artifact is unavailable for hash verification: {resolved}"
        )
    observed_hash = sha256_file(resolved)
    if observed_hash != expected_hash:
        raise EvaluationEvidenceError(
            f"{context} SHA256 mismatch: expected {expected_hash}, observed {observed_hash}"
        )
    return {
        "path": declared_path,
        "sha256": expected_hash,
        "verified_path": str(resolved.resolve()),
    }
