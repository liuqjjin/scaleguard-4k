#!/usr/bin/env python3
"""Write a source-bound receipt after the AutoDL preflight succeeds."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import stat
import sys
import tempfile

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scaleguard.provenance import (  # noqa: E402
    ENVIRONMENT_LOCK_PATHS,
    LOCK_PATHS,
    RuntimePreflightError,
    load_evidence_snapshot,
    require_clean_git_commit,
    resolve_materialization_sources,
    sha256,
    validate_runtime_preflight,
)


def _from_project(value: str) -> pathlib.Path:
    path = pathlib.Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _new_output_path(value: pathlib.Path) -> pathlib.Path:
    expanded = value.expanduser()
    candidate = expanded if expanded.is_absolute() else pathlib.Path.cwd() / expanded
    if candidate.name in {"", ".", ".."}:
        raise RuntimePreflightError("runtime preflight output has an invalid filename")
    output = candidate.parent.resolve() / candidate.name
    if output.is_symlink() or output.exists():
        raise RuntimePreflightError(f"runtime preflight output already exists: {output}")
    return output


def _snapshot_open_receipt(
    descriptor: int,
    *,
    context: str,
) -> tuple[str, os.stat_result]:
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimePreflightError(f"runtime preflight receipt is not regular {context}")
        os.lseek(descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
        after = os.fstat(descriptor)
    except OSError as error:
        raise RuntimePreflightError(
            f"cannot snapshot runtime preflight receipt {context}: {error}"
        ) from error
    if not stat.S_ISREG(after.st_mode) or any(
        getattr(before, field) != getattr(after, field) for field in stable_fields
    ):
        raise RuntimePreflightError(f"runtime preflight receipt changed {context}")
    return digest.hexdigest(), after


def _write_validated_receipt(
    output: pathlib.Path,
    document: dict[str, object],
    *,
    config: pathlib.Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    temporary = pathlib.Path(temporary_name)
    snapshot_descriptor: int | None = None
    published_identity: tuple[int, int] | None = None
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        validation = validate_runtime_preflight(
            temporary,
            config_path=config,
            project_root=PROJECT_ROOT,
        )
        expected_digest = validation.get("runtime_preflight_sha256")
        flags = os.O_RDONLY
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        flags |= getattr(os, "O_NONBLOCK", 0)
        try:
            snapshot_descriptor = os.open(temporary, flags)
            snapshot_identity = os.fstat(snapshot_descriptor)
        except OSError as error:
            if snapshot_descriptor is not None:
                os.close(snapshot_descriptor)
                snapshot_descriptor = None
            raise RuntimePreflightError(
                f"cannot reopen validated runtime preflight receipt: {error}"
            ) from error
        snapshot_digest, snapshot_identity = _snapshot_open_receipt(
            snapshot_descriptor,
            context="after validation",
        )
        if snapshot_digest != expected_digest:
            raise RuntimePreflightError("runtime preflight receipt changed after validation")
        try:
            os.link(temporary, output, follow_symlinks=False)
        except FileExistsError as error:
            raise RuntimePreflightError(
                f"runtime preflight output appeared concurrently: {output}"
            ) from error
        except OSError as error:
            raise RuntimePreflightError(
                f"cannot publish runtime preflight receipt without clobbering: {error}"
            ) from error
        try:
            published = os.lstat(output)
        except OSError as error:
            raise RuntimePreflightError(
                f"cannot inspect published runtime preflight receipt: {error}"
            ) from error
        published_identity = (published.st_dev, published.st_ino)
        if not stat.S_ISREG(published.st_mode) or published_identity != (
            snapshot_identity.st_dev,
            snapshot_identity.st_ino,
        ):
            raise RuntimePreflightError(
                "published runtime preflight receipt is not the validated inode"
            )
        published_digest, published_snapshot = _snapshot_open_receipt(
            snapshot_descriptor,
            context="during publication",
        )
        if published_digest != expected_digest or (
            published_snapshot.st_dev,
            published_snapshot.st_ino,
        ) != (snapshot_identity.st_dev, snapshot_identity.st_ino):
            raise RuntimePreflightError("runtime preflight receipt changed during publication")
        try:
            final_published = os.lstat(output)
        except OSError as error:
            raise RuntimePreflightError(
                f"cannot recheck published runtime preflight receipt: {error}"
            ) from error
        final_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if not stat.S_ISREG(final_published.st_mode) or any(
            getattr(final_published, field) != getattr(published_snapshot, field)
            for field in final_fields
        ):
            raise RuntimePreflightError(
                "published runtime preflight receipt changed during publication"
            )
        directory_descriptor = os.open(
            output.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except Exception:
        if published_identity is not None:
            try:
                current = os.lstat(output)
                if (current.st_dev, current.st_ino) == published_identity:
                    output.unlink()
            except OSError:
                pass
        raise
    finally:
        if snapshot_descriptor is not None:
            os.close(snapshot_descriptor)
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--materialization", type=pathlib.Path, required=True)
    parser.add_argument("--runtime-environments", type=pathlib.Path, required=True)
    parser.add_argument("--stage-started-at", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    config = args.config.resolve()
    materialization = args.materialization.resolve()
    output = _new_output_path(args.output)
    runtime_environments = args.runtime_environments
    expected_runtime_environments = output.parent / "runtime-environments"
    if (
        runtime_environments.is_symlink()
        or not runtime_environments.is_dir()
        or runtime_environments.resolve() != expected_runtime_environments.resolve()
    ):
        raise RuntimePreflightError(
            "runtime environment receipts must use this attempt's runtime-environments directory"
        )
    environment_records: dict[str, dict[str, object]] = {}
    for name in ENVIRONMENT_LOCK_PATHS:
        path = runtime_environments / f"{name}.json"
        if path.is_symlink() or not path.is_file():
            raise RuntimePreflightError(f"runtime environment receipt is missing or unsafe: {path}")
        document, digest = load_evidence_snapshot(
            path,
            f"runtime environment {name} receipt",
        )
        environment_records[name] = {
            "path": str(path.resolve()),
            "sha256": digest,
            "status": document.get("status"),
        }
    bootstrap = (PROJECT_ROOT / ".runtime" / "receipts" / "bootstrap.json").resolve()
    artifact_root = _from_project(
        os.environ.get(
            "SCALEGUARD_ARTIFACT_ROOT",
            str(PROJECT_ROOT / "artifacts" / "autodl"),
        )
    )
    override_text = os.environ.get("SCALEGUARD_WEIGHT_RECEIPT")
    override = _from_project(override_text) if override_text else None
    marker, weights_receipt = resolve_materialization_sources(
        materialization,
        artifact_root=artifact_root,
        weight_receipt_override=override,
    )
    commit = require_clean_git_commit(PROJECT_ROOT)
    document = {
        "schema_version": 2,
        "status": "passed",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "stage_started_at_utc": args.stage_started_at,
        "project_commit": commit,
        "config": {"path": str(config), "sha256": sha256(config)},
        "locks": {relative: sha256(PROJECT_ROOT / relative) for relative in LOCK_PATHS},
        "bootstrap": {"path": str(bootstrap), "sha256": sha256(bootstrap)},
        "runtime_environments": environment_records,
        "materialization": {
            "path": str(materialization),
            "sha256": sha256(materialization),
        },
        "materialization_marker": {
            "path": str(marker),
            "sha256": sha256(marker),
        },
        "source_weights_receipt": {
            "path": str(weights_receipt),
            "sha256": sha256(weights_receipt),
        },
        "claim": (
            "Source, environment, and materialized-weight preflight passed. "
            "This receipt contains no inference or quality-result claim."
        ),
    }
    _write_validated_receipt(output, document, config=config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
