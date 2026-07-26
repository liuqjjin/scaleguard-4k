#!/usr/bin/env python3
"""Safely materialize downloaded ScaleGuard weight layouts.

The download receipt authenticates acquisition. This hook performs deterministic
post-processing without touching either audited upstream checkout and emits a
content-addressed receipt bound to the original download evidence.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import shutil
import sys
import tarfile
import tempfile
from typing import Any

from scaleguard.strict_json import StrictJSONError, loads_object


class MaterializationError(ValueError):
    """Raised when source evidence or a derived layout is unsafe."""


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: pathlib.Path, label: str) -> dict[str, Any]:
    try:
        value = loads_object(path.read_text(encoding="utf-8"))
    except (OSError, StrictJSONError) as error:
        raise MaterializationError(f"cannot read {label} {path}: {error}") from error
    return value


def safe_destination(root: pathlib.Path, value: object) -> pathlib.Path:
    if not isinstance(value, str) or not value:
        raise MaterializationError("layout destination must be a non-empty string")
    relative = pathlib.PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise MaterializationError(f"unsafe layout destination: {value!r}")
    resolved = (root / pathlib.Path(*relative.parts)).resolve()
    if not resolved.is_relative_to(root):
        raise MaterializationError(f"layout destination escapes weight root: {value!r}")
    return resolved


def inventory(path: pathlib.Path) -> list[dict[str, object]]:
    if path.is_symlink():
        raise MaterializationError(f"materialized layout contains a symlink: {path}")
    if path.is_file():
        paths = [path]
        root = path.parent
    elif path.is_dir():
        paths = sorted(path.rglob("*"))
        if any(item.is_symlink() for item in paths):
            raise MaterializationError(f"materialized layout contains a symlink: {path}")
        paths = [item for item in paths if item.is_file()]
        root = path
    else:
        raise MaterializationError(f"materialized layout is missing: {path}")
    return [
        {
            "path": item.relative_to(root).as_posix(),
            "size_bytes": item.stat().st_size,
            "sha256": sha256(item),
        }
        for item in paths
    ]


def write_atomic(path: pathlib.Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def artifact_by_id(receipt: dict[str, Any], artifact_id: str) -> dict[str, Any]:
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, list):
        raise MaterializationError("download receipt has no artifacts list")
    matches = [
        artifact
        for artifact in artifacts
        if isinstance(artifact, dict) and artifact.get("id") == artifact_id
    ]
    if len(matches) != 1:
        raise MaterializationError(
            f"download receipt must contain exactly one {artifact_id!r} artifact"
        )
    artifact = matches[0]
    if artifact.get("status") not in {"downloaded", "recorded_manual"}:
        raise MaterializationError(
            f"{artifact_id}: source artifact is not complete: {artifact.get('status')!r}"
        )
    return artifact


def validate_download_receipt(
    receipt_path: pathlib.Path,
    weight_root: pathlib.Path,
) -> dict[str, Any]:
    receipt = load_object(receipt_path, "download receipt")
    if receipt.get("schema_version") not in {1, "1", "1.0"}:
        raise MaterializationError("unsupported download receipt schema")
    if receipt.get("status") != "passed":
        raise MaterializationError(
            f"download receipt status is not passed: {receipt.get('status')!r}"
        )
    recorded_root = receipt.get("weight_root")
    if not isinstance(recorded_root, str) or pathlib.Path(recorded_root).resolve() != weight_root:
        raise MaterializationError("download receipt is bound to another weight root")
    commit = receipt.get("git_commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise MaterializationError("download receipt has no immutable project git commit")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise MaterializationError("download receipt artifacts are missing")
    incomplete = [
        str(item.get("id"))
        for item in artifacts
        if isinstance(item, dict)
        and item.get("required") is True
        and item.get("status") not in {"downloaded", "recorded_manual"}
    ]
    if incomplete:
        raise MaterializationError(
            "required source artifacts are incomplete: " + ", ".join(incomplete)
        )
    return receipt


def source_file(
    weight_root: pathlib.Path,
    artifact: dict[str, Any],
    filename: str,
) -> pathlib.Path:
    destination = safe_destination(weight_root, artifact.get("destination"))
    path = destination if destination.is_file() else destination / filename
    files = artifact.get("files")
    if not isinstance(files, list):
        raise MaterializationError(f"{artifact.get('id')}: receipt file inventory is missing")
    entries = [item for item in files if isinstance(item, dict) and item.get("path") == filename]
    if len(entries) != 1:
        raise MaterializationError(f"{artifact.get('id')}: receipt does not identify {filename}")
    entry = entries[0]
    expected_hash = entry.get("sha256")
    expected_size = entry.get("size_bytes")
    if (
        not isinstance(expected_hash, str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_hash)
        or type(expected_size) is not int
    ):
        raise MaterializationError(f"{artifact.get('id')}: invalid source file evidence")
    if not path.is_file() or path.stat().st_size != expected_size or sha256(path) != expected_hash:
        raise MaterializationError(
            f"{artifact.get('id')}: source file no longer matches its download receipt"
        )
    return path


def normalized_tar_members(
    archive: tarfile.TarFile,
) -> list[tuple[tarfile.TarInfo, pathlib.PurePosixPath]]:
    raw: list[tuple[tarfile.TarInfo, pathlib.PurePosixPath]] = []
    seen: set[pathlib.PurePosixPath] = set()
    total_size = 0
    for member in archive.getmembers():
        if "\\" in member.name:
            raise MaterializationError(f"tar member uses a backslash path: {member.name!r}")
        path = pathlib.PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise MaterializationError(f"tar member escapes extraction root: {member.name!r}")
        if member.issym() or member.islnk() or member.isdev() or member.isfifo():
            raise MaterializationError(f"tar member type is not allowed: {member.name!r}")
        if not (member.isdir() or member.isfile()):
            raise MaterializationError(f"unsupported tar member: {member.name!r}")
        if path in seen:
            raise MaterializationError(f"duplicate tar member: {member.name!r}")
        seen.add(path)
        total_size += member.size
        raw.append((member, path))
    if not raw or len(raw) > 200_000:
        raise MaterializationError("toolbox archive has an invalid member count")
    if total_size > 100 * 1024**3:
        raise MaterializationError("toolbox archive expands beyond the 100 GiB safety limit")

    prefixes: set[tuple[str, ...]] = set()
    for _member, path in raw:
        if "pretrained_ckpts" in path.parts:
            index = path.parts.index("pretrained_ckpts")
            prefixes.add(path.parts[:index])
    if not prefixes:
        raise MaterializationError("toolbox archive has no pretrained_ckpts layout")
    shortest = min(len(prefix) for prefix in prefixes)
    candidates = {prefix for prefix in prefixes if len(prefix) == shortest}
    if len(candidates) != 1:
        raise MaterializationError("toolbox archive has ambiguous project roots")
    prefix = next(iter(candidates))
    if prefix and not all(path.parts[: len(prefix)] == prefix for _member, path in raw):
        raise MaterializationError("toolbox archive mixes entries inside and outside its root")

    normalized: list[tuple[tarfile.TarInfo, pathlib.PurePosixPath]] = []
    normalized_seen: set[pathlib.PurePosixPath] = set()
    for member, path in raw:
        stripped = pathlib.PurePosixPath(*path.parts[len(prefix) :])
        if not stripped.parts:
            continue
        if stripped in normalized_seen:
            raise MaterializationError(f"duplicate normalized tar member: {stripped}")
        normalized_seen.add(stripped)
        normalized.append((member, stripped))
    return normalized


def extract_toolbox(archive_path: pathlib.Path, destination: pathlib.Path) -> None:
    if destination.exists():
        raise MaterializationError(
            f"toolbox destination exists without matching evidence: {destination}"
        )
    staging_parent = destination.parent
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging = pathlib.Path(tempfile.mkdtemp(prefix=".runtime-root.", dir=staging_parent)).resolve()
    try:
        with tarfile.open(archive_path, mode="r:*") as archive:
            members = normalized_tar_members(archive)
            for member, relative in members:
                target = staging / pathlib.Path(*relative.parts)
                if not target.resolve().is_relative_to(staging):
                    raise MaterializationError(f"unsafe normalized tar member: {relative}")
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    target.chmod(0o755)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise MaterializationError(f"cannot read tar member: {member.name}")
                with source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                target.chmod(0o644)
        if not (staging / "pretrained_ckpts").is_dir():
            raise MaterializationError("normalized toolbox is missing pretrained_ckpts")
        os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def copy_verified(source: pathlib.Path, destination: pathlib.Path) -> None:
    source_hash = sha256(source)
    if destination.exists():
        if destination.is_file() and sha256(destination) == source_hash:
            return
        raise MaterializationError(f"refusing to overwrite materialized file: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = pathlib.Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        if sha256(temporary) != source_hash:
            raise MaterializationError(f"copy verification failed: {destination}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def verify_marker(
    marker: dict[str, Any],
    *,
    receipt_path: pathlib.Path,
    receipt: dict[str, Any],
    weight_root: pathlib.Path,
) -> None:
    expected = {
        "schema_version": 1,
        "status": "passed",
        "source_weights_receipt_sha256": sha256(receipt_path),
        "weights_root": str(weight_root),
        "source_git_commit": receipt["git_commit"],
        "checkout_mutations": False,
        "errors": [],
    }
    for key, value in expected.items():
        if marker.get(key) != value:
            raise MaterializationError(f"materialization marker mismatch: {key}")
    layouts = marker.get("layouts")
    if not isinstance(layouts, list) or not layouts:
        raise MaterializationError("materialization marker has no layouts")
    for layout in layouts:
        if not isinstance(layout, dict):
            raise MaterializationError("invalid materialization layout entry")
        destination = safe_destination(weight_root, layout.get("destination"))
        expected_files = layout.get("files")
        if not isinstance(expected_files, list) or inventory(destination) != expected_files:
            raise MaterializationError(
                f"materialized layout no longer matches evidence: {layout.get('id')}"
            )


def rebind_marker(
    marker: dict[str, Any],
    *,
    receipt_path: pathlib.Path,
    receipt: dict[str, Any],
    weight_root: pathlib.Path,
) -> dict[str, Any]:
    previous_hash = marker.get("source_weights_receipt_sha256")
    if not isinstance(previous_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", previous_hash):
        raise MaterializationError("existing marker has no valid source receipt hash")
    shadow = dict(marker)
    shadow["source_weights_receipt_sha256"] = sha256(receipt_path)
    verify_marker(
        shadow,
        receipt_path=receipt_path,
        receipt=receipt,
        weight_root=weight_root,
    )
    shadow["completed_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    return shadow


def materialize(
    receipt_path: pathlib.Path,
    receipt: dict[str, Any],
    weight_root: pathlib.Path,
) -> dict[str, Any]:
    toolbox_artifact = artifact_by_id(receipt, "4kagent-toolbox-archive")
    archive = source_file(
        weight_root,
        toolbox_artifact,
        "4KAgent_toolbox_pretrained_ckpts.tar.gz",
    )
    runtime_root = weight_root / "4kagent" / "runtime" / "toolbox-root"
    extract_toolbox(archive, runtime_root)

    depictqa_artifact = artifact_by_id(receipt, "4kagent-depictqa-dq495k")
    dq_source = source_file(weight_root, depictqa_artifact, "ckpt.pt")
    dq_destination = weight_root / "4kagent" / "runtime" / "depictqa" / "delta" / "DQ495K.pt"
    copy_verified(dq_source, dq_destination)

    return {
        "schema_version": 1,
        "status": "passed",
        "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_weights_receipt_sha256": sha256(receipt_path),
        "weights_root": str(weight_root),
        "source_git_commit": receipt["git_commit"],
        "layouts": [
            {
                "id": "4kagent-toolbox-runtime-root",
                "source_artifact_id": "4kagent-toolbox-archive",
                "destination": runtime_root.relative_to(weight_root).as_posix(),
                "files": inventory(runtime_root),
            },
            {
                "id": "4kagent-depictqa-dq495k",
                "source_artifact_id": "4kagent-depictqa-dq495k",
                "destination": dq_destination.relative_to(weight_root).as_posix(),
                "files": inventory(dq_destination),
            },
        ],
        "checkout_mutations": False,
        "errors": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-root", type=pathlib.Path, required=True)
    parser.add_argument("--receipt", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    weight_root = args.weights_root.resolve()
    receipt_path = args.receipt.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else receipt_path.with_name("materialization-receipt.json")
    )
    marker_path = weight_root / ".scaleguard-materialization.json"
    receipt = validate_download_receipt(receipt_path, weight_root)

    if args.verify_only:
        marker = load_object(marker_path, "materialization marker")
        verify_marker(
            marker,
            receipt_path=receipt_path,
            receipt=receipt,
            weight_root=weight_root,
        )
        if output != marker_path:
            write_atomic(output, marker_path.read_bytes())
        return 0

    if marker_path.exists():
        marker = load_object(marker_path, "materialization marker")
        marker = rebind_marker(
            marker,
            receipt_path=receipt_path,
            receipt=receipt,
            weight_root=weight_root,
        )
        payload = (json.dumps(marker, indent=2, sort_keys=True) + "\n").encode()
        write_atomic(marker_path, payload)
        if output != marker_path:
            write_atomic(output, payload)
        return 0

    marker = materialize(receipt_path, receipt, weight_root)
    payload = (json.dumps(marker, indent=2, sort_keys=True) + "\n").encode()
    write_atomic(marker_path, payload)
    if output != marker_path:
        write_atomic(output, payload)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (MaterializationError, OSError, tarfile.TarError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from None
