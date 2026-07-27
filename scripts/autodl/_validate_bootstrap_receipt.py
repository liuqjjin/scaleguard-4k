#!/usr/bin/env python3
"""Validate and snapshot the project AutoDL environment receipts."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import re
import sys
from typing import Any

from scaleguard.provenance import (
    RuntimePreflightError,
    load_regular_file_snapshot,
    validate_environment_receipt,
)
from scaleguard.strict_json import StrictJSONError, loads_object


class ReceiptError(ValueError):
    """Raised when a bootstrap receipt does not bind the current source tree."""


EXPECTED_LOCKS = (
    "uv.lock",
    "upstream-lock.yaml",
    "runtime-dependencies.yaml",
    "environments/uv.version",
    "environments/python-downloads.json",
    "environments/bootstrap/uv.lock",
    "environments/bootstrap/uv-binary.sha256",
    "environments/4kagent/requirements.lock",
    "environments/4kagent/requirements.resolved.lock",
    "environments/4kagent/pyiqa.override.lock",
    "environments/4kagent/hpsv2.override.lock",
    "environments/depictqa/requirements.lock",
    "environments/depictqa/requirements.resolved.lock",
    "environments/coz/requirements.lock",
    "environments/coz/requirements.resolved.lock",
)
EXPECTED_ENVIRONMENTS = ("scaleguard", "4kagent", "depictqa", "coz")
EXPECTED_PYTHON_DISTRIBUTION = "cpython-3.10.18-linux-x86_64-gnu"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_snapshot(path: pathlib.Path, label: str) -> tuple[bytes, dict[str, Any]]:
    try:
        payload = path.read_bytes()
        value = loads_object(payload)
    except (OSError, StrictJSONError) as exc:
        raise ReceiptError(f"cannot read {label} {path}: {exc}") from exc
    return payload, value


def timestamp(value: object, label: str) -> dt.datetime:
    if not isinstance(value, str):
        raise ReceiptError(f"{label} is not a timestamp")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ReceiptError(f"{label} is not valid ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ReceiptError(f"{label} must include a timezone")
    return parsed.astimezone(dt.timezone.utc)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--project-root", type=pathlib.Path, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--not-before", required=True)
    parser.add_argument("--destination", type=pathlib.Path, required=True)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    runtime_root = project_root / ".runtime"
    receipt_directory = runtime_root / "receipts"
    for candidate in (runtime_root, receipt_directory):
        if candidate.is_symlink():
            raise ReceiptError(f"runtime receipt path must not be a symlink: {candidate}")
    source = args.source.resolve()
    expected_source = receipt_directory / "bootstrap.json"
    if source != expected_source.resolve():
        raise ReceiptError(f"unexpected aggregate receipt path: {source}")
    if not re.fullmatch(r"[0-9a-f]{40}", args.git_commit):
        raise ReceiptError("git commit must be a full lowercase SHA-1")
    try:
        not_before = dt.datetime.strptime(
            args.not_before,
            "%Y%m%dT%H%M%SZ",
        ).replace(tzinfo=dt.timezone.utc)
    except ValueError as exc:
        raise ReceiptError("--not-before must use YYYYMMDDTHHMMSSZ") from exc

    aggregate_bytes, aggregate = load_snapshot(source, "aggregate receipt")
    if aggregate.get("schema_version") != 1 or aggregate.get("status") != "passed":
        raise ReceiptError("aggregate environment receipt did not pass")
    if aggregate.get("project_commit") != args.git_commit:
        raise ReceiptError("aggregate environment receipt is bound to another commit")
    if timestamp(aggregate.get("created_at_utc"), "aggregate created_at_utc") < not_before:
        raise ReceiptError("aggregate environment receipt predates this bootstrap attempt")
    if aggregate.get("python_version") != "3.10.18":
        raise ReceiptError("aggregate environment receipt has an unexpected Python version")
    expected_uv = (project_root / "environments" / "uv.version").read_text(encoding="utf-8").strip()
    if aggregate.get("uv_version") != expected_uv:
        raise ReceiptError("aggregate environment receipt has an unexpected uv version")
    expected_uv_binary = (
        (project_root / "environments/bootstrap/uv-binary.sha256")
        .read_text(encoding="utf-8")
        .strip()
    )
    if (
        re.fullmatch(r"[0-9a-f]{64}", expected_uv_binary) is None
        or aggregate.get("uv_binary_sha256") != expected_uv_binary
    ):
        raise ReceiptError("aggregate environment receipt has an unexpected uv binary identity")
    _python_download_bytes, python_downloads = load_snapshot(
        project_root / "environments/python-downloads.json",
        "managed Python distribution lock",
    )
    if set(python_downloads) != {EXPECTED_PYTHON_DISTRIBUTION}:
        raise ReceiptError("managed Python distribution lock has an unexpected build set")
    python_distribution = python_downloads[EXPECTED_PYTHON_DISTRIBUTION]
    if not isinstance(python_distribution, dict):
        raise ReceiptError("managed Python distribution lock is invalid")
    expected_python_distribution = {
        "key": EXPECTED_PYTHON_DISTRIBUTION,
        "build": python_distribution.get("build"),
        "url": python_distribution.get("url"),
        "archive_sha256": python_distribution.get("sha256"),
    }
    if (
        python_distribution.get("name") != "cpython"
        or python_distribution.get("major") != 3
        or python_distribution.get("minor") != 10
        or python_distribution.get("patch") != 18
        or python_distribution.get("os") != "linux"
        or python_distribution.get("arch") != {"family": "x86_64", "variant": None}
        or python_distribution.get("libc") != "gnu"
        or not isinstance(expected_python_distribution["build"], str)
        or not isinstance(expected_python_distribution["url"], str)
        or not str(expected_python_distribution["url"]).startswith(
            "https://github.com/astral-sh/python-build-standalone/releases/download/"
        )
        or not isinstance(expected_python_distribution["archive_sha256"], str)
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(expected_python_distribution["archive_sha256"]),
        )
        is None
        or aggregate.get("python_distribution") != expected_python_distribution
    ):
        raise ReceiptError(
            "aggregate environment receipt has an unexpected managed Python distribution"
        )
    platform = aggregate.get("platform")
    if (
        not isinstance(platform, dict)
        or platform.get("system") != "Linux"
        or platform.get("machine") != "x86_64"
        or not isinstance(platform.get("glibc"), str)
    ):
        raise ReceiptError("aggregate environment receipt has an unexpected platform")

    locks = aggregate.get("locks")
    if not isinstance(locks, dict) or set(locks) != set(EXPECTED_LOCKS):
        raise ReceiptError("aggregate environment receipt has an unexpected lock set")
    for relative in EXPECTED_LOCKS:
        lock_path = project_root / relative
        if not lock_path.is_file() or locks.get(relative) != sha256(lock_path):
            raise ReceiptError(f"aggregate receipt lock mismatch: {relative}")

    environments = aggregate.get("environments")
    if not isinstance(environments, dict) or set(environments) != set(EXPECTED_ENVIRONMENTS):
        raise ReceiptError("aggregate receipt has an unexpected environment set")

    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    environment_summary: dict[str, dict[str, str]] = {}
    receipt_root = source.parent.resolve()
    for name in EXPECTED_ENVIRONMENTS:
        record = environments[name]
        if not isinstance(record, dict):
            raise ReceiptError(f"invalid aggregate environment record: {name}")
        relative_path = record.get("path")
        expected_digest = record.get("sha256")
        observed_status = record.get("status")
        required_status = "passed_with_audited_override" if name == "4kagent" else "passed"
        if (
            not isinstance(relative_path, str)
            or not isinstance(expected_digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected_digest)
            or observed_status != required_status
        ):
            raise ReceiptError(f"invalid aggregate environment evidence: {name}")
        receipt_path = (project_root / relative_path).resolve()
        if not receipt_path.is_relative_to(receipt_root):
            raise ReceiptError(f"environment receipt escapes its runtime root: {name}")
        try:
            _validated_path, receipt, validated_digest = validate_environment_receipt(
                name,
                record,
                project_root=project_root,
                expected_path=receipt_path,
                context=f"bootstrap environment {name}",
            )
            payload, copied_digest = load_regular_file_snapshot(
                receipt_path,
                f"{name} receipt",
            )
        except RuntimePreflightError as exc:
            raise ReceiptError(f"environment receipt content mismatch: {name}: {exc}") from exc
        if (
            validated_digest != expected_digest
            or copied_digest != expected_digest
            or sha256_bytes(payload) != expected_digest
        ):
            raise ReceiptError(f"environment receipt hash mismatch: {name}")
        if timestamp(receipt.get("created_at_utc"), f"{name} created_at_utc") < not_before:
            raise ReceiptError(f"environment receipt predates this bootstrap attempt: {name}")
        copied = destination / f"{name}.json"
        copied.write_bytes(payload)
        environment_summary[name] = {
            "path": copied.name,
            "sha256": expected_digest,
            "status": observed_status,
        }

    aggregate_copy = destination / "bootstrap.json"
    aggregate_copy.write_bytes(aggregate_bytes)
    validation = {
        "schema_version": 1,
        "status": "passed",
        "project_commit": args.git_commit,
        "aggregate_receipt": {
            "path": aggregate_copy.name,
            "sha256": sha256_bytes(aggregate_bytes),
        },
        "environments": environment_summary,
        "claim": (
            "The project hook receipts match this commit and every declared lock. "
            "No GPU execution or model-quality result is asserted."
        ),
    }
    (destination / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ReceiptError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
