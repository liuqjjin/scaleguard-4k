#!/usr/bin/env python3
"""Write a source-bound receipt after the AutoDL preflight succeeds."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scaleguard.provenance import (  # noqa: E402
    LOCK_PATHS,
    require_clean_git_commit,
    resolve_materialization_sources,
    sha256,
    validate_runtime_preflight,
)


def _from_project(value: str) -> pathlib.Path:
    path = pathlib.Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--materialization", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    config = args.config.resolve()
    materialization = args.materialization.resolve()
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
        "schema_version": 1,
        "status": "passed",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "project_commit": commit,
        "config": {"path": str(config), "sha256": sha256(config)},
        "locks": {relative: sha256(PROJECT_ROOT / relative) for relative in LOCK_PATHS},
        "bootstrap": {"path": str(bootstrap), "sha256": sha256(bootstrap)},
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
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        validate_runtime_preflight(
            temporary,
            config_path=config,
            project_root=PROJECT_ROOT,
        )
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
