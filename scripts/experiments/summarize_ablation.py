#!/usr/bin/env python3
"""Expand ablation run directories into paired, non-imputed evidence tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scaleguard.errors import ScaleGuardError
from scaleguard.evaluation.summary import summarize_paired_manifests


def _expand(paths: list[Path] | None) -> list[Path]:
    manifests: set[Path] = set()
    for path in paths or []:
        if path.is_file():
            manifests.add(path)
        elif path.is_dir():
            manifests.update(path.rglob("manifest.json"))
        else:
            raise ValueError(f"manifest path does not exist: {path}")
    return sorted(manifests)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-only", type=Path, action="append")
    parser.add_argument("--b-only", type=Path, action="append")
    parser.add_argument("--ab-fixed", type=Path, action="append")
    parser.add_argument("--scaleguard", type=Path, action="append")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path)
    args = parser.parse_args()
    try:
        summary = summarize_paired_manifests(
            {
                "A-only": _expand(args.a_only),
                "B-only": _expand(args.b_only),
                "AB-fixed": _expand(args.ab_fixed),
                "ScaleGuard": _expand(args.scaleguard),
            },
            args.output_csv,
            args.output_json,
            artifact_root=args.artifact_root,
        )
    except (OSError, ScaleGuardError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({"status": "ok", **summary["counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
