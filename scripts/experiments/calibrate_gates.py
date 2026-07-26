#!/usr/bin/env python3
"""Expand run directories and create one evidence-bound calibration receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scaleguard.errors import ScaleGuardError
from scaleguard.evaluation.calibration import (
    CalibrationParameters,
    calibrate_from_manifests,
)


def _expand(paths: list[Path]) -> list[Path]:
    manifests: set[Path] = set()
    for path in paths:
        if path.is_file():
            manifests.add(path)
        elif path.is_dir():
            manifests.update(path.rglob("manifest.json"))
        else:
            raise ValueError(f"manifest path does not exist: {path}")
    if not manifests:
        raise ValueError("no manifest.json files were found")
    return sorted(manifests)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        type=Path,
        action="append",
        required=True,
        help="manifest file or run directory; repeat as needed",
    )
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--minimum-acceptable-samples", type=int, default=20)
    parser.add_argument("--quality-lower-quantile", type=float, default=0.05)
    parser.add_argument("--error-upper-quantile", type=float, default=0.95)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=20250727)
    parser.add_argument("--include-measurement", action="store_true")
    args = parser.parse_args()
    try:
        receipt = calibrate_from_manifests(
            _expand(args.runs),
            args.labels,
            args.output,
            parameters=CalibrationParameters(
                minimum_acceptable_samples=args.minimum_acceptable_samples,
                quality_lower_quantile=args.quality_lower_quantile,
                error_upper_quantile=args.error_upper_quantile,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_confidence=args.bootstrap_confidence,
                bootstrap_seed=args.bootstrap_seed,
                include_measurement=args.include_measurement,
            ),
            artifact_root=args.artifact_root,
        )
    except (OSError, ScaleGuardError, ValueError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "output": str(args.output.resolve()),
                "acceptable_real": receipt["sample_counts"]["acceptable_real"],
            }
        )
    )
    return 0 if receipt["status"] == "calibrated" else 1


if __name__ == "__main__":
    raise SystemExit(main())
