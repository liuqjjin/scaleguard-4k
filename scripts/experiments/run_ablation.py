#!/usr/bin/env python3
"""Execute four paired ablation groups through the AutoDL experiment wrapper."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from scaleguard.errors import ScaleGuardError
from scaleguard.experiments import run_ablation_suite

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "experiments" / "ablation.yaml"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize and execute paired A-only, B-only, AB-fixed, and "
            "ScaleGuard jobs. Every job goes through the fixed AutoDL experiment "
            "wrapper and creates a fresh runtime preflight."
        )
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_PROTOCOL,
        help="strict executable ablation protocol",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        required=True,
        help="real AutoDL runtime configuration used as the immutable base",
    )
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="authorized input image; repeat for multiple paired samples",
    )
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        help="declared CoZ seed; repeat for multiple seeds (default: base config seed)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="new or empty suite directory; partial and failed evidence is retained",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="write hash-bound input snapshots, configs, argv, and receipt without execution",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        receipt = run_ablation_suite(
            protocol_path=args.protocol,
            base_config_path=args.base_config,
            inputs=args.input,
            seeds=args.seed,
            output_directory=args.output_dir,
            project_root=PROJECT_ROOT,
            plan_only=args.plan_only,
        )
    except (OSError, ScaleGuardError, ValueError) as error:
        parser.error(str(error))
    summary = {
        "status": receipt["status"],
        "receipt": str(Path(receipt["output_directory"]) / "suite-receipt.json"),
        "counts": receipt["counts"],
    }
    print(json.dumps(summary, sort_keys=True))
    return 0 if receipt["status"] in {"planned", "passed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
