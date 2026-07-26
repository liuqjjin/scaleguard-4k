"""Command-line entry point."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from scaleguard import __version__
from scaleguard.config import PipelineConfig, load_config, validate_config
from scaleguard.controller.trusted_scale import TrustedScaleController
from scaleguard.doctor import run_doctor
from scaleguard.errors import ScaleGuardError
from scaleguard.evaluation.calibration import (
    CalibrationParameters,
    calibrate_from_manifests,
    verify_calibration_receipt,
)
from scaleguard.evaluation.metrics import (
    PYIQA_METRICS,
    SUPPORTED_METRICS,
    evaluate_metric_receipt,
)
from scaleguard.evaluation.summary import EXPERIMENT_GROUPS, summarize_paired_manifests
from scaleguard.factory import build_backends
from scaleguard.manifest import validate_run_manifest
from scaleguard.provenance import validate_runtime_preflight
from scaleguard.upstream import verify_upstreams


def find_project_root() -> Path:
    override = os.environ.get("SCALEGUARD_PROJECT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    candidates = [Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parents[2]]
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise ScaleGuardError(
        "cannot locate pyproject.toml; run from the repository or set SCALEGUARD_PROJECT_ROOT"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scaleguard",
        description="Trusted restoration and terminal scale control for 4KAgent and CoZ.",
        epilog="Configuration reference: docs/configuration.md in the source distribution.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
        help="print the installed ScaleGuard version and exit",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser(
        "run",
        help="run restoration and trusted terminal scaling",
        description="Run one configured restoration and trusted terminal-scale session.",
    )
    run.add_argument(
        "--config",
        type=Path,
        required=True,
        metavar="CONFIG",
        help="runtime YAML configuration",
    )
    run.add_argument(
        "--input",
        type=Path,
        required=True,
        metavar="IMAGE",
        help="source image",
    )
    run.add_argument(
        "--output",
        type=Path,
        required=True,
        metavar="IMAGE",
        help="destination for the validated final image",
    )
    run.add_argument(
        "--target-factor",
        type=int,
        choices=(1, 2, 4, 8, 16),
        metavar="FACTOR",
        help="override controller.target_factor with 1, 2, 4, 8, or 16",
    )
    run.add_argument(
        "--run-id",
        metavar="RUN_ID",
        help="safe unique run-directory name; generated when omitted",
    )
    run.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing published output after validation",
    )
    run.add_argument(
        "--runtime-preflight",
        type=Path,
        metavar="RECEIPT",
        help="validated AutoDL source/environment/weight receipt used for evidence promotion",
    )

    doctor = subparsers.add_parser(
        "doctor",
        help="check local/runtime readiness",
        description="Check configuration, source, environment, weight, and runtime readiness.",
    )
    doctor.add_argument(
        "--config",
        type=Path,
        required=True,
        metavar="CONFIG",
        help="runtime YAML configuration",
    )
    doctor.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    config = subparsers.add_parser("config", help="configuration operations")
    config_subparsers = config.add_subparsers(dest="config_command", required=True)
    config_validate = config_subparsers.add_parser(
        "validate",
        help="validate a strict runtime YAML",
        description="Validate a runtime YAML without starting either upstream model.",
    )
    config_validate.add_argument(
        "path",
        type=Path,
        metavar="CONFIG",
        help="runtime YAML configuration",
    )

    upstream = subparsers.add_parser("upstream", help="upstream checkout operations")
    upstream_subparsers = upstream.add_subparsers(dest="upstream_command", required=True)
    upstream_verify = upstream_subparsers.add_parser(
        "verify",
        help="verify locked checkout identities",
        description="Verify commit, tree, patch, and clean-checkout constraints.",
    )
    upstream_verify.add_argument(
        "--lock",
        type=Path,
        default=Path("upstream-lock.yaml"),
        metavar="LOCK",
        help="upstream lock YAML (default: upstream-lock.yaml)",
    )
    upstream_verify.add_argument(
        "--mapping",
        default="repositories",
        metavar="KEY",
        help="lock-file mapping to verify (default: repositories)",
    )
    upstream_verify.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    manifest = subparsers.add_parser("manifest", help="manifest operations")
    manifest_subparsers = manifest.add_subparsers(dest="manifest_command", required=True)
    manifest_validate = manifest_subparsers.add_parser(
        "validate",
        help="validate a run manifest and its artifacts",
        description="Validate manifest structure and hash every referenced image.",
    )
    manifest_validate.add_argument(
        "path",
        type=Path,
        metavar="MANIFEST",
        help="run manifest JSON",
    )
    manifest_validate.add_argument(
        "--artifact-root",
        type=Path,
        metavar="DIR",
        help="base directory for relative artifact paths (default: manifest directory)",
    )

    evaluation = subparsers.add_parser(
        "evaluation",
        help="calibrate, verify, summarize, or measure evidence",
    )
    evaluation_subparsers = evaluation.add_subparsers(
        dest="evaluation_command",
        required=True,
    )
    calibrate = evaluation_subparsers.add_parser(
        "calibrate",
        help="create a gate calibration receipt from labeled manifests",
    )
    calibrate.add_argument(
        "--manifest",
        type=Path,
        action="append",
        required=True,
        metavar="MANIFEST",
        help="labeled run manifest; repeat for multiple runs",
    )
    calibrate.add_argument(
        "--labels",
        type=Path,
        required=True,
        metavar="CSV",
        help="CSV with run_id, step_index, and acceptable columns",
    )
    calibrate.add_argument(
        "--output",
        type=Path,
        required=True,
        metavar="RECEIPT",
        help="destination calibration receipt JSON",
    )
    calibrate.add_argument(
        "--artifact-root",
        type=Path,
        metavar="DIR",
        help="base for relative artifacts; enables use outside a checkout",
    )
    calibrate.add_argument(
        "--minimum-acceptable-samples",
        type=int,
        default=20,
        metavar="COUNT",
        help="minimum real acceptable samples required (default: 20)",
    )
    calibrate.add_argument(
        "--quality-lower-quantile",
        type=float,
        default=0.05,
        metavar="Q",
        help="quality-gain lower quantile (default: 0.05)",
    )
    calibrate.add_argument(
        "--error-upper-quantile",
        type=float,
        default=0.95,
        metavar="Q",
        help="consistency-error upper quantile (default: 0.95)",
    )
    calibrate.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        metavar="COUNT",
        help="bootstrap resamples per threshold (default: 2000)",
    )
    calibrate.add_argument(
        "--bootstrap-confidence",
        type=float,
        default=0.95,
        metavar="LEVEL",
        help="bootstrap interval confidence (default: 0.95)",
    )
    calibrate.add_argument(
        "--bootstrap-seed",
        type=int,
        default=20250727,
        metavar="SEED",
        help="deterministic bootstrap seed (default: 20250727)",
    )
    calibrate.add_argument(
        "--include-measurement",
        action="store_true",
        help="also calibrate the configured observation-consistency gate",
    )

    verify = evaluation_subparsers.add_parser(
        "verify",
        help="verify a calibration receipt against a runtime config",
    )
    verify.add_argument(
        "--receipt",
        type=Path,
        required=True,
        metavar="RECEIPT",
        help="calibration receipt JSON",
    )
    verify.add_argument(
        "--config",
        type=Path,
        required=True,
        metavar="CONFIG",
        help="runtime YAML whose gates must match the receipt",
    )

    summarize = evaluation_subparsers.add_parser(
        "summarize",
        help="write paired CSV/JSON evidence for the four ablation groups",
    )
    summarize.add_argument(
        "--group",
        action="append",
        required=True,
        metavar="GROUP=MANIFEST",
        help="repeat for each manifest; GROUP is A-only, B-only, AB-fixed, or ScaleGuard",
    )
    summarize.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        metavar="CSV",
        help="destination paired-sample table",
    )
    summarize.add_argument(
        "--output-json",
        type=Path,
        required=True,
        metavar="JSON",
        help="destination summary receipt",
    )
    summarize.add_argument(
        "--artifact-root",
        type=Path,
        metavar="DIR",
        help="base for relative artifacts; enables use outside a checkout",
    )

    metrics = evaluation_subparsers.add_parser(
        "metrics",
        help="measure hash-verified final images against aligned references",
    )
    metrics.add_argument(
        "--manifest",
        type=Path,
        action="append",
        required=True,
        metavar="MANIFEST",
        help="hash-valid run manifest; repeat in sample order",
    )
    metrics.add_argument(
        "--reference",
        type=Path,
        action="append",
        required=True,
        metavar="IMAGE",
        help="aligned reference image; repeat in manifest order",
    )
    metrics.add_argument(
        "--metric",
        action="append",
        choices=SUPPORTED_METRICS,
        metavar="NAME",
        help="psnr, ssim, lpips, musiq, or clipiqa; repeat (default: psnr and ssim)",
    )
    metrics.add_argument(
        "--pyiqa-weight",
        action="append",
        default=[],
        metavar="METRIC=PATH",
        help="explicit local checkpoint; required for each requested PyIQA metric",
    )
    metrics.add_argument(
        "--pyiqa-backbone",
        action="append",
        default=[],
        metavar="lpips=PATH",
        help="explicit local AlexNet checkpoint required by LPIPS",
    )
    metrics.add_argument(
        "--device",
        default="cpu",
        metavar="DEVICE",
        help="PyIQA execution device (default: cpu)",
    )
    metrics.add_argument(
        "--crop-border",
        type=int,
        default=0,
        metavar="PIXELS",
        help="border excluded from full-reference metrics (default: 0)",
    )
    metrics.add_argument(
        "--output",
        type=Path,
        required=True,
        metavar="RECEIPT",
        help="destination metric receipt JSON",
    )
    metrics.add_argument(
        "--artifact-root",
        type=Path,
        metavar="DIR",
        help="base for relative artifacts; enables use outside a checkout",
    )
    return parser


def _override_target(config: PipelineConfig, target: int | None) -> PipelineConfig:
    if target is None:
        return config
    updated = dataclasses.replace(
        config,
        controller=dataclasses.replace(config.controller, target_factor=target),
    )
    validate_config(updated)
    return updated


def _run_command(args: argparse.Namespace, project_root: Path) -> int:
    if not args.input.is_file():
        raise ScaleGuardError(f"input image does not exist: {args.input}")
    if args.output.exists() and not args.overwrite:
        raise ScaleGuardError(
            f"output already exists: {args.output}; pass --overwrite to replace it"
        )
    config = _override_target(load_config(args.config), args.target_factor)
    if args.runtime_preflight is not None and args.target_factor is not None:
        raise ScaleGuardError(
            "--target-factor cannot be combined with --runtime-preflight; "
            "bind the requested target in the preflighted config"
        )
    provenance: dict[str, object] = {"scaleguard_version": __version__}
    if args.runtime_preflight is not None:
        provenance.update(
            validate_runtime_preflight(
                args.runtime_preflight,
                config_path=args.config,
                project_root=project_root,
            )
        )
    restoration, scale = build_backends(config, project_root=project_root)
    controller = TrustedScaleController(
        config,
        restoration,
        scale,
        provenance=provenance,
        project_root=project_root,
    )
    output = controller.run(args.input, args.output, run_id=args.run_id)
    if controller.last_run_dir is None:
        raise ScaleGuardError("controller did not retain a run directory")
    manifest = validate_run_manifest(controller.last_run_dir / "manifest.json")
    payload = {
        "status": "ok",
        "run_status": manifest["status"],
        "completion_level": manifest["completion_level"],
        "requested_factor": manifest["requested_factor"],
        "achieved_factor": manifest["achieved_factor"],
        "target_reached": manifest["target_reached"],
        "output": str(output.resolve()),
        "run_dir": str(controller.last_run_dir),
        "mock": config.is_mock,
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


def _doctor_command(args: argparse.Namespace, project_root: Path) -> int:
    checks = run_doctor(load_config(args.config), project_root)
    if args.json:
        print(json.dumps([dataclasses.asdict(check) for check in checks], indent=2))
    else:
        for check in checks:
            print(f"{check.status.upper():5} {check.name}: {check.detail}")
    return 1 if any(check.status == "fail" for check in checks) else 0


def _upstream_command(args: argparse.Namespace, project_root: Path) -> int:
    results = verify_upstreams(args.lock, project_root, args.mapping)
    if args.json:
        print(json.dumps([dataclasses.asdict(result) for result in results], indent=2))
    else:
        for result in results:
            status = "PASS" if result.ok else "FAIL"
            print(f"{status} {result.target}.{result.check}: {result.detail}")
    return 1 if any(not result.ok for result in results) else 0


def _manifest_command(args: argparse.Namespace) -> int:
    manifest = validate_run_manifest(
        args.path,
        artifact_root=args.artifact_root,
    )
    print(json.dumps({"status": "ok", "run_id": manifest["run_id"]}))
    return 0


def _parse_experiment_groups(specifications: Sequence[str]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for specification in specifications:
        group, separator, raw_path = specification.partition("=")
        if not separator or not raw_path:
            raise ScaleGuardError(f"invalid --group {specification!r}; expected GROUP=MANIFEST")
        if group not in EXPERIMENT_GROUPS:
            raise ScaleGuardError(
                f"invalid experiment group {group!r}; expected one of "
                + ", ".join(EXPERIMENT_GROUPS)
            )
        groups.setdefault(group, []).append(Path(raw_path))
    return groups


def _parse_metric_paths(
    specifications: Sequence[str],
    *,
    accepted: Sequence[str],
    option: str,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for specification in specifications:
        metric, separator, raw_path = specification.partition("=")
        if not separator or not raw_path:
            raise ScaleGuardError(f"invalid {option} {specification!r}; expected METRIC=PATH")
        if metric not in accepted:
            raise ScaleGuardError(
                f"invalid {option} metric {metric!r}; expected one of " + ", ".join(accepted)
            )
        if metric in paths:
            raise ScaleGuardError(f"duplicate {option} for {metric}")
        paths[metric] = Path(raw_path)
    return paths


def _evaluation_command(
    args: argparse.Namespace,
    project_root: Path | None = None,
) -> int:
    if args.evaluation_command == "verify":
        valid, reasons = verify_calibration_receipt(args.receipt, args.config)
        print(json.dumps({"valid": valid, "reasons": reasons}))
        return 0 if valid else 1
    artifact_root = getattr(args, "artifact_root", None)
    if artifact_root is None and project_root is None:
        raise ScaleGuardError(f"evaluation {args.evaluation_command} requires the project root")
    artifact_root = artifact_root or project_root
    if args.evaluation_command == "calibrate":
        parameters = CalibrationParameters(
            minimum_acceptable_samples=args.minimum_acceptable_samples,
            quality_lower_quantile=args.quality_lower_quantile,
            error_upper_quantile=args.error_upper_quantile,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_confidence=args.bootstrap_confidence,
            bootstrap_seed=args.bootstrap_seed,
            include_measurement=args.include_measurement,
        )
        receipt = calibrate_from_manifests(
            args.manifest,
            args.labels,
            args.output,
            parameters=parameters,
            artifact_root=artifact_root,
        )
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
    if args.evaluation_command == "summarize":
        summary = summarize_paired_manifests(
            _parse_experiment_groups(args.group),
            args.output_csv,
            args.output_json,
            artifact_root=artifact_root,
        )
        print(
            json.dumps(
                {
                    "status": "ok",
                    "output_csv": str(args.output_csv.resolve()),
                    "output_json": str(args.output_json.resolve()),
                    **summary["counts"],
                }
            )
        )
        return 0
    if args.evaluation_command == "metrics":
        receipt = evaluate_metric_receipt(
            args.manifest,
            args.reference,
            args.output,
            metric_names=args.metric or ("psnr", "ssim"),
            crop_border=args.crop_border,
            device=args.device,
            pyiqa_weights=_parse_metric_paths(
                args.pyiqa_weight,
                accepted=PYIQA_METRICS,
                option="--pyiqa-weight",
            ),
            pyiqa_backbones=_parse_metric_paths(
                args.pyiqa_backbone,
                accepted=("lpips",),
                option="--pyiqa-backbone",
            ),
            artifact_root=artifact_root,
        )
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "output": str(args.output.resolve()),
                    **receipt["counts"],
                }
            )
        )
        return 0 if receipt["status"] == "completed" else 1
    raise ScaleGuardError(f"unsupported evaluation command: {args.evaluation_command}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "config" and args.config_command == "validate":
            config = load_config(args.path)
            print(json.dumps({"status": "ok", "mock": config.is_mock}))
            return 0
        if args.command == "manifest" and args.manifest_command == "validate":
            return _manifest_command(args)
        if args.command == "evaluation" and args.evaluation_command == "verify":
            return _evaluation_command(args)
        if args.command == "evaluation" and args.artifact_root is not None:
            return _evaluation_command(args)

        project_root = find_project_root()
        if args.command == "run":
            return _run_command(args, project_root)
        if args.command == "doctor":
            return _doctor_command(args, project_root)
        if args.command == "upstream" and args.upstream_command == "verify":
            return _upstream_command(args, project_root)
        if args.command == "evaluation":
            return _evaluation_command(args, project_root)
        raise ScaleGuardError(f"unsupported command: {args.command}")
    except (ScaleGuardError, OSError, ValueError) as error:
        print(f"scaleguard: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
