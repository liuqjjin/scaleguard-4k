#!/usr/bin/env python3
"""Extract and verify the ScaleGuard run manifest named by CLI JSON output."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import sys
from typing import Any, NamedTuple

from PIL import Image, ImageChops

from scaleguard.config import PipelineConfig, parse_config  # type: ignore[import-untyped]
from scaleguard.errors import ConfigurationError
from scaleguard.manifest import ManifestValidationError, validate_run_manifest
from scaleguard.provenance import (
    RuntimePreflightError,
    bind_runtime_config,
    load_regular_file_snapshot,
    validate_runtime_preflight,
)
from scaleguard.strict_json import StrictJSONError, loads


class EvidenceError(ValueError):
    """Raised when a successful CLI exit lacks real pipeline evidence."""


class RuntimeContext(NamedTuple):
    """Preflight-validated configuration and its immutable source digests."""

    config: PipelineConfig
    provenance: dict[str, Any]
    config_sha256: str
    preflight_sha256: str


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_cli_result(log_path: pathlib.Path) -> dict[str, Any]:
    result: dict[str, Any] | None = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            candidate = loads(line)
        except StrictJSONError:
            continue
        if (
            isinstance(candidate, dict)
            and candidate.get("status") == "ok"
            and "run_dir" in candidate
            and "output" in candidate
        ):
            result = candidate
    if result is None:
        raise EvidenceError("ScaleGuard CLI emitted no successful run JSON")
    return result


def backend_name(manifest: dict[str, Any], key: str) -> str:
    provenance = manifest.get("provenance")
    value = provenance.get(key) if isinstance(provenance, dict) else None
    if not isinstance(value, str) or not value or "fake" in value.lower():
        raise EvidenceError(f"manifest has no real {key}")
    return value


def timestamp(value: object, label: str) -> dt.datetime:
    if not isinstance(value, str):
        raise EvidenceError(f"{label} is not a timestamp")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EvidenceError(f"{label} is not valid ISO-8601: {value!r}") from exc
    if parsed.tzinfo is None:
        raise EvidenceError(f"{label} must include a timezone")
    return parsed.astimezone(dt.timezone.utc)


def same_rgb_pixels(left: pathlib.Path, right: pathlib.Path) -> bool:
    try:
        with Image.open(left) as left_image, Image.open(right) as right_image:
            left_rgb = left_image.convert("RGB")
            right_rgb = right_image.convert("RGB")
            return (
                left_rgb.size == right_rgb.size
                and ImageChops.difference(left_rgb, right_rgb).getbbox() is None
            )
    except (OSError, ValueError) as exc:
        raise EvidenceError(f"cannot compare input images: {exc}") from exc


def _runtime_context(
    *,
    expected_config: pathlib.Path,
    runtime_preflight: pathlib.Path,
    project_root: pathlib.Path,
) -> RuntimeContext:
    try:
        config_payload, config_digest = load_regular_file_snapshot(
            expected_config,
            "invoked runtime config",
        )
        preflight_payload, preflight_digest = load_regular_file_snapshot(
            runtime_preflight,
            "runtime preflight receipt",
        )
        if not preflight_payload:
            raise RuntimePreflightError("runtime preflight receipt is empty")
        validated = validate_runtime_preflight(
            runtime_preflight,
            config_path=expected_config,
            project_root=project_root,
        )
        binding = validated.get("runtime_execution_binding")
        if not isinstance(binding, dict):
            raise RuntimePreflightError(
                "runtime preflight did not reconstruct an execution binding"
            )
        parsed = parse_config(config_payload, source=expected_config)
        bound = bind_runtime_config(
            parsed,
            project_root=project_root,
            binding=binding,
        )
    except (ConfigurationError, OSError, RuntimePreflightError) as exc:
        raise EvidenceError(f"runtime preflight/config binding is invalid: {exc}") from exc

    if validated.get("runtime_config_sha256") != config_digest:
        raise EvidenceError("runtime preflight config digest differs from the invoked config")
    if validated.get("runtime_preflight_sha256") != preflight_digest:
        raise EvidenceError("runtime preflight receipt digest differs from current bytes")
    if pathlib.Path(str(validated.get("runtime_config_path", ""))).resolve() != expected_config:
        raise EvidenceError("runtime preflight is bound to another config path")
    if (
        pathlib.Path(str(validated.get("runtime_preflight_receipt", ""))).resolve()
        != runtime_preflight
    ):
        raise EvidenceError("runtime preflight provenance names another receipt")
    if pathlib.Path(str(validated.get("project_root", ""))).resolve() != project_root:
        raise EvidenceError("runtime preflight is bound to another project root")
    return RuntimeContext(
        config=bound,
        provenance=validated,
        config_sha256=config_digest,
        preflight_sha256=preflight_digest,
    )


def _successful_process(value: object) -> bool:
    return isinstance(value, dict) and value.get("returncode") == 0


def _real_persistent_candidates(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    steps = manifest.get("steps")
    if not isinstance(steps, list):
        return candidates
    for value in steps:
        if not isinstance(value, dict):
            continue
        candidate = value.get("candidate")
        metadata = value.get("worker_metadata")
        if (
            isinstance(candidate, dict)
            and candidate.get("mock") is False
            and isinstance(metadata, dict)
            and metadata.get("backend") == "chain_of_zoom_persistent"
        ):
            candidates.append(value)
    return candidates


def _require_4kagent(manifest: dict[str, Any]) -> None:
    metadata = manifest.get("restoration_metadata")
    if (
        backend_name(manifest, "restoration_backend") != "4kagent_upstream"
        or not isinstance(metadata, dict)
        or metadata.get("backend") != "4kagent_upstream"
        or not _successful_process(manifest.get("restoration_process"))
    ):
        raise EvidenceError("experiment contract requires a successful real 4KAgent process")


def _require_identity_restoration(manifest: dict[str, Any]) -> None:
    metadata = manifest.get("restoration_metadata")
    if (
        backend_name(manifest, "restoration_backend") != "scaleguard_identity_observation"
        or not isinstance(metadata, dict)
        or metadata.get("backend") != "scaleguard_identity_observation"
        or metadata.get("algorithmic_restoration") is not False
        or manifest.get("restoration_process") is not None
    ):
        raise EvidenceError("B-only requires the non-algorithmic identity observation")


def _require_candidate_contract(
    manifest: dict[str, Any],
    *,
    expected_count: int,
) -> list[dict[str, Any]]:
    steps = manifest.get("steps")
    candidates = _real_persistent_candidates(manifest)
    if not isinstance(steps, list) or len(steps) != expected_count:
        raise EvidenceError(f"experiment contract requires exactly {expected_count} CoZ step(s)")
    if len(candidates) != expected_count:
        raise EvidenceError(
            f"experiment contract requires exactly {expected_count} real CoZ candidate(s)"
        )
    if expected_count == 0:
        if manifest.get("scale_session_process") is not None:
            raise EvidenceError("A-only must not start a Chain-of-Zoom session")
    elif not _successful_process(manifest.get("scale_session_process")):
        raise EvidenceError(
            "experiment contract requires a successful persistent Chain-of-Zoom session"
        )
    return candidates


def _validate_experiment_contract(
    manifest: dict[str, Any],
    *,
    group: str,
) -> int:
    status = manifest.get("status")
    completion = manifest.get("completion_level")
    target_reached = manifest.get("target_reached")
    if backend_name(manifest, "scale_backend") != "chain_of_zoom":
        raise EvidenceError("experiment contract requires the Chain-of-Zoom backend")

    if group == "A-only":
        if status != "succeeded" or completion != "STATIC_READY" or target_reached is not True:
            raise EvidenceError("A-only requires succeeded/STATIC_READY target-reaching evidence")
        _require_4kagent(manifest)
        return len(_require_candidate_contract(manifest, expected_count=0))

    candidates = _require_candidate_contract(manifest, expected_count=1)
    step = candidates[0]
    if group == "B-only":
        if status != "succeeded" or completion != "STATIC_READY" or target_reached is not True:
            raise EvidenceError("B-only requires succeeded/STATIC_READY target-reaching evidence")
        _require_identity_restoration(manifest)
        if step.get("accepted") is not True:
            raise EvidenceError("B-only fixed policy must accept its one real CoZ candidate")
    elif group == "AB-fixed":
        if status != "succeeded" or completion != "STATIC_READY" or target_reached is not True:
            raise EvidenceError("AB-fixed requires succeeded/STATIC_READY target-reaching evidence")
        _require_4kagent(manifest)
        if step.get("accepted") is not True:
            raise EvidenceError("AB-fixed must accept its one real CoZ candidate")
    elif group == "ScaleGuard":
        _require_4kagent(manifest)
        integrated = (
            status == "succeeded"
            and completion == "AB_INTEGRATED"
            and target_reached is True
            and step.get("accepted") is True
        )
        events = manifest.get("events")
        final_metrics = manifest.get("final_metrics")
        selected_scale = (
            final_metrics.get("selected_scale") if isinstance(final_metrics, dict) else None
        )
        final_gate_rollback = (
            isinstance(events, list)
            and any(
                isinstance(event, dict) and event.get("event") == "final_gate_rollback"
                for event in events
            )
            and isinstance(selected_scale, (int, float))
            and not isinstance(selected_scale, bool)
            and selected_scale < 4
        )
        rejected_step = step.get("accepted") is False and step.get("decision") in {
            "stop",
            "rollback",
        }
        post_final_gate_rollback = (
            step.get("accepted") is True and step.get("decision") == "stop" and final_gate_rollback
        )
        rollback = (
            status == "succeeded_with_rollback"
            and completion == "COMPONENT_REPRODUCED"
            and target_reached is False
            and (rejected_step or post_final_gate_rollback)
        )
        if not integrated and not rollback:
            raise EvidenceError(
                "ScaleGuard requires AB_INTEGRATED success or COMPONENT_REPRODUCED "
                "gate rollback evidence"
            )
    else:
        raise EvidenceError(f"undeclared experiment group: {group!r}")
    return len(candidates)


def _validate_integrated_contract(manifest: dict[str, Any]) -> int:
    if (
        manifest.get("status") != "succeeded"
        or manifest.get("completion_level") != "AB_INTEGRATED"
        or manifest.get("target_reached") is not True
    ):
        raise EvidenceError(
            "smoke/integration evidence requires successful AB_INTEGRATED completion"
        )
    _require_4kagent(manifest)
    if backend_name(manifest, "scale_backend") != "chain_of_zoom":
        raise EvidenceError("smoke/integration requires the Chain-of-Zoom backend")
    candidates = _require_candidate_contract(manifest, expected_count=1)
    if candidates[0].get("accepted") is not True:
        raise EvidenceError("AB_INTEGRATED evidence must accept its real CoZ candidate")
    return len(candidates)


def validate_manifest(
    manifest: dict[str, Any],
    expected_output: pathlib.Path,
    expected_input: pathlib.Path,
    expected_config: pathlib.Path,
    project_root: pathlib.Path,
    run_dir: pathlib.Path,
    wrapper_started_at: dt.datetime,
    expected_output_sha256: str,
    *,
    manifest_path: pathlib.Path,
    runtime_preflight: pathlib.Path,
    stage: str,
) -> dict[str, Any]:
    try:
        validated_manifest = validate_run_manifest(manifest_path)
    except (ManifestValidationError, OSError) as exc:
        raise EvidenceError(f"full run-manifest validation failed: {exc}") from exc
    if validated_manifest != manifest:
        raise EvidenceError("manifest changed while the evidence snapshot was validated")

    runtime = _runtime_context(
        expected_config=expected_config,
        runtime_preflight=runtime_preflight,
        project_root=project_root,
    )
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise EvidenceError("manifest has no runtime provenance")
    mismatched_provenance = [
        field for field, expected in runtime.provenance.items() if provenance.get(field) != expected
    ]
    if mismatched_provenance:
        raise EvidenceError(
            "manifest runtime provenance differs from the preflight: "
            + ", ".join(sorted(mismatched_provenance))
        )
    if manifest.get("config") != runtime.config.as_dict():
        raise EvidenceError("manifest config differs from the bound preflight configuration")

    if manifest.get("mock") is not False:
        raise EvidenceError("AutoDL evidence cannot use a mock backend")
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id or run_dir.name != run_id:
        raise EvidenceError("manifest run_id does not match the CLI run directory")
    started_at = timestamp(manifest.get("started_at"), "manifest.started_at")
    finished_at = timestamp(manifest.get("finished_at"), "manifest.finished_at")
    if started_at < wrapper_started_at:
        raise EvidenceError("ScaleGuard manifest predates this wrapper attempt")
    if finished_at < started_at:
        raise EvidenceError("ScaleGuard manifest finished before it started")

    expected_run_root = runtime.config.runtime.run_root
    if not expected_run_root.is_absolute():
        expected_run_root = project_root / expected_run_root
    if run_dir.parent != expected_run_root.resolve():
        raise EvidenceError(
            f"CLI run directory {run_dir} is outside configured run root "
            f"{expected_run_root.resolve()}"
        )

    input_image = manifest.get("input_image")
    if not isinstance(input_image, dict) or input_image.get("mock") is not False:
        raise EvidenceError("manifest has no real normalized input image")
    manifest_input = pathlib.Path(str(input_image.get("path", ""))).resolve()
    if manifest_input != (run_dir / "input.png").resolve() or not manifest_input.is_file():
        raise EvidenceError("manifest input path does not match the run directory")
    if input_image.get("sha256") != sha256(manifest_input):
        raise EvidenceError("manifest input SHA-256 does not match its file")
    if not same_rgb_pixels(expected_input, manifest_input):
        raise EvidenceError("manifest input pixels differ from the invoked input")

    events = manifest.get("events")
    restoration_events = [
        event
        for event in events or []
        if isinstance(event, dict) and event.get("event") == "restoration_completed"
    ]
    if not restoration_events:
        raise EvidenceError("manifest has no completed restoration event")

    group = runtime.config.runtime.experiment_group
    sample_id = runtime.config.runtime.experiment_sample_id
    if stage == "experiment":
        if group is None or sample_id is None:
            raise EvidenceError(
                "experiment stage requires a fixed group and sample id in its preflighted config"
            )
        successful_candidates = _validate_experiment_contract(manifest, group=group)
    elif stage in {"smoke", "integration"}:
        if group is not None or sample_id is not None:
            raise EvidenceError("smoke/integration cannot publish experiment-group evidence")
        successful_candidates = _validate_integrated_contract(manifest)
    else:
        raise EvidenceError(f"unsupported AutoDL evidence stage: {stage!r}")

    final_image = manifest.get("final_image")
    if not isinstance(final_image, dict) or final_image.get("mock") is not False:
        raise EvidenceError("manifest has no real final image")
    final_path = pathlib.Path(str(final_image.get("path", ""))).resolve()
    expected_final_path = (run_dir / "final.png").resolve()
    if final_path != expected_final_path or not final_path.is_file():
        raise EvidenceError("manifest final image is not the immutable run artifact")
    expected_digest = final_image.get("sha256")
    internal_digest = sha256(final_path)
    if not isinstance(expected_digest, str) or expected_digest != internal_digest:
        raise EvidenceError("internal final artifact SHA-256 does not match the manifest")
    if internal_digest != expected_output_sha256:
        raise EvidenceError("published output bytes differ from the internal final artifact")

    return {
        "run_id": run_id,
        "manifest_status": manifest["status"],
        "completion_level": manifest["completion_level"],
        "mock": False,
        "restoration_backend": backend_name(manifest, "restoration_backend"),
        "scale_backend": backend_name(manifest, "scale_backend"),
        "successful_coz_candidates": successful_candidates,
        "experiment_group": group,
        "experiment_sample_id": sample_id,
        "runtime_config_sha256": runtime.config_sha256,
        "runtime_preflight_sha256": runtime.preflight_sha256,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "integration", "experiment"), required=True)
    parser.add_argument("--runtime-preflight", type=pathlib.Path, required=True)
    parser.add_argument("log", type=pathlib.Path)
    parser.add_argument("expected_output", type=pathlib.Path)
    parser.add_argument("expected_input", type=pathlib.Path)
    parser.add_argument("expected_config", type=pathlib.Path)
    parser.add_argument("project_root", type=pathlib.Path)
    parser.add_argument("wrapper_started_at")
    parser.add_argument("cli_result", type=pathlib.Path)
    parser.add_argument("manifest_copy", type=pathlib.Path)
    parser.add_argument("output_copy", type=pathlib.Path)
    parser.add_argument("summary", type=pathlib.Path)
    args = parser.parse_args()

    if args.expected_output.is_symlink():
        raise EvidenceError("expected output must not be a symbolic link")
    expected_output = args.expected_output.resolve()
    if not expected_output.is_file() or not expected_output.stat().st_size:
        raise EvidenceError(f"expected output is not a non-empty file: {expected_output}")
    output_bytes = expected_output.read_bytes()
    expected_output_sha256 = hashlib.sha256(output_bytes).hexdigest()
    args.output_copy.write_bytes(output_bytes)
    if sha256(args.output_copy) != expected_output_sha256:
        raise EvidenceError("output evidence snapshot failed its SHA-256 check")

    cli_result = find_cli_result(args.log)
    if cli_result.get("mock") is not False:
        raise EvidenceError("ScaleGuard CLI reported a mock run")
    cli_output = pathlib.Path(str(cli_result["output"])).resolve()
    if cli_output != expected_output:
        raise EvidenceError(f"CLI output path {cli_output} differs from {expected_output}")

    run_dir = pathlib.Path(str(cli_result["run_dir"])).resolve()
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_symlink():
        raise EvidenceError("ScaleGuard run manifest must not be a symbolic link")
    try:
        manifest_bytes = manifest_path.read_bytes()
        args.manifest_copy.write_bytes(manifest_bytes)
        manifest = loads(args.manifest_copy.read_text(encoding="utf-8"))
    except (OSError, StrictJSONError) as exc:
        raise EvidenceError(f"cannot read ScaleGuard run manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise EvidenceError("ScaleGuard run manifest is not a JSON object")
    summary = validate_manifest(
        manifest,
        expected_output,
        args.expected_input.resolve(),
        args.expected_config.resolve(),
        args.project_root.resolve(),
        run_dir,
        timestamp(args.wrapper_started_at, "wrapper_started_at"),
        expected_output_sha256,
        manifest_path=args.manifest_copy.resolve(),
        runtime_preflight=args.runtime_preflight.resolve(),
        stage=args.stage,
    )

    args.cli_result.write_text(json.dumps(cli_result, indent=2) + "\n", encoding="utf-8")
    summary.update(
        {
            "status": "passed",
            "stage": args.stage,
            "source_manifest": str(manifest_path),
            "manifest_sha256": sha256(args.manifest_copy),
            "runtime_preflight_path": str(args.runtime_preflight.resolve()),
            "invoked_input_sha256": sha256(args.expected_input),
            "invoked_config_sha256": sha256(args.expected_config),
            "final_output_sha256": expected_output_sha256,
            "output_evidence_path": str(args.output_copy.resolve()),
            "output_evidence_sha256": sha256(args.output_copy),
        }
    )
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (EvidenceError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
