#!/usr/bin/env python3
"""Extract and verify the ScaleGuard run manifest named by CLI JSON output."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import sys
from typing import Any

from PIL import Image, ImageChops

from scaleguard.config import load_config  # type: ignore[import-untyped]
from scaleguard.strict_json import StrictJSONError, loads


class EvidenceError(ValueError):
    """Raised when a successful CLI exit lacks real pipeline evidence."""


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


def validate_manifest(
    manifest: dict[str, Any],
    expected_output: pathlib.Path,
    expected_input: pathlib.Path,
    expected_config: pathlib.Path,
    project_root: pathlib.Path,
    run_dir: pathlib.Path,
    wrapper_started_at: dt.datetime,
    expected_output_sha256: str,
) -> dict[str, Any]:
    if manifest.get("mock") is not False:
        raise EvidenceError("AutoDL evidence cannot use a mock backend")
    status = manifest.get("status")
    if status not in {"succeeded", "succeeded_with_rollback"}:
        raise EvidenceError(f"ScaleGuard run status is not successful: {status!r}")
    completion = manifest.get("completion_level")
    if completion not in {
        "AB_INTEGRATED",
        "SCALEGUARD_VALIDATED",
        "RESEARCH_EVALUATED",
    }:
        raise EvidenceError(f"manifest does not establish an A+B execution: {completion!r}")

    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id or run_dir.name != run_id:
        raise EvidenceError("manifest run_id does not match the CLI run directory")
    started_at = timestamp(manifest.get("started_at"), "manifest.started_at")
    finished_at = timestamp(manifest.get("finished_at"), "manifest.finished_at")
    if started_at < wrapper_started_at:
        raise EvidenceError("ScaleGuard manifest predates this wrapper attempt")
    if finished_at < started_at:
        raise EvidenceError("ScaleGuard manifest finished before it started")

    loaded_config = load_config(expected_config)
    expected_run_root = loaded_config.runtime.run_root
    if not expected_run_root.is_absolute():
        expected_run_root = project_root / expected_run_root
    if run_dir.parent != expected_run_root.resolve():
        raise EvidenceError(
            f"CLI run directory {run_dir} is outside configured run root "
            f"{expected_run_root.resolve()}"
        )
    if manifest.get("config") != loaded_config.as_dict():
        raise EvidenceError("manifest config differs from the invoked configuration")

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

    restoration_backend = backend_name(manifest, "restoration_backend")
    scale_backend = backend_name(manifest, "scale_backend")
    events = manifest.get("events")
    restoration_events = [
        event
        for event in events or []
        if isinstance(event, dict) and event.get("event") == "restoration_completed"
    ]
    if not restoration_events:
        raise EvidenceError("manifest has no completed 4KAgent restoration event")

    successful_candidates = []
    for step in manifest.get("steps") or []:
        if not isinstance(step, dict):
            continue
        candidate = step.get("candidate")
        metadata = step.get("worker_metadata")
        if (
            isinstance(candidate, dict)
            and candidate.get("mock") is False
            and isinstance(metadata, dict)
            and isinstance(metadata.get("backend"), str)
            and "chain_of_zoom" in metadata["backend"]
        ):
            successful_candidates.append(step)
    if not successful_candidates:
        raise EvidenceError("manifest has no successful non-mock Chain-of-Zoom candidate")

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
        "manifest_status": status,
        "completion_level": completion,
        "mock": False,
        "restoration_backend": restoration_backend,
        "scale_backend": scale_backend,
        "successful_coz_candidates": len(successful_candidates),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
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
    )

    args.cli_result.write_text(json.dumps(cli_result, indent=2) + "\n", encoding="utf-8")
    summary.update(
        {
            "status": "passed",
            "source_manifest": str(manifest_path),
            "manifest_sha256": sha256(args.manifest_copy),
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
