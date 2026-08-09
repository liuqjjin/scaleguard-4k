from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(path: Path, *, mock: bool = False) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "width": 1,
        "height": 1,
        "media_type": "application/octet-stream",
        "mock": mock,
        "stage": "test",
    }


def write_summary_calibration_receipt(path: Path) -> Path:
    threshold_values = {
        "min_quality_gain": -0.02,
        "max_scale_nrmse": 0.12,
        "max_scale_edge_mae": 0.10,
        "max_measurement_nrmse": 0.12,
    }
    receipt: dict[str, Any] = {
        "schema_version": "scaleguard.calibration-receipt/v1",
        "status": "calibrated",
        "inputs": {
            "labels": {"path": "labels.csv", "sha256": "1" * 64},
            "manifests": [
                {
                    "run_id": "calibration-run",
                    "sha256": "2" * 64,
                    "input_sha256": "3" * 64,
                }
            ],
        },
        "sample_counts": {"acceptable_real": 1},
        "metric_backend": {
            "quality": "pyiqa:musiq",
            "quality_is_proxy": False,
            "measurement": "resize_lanczos",
        },
        "algorithm": {"minimum_acceptable_samples": 1},
        "thresholds": {
            name: {
                "value": value,
                "bootstrap_ci": {
                    "lower": value,
                    "upper": value,
                    "confidence": 0.95,
                },
            }
            for name, value in threshold_values.items()
        },
        "issues": [],
    }
    encoded = json.dumps(
        receipt,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    receipt["receipt_sha256"] = hashlib.sha256(encoded).hexdigest()
    path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def write_calibration_manifest(
    path: Path,
    *,
    run_id: str,
    trusted: Path,
    candidate: Path,
    values: list[tuple[float, float, float]],
    mock: bool = False,
    quality_backend: str = "gradient_proxy_v1",
    measurement: tuple[float, str] | None = None,
    input_image: Path | None = None,
) -> Path:
    if input_image is None:
        input_image = path.with_name(f"{path.stem}-input.bin")
        input_image.write_bytes(f"input:{run_id}".encode())
    steps: list[dict[str, Any]] = []
    for index, (quality_gain, scale_nrmse, scale_edge_mae) in enumerate(values, start=1):
        steps.append(
            {
                "index": index,
                "trusted_before": artifact(trusted, mock=mock),
                "candidate": artifact(candidate, mock=mock),
                "metrics": {
                    "quality_backend": quality_backend,
                    "quality_gain": quality_gain,
                    "scale_nrmse": scale_nrmse,
                    "scale_edge_mae": scale_edge_mae,
                    "measurement_nrmse": measurement[0] if measurement else None,
                    "measurement_model": measurement[1] if measurement else None,
                },
            }
        )
    path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": "succeeded",
                "mock": mock,
                "input_image": artifact(input_image, mock=mock),
                "steps": steps,
            }
        ),
        encoding="utf-8",
    )
    return path


def write_summary_manifest(
    path: Path,
    *,
    run_id: str,
    source: Path,
    final: Path,
    mock: bool = False,
    status: str = "succeeded",
    metrics: dict[str, float | None] | None = None,
    group: str = "A-only",
    sample_id: str = "sample-1",
) -> Path:
    calibration_receipt = write_summary_calibration_receipt(
        path.parent / "quality-calibration.json"
    )
    fourkagent_mode = "identity" if group == "B-only" else "upstream"
    target_factor = 1 if group == "A-only" else 4
    max_coz_steps = 0 if group == "A-only" else 1
    acceptance_policy = "trusted" if group == "ScaleGuard" else "fixed"
    candidate = artifact(final, mock=mock)
    steps = (
        []
        if group == "A-only"
        else [
            {
                "candidate": candidate,
                "accepted": True,
                "decision": "stop",
                "worker_metadata": {
                    "backend": "chain_of_zoom_persistent",
                    "candidate_sha256": candidate["sha256"],
                },
            }
        ]
    )
    path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": status,
                "completion_level": ("AB_INTEGRATED" if group == "ScaleGuard" else "STATIC_READY"),
                "target_reached": status == "succeeded",
                "mock": mock,
                "config": {
                    "runtime": {
                        "run_root": "ignored-per-job",
                        "process_timeout_seconds": 10.0,
                        "experiment_group": group,
                        "experiment_sample_id": sample_id,
                    },
                    "fourkagent": {
                        "mode": fourkagent_mode,
                        "profile": "FastGen4K_P",
                        "llm_provider": "dashscope",
                        "llm_base_url": ("https://dashscope.aliyuncs.com/compatible-mode/v1"),
                        "llm_region": "cn-beijing",
                        "llm_model": "qwen3.7-flash-2026-07-15",
                        "api_key_env": "DASHSCOPE_API_KEY",
                    },
                    "coz": {
                        "mode": "persistent",
                        "seed": 7,
                        "tile_size": 512,
                        "tile_overlap": 64,
                    },
                    "metrics": {
                        "quality_backend": "pyiqa",
                        "quality_metric": "musiq",
                        "quality_model_path": "/weights/musiq.pth",
                        "min_quality_gain": -0.02,
                        "max_scale_nrmse": 0.12,
                        "max_scale_edge_mae": 0.10,
                        "measurement_enabled": True,
                        "measurement_model": "resize",
                        "measurement_parameters": {},
                        "max_measurement_nrmse": 0.12,
                        "calibration_receipt": str(calibration_receipt.resolve()),
                    },
                    "controller": {
                        "target_factor": target_factor,
                        "max_coz_steps": max_coz_steps,
                        "color_strategy": "adain",
                        "acceptance_policy": acceptance_policy,
                    },
                },
                "provenance": {
                    "runtime_evidence_verified": True,
                    "runtime_profile_bound": True,
                    "bootstrap_receipt_sha256": "a" * 64,
                    "materialization_marker_sha256": "b" * 64,
                    "source_weights_receipt_sha256": "c" * 64,
                    "weights_root": "/weights",
                    "project_commit": "d" * 40,
                    "project_root": "/project",
                    "runtime_execution_binding": {"schema_version": 1},
                    "runtime_execution_binding_sha256": "e" * 64,
                    "quality_backend_is_proxy": False,
                    "quality_thresholds_calibrated": True,
                    "quality_calibration_receipt": str(calibration_receipt.resolve()),
                    "quality_calibration_receipt_size_bytes": (calibration_receipt.stat().st_size),
                    "quality_calibration_receipt_sha256": file_sha256(calibration_receipt),
                    "restoration_backend": (
                        "scaleguard_identity_observation"
                        if group == "B-only"
                        else "4kagent_upstream"
                    ),
                    "scale_backend": "chain_of_zoom",
                },
                "input_image": artifact(source),
                "restoration_metadata": (
                    {
                        "backend": "scaleguard_identity_observation",
                        "algorithmic_restoration": False,
                    }
                    if group == "B-only"
                    else {"backend": "4kagent_upstream"}
                ),
                "restoration_process": (None if group == "B-only" else {"returncode": 0}),
                "scale_session_process": (None if group == "A-only" else {"returncode": 0}),
                "steps": steps,
                "final_image": artifact(final, mock=mock),
                "final_metrics": {
                    "metrics": metrics
                    or {
                        "quality_gain": 0.1,
                        "scale_nrmse": 0.02,
                        "scale_edge_mae": 0.03,
                        "measurement_nrmse": 0.04,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return path
