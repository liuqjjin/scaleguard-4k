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
) -> Path:
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
        json.dumps({"run_id": run_id, "mock": mock, "steps": steps}),
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
) -> Path:
    path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": status,
                "mock": mock,
                "input_image": artifact(source),
                "final_image": artifact(final, mock=mock),
                "final_metrics": {
                    "metrics": metrics
                    or {
                        "quality_gain": 0.1,
                        "scale_nrmse": 0.02,
                        "scale_edge_mae": 0.03,
                        "measurement_nrmse": None,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return path
