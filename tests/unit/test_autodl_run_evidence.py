from __future__ import annotations

import importlib.util
import shutil
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from scaleguard.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_evidence_module() -> ModuleType:
    path = PROJECT_ROOT / "scripts" / "autodl" / "_extract_run_evidence.py"
    specification = importlib.util.spec_from_file_location(
        "scaleguard_autodl_run_evidence",
        path,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _fixture(
    tmp_path: Path,
    make_image: Callable[..., Path],
) -> tuple[ModuleType, dict[str, Any], dict[str, Any]]:
    module = _load_evidence_module()
    run_root = tmp_path / "runs"
    run_dir = run_root / "attempt-1"
    run_dir.mkdir(parents=True)
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text(
        f"""
runtime:
  run_root: "{run_root}"
fourkagent:
  mode: command
  command: ["4kagent-worker"]
coz:
  mode: command
  command: ["coz-worker"]
metrics:
  min_quality_gain: -1.0
  max_scale_nrmse: 10.0
  max_scale_edge_mae: 10.0
controller:
  target_factor: 4
  color_strategy: none
  accept_unvalidated_quality_proxy: true
""",
        encoding="utf-8",
    )
    expected_input = make_image(tmp_path / "source.png", size=(8, 6))
    normalized_input = run_dir / "input.png"
    shutil.copy2(expected_input, normalized_input)
    internal_final = make_image(run_dir / "final.png", size=(32, 24))
    external_output = tmp_path / "published.png"
    shutil.copy2(internal_final, external_output)
    candidate = make_image(run_dir / "candidate.png", size=(32, 24))
    manifest = {
        "mock": False,
        "status": "succeeded",
        "completion_level": "AB_INTEGRATED",
        "run_id": run_dir.name,
        "started_at": "2026-07-27T00:00:00Z",
        "finished_at": "2026-07-27T00:01:00Z",
        "config": load_config(config_path).as_dict(),
        "provenance": {
            "restoration_backend": "4kagent_upstream",
            "scale_backend": "chain_of_zoom",
        },
        "input_image": {
            "path": str(normalized_input.resolve()),
            "sha256": module.sha256(normalized_input),
            "mock": False,
        },
        "events": [{"event": "restoration_completed"}],
        "steps": [
            {
                "candidate": {"path": str(candidate.resolve()), "mock": False},
                "worker_metadata": {"backend": "chain_of_zoom_persistent"},
            }
        ],
        "final_image": {
            "path": str(internal_final.resolve()),
            "sha256": module.sha256(internal_final),
            "mock": False,
        },
    }
    arguments = {
        "expected_output": external_output.resolve(),
        "expected_input": expected_input.resolve(),
        "expected_config": config_path.resolve(),
        "project_root": PROJECT_ROOT,
        "run_dir": run_dir.resolve(),
        "wrapper_started_at": datetime(2026, 7, 26, tzinfo=timezone.utc),
        "expected_output_sha256": module.sha256(external_output),
    }
    return module, manifest, arguments


def test_autodl_evidence_accepts_an_external_copy_of_the_internal_final_artifact(
    tmp_path: Path,
    make_image: Callable[..., Path],
) -> None:
    module, manifest, arguments = _fixture(tmp_path, make_image)

    summary = module.validate_manifest(manifest, **arguments)

    assert summary["completion_level"] == "AB_INTEGRATED"
    assert Path(manifest["final_image"]["path"]) != arguments["expected_output"]


def test_autodl_evidence_rejects_published_bytes_that_drift_from_the_run_artifact(
    tmp_path: Path,
    make_image: Callable[..., Path],
) -> None:
    module, manifest, arguments = _fixture(tmp_path, make_image)
    external_output = arguments["expected_output"]
    make_image(external_output, size=(32, 24), color=(200, 20, 10))
    arguments["expected_output_sha256"] = module.sha256(external_output)

    with pytest.raises(
        module.EvidenceError,
        match="published output bytes differ from the internal final artifact",
    ):
        module.validate_manifest(manifest, **arguments)


def test_autodl_evidence_rejects_ambiguous_cli_result(tmp_path: Path) -> None:
    module = _load_evidence_module()
    log = tmp_path / "run.log"
    log.write_text(
        '{"status":"ok","status":"failed","run_dir":"/trusted","output":"/trusted.png"}\n',
        encoding="utf-8",
    )

    with pytest.raises(module.EvidenceError, match="emitted no successful run JSON"):
        module.find_cli_result(log)
