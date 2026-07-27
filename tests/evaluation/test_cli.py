from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from scaleguard.cli import main

from ._fixtures import (
    write_calibration_manifest,
    write_summary_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _fail_project_root_resolution() -> Path:
    raise AssertionError("project root must not be resolved")


def test_evaluation_calibrate_and_verify_cli(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.delenv("SCALEGUARD_PROJECT_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("scaleguard.cli.find_project_root", _fail_project_root_resolution)
    trusted = tmp_path / "trusted.bin"
    candidate = tmp_path / "candidate.bin"
    trusted.write_bytes(b"trusted")
    candidate.write_bytes(b"candidate")
    manifest = write_calibration_manifest(
        tmp_path / "manifest.json",
        run_id="run",
        trusted=trusted,
        candidate=candidate,
        values=[(0.1, 0.2, 0.3)],
    )
    labels = tmp_path / "labels.csv"
    labels.write_text("run_id,step_index,acceptable\nrun,1,true\n", encoding="utf-8")
    receipt = tmp_path / "receipt.json"

    exit_code = main(
        [
            "evaluation",
            "calibrate",
            "--manifest",
            str(manifest),
            "--labels",
            str(labels),
            "--output",
            str(receipt),
            "--artifact-root",
            str(tmp_path),
            "--minimum-acceptable-samples",
            "1",
            "--bootstrap-samples",
            "5",
        ]
    )
    response = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert response["status"] == "calibrated"

    thresholds = json.loads(receipt.read_text(encoding="utf-8"))["thresholds"]
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
metrics:
  min_quality_gain: {thresholds["min_quality_gain"]["value"]}
  max_scale_nrmse: {thresholds["max_scale_nrmse"]["value"]}
  max_scale_edge_mae: {thresholds["max_scale_edge_mae"]["value"]}
""",
        encoding="utf-8",
    )
    exit_code = main(
        [
            "evaluation",
            "verify",
            "--receipt",
            str(receipt),
            "--config",
            str(config),
        ]
    )
    response = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert response == {"valid": True, "reasons": []}


def test_evaluation_cli_returns_one_for_insufficient_data(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("SCALEGUARD_PROJECT_ROOT", str(PROJECT_ROOT))
    trusted = tmp_path / "trusted.bin"
    candidate = tmp_path / "candidate.bin"
    trusted.write_bytes(b"trusted")
    candidate.write_bytes(b"candidate")
    manifest = write_calibration_manifest(
        tmp_path / "manifest.json",
        run_id="run",
        trusted=trusted,
        candidate=candidate,
        values=[(0.1, 0.2, 0.3)],
    )
    labels = tmp_path / "labels.csv"
    labels.write_text("run_id,step_index,acceptable\nrun,1,true\n", encoding="utf-8")

    exit_code = main(
        [
            "evaluation",
            "calibrate",
            "--manifest",
            str(manifest),
            "--labels",
            str(labels),
            "--output",
            str(tmp_path / "receipt.json"),
            "--bootstrap-samples",
            "2",
        ]
    )

    response = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert response["status"] == "insufficient_data"


def test_evaluation_summarize_cli_and_invalid_group(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.delenv("SCALEGUARD_PROJECT_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("scaleguard.cli.find_project_root", _fail_project_root_resolution)
    source = tmp_path / "source.bin"
    final = tmp_path / "final.bin"
    source.write_bytes(b"source")
    final.write_bytes(b"final")
    manifest = write_summary_manifest(
        tmp_path / "manifest.json",
        run_id="run",
        source=source,
        final=final,
    )
    monkeypatch.setattr(
        "scaleguard.evaluation.summary.validate_run_manifest",
        lambda path, **_kwargs: json.loads(path.read_text(encoding="utf-8")),
    )
    output_csv = tmp_path / "summary.csv"
    output_json = tmp_path / "summary.json"

    exit_code = main(
        [
            "evaluation",
            "summarize",
            "--group",
            f"A-only={manifest}",
            "--output-csv",
            str(output_csv),
            "--output-json",
            str(output_json),
            "--artifact-root",
            str(tmp_path),
        ]
    )
    response = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert response["pairs"] == 1
    assert response["complete_pairs"] == 0
    assert response["research_eligible_pairs"] == 0
    assert output_csv.is_file()
    assert output_json.is_file()
    summary = json.loads(output_json.read_text(encoding="utf-8"))
    assert summary["pairs"][0]["issues"][-1] == "suite_receipt_unverified"

    exit_code = main(
        [
            "evaluation",
            "summarize",
            "--group",
            f"unknown={manifest}",
            "--output-csv",
            str(output_csv),
            "--output-json",
            str(output_json),
            "--artifact-root",
            str(tmp_path),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "invalid experiment group" in captured.err


def test_evaluation_summarize_cli_forwards_suite_receipt(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    receipt = tmp_path / "suite-receipt.json"
    receipt.write_text("{}\n", encoding="utf-8")
    observed: dict[str, Any] = {}

    def summarize(
        manifests: dict[str, list[Path]],
        output_csv: Path,
        output_json: Path,
        *,
        artifact_root: Path,
        suite_receipt: Path | None,
    ) -> dict[str, Any]:
        observed.update(
            {
                "manifests": manifests,
                "output_csv": output_csv,
                "output_json": output_json,
                "artifact_root": artifact_root,
                "suite_receipt": suite_receipt,
            }
        )
        return {
            "counts": {
                "manifests": 1,
                "pairs": 1,
                "complete_pairs": 0,
                "research_eligible_pairs": 0,
                "mock_pairs": 0,
            }
        }

    monkeypatch.setattr("scaleguard.cli.summarize_paired_manifests", summarize)

    exit_code = main(
        [
            "evaluation",
            "summarize",
            "--group",
            f"A-only={manifest}",
            "--output-csv",
            str(tmp_path / "summary.csv"),
            "--output-json",
            str(tmp_path / "summary.json"),
            "--artifact-root",
            str(tmp_path),
            "--suite-receipt",
            str(receipt),
        ]
    )

    assert exit_code == 0
    assert observed["suite_receipt"] == receipt
    assert json.loads(capsys.readouterr().out)["research_eligible_pairs"] == 0


def test_summarize_ablation_script_exposes_suite_receipt_option() -> None:
    script = PROJECT_ROOT / "scripts" / "experiments" / "summarize_ablation.py"

    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--suite-receipt SUITE_RECEIPT" in completed.stdout
