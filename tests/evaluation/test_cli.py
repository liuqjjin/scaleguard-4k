from __future__ import annotations

import json
from pathlib import Path

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
    assert output_csv.is_file()
    assert output_json.is_file()

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
