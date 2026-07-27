from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from scaleguard.cli import main

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _fail_project_root_resolution() -> Path:
    raise AssertionError("project root must not be resolved")


def test_public_help_labels_file_and_artifact_arguments(capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["evaluation", "metrics", "--help"])

    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert "--manifest MANIFEST" in output
    assert "--reference IMAGE" in output
    assert "--output RECEIPT" in output
    assert "--artifact-root DIR" in output
    assert "base for relative artifacts" in output
    assert "checkout" in output

    with pytest.raises(SystemExit) as root_exit:
        main(["--help"])
    assert root_exit.value.code == 0
    assert "docs/configuration.md" in capsys.readouterr().out


def test_cli_runs_the_fake_pipeline_with_unicode_and_space_paths(
    tmp_path: Path,
    make_image: Callable[..., Path],
    monkeypatch,
    capsys,
) -> None:
    run_root = tmp_path / "运行 记录"
    config = tmp_path / "配置 文件.yaml"
    config.write_text(
        f"""
runtime:
  run_root: "{run_root}"
fourkagent:
  mode: fake
coz:
  mode: fake
metrics:
  min_quality_gain: -10.0
  max_scale_nrmse: 10.0
  max_scale_edge_mae: 10.0
controller:
  target_factor: 1
  color_strategy: none
""",
        encoding="utf-8",
    )
    source = make_image(
        tmp_path / "输入 图片.JPG",
        size=(7, 5),
        image_format="JPEG",
    )
    output = tmp_path / "输出 文件" / "结果.png"
    monkeypatch.setenv("SCALEGUARD_PROJECT_ROOT", str(PROJECT_ROOT))

    exit_code = main(
        [
            "run",
            "--config",
            str(config),
            "--input",
            str(source),
            "--output",
            str(output),
            "--target-factor",
            "4",
            "--run-id",
            "命令行-run",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    assert payload["status"] == "ok"
    assert payload["run_status"] == "succeeded"
    assert payload["completion_level"] == "STATIC_READY"
    assert payload["requested_factor"] == 4
    assert payload["achieved_factor"] == 4
    assert payload["target_reached"] is True
    assert payload["mock"] is True
    assert payload["output"] == str(output.resolve())
    assert payload["run_dir"] == str((run_root / "命令行-run").resolve())
    assert output.is_file()


def test_cli_requires_recent_runtime_preflight_for_a_real_run(
    tmp_path: Path,
    make_image: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = tmp_path / "runtime.yaml"
    config.write_text(
        f"""
runtime:
  run_root: "{tmp_path / "runs"}"
fourkagent:
  mode: fake
coz:
  mode: fake
controller:
  target_factor: 1
  color_strategy: none
""",
        encoding="utf-8",
    )
    source = make_image(tmp_path / "input.png")
    output = tmp_path / "output.png"
    preflight = tmp_path / "runtime-preflight.json"
    observed: list[bool] = []
    config_digest = hashlib.sha256(config.read_bytes()).hexdigest()

    def validate(
        receipt_path: Path,
        *,
        config_path: Path | None,
        project_root: Path,
        require_recent: bool = False,
    ) -> dict[str, object]:
        assert receipt_path == preflight
        assert config_path == config
        assert project_root == PROJECT_ROOT
        observed.append(require_recent)
        return {
            "runtime_evidence_verified": True,
            "runtime_config_sha256": config_digest,
            "runtime_profile_bound": True,
            "runtime_execution_binding": {},
        }

    monkeypatch.setenv("SCALEGUARD_PROJECT_ROOT", str(PROJECT_ROOT))
    monkeypatch.setattr("scaleguard.cli.validate_runtime_preflight", validate)
    monkeypatch.setattr(
        "scaleguard.cli.bind_runtime_config",
        lambda config, **_kwargs: config,
    )

    exit_code = main(
        [
            "run",
            "--config",
            str(config),
            "--input",
            str(source),
            "--output",
            str(output),
            "--runtime-preflight",
            str(preflight),
            "--run-id",
            "recent-preflight",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert observed == [True]


def test_cli_rejects_an_atomic_config_replacement_before_starting_backends(
    tmp_path: Path,
    make_image: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = tmp_path / "runtime.yaml"
    config.write_text(
        f"""
runtime:
  run_root: "{tmp_path / "runs"}"
controller:
  target_factor: 1
  color_strategy: none
""",
        encoding="utf-8",
    )
    validated_digest = hashlib.sha256(config.read_bytes()).hexdigest()
    replacement = tmp_path / "replacement.yaml"
    replacement.write_text(
        f"""
runtime:
  run_root: "{tmp_path / "unexpected-runs"}"
controller:
  target_factor: 8
""",
        encoding="utf-8",
    )
    source = make_image(tmp_path / "input.png")
    output = tmp_path / "output.png"
    preflight = tmp_path / "runtime-preflight.json"

    def validate(*_args: object, **_kwargs: object) -> dict[str, object]:
        replacement.replace(config)
        return {
            "runtime_evidence_verified": True,
            "runtime_config_sha256": validated_digest,
        }

    def reject_backend_start(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("backends must not start for a replaced config")

    monkeypatch.setenv("SCALEGUARD_PROJECT_ROOT", str(PROJECT_ROOT))
    monkeypatch.setattr("scaleguard.cli.validate_runtime_preflight", validate)
    monkeypatch.setattr("scaleguard.cli.build_backends", reject_backend_start)

    exit_code = main(
        [
            "run",
            "--config",
            str(config),
            "--input",
            str(source),
            "--output",
            str(output),
            "--runtime-preflight",
            str(preflight),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "config snapshot digest disagrees" in captured.err
    assert not output.exists()


def test_cli_refuses_to_overwrite_an_existing_output(
    tmp_path: Path,
    make_image: Callable[..., Path],
    monkeypatch,
    capsys,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
runtime:
  run_root: "{tmp_path / "runs"}"
controller:
  target_factor: 1
  color_strategy: none
""",
        encoding="utf-8",
    )
    source = make_image(tmp_path / "input.png")
    output = make_image(tmp_path / "existing.png")
    monkeypatch.setenv("SCALEGUARD_PROJECT_ROOT", str(PROJECT_ROOT))

    exit_code = main(
        [
            "run",
            "--config",
            str(config),
            "--input",
            str(source),
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "pass --overwrite to replace it" in captured.err


def test_shipped_cpu_mock_config_passes_strict_cli_validation(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.delenv("SCALEGUARD_PROJECT_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("scaleguard.cli.find_project_root", _fail_project_root_resolution)

    exit_code = main(
        [
            "config",
            "validate",
            str(PROJECT_ROOT / "configs" / "runtime" / "cpu-mock.yaml"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert json.loads(captured.out) == {"status": "ok", "mock": True}
    assert captured.err == ""


def test_cli_validates_a_complete_manifest(
    tmp_path: Path,
    make_image: Callable[..., Path],
    monkeypatch,
    capsys,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
runtime:
  run_root: "{tmp_path / "runs"}"
controller:
  target_factor: 1
  color_strategy: none
""",
        encoding="utf-8",
    )
    source = make_image(tmp_path / "input.png")
    output = tmp_path / "output.png"
    monkeypatch.setenv("SCALEGUARD_PROJECT_ROOT", str(PROJECT_ROOT))
    assert (
        main(
            [
                "run",
                "--config",
                str(config),
                "--input",
                str(source),
                "--output",
                str(output),
                "--run-id",
                "run-1",
            ]
        )
        == 0
    )
    capsys.readouterr()
    manifest = tmp_path / "runs" / "run-1" / "manifest.json"
    monkeypatch.delenv("SCALEGUARD_PROJECT_ROOT")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("scaleguard.cli.find_project_root", _fail_project_root_resolution)

    exit_code = main(["manifest", "validate", str(manifest)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert json.loads(captured.out) == {"status": "ok", "run_id": "run-1"}
    assert captured.err == ""


def test_cli_rejects_an_incomplete_or_invalid_manifest(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("SCALEGUARD_PROJECT_ROOT", str(PROJECT_ROOT))
    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text('{"run_id": "run-1"}', encoding="utf-8")

    exit_code = main(["manifest", "validate", str(incomplete)])

    first = capsys.readouterr()
    assert exit_code == 2
    assert "manifest is missing fields" in first.err
    assert "schema_version" in first.err

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not-json", encoding="utf-8")
    exit_code = main(["manifest", "validate", str(malformed)])

    second = capsys.readouterr()
    assert exit_code == 2
    assert "invalid manifest" in second.err
