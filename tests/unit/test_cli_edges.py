from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import scaleguard.cli as cli
from scaleguard.doctor import DoctorCheck
from scaleguard.errors import ScaleGuardError
from scaleguard.upstream import Verification


def test_project_root_resolution_uses_override_parent_search_and_fails_cleanly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    override = tmp_path / "override"
    monkeypatch.setenv("SCALEGUARD_PROJECT_ROOT", str(override))
    assert cli.find_project_root() == override.resolve()

    monkeypatch.delenv("SCALEGUARD_PROJECT_ROOT")
    project = tmp_path / "project"
    nested = project / "a" / "b"
    nested.mkdir(parents=True)
    (project / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    monkeypatch.chdir(nested)
    monkeypatch.setattr(cli, "__file__", str(tmp_path / "package" / "scaleguard" / "cli.py"))
    assert cli.find_project_root() == project

    (project / "pyproject.toml").unlink()
    with pytest.raises(ScaleGuardError, match=r"cannot locate pyproject\.toml"):
        cli.find_project_root()


@pytest.mark.parametrize(
    ("specifications", "message"),
    [
        (["A-only"], "expected GROUP=MANIFEST"),
        (["unknown=/tmp/manifest.json"], "invalid experiment group"),
    ],
)
def test_experiment_group_parser_rejects_malformed_assignments(
    specifications: list[str],
    message: str,
) -> None:
    with pytest.raises(ScaleGuardError, match=message):
        cli._parse_experiment_groups(specifications)


def test_metric_path_parser_rejects_malformed_unknown_and_duplicate_values() -> None:
    with pytest.raises(ScaleGuardError, match="expected METRIC=PATH"):
        cli._parse_metric_paths(["lpips"], accepted=("lpips",), option="--weight")
    with pytest.raises(ScaleGuardError, match="invalid --weight metric"):
        cli._parse_metric_paths(["musiq=/tmp/model"], accepted=("lpips",), option="--weight")
    with pytest.raises(ScaleGuardError, match="duplicate --weight"):
        cli._parse_metric_paths(
            ["lpips=/tmp/one", "lpips=/tmp/two"],
            accepted=("lpips",),
            option="--weight",
        )
    assert cli._parse_metric_paths(
        ["lpips=/tmp/model"],
        accepted=("lpips",),
        option="--weight",
    ) == {"lpips": Path("/tmp/model")}


@pytest.mark.parametrize("as_json", [False, True])
def test_doctor_command_renders_both_formats_and_propagates_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    as_json: bool,
) -> None:
    checks = [
        DoctorCheck("python", "pass", "3.14"),
        DoctorCheck("weights", "fail", "missing"),
    ]
    monkeypatch.setattr(cli, "load_config", lambda _path: object())
    monkeypatch.setattr(cli, "run_doctor", lambda _config, _root: checks)

    result = cli._doctor_command(
        argparse.Namespace(config=tmp_path / "config.yaml", json=as_json),
        tmp_path,
    )

    assert result == 1
    output = capsys.readouterr().out
    if as_json:
        assert json.loads(output)[1]["status"] == "fail"
    else:
        assert "FAIL  weights: missing" in output


@pytest.mark.parametrize("as_json", [False, True])
def test_upstream_command_renders_both_formats_and_propagates_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    as_json: bool,
) -> None:
    results = [
        Verification("fourkagent", "commit", True, "locked"),
        Verification("coz", "tree", False, "changed"),
    ]
    monkeypatch.setattr(cli, "verify_upstreams", lambda *_args: results)

    result = cli._upstream_command(
        argparse.Namespace(lock=tmp_path / "lock.yaml", mapping="repositories", json=as_json),
        tmp_path,
    )

    assert result == 1
    output = capsys.readouterr().out
    if as_json:
        assert json.loads(output)[0]["ok"] is True
    else:
        assert "FAIL coz.tree: changed" in output


def test_evaluation_dispatch_requires_a_root_and_rejects_unknown_operations(
    tmp_path: Path,
) -> None:
    with pytest.raises(ScaleGuardError, match="requires the project root"):
        cli._evaluation_command(
            argparse.Namespace(evaluation_command="metrics", artifact_root=None),
        )
    with pytest.raises(ScaleGuardError, match="unsupported evaluation command"):
        cli._evaluation_command(
            argparse.Namespace(evaluation_command="unknown", artifact_root=tmp_path),
        )


def _run_args(tmp_path: Path, source: Path) -> argparse.Namespace:
    return argparse.Namespace(
        input=source,
        output=tmp_path / "output.png",
        overwrite=False,
        runtime_preflight=None,
        target_factor=None,
        config=tmp_path / "config.yaml",
        run_id="run",
    )


def test_run_command_rejects_missing_input_and_conflicting_preflight_override(
    tmp_path: Path,
) -> None:
    args = _run_args(tmp_path, tmp_path / "missing.png")
    with pytest.raises(ScaleGuardError, match="input image does not exist"):
        cli._run_command(args, tmp_path)

    source = tmp_path / "input.png"
    source.write_bytes(b"input")
    args = _run_args(tmp_path, source)
    args.runtime_preflight = tmp_path / "preflight.json"
    args.target_factor = 4
    with pytest.raises(ScaleGuardError, match="cannot be combined"):
        cli._run_command(args, tmp_path)


def test_run_command_rechecks_preflight_config_and_profile_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "input.png"
    source.write_bytes(b"input")
    args = _run_args(tmp_path, source)
    args.runtime_preflight = tmp_path / "preflight.json"
    monkeypatch.setattr(
        cli,
        "validate_runtime_preflight",
        lambda *_args, **_kwargs: {"runtime_config_sha256": "a" * 64},
    )
    monkeypatch.setattr(cli, "load_regular_file_snapshot", lambda *_args: (b"{}", "b" * 64))

    with pytest.raises(ScaleGuardError, match="snapshot digest disagrees"):
        cli._run_command(args, tmp_path)

    monkeypatch.setattr(
        cli,
        "load_regular_file_snapshot",
        lambda *_args: (b"{}", "a" * 64),
    )
    monkeypatch.setattr(cli, "parse_config", lambda *_args, **_kwargs: object())
    with pytest.raises(ScaleGuardError, match="did not bind the audited execution profile"):
        cli._run_command(args, tmp_path)


def test_run_command_requires_the_controller_run_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "input.png"
    source.write_bytes(b"input")
    args = _run_args(tmp_path, source)
    args.overwrite = True
    config = SimpleNamespace(is_mock=True)
    monkeypatch.setattr(cli, "load_config", lambda _path: config)
    monkeypatch.setattr(cli, "_override_target", lambda value, _target: value)
    monkeypatch.setattr(cli, "build_backends", lambda *_args, **_kwargs: (object(), object()))

    class Controller:
        last_run_dir = None
        received_overwrite: bool | None = None

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def run(
            self,
            _source: Path,
            destination: Path,
            *,
            run_id: str,
            overwrite: bool,
        ) -> Path:
            del run_id
            type(self).received_overwrite = overwrite
            return destination

    monkeypatch.setattr(cli, "TrustedScaleController", Controller)

    with pytest.raises(ScaleGuardError, match="did not retain a run directory"):
        cli._run_command(args, tmp_path)
    assert Controller.received_overwrite is True


@pytest.mark.parametrize(
    ("argv", "handler"),
    [
        (["doctor", "--config", "config.yaml"], "_doctor_command"),
        (["upstream", "verify"], "_upstream_command"),
        (
            [
                "evaluation",
                "calibrate",
                "--manifest",
                "manifest.json",
                "--labels",
                "labels.csv",
                "--output",
                "receipt.json",
            ],
            "_evaluation_command",
        ),
    ],
)
def test_main_dispatches_project_bound_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    handler: str,
) -> None:
    monkeypatch.setattr(cli, "find_project_root", lambda: tmp_path)
    monkeypatch.setattr(cli, handler, lambda *_args, **_kwargs: 7)

    assert cli.main(argv) == 7
