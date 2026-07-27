from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from scaleguard.backends.fourkagent import FourKAgentBackend
from scaleguard.config import FourKAgentConfig, RuntimeConfig
from scaleguard.contracts import ProcessEvidence
from scaleguard.errors import ArtifactError, WorkerError
from scaleguard.runtime.process import ProcessRunner


@pytest.fixture(autouse=True)
def reachable_external_depictqa(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("scaleguard.backends.fourkagent.tcp_ready", lambda _host, _port: True)
    monkeypatch.setenv("OPENAI_API_KEY", "contract-credential-placeholder")


def evidence(argv: Sequence[str], cwd: Path, log_dir: Path, label: str) -> ProcessEvidence:
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout = log_dir / f"{label}.stdout.log"
    stderr = log_dir / f"{label}.stderr.log"
    stdout.write_text("", encoding="utf-8")
    stderr.write_text("", encoding="utf-8")
    return ProcessEvidence(
        argv=tuple(argv),
        cwd=str(cwd),
        returncode=0,
        duration_seconds=0.01,
        stdout_path=str(stdout),
        stderr_path=str(stderr),
    )


def backend(
    tmp_path: Path,
    config: FourKAgentConfig | None = None,
) -> FourKAgentBackend:
    return FourKAgentBackend(
        config
        or FourKAgentConfig(
            mode="upstream",
            checkout=tmp_path / "4KAgent",
            toolbox_root=Path("weights/4kagent/runtime/toolbox-root"),
            hps_root=Path("weights/4kagent/hpsv2"),
            quality_model_path=Path("weights/metrics/pyiqa/musiq_koniq_ckpt-e95806b9.pth"),
            perception_model_path="weights/4kagent/models/Qwen2.5-VL-7B-Instruct",
        ),
        RuntimeConfig(process_timeout_seconds=2.0),
        project_root=tmp_path,
    )


def test_fourkagent_adapter_requires_an_explicit_checkout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="4KAgent checkout is required"):
        FourKAgentBackend(FourKAgentConfig(), RuntimeConfig(), project_root=tmp_path)


def test_fourkagent_adapter_normalizes_the_locked_result_and_records_execution_path(
    tmp_path: Path,
    make_image: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = make_image(tmp_path / "source.jpg", size=(6, 4), image_format="JPEG")
    destination = tmp_path / "restored.png"

    def fake_run(
        _runner: ProcessRunner,
        argv: Sequence[str],
        *,
        cwd: Path,
        log_dir: Path,
        env: dict[str, str] | None = None,
        label: str,
        **_kwargs: Any,
    ) -> ProcessEvidence:
        assert env == {
            "CUDA_VISIBLE_DEVICES": "0",
            "HPS_ROOT": str((tmp_path / "weights/4kagent/hpsv2").resolve()),
            "OUTLINES_CACHE_DIR": str((tmp_path / "run/workers/4kagent/outlines-cache").resolve()),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TORCH_FORCE_WEIGHTS_ONLY_LOAD": "1",
            "OPENAI_API_KEY": "contract-credential-placeholder",
        }
        cache_dir = Path(env["OUTLINES_CACHE_DIR"])
        assert cache_dir.is_dir()
        assert cache_dir.stat().st_mode & 0o077 == 0
        assert Path(argv[argv.index("--runtime-view") + 1]).is_relative_to(
            (tmp_path / "run").resolve()
        )
        assert argv[argv.index("--llm-model") + 1] == "gpt-4-turbo"
        assert argv[argv.index("--api-key-env") + 1] == "OPENAI_API_KEY"
        assert (
            Path(argv[argv.index("--perception-model-path") + 1])
            == (tmp_path / "weights/4kagent/models/Qwen2.5-VL-7B-Instruct").resolve()
        )
        output_dir = Path(argv[argv.index("--output-dir") + 1])
        input_path = Path(argv[argv.index("--input") + 1])
        result_path = output_dir / "nested" / "restored.jpg"
        result_path.parent.mkdir(parents=True)
        with Image.open(input_path) as image:
            image.convert("RGB").save(result_path, "JPEG")
        (output_dir / "scaleguard-result.json").write_text(
            json.dumps(
                {
                    "result": str(result_path.resolve()),
                    "execution_path": {"tasks": ["denoise", "deblur"]},
                }
            ),
            encoding="utf-8",
        )
        return evidence(argv, cwd, log_dir, label)

    monkeypatch.setattr(ProcessRunner, "run", fake_run)

    result = backend(tmp_path).restore(
        source,
        destination,
        bridge_factor=1,
        run_dir=tmp_path / "run",
    )

    assert result.image.path == destination.resolve()
    assert result.image.media_type == "image/png"
    assert result.image.mock is False
    assert result.metadata["backend"] == "4kagent_upstream"
    assert result.metadata["execution_path"] == {"tasks": ["denoise", "deblur"]}
    assert result.metadata["terminal_generative_sr"] is False
    assert result.metadata["depictqa_service"] == {
        "managed": False,
        "host": "127.0.0.1",
        "port": 5001,
    }
    assert result.process is not None


@pytest.mark.parametrize(
    "payload",
    [
        None,
        "{not-json",
        "{}",
        '{"result":"trusted.png","result":"forged.png"}',
        '{"result":NaN}',
    ],
)
def test_fourkagent_adapter_rejects_missing_or_invalid_result_evidence(
    tmp_path: Path,
    make_image: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
    payload: str | None,
) -> None:
    source = make_image(tmp_path / "source.png")

    def fake_run(
        _runner: ProcessRunner,
        argv: Sequence[str],
        *,
        cwd: Path,
        log_dir: Path,
        label: str,
        **_kwargs: Any,
    ) -> ProcessEvidence:
        output_dir = Path(argv[argv.index("--output-dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        if payload is not None:
            (output_dir / "scaleguard-result.json").write_text(payload, encoding="utf-8")
        return evidence(argv, cwd, log_dir, label)

    monkeypatch.setattr(ProcessRunner, "run", fake_run)

    with pytest.raises(ArtifactError, match="did not write a valid adapter result"):
        backend(tmp_path).restore(
            source,
            tmp_path / "output.png",
            bridge_factor=1,
            run_dir=tmp_path / "run",
        )


def test_fourkagent_adapter_rejects_a_result_outside_its_private_directory(
    tmp_path: Path,
    make_image: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = make_image(tmp_path / "source.png")
    escaped = make_image(tmp_path / "escaped.png")

    def fake_run(
        _runner: ProcessRunner,
        argv: Sequence[str],
        *,
        cwd: Path,
        log_dir: Path,
        label: str,
        **_kwargs: Any,
    ) -> ProcessEvidence:
        output_dir = Path(argv[argv.index("--output-dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "scaleguard-result.json").write_text(
            json.dumps({"result": str(escaped.resolve())}),
            encoding="utf-8",
        )
        return evidence(argv, cwd, log_dir, label)

    monkeypatch.setattr(ProcessRunner, "run", fake_run)

    with pytest.raises(ArtifactError, match="escaped its private output directory"):
        backend(tmp_path).restore(
            source,
            tmp_path / "output.png",
            bridge_factor=1,
            run_dir=tmp_path / "run",
        )


def test_fourkagent_adapter_rejects_a_result_symlink_outside_its_private_directory(
    tmp_path: Path,
    make_image: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = make_image(tmp_path / "source.png")
    escaped = make_image(tmp_path / "escaped.png")

    def fake_run(
        _runner: ProcessRunner,
        argv: Sequence[str],
        *,
        cwd: Path,
        log_dir: Path,
        label: str,
        **_kwargs: Any,
    ) -> ProcessEvidence:
        output_dir = Path(argv[argv.index("--output-dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        linked_result = output_dir / "linked.png"
        linked_result.symlink_to(escaped)
        (output_dir / "scaleguard-result.json").write_text(
            json.dumps({"result": str(linked_result)}),
            encoding="utf-8",
        )
        return evidence(argv, cwd, log_dir, label)

    monkeypatch.setattr(ProcessRunner, "run", fake_run)

    with pytest.raises(ArtifactError, match="escaped its private output directory"):
        backend(tmp_path).restore(
            source,
            tmp_path / "output.png",
            bridge_factor=1,
            run_dir=tmp_path / "run",
        )


def test_fourkagent_adapter_accepts_a_result_relative_to_its_private_output(
    tmp_path: Path,
    make_image: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = make_image(tmp_path / "source.png")

    def fake_run(
        _runner: ProcessRunner,
        argv: Sequence[str],
        *,
        cwd: Path,
        log_dir: Path,
        label: str,
        **_kwargs: Any,
    ) -> ProcessEvidence:
        output_dir = Path(argv[argv.index("--output-dir") + 1])
        result_path = output_dir / "nested" / "result.png"
        make_image(result_path)
        (output_dir / "scaleguard-result.json").write_text(
            json.dumps({"result": "nested/result.png"}),
            encoding="utf-8",
        )
        return evidence(argv, cwd, log_dir, label)

    monkeypatch.setattr(ProcessRunner, "run", fake_run)

    result = backend(tmp_path).restore(
        source,
        tmp_path / "output.png",
        bridge_factor=1,
        run_dir=tmp_path / "run",
    )

    assert result.image.path == (tmp_path / "output.png").resolve()


def test_fourkagent_adapter_requires_a_reachable_external_depictqa_service(
    tmp_path: Path,
    make_image: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("scaleguard.backends.fourkagent.tcp_ready", lambda _host, _port: False)
    source = make_image(tmp_path / "source.png")

    with pytest.raises(WorkerError, match="DepictQA is not reachable"):
        backend(tmp_path).restore(
            source,
            tmp_path / "output.png",
            bridge_factor=1,
            run_dir=tmp_path / "run",
        )


def test_fourkagent_adapter_requires_a_cwd_for_a_managed_depictqa_service(
    tmp_path: Path,
    make_image: Callable[..., Path],
) -> None:
    source = make_image(tmp_path / "source.png")
    config = FourKAgentConfig(
        mode="upstream",
        checkout=Path("third_party/checkouts/4KAgent"),
        depictqa_command=("python", "server.py"),
    )

    with pytest.raises(WorkerError, match="requires depictqa_cwd"):
        backend(tmp_path, config).restore(
            source,
            tmp_path / "output.png",
            bridge_factor=1,
            run_dir=tmp_path / "run",
        )


def test_fourkagent_adapter_manages_depictqa_and_expands_project_relative_paths(
    tmp_path: Path,
    make_image: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = make_image(tmp_path / "source.png", size=(6, 4))
    destination = tmp_path / "output.png"
    interpreter_target = tmp_path / "shared-python"
    interpreter_target.write_text("#!/bin/sh\n", encoding="utf-8")
    interpreter_entrypoint = tmp_path / ".runtime/envs/4kagent/bin/python"
    interpreter_entrypoint.parent.mkdir(parents=True)
    interpreter_entrypoint.symlink_to(interpreter_target)
    config = FourKAgentConfig(
        mode="upstream",
        checkout=Path("third_party/checkouts/4KAgent"),
        python_executable=".runtime/envs/4kagent/bin/python",
        depictqa_command=(
            "{project_root}/.runtime/envs/depictqa/bin/python",
            "server.py",
            "--checkout",
            "{checkout}",
        ),
        depictqa_cwd=Path("third_party/checkouts/4KAgent/DepictQA"),
        depictqa_port=5002,
        depictqa_visible_devices="1",
    )
    managed_calls: list[dict[str, Any]] = []

    class FakeManagedService:
        def __init__(self, argv: Sequence[str], **kwargs: Any) -> None:
            self.call = {"argv": tuple(argv), **kwargs}
            self.entered = False
            self.exited = False
            managed_calls.append(self.call)

        def __enter__(self) -> FakeManagedService:
            self.entered = True
            self.call["entered"] = True
            return self

        def __exit__(self, *_args: object) -> None:
            self.exited = True
            self.call["exited"] = True

        def evidence(self) -> dict[str, object]:
            return {
                "managed": True,
                "host": self.call["host"],
                "port": self.call["port"],
                "stopped": self.exited,
            }

    def fake_run(
        _runner: ProcessRunner,
        argv: Sequence[str],
        *,
        cwd: Path,
        log_dir: Path,
        label: str,
        **_kwargs: Any,
    ) -> ProcessEvidence:
        output_dir = Path(argv[argv.index("--output-dir") + 1])
        input_path = Path(argv[argv.index("--input") + 1])
        result_path = output_dir / "result.png"
        result_path.parent.mkdir(parents=True)
        with Image.open(input_path) as image:
            image.save(result_path, "PNG")
        (output_dir / "scaleguard-result.json").write_text(
            json.dumps({"result": str(result_path.resolve())}),
            encoding="utf-8",
        )
        return evidence(argv, cwd, log_dir, label)

    monkeypatch.setattr("scaleguard.backends.fourkagent.ManagedService", FakeManagedService)
    monkeypatch.setattr(ProcessRunner, "run", fake_run)
    adapter = backend(tmp_path, config)

    result = adapter.restore(
        source,
        destination,
        bridge_factor=1,
        run_dir=tmp_path / "run",
    )

    project_root = tmp_path.resolve()
    checkout = (project_root / "third_party/checkouts/4KAgent").resolve()
    assert adapter.checkout == checkout
    assert adapter.python_executable == str(interpreter_entrypoint)
    assert Path(adapter.python_executable).is_symlink()
    assert adapter.python_executable != str(interpreter_target.resolve())
    assert managed_calls == [
        {
            "argv": (
                str((project_root / ".runtime/envs/depictqa/bin/python").resolve()),
                "server.py",
                "--checkout",
                str(checkout),
            ),
            "cwd": (project_root / "third_party/checkouts/4KAgent/DepictQA").resolve(),
            "log_dir": tmp_path / "run/workers/4kagent/depictqa",
            "host": "127.0.0.1",
            "port": 5002,
            "startup_timeout_seconds": 600.0,
            "env": {
                "CUDA_VISIBLE_DEVICES": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            "label": "depictqa-eval",
            "entered": True,
            "exited": True,
        }
    ]
    assert result.metadata["depictqa_service"] == {
        "managed": True,
        "host": "127.0.0.1",
        "port": 5002,
        "stopped": True,
    }
