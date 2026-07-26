from __future__ import annotations

import signal
import subprocess
import sys
from pathlib import Path

import pytest

from scaleguard.errors import WorkerError, WorkerTimeoutError
from scaleguard.runtime.process import ProcessRunner, _GpuSampler, format_command, redact_argv


def test_process_preserves_stdout_stderr_and_evidence_on_success(tmp_path: Path) -> None:
    runner = ProcessRunner(timeout_seconds=2.0, gpu_poll_interval_seconds=0.01)
    command = [
        sys.executable,
        "-c",
        "import sys; print('标准输出'); print('warning on stderr', file=sys.stderr)",
    ]

    evidence = runner.run(
        command,
        cwd=tmp_path,
        log_dir=tmp_path / "日志 with spaces",
        label="fake-worker",
    )

    assert evidence.returncode == 0
    assert evidence.argv == tuple(command)
    assert Path(evidence.stdout_path).read_text(encoding="utf-8").strip() == "标准输出"
    assert Path(evidence.stderr_path).read_text(encoding="utf-8").strip() == "warning on stderr"
    assert evidence.peak_vram_mib == {}


def test_nonzero_process_reports_only_log_path_and_keeps_stderr_private(tmp_path: Path) -> None:
    runner = ProcessRunner(timeout_seconds=2.0)
    command = [
        sys.executable,
        "-c",
        "import sys; print('fatal worker detail', file=sys.stderr); raise SystemExit(7)",
    ]

    with pytest.raises(WorkerError) as captured:
        runner.run(command, cwd=tmp_path, log_dir=tmp_path / "logs", label="broken-worker")

    message = str(captured.value)
    assert "exited with code 7" in message
    assert "fatal worker detail" not in message
    assert "broken-worker.stderr.log" in message
    stderr_path = tmp_path / "logs" / "broken-worker.stderr.log"
    assert stderr_path.read_text(encoding="utf-8").strip() == "fatal worker detail"


def test_process_does_not_inherit_ambient_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-credential-placeholder")
    monkeypatch.setenv("SLURM_NODELIST", "node[01-02]; touch /tmp/not-allowed")
    monkeypatch.setenv("OUTLINES_CACHE_DIR", "/tmp/ambient-cache")
    runner = ProcessRunner(timeout_seconds=2.0)
    command = [
        sys.executable,
        "-c",
        (
            "import os; "
            "assert 'OPENAI_API_KEY' not in os.environ; "
            "assert 'SLURM_NODELIST' not in os.environ; "
            "assert 'OUTLINES_CACHE_DIR' not in os.environ; "
            "assert os.environ['SCALEGUARD_EXPLICIT'] == 'yes'"
        ),
    ]

    runner.run(
        command,
        cwd=tmp_path,
        log_dir=tmp_path / "logs",
        env={"SCALEGUARD_EXPLICIT": "yes"},
        label="sanitized-worker",
    )


def test_gpu_sampler_does_not_pass_ambient_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sampler-secret")
    monkeypatch.setenv("GITHUB_TOKEN", "github-secret")
    sampler = _GpuSampler(interval_seconds=0.01)
    observed_environment: dict[str, str] = {}

    def run(*_args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        environment = kwargs.get("env")
        assert isinstance(environment, dict)
        observed_environment.update(environment)
        sampler._stop.set()
        return subprocess.CompletedProcess(
            args=["nvidia-smi"],
            returncode=0,
            stdout="0, 123\n",
            stderr="",
        )

    monkeypatch.setattr("scaleguard.runtime.process.subprocess.run", run)

    sampler._sample_loop()

    assert sampler.peaks == {"0": 123}
    assert "OPENAI_API_KEY" not in observed_environment
    assert "GITHUB_TOKEN" not in observed_environment


def test_process_interrupt_terminates_the_worker_group_and_preserves_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InterruptedProcess:
        pid = 4242

        def __init__(self) -> None:
            self.waits = 0

        def wait(self, timeout: float) -> int:
            del timeout
            self.waits += 1
            if self.waits == 1:
                raise KeyboardInterrupt
            return -signal.SIGTERM

    process = InterruptedProcess()
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(
        "scaleguard.runtime.process.os.killpg",
        lambda pid, sig: signals.append((pid, sig)),
    )
    monkeypatch.setattr("scaleguard.runtime.process._GpuSampler.start", lambda _self: None)
    monkeypatch.setattr("scaleguard.runtime.process._GpuSampler.stop", lambda _self: {})

    with pytest.raises(KeyboardInterrupt):
        ProcessRunner(timeout_seconds=2.0).run(
            ["worker"],
            cwd=tmp_path,
            log_dir=tmp_path / "logs",
            label="interrupted-worker",
        )

    assert signals == [(4242, signal.SIGTERM)]
    assert process.waits == 2


def test_process_timeout_terminates_the_worker_and_preserves_stderr(tmp_path: Path) -> None:
    runner = ProcessRunner(timeout_seconds=0.1)
    command = [
        sys.executable,
        "-c",
        (
            "import sys,time; "
            "print('started before timeout', file=sys.stderr, flush=True); "
            "time.sleep(30)"
        ),
    ]

    with pytest.raises(WorkerTimeoutError, match=r"timed out after 0\.1s"):
        runner.run(command, cwd=tmp_path, log_dir=tmp_path / "logs", label="slow-worker")

    stderr_path = tmp_path / "logs" / "slow-worker.stderr.log"
    assert stderr_path.read_text(encoding="utf-8").strip() == "started before timeout"


def test_command_placeholders_expand_as_argv_without_a_shell() -> None:
    template = (
        "python",
        "worker.py",
        "--input",
        "{input}",
        "--output={output}",
        "--literal",
        "$HOME; touch should-not-exist",
    )

    result = format_command(
        template,
        {
            "input": "/tmp/含 空格/input.jpg",
            "output": "/tmp/output.png",
        },
    )

    assert result == (
        "python",
        "worker.py",
        "--input",
        "/tmp/含 空格/input.jpg",
        "--output=/tmp/output.png",
        "--literal",
        "$HOME; touch should-not-exist",
    )


def test_unknown_command_placeholder_is_an_actionable_worker_error() -> None:
    with pytest.raises(WorkerError, match="unknown command placeholder 'missing'"):
        format_command(("python", "{missing}"), {"input": "image.png"})


def test_command_evidence_redacts_secret_flags_and_bare_tokens() -> None:
    assert redact_argv(
        (
            "worker",
            "--api-key",
            "explicit-secret",
            "--token=inline-secret",
            "sk-baresecret123",
            "--safe=value",
        )
    ) == (
        "worker",
        "--api-key",
        "<redacted>",
        "--token=<redacted>",
        "<redacted>",
        "--safe=value",
    )
