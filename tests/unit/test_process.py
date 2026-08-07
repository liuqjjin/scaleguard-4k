from __future__ import annotations

import io
import json
import os
import signal
import subprocess
import sys
import time
import venv
from pathlib import Path

import pytest

from scaleguard.errors import WorkerError, WorkerTimeoutError
from scaleguard.runtime.process import (
    ProcessRunner,
    _copy_redacted,
    _GpuSampler,
    _secret_replacements,
    format_command,
    minimal_subprocess_environment,
    project_executable,
    redact_argv,
)


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


def test_process_redacts_explicit_credentials_before_writing_logs(
    tmp_path: Path,
) -> None:
    secret = "scheduler-secret-that-must-not-reach-disk"
    runner = ProcessRunner(timeout_seconds=2.0)
    command = [
        sys.executable,
        "-c",
        (
            "import os,sys; "
            "value=os.environ['CUSTOM_SCHEDULER_CREDENTIAL']; "
            "sys.stdout.write('prefix-' + value + '-suffix'); "
            "sys.stderr.write(value)"
        ),
    ]

    evidence = runner.run(
        command,
        cwd=tmp_path,
        log_dir=tmp_path / "logs",
        env={"CUSTOM_SCHEDULER_CREDENTIAL": secret},
        label="credential-printing-worker",
    )

    stdout = Path(evidence.stdout_path).read_text(encoding="utf-8")
    stderr = Path(evidence.stderr_path).read_text(encoding="utf-8")
    assert secret not in stdout
    assert secret not in stderr
    assert stdout == "prefix-[REDACTED:CUSTOM_SCHEDULER_CREDENTIAL]-suffix"
    assert stderr == "[REDACTED:CUSTOM_SCHEDULER_CREDENTIAL]"


def test_stream_redaction_handles_chunk_boundaries_and_overlapping_secrets() -> None:
    class ChunkedReader(io.BytesIO):
        def read(self, _size: int = -1) -> bytes:
            return super().read(2)

    replacements = _secret_replacements(
        {
            "SHORT_TOKEN": "orchid",
            "LONG_API_KEY": "orchid-lantern",
        }
    )
    destination = io.BytesIO()
    _copy_redacted(
        ChunkedReader(b"prefix-orchid-lantern-orchid-suffix"),
        destination,
        replacements,
    )

    assert destination.getvalue() == (
        b"prefix-[REDACTED:LONG_API_KEY]-[REDACTED:SHORT_TOKEN]-suffix"
    )


def test_stream_copy_without_secrets_and_write_failure_are_explicit() -> None:
    source = io.BytesIO(b"no newline")
    destination = io.BytesIO()
    _copy_redacted(source, destination, ())
    assert destination.getvalue() == b"no newline"

    class FailingWriter(io.BytesIO):
        def write(self, _payload: bytes) -> int:
            raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        _copy_redacted(io.BytesIO(b"payload"), FailingWriter(), ())


def test_minimal_environment_does_not_treat_arbitrary_locale_names_as_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LC_VENDOR_API_TOKEN", "locale-secret")
    monkeypatch.setenv("LC_TIME", "C")
    monkeypatch.setenv("PATH", "/tmp/attacker-bin")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/tmp/attacker-libraries")
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", "/tmp/attacker-ca.pem")
    monkeypatch.setenv("HOME", "/tmp/ambient-home")

    environment = minimal_subprocess_environment()

    assert "LC_VENDOR_API_TOKEN" not in environment
    assert environment["LC_TIME"] == "C"
    assert environment["PATH"] == os.defpath
    assert "LD_LIBRARY_PATH" not in environment
    assert "REQUESTS_CA_BUNDLE" not in environment
    assert "HOME" not in environment


@pytest.mark.skipif(os.name == "nt", reason="the production runtime uses POSIX venv symlinks")
def test_project_executable_preserves_a_real_venv_entrypoint_identity(tmp_path: Path) -> None:
    environment = tmp_path / "runtime" / "worker"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
    entrypoint = environment / "bin" / "python"
    assert entrypoint.is_symlink()

    executable = project_executable(tmp_path, "runtime/worker/bin/python")
    identity = json.loads(
        subprocess.check_output(
            [
                executable,
                "-c",
                (
                    "import json,sys; "
                    "print(json.dumps({'executable':sys.executable,"
                    "'prefix':sys.prefix,'base_prefix':sys.base_prefix}))"
                ),
            ],
            text=True,
        )
    )

    assert executable == str(entrypoint)
    assert Path(executable).is_symlink()
    assert identity["executable"] == executable
    assert identity["prefix"] == str(environment)
    assert identity["base_prefix"] != identity["prefix"]


def test_gpu_sampler_does_not_pass_ambient_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sampler-secret")
    monkeypatch.setenv("GITHUB_TOKEN", "github-secret")
    sampler = _GpuSampler(interval_seconds=0.01, visible_devices="0,1")
    observed_environment: dict[str, str] = {}
    observed_command: list[str] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        observed_command.extend(command)
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
    assert observed_command[:3] == ["nvidia-smi", "-i", "0,1"]
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
            self.stdout = io.BytesIO()
            self.stderr = io.BytesIO()

        def wait(self, timeout: float) -> int:
            del timeout
            self.waits += 1
            if self.waits == 1:
                raise KeyboardInterrupt
            return -signal.SIGTERM

    process = InterruptedProcess()
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: process)

    def terminate(interrupted: InterruptedProcess) -> None:
        signals.append((interrupted.pid, signal.SIGTERM))
        interrupted.wait(timeout=1.0)

    monkeypatch.setattr(
        "scaleguard.runtime.process.terminate_process_group",
        terminate,
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


def test_process_deadline_includes_descendants_holding_log_pipes(tmp_path: Path) -> None:
    runner = ProcessRunner(timeout_seconds=0.15)
    command = [
        sys.executable,
        "-I",
        "-c",
        (
            "import subprocess,sys; "
            "subprocess.Popen([sys.executable, '-I', '-c', "
            "'import time; time.sleep(30)']); "
            "print('parent exited', flush=True)"
        ),
    ]
    started = time.monotonic()

    with pytest.raises(WorkerTimeoutError, match=r"timed out after 0\.1s"):
        runner.run(
            command,
            cwd=tmp_path,
            log_dir=tmp_path / "logs",
            label="descendant-pipe-holder",
        )

    assert time.monotonic() - started < 2.0
    assert (tmp_path / "logs" / "descendant-pipe-holder.stdout.log").read_text(
        encoding="utf-8"
    ).strip() == "parent exited"


def test_process_promptly_reaps_a_log_detached_descendant_after_leader_exit(
    tmp_path: Path,
) -> None:
    child_pid_path = tmp_path / "child.pid"
    runner = ProcessRunner(timeout_seconds=60.0)
    child_code = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"
    command = [
        sys.executable,
        "-I",
        "-c",
        (
            "import pathlib,subprocess,sys; "
            f"child=subprocess.Popen([sys.executable, '-I', '-c', {child_code!r}], "
            "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); "
            f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid))"
        ),
    ]
    started = time.monotonic()

    evidence = runner.run(
        command,
        cwd=tmp_path,
        log_dir=tmp_path / "logs",
        label="detached-log-descendant",
    )

    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    assert evidence.returncode == 0
    assert time.monotonic() - started < 3.0
    deadline = time.monotonic() + 1.0
    while _process_is_running(child_pid) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not _process_is_running(child_pid)


def _process_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    status = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)],
        check=False,
        capture_output=True,
        text=True,
    )
    return status.returncode == 0 and not status.stdout.strip().startswith("Z")


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
    fake_bare_token = "sk-" + "fixture-bare-token-123"
    assert redact_argv(
        (
            "worker",
            "--api-key",
            "explicit-secret",
            "--token=inline-secret",
            fake_bare_token,
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
