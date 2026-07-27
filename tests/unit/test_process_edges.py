from __future__ import annotations

import io
import os
import signal
import subprocess
from pathlib import Path
from typing import Any

import pytest

from scaleguard.errors import WorkerError
from scaleguard.runtime.process import (
    ProcessRunner,
    _GpuSampler,
    process_group_exists,
    terminate_process_group,
)


class _FinishedProcess:
    pid = 4242
    stdout: io.BytesIO | None = None
    stderr: io.BytesIO | None = None

    def poll(self) -> int:
        return 0

    def wait(self, timeout: float) -> int:
        del timeout
        return 0


class _RunningProcess(_FinishedProcess):
    def poll(self) -> None:
        return None

    def wait(self, timeout: float) -> int:
        raise subprocess.TimeoutExpired(["worker"], timeout)


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (ProcessLookupError(), False),
        (PermissionError(), True),
        (None, True),
    ],
)
def test_process_group_probe_handles_kernel_outcomes(
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException | None,
    expected: bool,
) -> None:
    def killpg(_process_group: int, _signal_number: int) -> None:
        if failure is not None:
            raise failure

    monkeypatch.setattr(os, "killpg", killpg)

    assert process_group_exists(4242) is expected


@pytest.mark.parametrize("process", [_FinishedProcess(), _RunningProcess()])
def test_terminate_waits_for_a_group_after_term(
    monkeypatch: pytest.MonkeyPatch,
    process: _FinishedProcess,
) -> None:
    group_states = iter([True, False, False, False, False])
    signals: list[int] = []
    sleeps: list[float] = []
    monkeypatch.setattr(
        "scaleguard.runtime.process.process_group_exists",
        lambda _process_group: next(group_states),
    )
    monkeypatch.setattr(
        os,
        "killpg",
        lambda _process_group, signal_number: signals.append(signal_number),
    )
    monkeypatch.setattr(
        "scaleguard.runtime.process.time.sleep",
        lambda duration: sleeps.append(duration),
    )

    terminate_process_group(process, term_timeout_seconds=1.0)

    assert signals == [signal.SIGTERM]
    if isinstance(process, _FinishedProcess) and not isinstance(process, _RunningProcess):
        assert sleeps


def test_terminate_escalates_and_reports_a_persistent_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals: list[int] = []
    monkeypatch.setattr(
        "scaleguard.runtime.process.process_group_exists",
        lambda _process_group: True,
    )
    monkeypatch.setattr(
        os,
        "killpg",
        lambda _process_group, signal_number: signals.append(signal_number),
    )

    with pytest.raises(WorkerError, match="did not exit after SIGKILL"):
        terminate_process_group(
            _FinishedProcess(),
            term_timeout_seconds=0.0,
            kill_timeout_seconds=0.0,
        )

    assert signals == [signal.SIGTERM, signal.SIGKILL]


def test_terminate_accepts_a_group_disappearing_during_sigkill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals: list[int] = []
    monkeypatch.setattr(
        "scaleguard.runtime.process.process_group_exists",
        lambda _process_group: True,
    )

    def killpg(_process_group: int, signal_number: int) -> None:
        signals.append(signal_number)
        if signal_number == signal.SIGKILL:
            raise ProcessLookupError

    monkeypatch.setattr(os, "killpg", killpg)

    terminate_process_group(
        _FinishedProcess(),
        term_timeout_seconds=0.0,
        kill_timeout_seconds=0.0,
    )

    assert signals == [signal.SIGTERM, signal.SIGKILL]


def test_gpu_sampler_start_and_stop_own_the_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, Any]] = []

    class Thread:
        def __init__(self, **kwargs: Any) -> None:
            events.append(("create", kwargs))

        def start(self) -> None:
            events.append(("start", None))

        def join(self, timeout: float) -> None:
            events.append(("join", timeout))

    monkeypatch.setattr("scaleguard.runtime.process.shutil.which", lambda _name: "/gpu")
    monkeypatch.setattr("scaleguard.runtime.process.threading.Thread", Thread)
    sampler = _GpuSampler(interval_seconds=0.25)
    sampler.peaks = {"1": 8, "0": 16}

    sampler.start()
    peaks = sampler.stop()

    assert [event[0] for event in events] == ["create", "start", "join"]
    assert events[-1] == ("join", 1.0)
    assert peaks == {"0": 16, "1": 8}


def test_gpu_sampler_ignores_malformed_rows_and_stops_on_invalid_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampler = _GpuSampler(interval_seconds=0.01)
    monkeypatch.setattr(
        "scaleguard.runtime.process.subprocess.run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=["nvidia-smi"],
            returncode=0,
            stdout="missing separator\n0, not-an-integer\n",
            stderr="",
        ),
    )

    sampler._sample_loop()

    assert sampler.peaks == {}


def test_process_runner_rejects_empty_and_unstartable_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = ProcessRunner(timeout_seconds=1.0)
    with pytest.raises(WorkerError, match="command is empty"):
        runner.run([], cwd=tmp_path, log_dir=tmp_path / "logs", label="empty")

    def fail_to_start(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("executable unavailable")

    monkeypatch.setattr(subprocess, "Popen", fail_to_start)
    with pytest.raises(
        WorkerError,
        match=r"cannot start missing command .*executable unavailable",
    ):
        runner.run(
            ["missing", "--api-key", "must-not-appear"],
            cwd=tmp_path,
            log_dir=tmp_path / "logs",
            label="missing",
        )


def test_cleanup_helpers_tolerate_races_and_closed_pipes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ClosedPipe:
        def fileno(self) -> int:
            raise ValueError("already closed")

    process = _FinishedProcess()
    process.stdout = ClosedPipe()  # type: ignore[assignment]
    process.stderr = None

    monkeypatch.setattr(
        os,
        "killpg",
        lambda _process_group, _signal_number: (_ for _ in ()).throw(ProcessLookupError()),
    )
    ProcessRunner._signal_process_group(process.pid, signal.SIGKILL)
    ProcessRunner._close_process_pipes(process)  # type: ignore[arg-type]

    monkeypatch.setattr(
        ProcessRunner,
        "_terminate",
        staticmethod(lambda _process: (_ for _ in ()).throw(WorkerError("race"))),
    )
    ProcessRunner._terminate_preserving_exception(process)  # type: ignore[arg-type]
