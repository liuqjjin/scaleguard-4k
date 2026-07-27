from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scaleguard.errors import WorkerError
from scaleguard.runtime.service import ManagedService, tcp_ready

HOST = "127.0.0.1"


def unused_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((HOST, 0))
        return int(listener.getsockname()[1])


def service_command(port: int) -> tuple[str, ...]:
    code = (
        "import socket\n"
        f"listener = socket.socket(); listener.bind(({HOST!r}, {port})); listener.listen()\n"
        "print('ready', flush=True)\n"
        "while True:\n"
        "    connection, _ = listener.accept()\n"
        "    connection.close()\n"
    )
    return (sys.executable, "-u", "-c", code)


def test_managed_service_stops_its_process_group_after_successful_readiness(
    tmp_path: Path,
) -> None:
    port = unused_port()
    service = ManagedService(
        service_command(port),
        cwd=tmp_path,
        log_dir=tmp_path / "logs",
        host=HOST,
        port=port,
        startup_timeout_seconds=3.0,
        env={"SCALEGUARD_SERVICE_TEST": "1"},
        label="fake-service",
    )

    with service as running:
        assert running.process is not None
        assert running.process.poll() is None
        assert tcp_ready(HOST, port)
        assert running.evidence()["managed"] is True

    assert service.process is not None
    assert service.process.poll() is not None
    assert service.returncode == -signal.SIGTERM
    assert service.stopped_at is not None
    assert service.stdout is not None
    assert service.stdout.closed
    assert service.stderr is not None
    assert service.stderr.closed
    evidence = service.evidence()
    assert evidence["returncode"] == -signal.SIGTERM
    assert isinstance(evidence["duration_seconds"], float)
    assert Path(evidence["stdout_path"]).read_text(encoding="utf-8").strip() == "ready"


def test_managed_service_reports_a_process_that_exits_before_listening(
    tmp_path: Path,
) -> None:
    port = unused_port()
    service = ManagedService(
        (
            sys.executable,
            "-c",
            "import sys; print('startup failed', file=sys.stderr, flush=True); raise SystemExit(7)",
        ),
        cwd=tmp_path,
        log_dir=tmp_path / "logs",
        host=HOST,
        port=port,
        startup_timeout_seconds=2.0,
        label="early-exit",
    )

    with pytest.raises(WorkerError, match="exited with code 7 before readiness"):
        service.__enter__()

    assert service.returncode == 7
    assert service.process is not None
    assert service.process.poll() == 7
    assert service.stdout is not None
    assert service.stdout.closed
    assert service.stderr is not None
    assert service.stderr.closed
    assert (tmp_path / "logs" / "early-exit.stderr.log").read_text(
        encoding="utf-8"
    ).strip() == "startup failed"


def test_managed_service_reaps_descendants_when_the_leader_exits_during_startup(
    tmp_path: Path,
) -> None:
    port = unused_port()
    child_pid_path = tmp_path / "child.pid"
    code = (
        "import pathlib,subprocess,sys\n"
        "child = subprocess.Popen("
        "[sys.executable, '-I', '-c', 'import time; time.sleep(30)'], "
        "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid))\n"
        "raise SystemExit(7)\n"
    )
    service = ManagedService(
        (sys.executable, "-I", "-c", code),
        cwd=tmp_path,
        log_dir=tmp_path / "logs",
        host=HOST,
        port=port,
        startup_timeout_seconds=2.0,
        label="orphaning-service",
    )

    with pytest.raises(WorkerError, match="exited with code 7 before readiness"):
        service.__enter__()

    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    try:
        deadline = time.monotonic() + 1.0
        while _process_is_running(child_pid) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not _process_is_running(child_pid)
    finally:
        if _process_is_running(child_pid):
            os.kill(child_pid, signal.SIGKILL)


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


def test_managed_service_refuses_to_take_over_an_occupied_port(tmp_path: Path) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        occupied.bind((HOST, 0))
        occupied.listen()
        port = int(occupied.getsockname()[1])
        service = ManagedService(
            service_command(port),
            cwd=tmp_path,
            log_dir=tmp_path / "logs",
            host=HOST,
            port=port,
            startup_timeout_seconds=1.0,
            label="occupied",
        )

        with pytest.raises(WorkerError, match=r"already in use.*refusing to take ownership"):
            service.__enter__()

    assert service.process is None
    assert not (tmp_path / "logs").exists()


def test_managed_service_rejects_an_empty_command(tmp_path: Path) -> None:
    service = ManagedService(
        (),
        cwd=tmp_path,
        log_dir=tmp_path / "logs",
        host=HOST,
        port=unused_port(),
        startup_timeout_seconds=1.0,
        label="empty",
    )

    with pytest.raises(WorkerError, match="service command is empty"):
        service.__enter__()
