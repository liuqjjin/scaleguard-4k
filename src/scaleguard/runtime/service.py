"""Lifecycle ownership for local model services used by an upstream phase."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, TextIO

from scaleguard.errors import WorkerError
from scaleguard.runtime.process import minimal_subprocess_environment, redact_argv


def tcp_ready(host: str, port: int, *, timeout_seconds: float = 0.25) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            return True
    except OSError:
        return False


class ManagedService:
    """Start one process group, wait for its TCP listener, and always stop it."""

    def __init__(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        log_dir: Path,
        host: str,
        port: int,
        startup_timeout_seconds: float,
        env: Mapping[str, str] | None = None,
        label: str,
    ) -> None:
        self.argv = tuple(argv)
        self.cwd = cwd
        self.log_dir = log_dir
        self.host = host
        self.port = port
        self.startup_timeout_seconds = startup_timeout_seconds
        self.env = env
        self.label = label
        self.process: subprocess.Popen[str] | None = None
        self.stdout: TextIO | None = None
        self.stderr: TextIO | None = None
        self.started_at: float | None = None
        self.stopped_at: float | None = None
        self.returncode: int | None = None

    def __enter__(self) -> ManagedService:
        if not self.argv or not self.argv[0]:
            raise WorkerError(f"{self.label} service command is empty")
        if tcp_ready(self.host, self.port):
            raise WorkerError(
                f"{self.label} endpoint {self.host}:{self.port} is already in use; "
                "refusing to take ownership of an unknown process"
            )
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.stdout = (self.log_dir / f"{self.label}.stdout.log").open("w", encoding="utf-8")
        self.stderr = (self.log_dir / f"{self.label}.stderr.log").open("w", encoding="utf-8")
        process_env = minimal_subprocess_environment(self.env)
        self.started_at = time.monotonic()
        try:
            self.process = subprocess.Popen(
                list(self.argv),
                cwd=self.cwd,
                env=process_env,
                stdin=subprocess.DEVNULL,
                stdout=self.stdout,
                stderr=self.stderr,
                text=True,
                start_new_session=True,
            )
            self._wait_until_ready()
        except BaseException:
            self.stop()
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        self.stop()

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.startup_timeout_seconds
        while time.monotonic() < deadline:
            if self.process is None:
                raise WorkerError(f"{self.label} service was not started")
            returncode = self.process.poll()
            if returncode is not None:
                self.returncode = returncode
                raise WorkerError(
                    f"{self.label} service exited with code {returncode} before readiness; "
                    f"stderr: {self.log_dir / f'{self.label}.stderr.log'}"
                )
            if tcp_ready(self.host, self.port):
                return
            time.sleep(0.2)
        raise WorkerError(
            f"{self.label} service did not listen on {self.host}:{self.port} "
            f"within {self.startup_timeout_seconds:.1f}s"
        )

    def stop(self) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
        if process is not None:
            self.returncode = process.poll()
        self.stopped_at = time.monotonic()
        for stream in (self.stdout, self.stderr):
            if stream is not None and not stream.closed:
                stream.close()

    def evidence(self) -> dict[str, Any]:
        duration: float | None = None
        if self.started_at is not None:
            duration = (self.stopped_at or time.monotonic()) - self.started_at
        return {
            "managed": True,
            "argv": list(redact_argv(self.argv)),
            "cwd": str(self.cwd.resolve()),
            "host": self.host,
            "port": self.port,
            "returncode": self.returncode,
            "duration_seconds": duration,
            "stdout_path": str((self.log_dir / f"{self.label}.stdout.log").resolve()),
            "stderr_path": str((self.log_dir / f"{self.label}.stderr.log").resolve()),
        }
