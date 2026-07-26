"""Subprocess execution with preserved logs, deadlines, and GPU evidence."""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from scaleguard.contracts import ProcessEvidence
from scaleguard.errors import WorkerError, WorkerTimeoutError

_SECRET_FLAG = re.compile(r"(?i)(token|api[-_]?key|password|secret)")
_SECRET_VALUE = re.compile(r"^(?:hf_|sk-)[A-Za-z0-9_-]{8,}$")
_SAFE_ENVIRONMENT_NAMES = frozenset(
    {
        "COMSPEC",
        "HOME",
        "LANG",
        "LD_LIBRARY_PATH",
        "LOGNAME",
        "NO_PROXY",
        "PATH",
        "PATHEXT",
        "REQUESTS_CA_BUNDLE",
        "SHELL",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "SystemRoot",
        "TEMP",
        "TMP",
        "TMPDIR",
        "USER",
    }
)


def minimal_subprocess_environment(
    overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return a small runtime environment without ambient credentials."""

    environment = {
        name: value
        for name, value in os.environ.items()
        if name in _SAFE_ENVIRONMENT_NAMES or name.startswith("LC_")
    }
    if overrides:
        environment.update(overrides)
    return environment


def redact_argv(argv: Sequence[str]) -> tuple[str, ...]:
    redacted: list[str] = []
    hide_next = False
    for token in argv:
        if hide_next:
            redacted.append("<redacted>")
            hide_next = False
            continue
        if token.startswith("--") and "=" in token:
            key, _, _value = token.partition("=")
            redacted.append(f"{key}=<redacted>" if _SECRET_FLAG.search(key) else token)
            continue
        redacted.append("<redacted>" if _SECRET_VALUE.match(token) else token)
        if token.startswith("-") and _SECRET_FLAG.search(token):
            hide_next = True
    return tuple(redacted)


class _GpuSampler:
    def __init__(self, interval_seconds: float) -> None:
        self.interval_seconds = interval_seconds
        self.peaks: dict[str, int] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if shutil.which("nvidia-smi") is None:
            return
        self._thread = threading.Thread(target=self._sample_loop, name="gpu-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, int]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 2))
        return dict(sorted(self.peaks.items()))

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=5,
                    env=minimal_subprocess_environment(),
                )
                if result.returncode == 0:
                    for line in result.stdout.splitlines():
                        index, separator, memory = line.partition(",")
                        if not separator:
                            continue
                        used = int(memory.strip())
                        key = index.strip()
                        self.peaks[key] = max(used, self.peaks.get(key, 0))
            except (OSError, ValueError, subprocess.SubprocessError):
                return
            self._stop.wait(self.interval_seconds)


@dataclass(frozen=True, slots=True)
class ProcessRunner:
    timeout_seconds: float
    gpu_poll_interval_seconds: float = 0.5

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        log_dir: Path,
        env: Mapping[str, str] | None = None,
        label: str,
    ) -> ProcessEvidence:
        if not argv or not argv[0]:
            raise WorkerError(f"{label} command is empty")
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / f"{label}.stdout.log"
        stderr_path = log_dir / f"{label}.stderr.log"
        process_env = minimal_subprocess_environment(env)
        sampler = _GpuSampler(self.gpu_poll_interval_seconds)
        started = time.monotonic()
        process: subprocess.Popen[str]
        returncode: int
        timed_out: subprocess.TimeoutExpired | None = None
        with (
            stdout_path.open("w", encoding="utf-8") as stdout,
            stderr_path.open("w", encoding="utf-8") as stderr,
        ):
            try:
                process = subprocess.Popen(
                    list(argv),
                    cwd=cwd,
                    env=process_env,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                    start_new_session=True,
                )
            except OSError as error:
                raise WorkerError(
                    f"cannot start {label} command {redact_argv(argv)!r} in {cwd}: {error}"
                ) from error
            try:
                sampler.start()
                returncode = process.wait(timeout=self.timeout_seconds)
            except subprocess.TimeoutExpired as error:
                timed_out = error
                self._terminate_preserving_exception(process)
                returncode = -signal.SIGTERM
            except BaseException:
                self._terminate_preserving_exception(process)
                raise
            finally:
                peaks = sampler.stop()
        evidence = ProcessEvidence(
            argv=redact_argv(argv),
            cwd=str(cwd.resolve()),
            returncode=returncode,
            duration_seconds=time.monotonic() - started,
            stdout_path=str(stdout_path.resolve()),
            stderr_path=str(stderr_path.resolve()),
            peak_vram_mib=peaks,
        )
        if timed_out is not None:
            raise WorkerTimeoutError(
                f"{label} timed out after {self.timeout_seconds:.1f}s; "
                f"stderr preserved at {evidence.stderr_path}"
            ) from timed_out
        if returncode != 0:
            raise WorkerError(
                f"{label} exited with code {returncode}; stderr preserved at {stderr_path}"
            )
        return evidence

    @classmethod
    def _terminate_preserving_exception(cls, process: subprocess.Popen[str]) -> None:
        try:
            cls._terminate(process)
        except (OSError, subprocess.SubprocessError):
            pass

    @staticmethod
    def _terminate(process: subprocess.Popen[str]) -> None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)


def format_command(template: Sequence[str], values: Mapping[str, str]) -> tuple[str, ...]:
    """Expand known placeholders without invoking a shell."""

    formatted: list[str] = []
    for token in template:
        try:
            formatted.append(token.format_map(values))
        except KeyError as error:
            raise WorkerError(
                f"unknown command placeholder {error.args[0]!r} in {token!r}"
            ) from error
    return tuple(formatted)
