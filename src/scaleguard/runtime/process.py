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
from typing import Any, BinaryIO

from scaleguard.contracts import ProcessEvidence
from scaleguard.errors import WorkerError, WorkerTimeoutError

_SECRET_FLAG = re.compile(r"(?i)(token|api[-_]?key|password|credential|secret)")
_SECRET_VALUE = re.compile(r"^(?:hf_|sk-)[A-Za-z0-9_-]{8,}$")
_DRAIN_SHUTDOWN_SECONDS = 1.0
_POST_LEADER_GROUP_GRACE_SECONDS = 0.2
_SAFE_ENVIRONMENT_NAMES = frozenset(
    {
        "COMSPEC",
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_COLLATE",
        "LC_CTYPE",
        "LC_MESSAGES",
        "LC_MONETARY",
        "LC_NUMERIC",
        "LC_TIME",
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
        name: value for name, value in os.environ.items() if name in _SAFE_ENVIRONMENT_NAMES
    }
    if overrides:
        environment.update(overrides)
    return environment


def project_executable(project_root: Path, executable: str) -> str:
    """Make an explicit project-relative command absolute without resolving symlinks."""

    path = Path(executable)
    if not path.is_absolute() and "/" not in executable:
        return executable
    candidate = path if path.is_absolute() else project_root / path
    return os.path.abspath(os.fspath(candidate))


def process_group_exists(process_group: int) -> bool:
    """Return whether an owned POSIX process group still has members."""

    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def terminate_process_group(
    process: subprocess.Popen[Any],
    *,
    term_timeout_seconds: float = 1.0,
    kill_timeout_seconds: float = 1.0,
) -> None:
    """Terminate an owned session even when its original leader has exited."""

    process_group = process.pid

    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + term_timeout_seconds
    while process_group_exists(process_group) and time.monotonic() < deadline:
        if process.poll() is None:
            try:
                process.wait(timeout=min(0.05, max(0.0, deadline - time.monotonic())))
            except subprocess.TimeoutExpired:
                pass
        else:
            time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))
    if process_group_exists(process_group):
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            return
    kill_deadline = time.monotonic() + kill_timeout_seconds
    while process_group_exists(process_group) and time.monotonic() < kill_deadline:
        if process.poll() is None:
            try:
                process.wait(timeout=min(0.05, max(0.0, kill_deadline - time.monotonic())))
            except subprocess.TimeoutExpired:
                pass
        else:
            time.sleep(min(0.02, max(0.0, kill_deadline - time.monotonic())))
    if process_group_exists(process_group):
        raise WorkerError(f"owned process group {process_group} did not exit after SIGKILL")


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


def _secret_replacements(
    environment: Mapping[str, str],
) -> tuple[tuple[bytes, bytes], ...]:
    replacements: list[tuple[bytes, bytes]] = []
    for name, value in environment.items():
        if not value or not _SECRET_FLAG.search(name):
            continue
        encoded = value.encode("utf-8")
        if not encoded:
            continue
        safe_name = re.sub(r"[^A-Za-z0-9_]", "_", name)
        replacements.append((encoded, f"[REDACTED:{safe_name}]".encode("ascii")))
    replacements.sort(key=lambda item: len(item[0]), reverse=True)
    return tuple(replacements)


def _redact_available(
    payload: bytes,
    replacements: Sequence[tuple[bytes, bytes]],
    *,
    final: bool,
) -> tuple[bytes, bytes]:
    if not replacements:
        return payload, b""
    maximum = max(len(secret) for secret, _replacement in replacements)
    safe_start_limit = len(payload) if final else max(0, len(payload) - maximum + 1)
    if safe_start_limit == 0:
        return b"", payload

    output = bytearray()
    cursor = 0
    while cursor < safe_start_limit:
        matches = [
            (index, -len(secret), secret, replacement)
            for secret, replacement in replacements
            if (index := payload.find(secret, cursor)) >= 0 and index < safe_start_limit
        ]
        if not matches:
            break
        index, _negative_length, secret, replacement = min(matches)
        output.extend(payload[cursor:index])
        output.extend(replacement)
        cursor = index + len(secret)
    if cursor < safe_start_limit:
        output.extend(payload[cursor:safe_start_limit])
    consumed = max(cursor, safe_start_limit)
    return bytes(output), payload[consumed:]


def _copy_redacted(
    source: BinaryIO,
    destination: BinaryIO,
    replacements: Sequence[tuple[bytes, bytes]],
) -> None:
    pending = b""
    for chunk in iter(lambda: source.read(64 * 1024), b""):
        emitted, pending = _redact_available(
            pending + chunk,
            replacements,
            final=False,
        )
        destination.write(emitted)
        destination.flush()
    emitted, pending = _redact_available(pending, replacements, final=True)
    destination.write(emitted)
    destination.write(pending)
    destination.flush()


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
        process: subprocess.Popen[bytes]
        returncode: int
        timed_out: subprocess.TimeoutExpired | None = None
        drain_errors: list[BaseException] = []
        drain_threads: list[threading.Thread] = []
        forced_pipe_close = threading.Event()
        terminate_group = False
        replacements = _secret_replacements(process_env)
        with (
            stdout_path.open("wb") as stdout,
            stderr_path.open("wb") as stderr,
        ):
            try:
                process = subprocess.Popen(
                    list(argv),
                    cwd=cwd,
                    env=process_env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=False,
                    start_new_session=True,
                )
            except OSError as error:
                raise WorkerError(
                    f"cannot start {label} command {redact_argv(argv)!r} in {cwd}: {error}"
                ) from error
            assert process.stdout is not None
            assert process.stderr is not None

            def drain(source: BinaryIO, destination: BinaryIO) -> None:
                try:
                    _copy_redacted(source, destination, replacements)
                except BaseException as error:
                    if not forced_pipe_close.is_set():
                        drain_errors.append(error)
                finally:
                    try:
                        source.close()
                    except OSError as error:
                        if not forced_pipe_close.is_set():
                            drain_errors.append(error)

            drain_threads = [
                threading.Thread(
                    target=drain,
                    args=(process.stdout, stdout),
                    name=f"{label}-stdout-redactor",
                    daemon=True,
                ),
                threading.Thread(
                    target=drain,
                    args=(process.stderr, stderr),
                    name=f"{label}-stderr-redactor",
                    daemon=True,
                ),
            ]
            for thread in drain_threads:
                thread.start()
            try:
                sampler.start()
                remaining = max(0.0, started + self.timeout_seconds - time.monotonic())
                returncode = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired as error:
                timed_out = error
                terminate_group = True
                returncode = -signal.SIGTERM
            except BaseException:
                terminate_group = True
                raise
            finally:
                if not terminate_group:
                    run_deadline = started + self.timeout_seconds
                    post_leader_deadline = min(
                        run_deadline,
                        time.monotonic() + _POST_LEADER_GROUP_GRACE_SECONDS,
                    )
                    self._wait_for_process_group_exit(
                        process.pid,
                        deadline=post_leader_deadline,
                    )
                    if process_group_exists(process.pid):
                        terminate_group = True
                        if time.monotonic() >= run_deadline:
                            timed_out = subprocess.TimeoutExpired(
                                list(argv),
                                self.timeout_seconds,
                            )
                            returncode = -signal.SIGTERM
                    else:
                        self._join_threads(
                            drain_threads,
                            deadline=min(
                                run_deadline,
                                time.monotonic() + _DRAIN_SHUTDOWN_SECONDS,
                            ),
                        )
                    if any(thread.is_alive() for thread in drain_threads):
                        timed_out = subprocess.TimeoutExpired(
                            list(argv),
                            self.timeout_seconds,
                        )
                        terminate_group = True
                        returncode = -signal.SIGTERM
                if terminate_group:
                    self._terminate_preserving_exception(process)
                    self._join_threads(
                        drain_threads,
                        deadline=time.monotonic() + _DRAIN_SHUTDOWN_SECONDS,
                    )
                if any(thread.is_alive() for thread in drain_threads):
                    self._signal_process_group(process.pid, signal.SIGKILL)
                    forced_pipe_close.set()
                    self._close_process_pipes(process)
                    self._join_threads(
                        drain_threads,
                        deadline=time.monotonic() + _DRAIN_SHUTDOWN_SECONDS,
                    )
                peaks = sampler.stop()
        if process_group_exists(process.pid):
            raise WorkerError(f"cannot stop owned {label} process group {process.pid}")
        if any(thread.is_alive() for thread in drain_threads):
            raise WorkerError(f"cannot stop {label} process log drains after termination")
        if drain_errors:
            raise WorkerError(
                f"cannot preserve redacted {label} process logs: {type(drain_errors[0]).__name__}"
            ) from drain_errors[0]
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
    def _terminate_preserving_exception(cls, process: subprocess.Popen[bytes]) -> None:
        try:
            cls._terminate(process)
        except (OSError, subprocess.SubprocessError, WorkerError):
            pass

    @staticmethod
    def _join_threads(
        threads: Sequence[threading.Thread],
        *,
        deadline: float,
    ) -> None:
        for thread in threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))

    @staticmethod
    def _wait_for_process_group_exit(process_group: int, *, deadline: float) -> None:
        while process_group_exists(process_group) and time.monotonic() < deadline:
            time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))

    @staticmethod
    def _signal_process_group(pid: int, signal_number: int) -> None:
        try:
            os.killpg(pid, signal_number)
        except ProcessLookupError:
            pass

    @staticmethod
    def _close_process_pipes(process: subprocess.Popen[bytes]) -> None:
        for source in (process.stdout, process.stderr):
            if source is None:
                continue
            try:
                os.close(source.fileno())
            except (OSError, ValueError):
                pass

    @staticmethod
    def _terminate(process: subprocess.Popen[bytes]) -> None:
        terminate_process_group(process)


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
