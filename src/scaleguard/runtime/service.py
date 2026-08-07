"""Lifecycle ownership for local model services used by an upstream phase."""

from __future__ import annotations

import fcntl
import hashlib
import os
import socket
import stat
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, TextIO

from scaleguard.errors import WorkerError
from scaleguard.runtime.process import (
    minimal_subprocess_environment,
    redact_argv,
    terminate_process_group,
)


def tcp_ready(host: str, port: int, *, timeout_seconds: float = 0.25) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            return True
    except OSError:
        return False


def _listener_pids_from_lsof(host: str, port: int) -> set[int] | None:
    """Return listener PIDs when lsof is available, otherwise ``None``."""

    executable = "/usr/sbin/lsof"
    if not os.path.isfile(executable):
        executable = "/usr/bin/lsof"
    if not os.path.isfile(executable):
        return None
    try:
        completed = subprocess.run(
            [
                executable,
                "-nP",
                "-a",
                f"-iTCP:{port}",
                "-sTCP:LISTEN",
                "-Fpn",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
            env=minimal_subprocess_environment(),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode not in {0, 1}:
        return None
    candidates: set[int] = set()
    current_pid: int | None = None
    accepted_hosts = {host}
    try:
        for _family, _kind, _proto, _canonical, address in socket.getaddrinfo(
            host,
            port,
            type=socket.SOCK_STREAM,
        ):
            resolved_host = address[0]
            if isinstance(resolved_host, str):
                accepted_hosts.add(resolved_host)
    except OSError:
        pass
    accepted_names = {f"*:{port}"}
    for accepted_host in accepted_hosts:
        accepted_names.add(f"{accepted_host}:{port}")
        accepted_names.add(f"[{accepted_host}]:{port}")
    for line in completed.stdout.splitlines():
        if line.startswith("p") and line[1:].isdigit():
            current_pid = int(line[1:])
        elif line.startswith("n") and current_pid is not None:
            endpoint = line[1:].split("->", 1)[0]
            if endpoint in accepted_names:
                candidates.add(current_pid)
    return candidates


def _linux_listener_inodes(host: str, port: int) -> set[str] | None:
    proc = Path("/proc")
    if not proc.is_dir():
        return None
    allowed_addresses: set[str] = {"00000000", "00000000000000000000000000000000"}
    try:
        for family, _kind, _proto, _canonical, socket_address in socket.getaddrinfo(
            host,
            port,
            type=socket.SOCK_STREAM,
        ):
            host_address = socket_address[0]
            if not isinstance(host_address, str):
                continue
            packed = socket.inet_pton(family, host_address)
            if family == socket.AF_INET:
                allowed_addresses.add(packed[::-1].hex().upper())
            elif family == socket.AF_INET6:
                # /proc/net/tcp6 writes each 32-bit word in host byte order.
                allowed_addresses.add(
                    b"".join(packed[index : index + 4][::-1] for index in range(0, len(packed), 4))
                    .hex()
                    .upper()
                )
    except OSError:
        return set()

    inodes: set[str] = set()
    expected_port = f"{port:04X}"
    for table in (proc / "net" / "tcp", proc / "net" / "tcp6"):
        try:
            lines = table.read_text(encoding="ascii").splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) < 10 or fields[3] != "0A":
                continue
            local_address, separator, local_port = fields[1].rpartition(":")
            if separator and local_port == expected_port and local_address in allowed_addresses:
                inodes.add(fields[9])
    return inodes


def _listener_owned_by_process_group(host: str, port: int, process_group: int) -> bool:
    """Fail closed unless the listening socket belongs to the owned process group."""

    listener_pids = _listener_pids_from_lsof(host, port)
    if listener_pids is not None:
        for pid in listener_pids:
            try:
                if os.getpgid(pid) == process_group:
                    return True
            except (ProcessLookupError, PermissionError):
                continue
        return False

    inodes = _linux_listener_inodes(host, port)
    if inodes is None or not inodes:
        return False
    socket_targets = {f"socket:[{inode}]" for inode in inodes}
    for process_dir in Path("/proc").iterdir():
        if not process_dir.name.isdigit():
            continue
        pid = int(process_dir.name)
        try:
            if os.getpgid(pid) != process_group:
                continue
            for descriptor in (process_dir / "fd").iterdir():
                try:
                    if os.readlink(descriptor) in socket_targets:
                        return True
                except OSError:
                    continue
        except (OSError, ProcessLookupError, PermissionError):
            continue
    return False


class _PortLease:
    """Cooperative, process-scoped lease for one TCP endpoint."""

    def __init__(self, host: str, port: int) -> None:
        identity = hashlib.sha256(f"tcp\0{host}\0{port}".encode()).hexdigest()
        self.path = Path(tempfile.gettempdir()) / f"scaleguard-port-leases-{os.getuid()}" / identity
        self.descriptor: int | None = None

    def acquire(self) -> None:
        root = self.path.parent
        try:
            root.mkdir(mode=0o700)
        except FileExistsError:
            pass
        metadata = root.lstat()
        if (
            root.is_symlink()
            or not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.getuid()
        ):
            raise WorkerError(f"unsafe service lease directory: {root}")
        os.chmod(root, 0o700)
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self.path, flags, 0o600)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid():
                raise WorkerError(f"unsafe service lease file: {self.path}")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise WorkerError(
                    f"service endpoint {self.path.name} is leased by another ScaleGuard run"
                ) from error
        except BaseException:
            os.close(descriptor)
            raise
        self.descriptor = descriptor

    def release(self) -> None:
        descriptor = self.descriptor
        self.descriptor = None
        if descriptor is None:
            return
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


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
        self._lease = _PortLease(host, port)

    def __enter__(self) -> ManagedService:
        if not self.argv or not self.argv[0]:
            raise WorkerError(f"{self.label} service command is empty")
        try:
            self._lease.acquire()
            if tcp_ready(self.host, self.port):
                raise WorkerError(
                    f"{self.label} endpoint {self.host}:{self.port} is already in use; "
                    "refusing to take ownership of an unknown process"
                )
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.stdout = (self.log_dir / f"{self.label}.stdout.log").open("w", encoding="utf-8")
            self.stderr = (self.log_dir / f"{self.label}.stderr.log").open("w", encoding="utf-8")
            private_home = self.log_dir / "service-home"
            private_home.mkdir(mode=0o700, exist_ok=True)
            private_home.chmod(0o700)
            process_env = minimal_subprocess_environment(self.env)
            process_env.setdefault("HOME", str(private_home.resolve()))
            self.started_at = time.monotonic()
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
                if _listener_owned_by_process_group(self.host, self.port, self.process.pid):
                    return
            time.sleep(0.2)
        raise WorkerError(
            f"{self.label} service did not own a listener on {self.host}:{self.port} "
            f"within {self.startup_timeout_seconds:.1f}s"
        )

    def stop(self) -> None:
        process = self.process
        try:
            if process is not None:
                terminate_process_group(process)
            if process is not None:
                self.returncode = process.poll()
        finally:
            self.stopped_at = time.monotonic()
            try:
                for stream in (self.stdout, self.stderr):
                    if stream is not None and not stream.closed:
                        stream.close()
            finally:
                self._lease.release()

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
