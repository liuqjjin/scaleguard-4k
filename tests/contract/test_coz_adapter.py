from __future__ import annotations

import importlib.util
import io
import json
import os
import signal
import subprocess
import sys
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from scaleguard.backends.coz import (
    _MAX_PROTOCOL_RESPONSE_BYTES,
    CoZBackend,
    OneShotCoZSession,
    PersistentCoZSession,
)
from scaleguard.config import CoZConfig, RuntimeConfig
from scaleguard.contracts import ProcessEvidence, WorkerResult
from scaleguard.errors import ArtifactError, WorkerError, WorkerTimeoutError
from scaleguard.images import file_sha256, inspect_image
from scaleguard.runtime.process import ProcessRunner


def coz_config(tmp_path: Path, *, mode: str = "upstream") -> CoZConfig:
    return CoZConfig(
        mode=mode,
        checkout=tmp_path / "Chain-of-Zoom",
        sr_lora_path=tmp_path / "weights" / "sr.pkl",
        vae_path=tmp_path / "weights" / "vae.pt",
        vlm_lora_path=tmp_path / "weights" / "vlm",
        seed=41,
    )


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
        duration_seconds=0.1,
        stdout_path=str(stdout),
        stderr_path=str(stderr),
    )


def worker_metadata(
    source: Path,
    output: Path,
    *,
    seed: int,
    step_index: int,
    root_sha256: str | None = None,
) -> dict[str, Any]:
    with Image.open(source) as source_image, Image.open(output) as output_image:
        source_size = list(source_image.size)
        output_size = list(output_image.size)
    placements = {
        name: {"device": "cuda:0", "dtype": "torch.float32"}
        for name in (
            "text_encoder_1",
            "text_encoder_2",
            "text_encoder_3",
            "transformer",
            "vae",
            "vlm_first_parameter",
        )
    }
    return {
        "source_size": source_size,
        "output_size": output_size,
        "seed": seed,
        "step_index": step_index,
        "root_sha256": root_sha256 or file_sha256(source),
        "input_sha256": file_sha256(source),
        "candidate_sha256": file_sha256(output),
        "prompts": ["fixture prompt"],
        "duration_seconds": 0.1,
        "peak_torch_allocated_mib": {"0": 1024, "1": 2048},
        "requested_precision": "fp32",
        "actual_precision": {"transformer": "torch.float32", "vae": "torch.float32"},
        "component_placement": placements,
        "semantic_anchor": str(source.parent / "semantic_anchor.png"),
        "gpu_inventory": [
            {
                "logical_index": str(index),
                "uuid": f"GPU-{index}",
                "name": "fixture-gpu",
                "memory_total_mib": "24564",
            }
            for index in range(2)
        ],
        "mock": False,
    }


def test_coz_backend_requires_a_checkout_and_selects_the_configured_session(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="CoZ checkout is required"):
        CoZBackend(CoZConfig(), RuntimeConfig(), project_root=tmp_path)

    one_shot = CoZBackend(
        coz_config(tmp_path),
        RuntimeConfig(),
        project_root=tmp_path,
    ).session(tmp_path / "run")
    persistent = CoZBackend(
        coz_config(tmp_path, mode="persistent"),
        RuntimeConfig(),
        project_root=tmp_path,
    ).session(tmp_path / "run")

    assert isinstance(one_shot, OneShotCoZSession)
    assert isinstance(persistent, PersistentCoZSession)


def test_one_shot_coz_contract_builds_a_single_scale_request(
    tmp_path: Path,
    make_image: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = make_image(tmp_path / "source.jpg", size=(5, 3), image_format="JPEG")
    destination = tmp_path / "candidate.png"
    interpreter_target = tmp_path / "shared-python"
    interpreter_target.write_text("#!/bin/sh\n", encoding="utf-8")
    interpreter_entrypoint = tmp_path / ".runtime/envs/coz/bin/python"
    interpreter_entrypoint.parent.mkdir(parents=True)
    interpreter_entrypoint.symlink_to(interpreter_target)
    config = replace(
        coz_config(tmp_path),
        python_executable=".runtime/envs/coz/bin/python",
    )
    backend = CoZBackend(config, RuntimeConfig(), project_root=tmp_path)

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
        assert argv[0] == str(interpreter_entrypoint)
        assert Path(argv[0]).is_symlink()
        assert argv[0] != str(interpreter_target.resolve())
        assert env == {
            "CUDA_VISIBLE_DEVICES": "0,1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TORCH_FORCE_WEIGHTS_ONLY_LOAD": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
        assert argv[argv.index("--seed") + 1] == "42"
        assert argv[argv.index("--one-shot-step-index") + 1] == "2"
        assert argv[argv.index("--mixed-precision") + 1] == "fp32"
        assert argv[argv.index("--qwen-path") + 1] == config.qwen_model_path
        private_input = Path(argv[argv.index("--one-shot-input") + 1])
        output = Path(argv[argv.index("--one-shot-output") + 1])
        metadata = Path(argv[argv.index("--one-shot-metadata") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(private_input) as image:
            image.resize((image.width * 4, image.height * 4)).save(output, "PNG")
        metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text(
            json.dumps(worker_metadata(private_input, output, seed=42, step_index=2)),
            encoding="utf-8",
        )
        return evidence(argv, cwd, log_dir, label)

    monkeypatch.setattr(ProcessRunner, "run", fake_run)

    with backend.session(tmp_path / "run") as session:
        result = session.upscale_once(source, destination, step_index=2, seed=42)
        session.accept(result, step_index=2)
        session.rollback(step_index=2)

    assert (result.image.width, result.image.height) == (20, 12)
    assert result.image.mock is False
    assert result.metadata == {
        **worker_metadata(
            tmp_path / "run/workers/coz/step_02/input.png",
            destination,
            seed=42,
            step_index=2,
        ),
        "backend": "chain_of_zoom_subprocess",
        "persistent": False,
    }
    assert result.process is not None


@pytest.mark.parametrize(
    "metadata_payload",
    [
        None,
        "{invalid",
        '{"candidate_sha256":"trusted","candidate_sha256":"forged"}',
        '{"seed":NaN}',
    ],
)
def test_one_shot_coz_rejects_missing_or_invalid_metadata(
    tmp_path: Path,
    make_image: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
    metadata_payload: str | None,
) -> None:
    source = make_image(tmp_path / "source.png", size=(3, 2))
    config = coz_config(tmp_path)
    session = CoZBackend(config, RuntimeConfig(), project_root=tmp_path).session(tmp_path / "run")
    assert isinstance(session, OneShotCoZSession)

    def fake_run(
        _runner: ProcessRunner,
        argv: Sequence[str],
        *,
        cwd: Path,
        log_dir: Path,
        label: str,
        **_kwargs: Any,
    ) -> ProcessEvidence:
        output = Path(argv[argv.index("--one-shot-output") + 1])
        metadata = Path(argv[argv.index("--one-shot-metadata") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        make_image(output, size=(12, 8))
        if metadata_payload is not None:
            metadata.write_text(metadata_payload, encoding="utf-8")
        return evidence(argv, cwd, log_dir, label)

    monkeypatch.setattr(ProcessRunner, "run", fake_run)

    with pytest.raises(ArtifactError, match="metadata is missing or invalid"):
        session.upscale_once(source, tmp_path / "candidate.png", step_index=1, seed=41)


def test_one_shot_coz_requires_the_sr_and_vae_weight_paths(
    tmp_path: Path,
    make_image: Callable[..., Path],
) -> None:
    source = make_image(tmp_path / "source.png")
    config = CoZConfig(mode="upstream", checkout=tmp_path / "Chain-of-Zoom")
    session = CoZBackend(config, RuntimeConfig(), project_root=tmp_path).session(tmp_path / "run")

    with pytest.raises(ValueError, match="SR LoRA and VAE paths are required"):
        session.upscale_once(source, tmp_path / "candidate.png", step_index=1, seed=0)


class MemoryStdout:
    def __init__(self, lines: list[str] | None = None) -> None:
        self.lines = deque(lines or [])
        self.closed = False

    def readline(self, size: int = -1) -> str:
        if not self.lines:
            return ""
        line = self.lines.popleft()
        if size >= 0 and len(line) > size:
            self.lines.appendleft(line[size:])
            return line[:size]
        return line

    def close(self) -> None:
        self.closed = True


class MemoryProcess:
    def __init__(self, *, ready: bool = False, wait_returncode: int = 0) -> None:
        initial = (
            [
                json.dumps(
                    {
                        "status": "ready",
                        "initialization_duration_seconds": 0.25,
                    }
                )
                + "\n"
            ]
            if ready
            else []
        )
        self.stdout = MemoryStdout(initial)
        self.stdin = MemoryStdin(self)
        self.requests: list[dict[str, Any]] = []
        self.returncode: int | None = None
        self.wait_returncode = wait_returncode
        self.pid = 999_999
        self.root_sha256: str | None = None

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self.returncode = self.wait_returncode
        return self.wait_returncode


class MemoryStdin:
    def __init__(self, process: MemoryProcess) -> None:
        self.process = process
        self.fail = False
        self.closed = False

    def write(self, value: str) -> int:
        if self.fail:
            raise BrokenPipeError("closed fake pipe")
        request = json.loads(value)
        self.process.requests.append(request)
        response: dict[str, Any] = {
            "request_id": request["request_id"],
            "status": "ok",
        }
        if request["op"] == "upscale":
            source = Path(request["input"])
            output = Path(request["output"])
            output.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(source) as image:
                image.resize((image.width * 4, image.height * 4)).save(output, "PNG")
            if self.process.root_sha256 is None:
                self.process.root_sha256 = file_sha256(source)
            response.update(
                {
                    "output": str(output.resolve()),
                    "metadata": worker_metadata(
                        source,
                        output,
                        seed=request["seed"],
                        step_index=request["step_index"],
                        root_sha256=self.process.root_sha256,
                    ),
                }
            )
        self.process.stdout.lines.append(json.dumps(response) + "\n")
        return len(value)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


def test_persistent_coz_full_protocol_lifecycle_uses_an_in_memory_worker(
    tmp_path: Path,
    make_image: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = MemoryProcess(ready=True)
    popen_calls: list[tuple[Sequence[str], dict[str, Any]]] = []

    def fake_popen(argv: Sequence[str], **kwargs: Any) -> MemoryProcess:
        popen_calls.append((argv, kwargs))
        return process

    monkeypatch.setattr("scaleguard.backends.coz.subprocess.Popen", fake_popen)
    source = make_image(tmp_path / "source.png", size=(4, 3))
    destination = tmp_path / "candidate.png"
    backend = CoZBackend(
        coz_config(tmp_path, mode="persistent"),
        RuntimeConfig(process_timeout_seconds=1.0),
        project_root=tmp_path,
    )
    session = backend.session(tmp_path / "run")
    assert isinstance(session, PersistentCoZSession)

    with session:
        result = session.upscale_once(source, destination, step_index=1, seed=41)
        session.accept(result, step_index=1)

    assert (result.image.width, result.image.height) == (16, 12)
    assert result.metadata["persistent"] is True
    assert result.metadata["initialization_duration_seconds"] == 0.25
    assert [request["op"] for request in process.requests] == [
        "health",
        "upscale",
        "accept",
        "close",
    ]
    assert popen_calls[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert popen_calls[0][1]["env"]["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert popen_calls[0][1]["env"]["HF_HUB_OFFLINE"] == "1"
    assert popen_calls[0][1]["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert popen_calls[0][1]["env"]["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] == "1"
    assert popen_calls[0][1]["env"]["TRANSFORMERS_OFFLINE"] == "1"
    session_evidence = session.evidence()
    assert session_evidence.returncode == 0
    assert session_evidence.argv
    assert Path(session_evidence.stdout_path).name == "protocol.jsonl"
    assert Path(session_evidence.stderr_path).name == "worker.stderr.log"
    assert session.protocol_log is None
    assert session.stderr_stream is None
    assert process.stdin.closed is True
    assert process.stdout.closed is True


def test_jsonl_worker_ready_response_records_initialization_duration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_path = (
        Path(__file__).parents[2]
        / "third_party"
        / "overlays"
        / "chain-of-zoom"
        / "coz_session_worker.py"
    )
    spec = importlib.util.spec_from_file_location("scaleguard_test_coz_worker", worker_path)
    assert spec is not None
    assert spec.loader is not None
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)

    class Session:
        def __init__(self, _args: object) -> None:
            pass

        def close(self) -> None:
            pass

    monkeypatch.setattr(worker, "CoZSession", Session)
    monkeypatch.setattr(
        worker.sys,
        "stdin",
        io.StringIO('{"request_id":"close-1","op":"close"}\n'),
    )
    protocol = io.StringIO()
    arguments = SimpleNamespace(
        strict_json_helper=(Path(__file__).parents[2] / "src" / "scaleguard" / "strict_json.py")
    )

    assert worker.run_jsonl(arguments, protocol) == 0

    ready, closed = [json.loads(line) for line in protocol.getvalue().splitlines()]
    duration = ready["initialization_duration_seconds"]
    assert ready["status"] == "ready"
    assert isinstance(duration, float)
    assert duration >= 0.0
    assert closed == {"request_id": "close-1", "status": "ok", "op": "close"}


def test_persistent_initialization_duration_is_bound_to_the_first_step_only(
    tmp_path: Path,
    make_image: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = MemoryProcess(ready=True)
    monkeypatch.setattr(
        "scaleguard.backends.coz.subprocess.Popen",
        lambda *_args, **_kwargs: process,
    )
    source = make_image(tmp_path / "source.png", size=(3, 2))
    first_destination = tmp_path / "candidate-1.png"
    second_destination = tmp_path / "candidate-2.png"
    session = bare_persistent_session(tmp_path)

    with session:
        first = session.upscale_once(source, first_destination, step_index=1, seed=41)
        session.accept(first, step_index=1)
        second = session.upscale_once(
            first.image.path,
            second_destination,
            step_index=2,
            seed=42,
        )
        session.accept(second, step_index=2)

    assert first.metadata["initialization_duration_seconds"] == 0.25
    assert "initialization_duration_seconds" not in second.metadata


@pytest.mark.parametrize(
    ("ready_payload", "health_outcome", "message"),
    [
        ({"status": "not-ready"}, {"status": "ok"}, "did not become ready"),
        (
            {"status": "ready", "initialization_duration_seconds": 0.25},
            {"status": "error"},
            "failed health check",
        ),
    ],
)
def test_persistent_coz_startup_failure_terminates_and_closes_every_stream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ready_payload: dict[str, Any],
    health_outcome: dict[str, str],
    message: str,
) -> None:
    process = MemoryProcess()
    process.stdout.lines.append(json.dumps(ready_payload) + "\n")
    session = bare_persistent_session(tmp_path)
    terminated: list[bool] = []

    monkeypatch.setattr(
        "scaleguard.backends.coz.subprocess.Popen",
        lambda *_args, **_kwargs: process,
    )
    monkeypatch.setattr(session, "_terminate", lambda: terminated.append(True))
    monkeypatch.setattr(session, "_request", lambda _operation: health_outcome)

    with pytest.raises(WorkerError, match=message):
        session.__enter__()

    assert terminated == [True]
    assert process.stdin.closed is True
    assert process.stdout.closed is True
    assert session.protocol_log is None
    assert session.stderr_stream is None


@pytest.mark.parametrize(
    "invalid_duration",
    [None, True, -0.01, "0.25", 10**400],
)
def test_persistent_coz_rejects_invalid_initialization_duration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_duration: object,
) -> None:
    process = MemoryProcess()
    process.stdout.lines.append(
        json.dumps(
            {
                "status": "ready",
                "initialization_duration_seconds": invalid_duration,
            }
        )
        + "\n"
    )
    session = bare_persistent_session(tmp_path)
    terminated: list[bool] = []
    monkeypatch.setattr(
        "scaleguard.backends.coz.subprocess.Popen",
        lambda *_args, **_kwargs: process,
    )
    monkeypatch.setattr(session, "_terminate", lambda: terminated.append(True))

    with pytest.raises(WorkerError, match="invalid initialization_duration_seconds"):
        session.__enter__()

    assert terminated == [True]
    assert process.stdin.closed is True
    assert process.stdout.closed is True


def test_persistent_coz_nonzero_exit_after_close_invalidates_the_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = MemoryProcess(ready=True, wait_returncode=17)
    session = bare_persistent_session(tmp_path)

    monkeypatch.setattr(
        "scaleguard.backends.coz.subprocess.Popen",
        lambda *_args, **_kwargs: process,
    )

    with pytest.raises(WorkerError, match="exited with code 17 after close"):
        with session:
            pass

    assert process.stdin.closed is True
    assert process.stdout.closed is True
    assert session.protocol_log is None
    assert session.stderr_stream is None


def bare_persistent_session(tmp_path: Path) -> PersistentCoZSession:
    return PersistentCoZSession(
        coz_config(tmp_path, mode="persistent"),
        RuntimeConfig(process_timeout_seconds=0.1),
        tmp_path / "Chain-of-Zoom",
        tmp_path / "worker.py",
        tmp_path / "run",
        tmp_path,
    )


def test_persistent_request_rejects_missing_process_broken_pipe_and_id_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = bare_persistent_session(tmp_path)
    with pytest.raises(WorkerError, match="is not running"):
        session._request("health")

    process = MemoryProcess()
    session.process = process  # type: ignore[assignment]
    process.stdin.fail = True
    with pytest.raises(WorkerError, match="cannot send health"):
        session._request("health")

    process.stdin.fail = False
    original_write = process.stdin.write

    def mismatched_write(value: str) -> int:
        written = original_write(value)
        response = json.loads(process.stdout.lines.pop())
        response["request_id"] = "wrong-id"
        process.stdout.lines.append(json.dumps(response) + "\n")
        return written

    process.stdin.write = mismatched_write  # type: ignore[method-assign]
    with pytest.raises(WorkerError, match="protocol request mismatch"):
        session._request("health")


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("", "closed its protocol stream"),
        ("not-json\n", "invalid JSON"),
        ("[1, 2]\n", "invalid CoZ protocol response"),
        ('{"ok":true,"ok":false}\n', "invalid JSON"),
        ('{"ok":NaN}\n', "invalid JSON"),
    ],
)
def test_persistent_response_rejects_closed_or_malformed_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    line: str,
    message: str,
) -> None:
    session = bare_persistent_session(tmp_path)
    process = MemoryProcess()
    process.stdout.lines.append(line)
    session.process = process  # type: ignore[assignment]
    session.protocol_log = io.StringIO()
    terminated: list[bool] = []
    monkeypatch.setattr(session, "_terminate", lambda: terminated.append(True))

    with pytest.raises(WorkerError, match=message):
        session._read_response(0.1)

    assert terminated == [True]


def test_persistent_response_timeout_raises_without_touching_a_gpu_process(
    tmp_path: Path,
) -> None:
    session = bare_persistent_session(tmp_path)
    process = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys,time; sys.stdout.write('{'); sys.stdout.flush(); time.sleep(30)",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )
    session.process = process
    started = time.monotonic()

    with pytest.raises(WorkerTimeoutError, match="response timed out"):
        session._read_response(0.1)

    assert time.monotonic() - started < 3.0
    assert process.poll() is not None


def test_persistent_response_rejects_incomplete_and_oversized_lines(
    tmp_path: Path,
) -> None:
    session = bare_persistent_session(tmp_path)
    process = MemoryProcess()
    session.process = process  # type: ignore[assignment]
    process.stdout.lines.append('{"status":"ok"}')

    with pytest.raises(WorkerError, match="incomplete protocol response"):
        session._read_response(0.1)

    process.stdout.lines.append("x" * (_MAX_PROTOCOL_RESPONSE_BYTES + 1))
    with pytest.raises(WorkerError, match="exceeded"):
        session._read_response(0.1)


def test_persistent_close_reaps_descendants_after_the_worker_leader_exits(
    tmp_path: Path,
) -> None:
    child_pid_path = tmp_path / "child.pid"
    code = (
        "import pathlib,subprocess,sys\n"
        "child = subprocess.Popen("
        "[sys.executable, '-I', '-c', 'import time; time.sleep(30)'], "
        "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid))\n"
    )
    process = subprocess.Popen(
        [sys.executable, "-I", "-c", code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )
    process.wait(timeout=2.0)
    session = bare_persistent_session(tmp_path)
    session.process = process

    session.__exit__(None, None, None)

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


def test_persistent_scale_and_state_operations_validate_worker_responses(
    tmp_path: Path,
    make_image: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = make_image(tmp_path / "source.png")
    candidate = make_image(tmp_path / "candidate.png", size=(36, 24))
    artifact = inspect_image(candidate, mock=False, stage="candidate")
    result = WorkerResult(image=artifact)
    session = bare_persistent_session(tmp_path)

    monkeypatch.setattr(
        session,
        "_request",
        lambda _op, **_payload: {"status": "error", "message": "rejected"},
    )
    with pytest.raises(WorkerError, match="persistent step 1 failed"):
        session.upscale_once(source, candidate, step_index=1, seed=0)
    session._pending_step = (1, result.image.sha256)
    with pytest.raises(WorkerError, match="rejected candidate"):
        session.accept(result, step_index=1)
    session._pending_step = (1, result.image.sha256)
    with pytest.raises(WorkerError, match="rejected rollback"):
        session.rollback(step_index=1)
    session._pending_step = None

    monkeypatch.setattr(
        session,
        "_request",
        lambda _op, **_payload: {
            "status": "ok",
            "output": str(tmp_path / "wrong.png"),
            "metadata": {},
        },
    )
    with pytest.raises(ArtifactError, match="unexpected output path"):
        session.upscale_once(source, candidate, step_index=1, seed=0)

    monkeypatch.setattr(
        session,
        "_request",
        lambda _op, **_payload: {
            "status": "ok",
            "output": str(candidate.resolve()),
            "metadata": [],
        },
    )
    with pytest.raises(ArtifactError, match="invalid metadata"):
        session.upscale_once(source, candidate, step_index=1, seed=0)


def test_persistent_decision_is_exactly_once_per_pending_candidate(
    tmp_path: Path,
    make_image: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = make_image(tmp_path / "candidate.png")
    result = WorkerResult(image=inspect_image(candidate, mock=False, stage="candidate"))
    session = bare_persistent_session(tmp_path)
    operations: list[str] = []

    def request(operation: str, **_payload: object) -> dict[str, str]:
        operations.append(operation)
        return {"status": "ok"}

    monkeypatch.setattr(session, "_request", request)
    session._pending_step = (1, result.image.sha256)
    session.accept(result, step_index=1)

    with pytest.raises(WorkerError, match="does not match the session's pending step"):
        session.rollback(step_index=1)

    assert operations == ["accept"]
