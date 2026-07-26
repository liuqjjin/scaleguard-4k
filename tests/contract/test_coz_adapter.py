from __future__ import annotations

import io
import json
from collections import deque
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from scaleguard.backends.coz import (
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
    config = coz_config(tmp_path)
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
            json.dumps(
                {
                    "prompt": "fake prompt",
                    "seed": 42,
                    "step_index": 2,
                    "input_sha256": file_sha256(private_input),
                    "candidate_sha256": file_sha256(output),
                    "requested_precision": "fp32",
                    "mock": False,
                }
            ),
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
        "backend": "chain_of_zoom_subprocess",
        "persistent": False,
        "prompt": "fake prompt",
        "seed": 42,
        "step_index": 2,
        "input_sha256": file_sha256(tmp_path / "run/workers/coz/step_02/input.png"),
        "candidate_sha256": file_sha256(destination),
        "requested_precision": "fp32",
        "mock": False,
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

    def readline(self) -> str:
        return self.lines.popleft() if self.lines else ""

    def close(self) -> None:
        self.closed = True


class MemoryProcess:
    def __init__(self, *, ready: bool = False, wait_returncode: int = 0) -> None:
        initial = [json.dumps({"status": "ready"}) + "\n"] if ready else []
        self.stdout = MemoryStdout(initial)
        self.stdin = MemoryStdin(self)
        self.requests: list[dict[str, Any]] = []
        self.returncode: int | None = None
        self.wait_returncode = wait_returncode
        self.pid = 999_999

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
            response.update(
                {
                    "output": str(output.resolve()),
                    "metadata": {
                        "prompt": "memory protocol",
                        "seed": request["seed"],
                        "step_index": request["step_index"],
                        "input_sha256": file_sha256(source),
                        "candidate_sha256": file_sha256(output),
                        "requested_precision": "fp32",
                        "mock": False,
                    },
                }
            )
        self.process.stdout.lines.append(json.dumps(response) + "\n")
        return len(value)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class ReadySelector:
    def __init__(self, *, force_timeout: bool = False) -> None:
        self.stream: MemoryStdout | None = None
        self.force_timeout = force_timeout

    def register(self, stream: MemoryStdout, _event: int) -> None:
        self.stream = stream

    def select(self, _timeout: float) -> list[tuple[object, int]]:
        if self.force_timeout or self.stream is None or not self.stream.lines:
            return []
        return [(object(), 1)]

    def close(self) -> None:
        return None


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
    monkeypatch.setattr(
        "scaleguard.backends.coz.selectors.DefaultSelector",
        ReadySelector,
    )
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
        session.rollback(step_index=1)

    assert (result.image.width, result.image.height) == (16, 12)
    assert result.metadata["persistent"] is True
    assert [request["op"] for request in process.requests] == [
        "health",
        "upscale",
        "accept",
        "rollback",
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


@pytest.mark.parametrize(
    ("ready_payload", "health_outcome", "message"),
    [
        ({"status": "not-ready"}, {"status": "ok"}, "did not become ready"),
        ({"status": "ready"}, {"status": "error"}, "failed health check"),
    ],
)
def test_persistent_coz_startup_failure_terminates_and_closes_every_stream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ready_payload: dict[str, str],
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
    monkeypatch.setattr(
        "scaleguard.backends.coz.selectors.DefaultSelector",
        ReadySelector,
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
    monkeypatch.setattr(
        "scaleguard.backends.coz.selectors.DefaultSelector",
        ReadySelector,
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
    monkeypatch.setattr(
        "scaleguard.backends.coz.selectors.DefaultSelector",
        ReadySelector,
    )
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
    monkeypatch.setattr(
        "scaleguard.backends.coz.selectors.DefaultSelector",
        ReadySelector,
    )

    with pytest.raises(WorkerError, match=message):
        session._read_response(0.1)


def test_persistent_response_timeout_raises_without_touching_a_gpu_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = bare_persistent_session(tmp_path)
    process = MemoryProcess()
    process.returncode = 0
    session.process = process  # type: ignore[assignment]
    monkeypatch.setattr(
        "scaleguard.backends.coz.selectors.DefaultSelector",
        lambda: ReadySelector(force_timeout=True),
    )

    with pytest.raises(WorkerTimeoutError, match="response timed out"):
        session._read_response(0.1)


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
    with pytest.raises(WorkerError, match="rejected candidate"):
        session.accept(result, step_index=1)
    with pytest.raises(WorkerError, match="rejected rollback"):
        session.rollback(step_index=1)

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
