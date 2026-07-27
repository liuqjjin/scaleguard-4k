"""One-shot and persistent adapters for the audited Chain-of-Zoom checkout."""

from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
import uuid
from pathlib import Path
from types import TracebackType
from typing import Any, TextIO

from scaleguard.config import CoZConfig, RuntimeConfig
from scaleguard.contracts import ProcessEvidence, WorkerResult
from scaleguard.errors import ArtifactError, WorkerError, WorkerTimeoutError
from scaleguard.images import file_sha256, inspect_image, normalize_to_png
from scaleguard.runtime.process import (
    ProcessRunner,
    minimal_subprocess_environment,
    project_executable,
    terminate_process_group,
)
from scaleguard.strict_json import StrictJSONError, loads

_MAX_PROTOCOL_RESPONSE_BYTES = 1024 * 1024
_PROTOCOL_READER_SHUTDOWN_SECONDS = 1.0


def _project_path(project_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _model_location(project_root: Path, value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    if value.startswith(("./", "../", "weights/", "third_party/")):
        return str((project_root / path).resolve())
    return value


class CoZBackend:
    name = "chain_of_zoom"
    mock = False

    def __init__(
        self,
        config: CoZConfig,
        runtime: RuntimeConfig,
        *,
        project_root: Path,
    ) -> None:
        if config.checkout is None:
            raise ValueError("CoZ checkout is required")
        self.config = config
        self.runtime = runtime
        self.project_root = project_root.resolve()
        self.checkout = _project_path(self.project_root, config.checkout)
        self.worker = (
            project_root / "third_party" / "overlays" / "chain-of-zoom" / "coz_session_worker.py"
        ).resolve()

    def session(self, run_dir: Path) -> OneShotCoZSession | PersistentCoZSession:
        if self.config.mode == "persistent":
            return PersistentCoZSession(
                self.config,
                self.runtime,
                self.checkout,
                self.worker,
                run_dir,
                self.project_root,
            )
        return OneShotCoZSession(
            self.config,
            self.runtime,
            self.checkout,
            self.worker,
            run_dir,
            self.project_root,
        )


class _Arguments:
    def __init__(
        self,
        config: CoZConfig,
        checkout: Path,
        worker: Path,
        session_dir: Path,
        project_root: Path,
    ) -> None:
        if config.sr_lora_path is None or config.vae_path is None:
            raise ValueError("CoZ SR LoRA and VAE paths are required")
        self.config = config
        self.argv = [
            project_executable(project_root, config.python_executable),
            str(worker),
            "--checkout",
            str(checkout),
            "--model-path",
            _model_location(project_root, config.model_path),
            "--qwen-path",
            _model_location(project_root, config.qwen_model_path),
            "--sr-lora",
            str(_project_path(project_root, config.sr_lora_path)),
            "--vae",
            str(_project_path(project_root, config.vae_path)),
            "--prompt-type",
            config.prompt_type,
            "--mixed-precision",
            config.mixed_precision,
            "--vae-encoder-tile",
            str(config.tile_size),
            "--vae-decoder-tile",
            "128",
            "--latent-tile",
            str(max(8, config.tile_size // 8)),
            "--latent-overlap",
            str(max(1, config.tile_overlap // 8)),
            "--session-dir",
            str(session_dir),
            "--strict-json-helper",
            str((project_root / "src" / "scaleguard" / "strict_json.py").resolve()),
        ]
        if config.vlm_lora_path is not None:
            self.argv.extend(["--vlm-lora", str(_project_path(project_root, config.vlm_lora_path))])


class OneShotCoZSession:
    name = "chain_of_zoom_subprocess"
    mock = False

    def __init__(
        self,
        config: CoZConfig,
        runtime: RuntimeConfig,
        checkout: Path,
        worker: Path,
        run_dir: Path,
        project_root: Path,
    ) -> None:
        self.config = config
        self.checkout = checkout
        self.worker = worker
        self.run_dir = run_dir
        self.project_root = project_root
        self.runner = ProcessRunner(
            timeout_seconds=runtime.process_timeout_seconds,
            gpu_poll_interval_seconds=runtime.gpu_poll_interval_seconds,
        )

    def __enter__(self) -> OneShotCoZSession:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback

    def upscale_once(
        self,
        source: Path,
        destination: Path,
        *,
        step_index: int,
        seed: int,
    ) -> WorkerResult:
        step_dir = self.run_dir / "workers" / "coz" / f"step_{step_index:02d}"
        private_input = step_dir / "input.png"
        metadata_path = step_dir / "metadata.json"
        normalize_to_png(source, private_input)
        arguments = _Arguments(
            self.config,
            self.checkout,
            self.worker,
            self.run_dir / "workers" / "coz" / "subprocess-session",
            self.project_root,
        )
        argv = [
            *arguments.argv,
            "--one-shot-input",
            str(private_input),
            "--one-shot-output",
            str(destination.resolve()),
            "--one-shot-metadata",
            str(metadata_path.resolve()),
            "--one-shot-step-index",
            str(step_index),
            "--seed",
            str(seed),
        ]
        evidence = self.runner.run(
            argv,
            cwd=self.checkout,
            log_dir=step_dir / "logs",
            env={
                "CUDA_VISIBLE_DEVICES": self.config.visible_devices,
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "HF_HUB_OFFLINE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "TORCH_FORCE_WEIGHTS_ONLY_LOAD": "1",
                "TRANSFORMERS_OFFLINE": "1",
            },
            label=f"coz-step-{step_index:02d}",
        )
        try:
            metadata = loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, StrictJSONError) as error:
            raise ArtifactError(
                f"CoZ metadata is missing or invalid at {metadata_path}: {error}"
            ) from error
        metadata = _validate_step_metadata(
            metadata,
            source=private_input,
            destination=destination,
            seed=seed,
            step_index=step_index,
            requested_precision=self.config.mixed_precision,
        )
        return WorkerResult(
            image=inspect_image(destination, mock=False, stage=f"coz_scale_{step_index}"),
            metadata={**metadata, "backend": self.name, "persistent": False},
            process=evidence,
        )

    def accept(self, candidate: WorkerResult, *, step_index: int) -> None:
        del candidate, step_index

    def rollback(self, *, step_index: int) -> None:
        del step_index


class PersistentCoZSession:
    name = "chain_of_zoom_persistent"
    mock = False

    def __init__(
        self,
        config: CoZConfig,
        runtime: RuntimeConfig,
        checkout: Path,
        worker: Path,
        run_dir: Path,
        project_root: Path,
    ) -> None:
        self.config = config
        self.runtime = runtime
        self.checkout = checkout
        self.worker = worker
        self.run_dir = run_dir
        self.project_root = project_root
        self.process: subprocess.Popen[str] | None = None
        self.protocol_log: TextIO | None = None
        self.stderr_stream: TextIO | None = None
        self.started = 0.0
        self.stopped = 0.0
        self.argv: tuple[str, ...] = ()
        self.worker_dir: Path | None = None
        self.peak_vram_mib: dict[str, int] = {}

    def __enter__(self) -> PersistentCoZSession:
        worker_dir = self.run_dir / "workers" / "coz" / "persistent"
        worker_dir.mkdir(parents=True, exist_ok=True)
        arguments = _Arguments(
            self.config,
            self.checkout,
            self.worker,
            worker_dir / "session",
            self.project_root,
        )
        self.argv = tuple(arguments.argv)
        self.worker_dir = worker_dir
        env = minimal_subprocess_environment(
            {
                "CUDA_VISIBLE_DEVICES": self.config.visible_devices,
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "HF_HUB_OFFLINE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "TORCH_FORCE_WEIGHTS_ONLY_LOAD": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
        self.protocol_log = (worker_dir / "protocol.jsonl").open("w", encoding="utf-8")
        self.stderr_stream = (worker_dir / "worker.stderr.log").open("w", encoding="utf-8")
        self.started = time.monotonic()
        try:
            self.process = subprocess.Popen(
                arguments.argv,
                cwd=self.checkout,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.stderr_stream,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            ready = self._read_response(self.runtime.process_timeout_seconds)
            if ready.get("status") != "ready":
                raise WorkerError(f"CoZ persistent worker did not become ready: {ready}")
            health = self._request("health")
            if health.get("status") != "ok":
                raise WorkerError(f"CoZ persistent worker failed health check: {health}")
        except BaseException:
            self._terminate()
            self._close_streams()
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del traceback
        try:
            if self.process is not None and self.process.poll() is None:
                response = self._request("close")
                if response.get("status") != "ok" and exc_type is None:
                    raise WorkerError(f"CoZ persistent worker rejected close: {response}")
                returncode = self.process.wait(timeout=10)
                if returncode != 0 and exc_type is None:
                    raise WorkerError(
                        f"CoZ persistent worker exited with code {returncode} after close"
                    )
        except (WorkerError, WorkerTimeoutError, subprocess.TimeoutExpired):
            if exc_type is None:
                raise
        finally:
            self._terminate()
            self.stopped = time.monotonic()
            self._close_streams()

    def upscale_once(
        self,
        source: Path,
        destination: Path,
        *,
        step_index: int,
        seed: int,
    ) -> WorkerResult:
        response = self._request(
            "upscale",
            input=str(source.resolve()),
            output=str(destination.resolve()),
            seed=seed,
            step_index=step_index,
        )
        if response.get("status") != "ok":
            raise WorkerError(
                f"CoZ persistent step {step_index} failed: "
                f"{response.get('error_type', 'WorkerError')}: {response.get('message', response)}"
            )
        output_path = Path(str(response.get("output", "")))
        if output_path != destination.resolve():
            expected = destination.resolve()
            raise ArtifactError(
                f"CoZ worker returned unexpected output path {output_path}; expected {expected}"
            )
        metadata = response.get("metadata")
        metadata = _validate_step_metadata(
            metadata,
            source=source,
            destination=destination,
            seed=seed,
            step_index=step_index,
            requested_precision=self.config.mixed_precision,
        )
        peaks = metadata.get("peak_torch_allocated_mib")
        if isinstance(peaks, dict):
            for device, value in peaks.items():
                if isinstance(device, str) and type(value) is int and value >= 0:
                    self.peak_vram_mib[device] = max(
                        value,
                        self.peak_vram_mib.get(device, 0),
                    )
        return WorkerResult(
            image=inspect_image(destination, mock=False, stage=f"coz_scale_{step_index}"),
            metadata={**metadata, "backend": self.name, "persistent": True},
        )

    def evidence(self) -> ProcessEvidence:
        if self.process is None or self.worker_dir is None or not self.argv:
            raise WorkerError("CoZ persistent worker has no process evidence")
        returncode = self.process.poll()
        if returncode is None or self.stopped <= 0:
            raise WorkerError("CoZ persistent process evidence requested before close")
        return ProcessEvidence(
            argv=self.argv,
            cwd=str(self.checkout.resolve()),
            returncode=returncode,
            duration_seconds=self.stopped - self.started,
            stdout_path=str((self.worker_dir / "protocol.jsonl").resolve()),
            stderr_path=str((self.worker_dir / "worker.stderr.log").resolve()),
            peak_vram_mib=dict(sorted(self.peak_vram_mib.items())),
        )

    def accept(self, candidate: WorkerResult, *, step_index: int) -> None:
        response = self._request(
            "accept",
            step_index=step_index,
            candidate=str(candidate.image.path),
            candidate_sha256=candidate.image.sha256,
        )
        if response.get("status") != "ok":
            raise WorkerError(f"CoZ persistent worker rejected candidate: {response}")

    def rollback(self, *, step_index: int) -> None:
        response = self._request("rollback", step_index=step_index)
        if response.get("status") != "ok":
            raise WorkerError(f"CoZ persistent worker rejected rollback: {response}")

    def _request(self, operation: str, **payload: Any) -> dict[str, Any]:
        if self.process is None or self.process.stdin is None:
            raise WorkerError("CoZ persistent worker is not running")
        request_id = uuid.uuid4().hex
        request = {"request_id": request_id, "op": operation, **payload}
        try:
            self.process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise WorkerError(
                f"cannot send {operation} to CoZ persistent worker: {error}"
            ) from error
        response = self._read_response(self.runtime.process_timeout_seconds)
        if response.get("request_id") != request_id:
            raise WorkerError(
                f"CoZ protocol request mismatch: sent {request_id}, "
                f"received {response.get('request_id')}"
            )
        return response

    def _read_response(self, timeout: float) -> dict[str, Any]:
        if self.process is None or self.process.stdout is None:
            raise WorkerError("CoZ persistent worker has no protocol stream")
        process = self.process
        stream = process.stdout
        assert stream is not None
        deadline = time.monotonic() + timeout
        result: queue.Queue[tuple[str | None, BaseException | None]] = queue.Queue(maxsize=1)

        def read_line() -> None:
            try:
                result.put((stream.readline(_MAX_PROTOCOL_RESPONSE_BYTES + 1), None))
            except BaseException as error:
                result.put((None, error))

        reader = threading.Thread(
            target=read_line,
            name="coz-protocol-reader",
            daemon=True,
        )
        reader.start()
        try:
            line, read_error = result.get(timeout=max(0.0, deadline - time.monotonic()))
        except queue.Empty as error:
            self._terminate()
            reader.join(timeout=_PROTOCOL_READER_SHUTDOWN_SECONDS)
            raise WorkerTimeoutError(
                f"CoZ persistent worker response timed out after {timeout:.1f}s"
            ) from error
        if read_error is not None:
            self._terminate()
            raise WorkerError(
                f"cannot read CoZ persistent worker response: "
                f"{type(read_error).__name__}: {read_error}"
            ) from read_error
        assert line is not None
        if not line:
            self._terminate()
            returncode = process.poll()
            raise WorkerError(
                f"CoZ persistent worker closed its protocol stream (returncode={returncode})"
            )
        encoded = line.encode("utf-8")
        if len(encoded) > _MAX_PROTOCOL_RESPONSE_BYTES:
            self._terminate()
            raise WorkerError(
                "CoZ persistent worker protocol response exceeded "
                f"{_MAX_PROTOCOL_RESPONSE_BYTES} bytes"
            )
        if not line.endswith("\n"):
            self._terminate()
            raise WorkerError("CoZ persistent worker returned an incomplete protocol response")
        if self.protocol_log is not None:
            self.protocol_log.write(line)
            self.protocol_log.flush()
        try:
            response = loads(line)
        except StrictJSONError as error:
            raise WorkerError(f"invalid JSON from CoZ persistent worker: {line[:500]!r}") from error
        if not isinstance(response, dict):
            raise WorkerError(f"invalid CoZ protocol response: {response!r}")
        return response

    def _terminate(self) -> None:
        if self.process is None:
            return
        terminate_process_group(self.process)

    def _close_streams(self) -> None:
        if self.process is not None:
            for stream in (self.process.stdin, self.process.stdout):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
        if self.protocol_log is not None:
            self.protocol_log.close()
            self.protocol_log = None
        if self.stderr_stream is not None:
            self.stderr_stream.close()
            self.stderr_stream = None


def _validate_step_metadata(
    raw: object,
    *,
    source: Path,
    destination: Path,
    seed: int,
    step_index: int,
    requested_precision: str,
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ArtifactError(f"CoZ worker returned invalid metadata: {raw!r}")
    expected: dict[str, object] = {
        "step_index": step_index,
        "seed": seed,
        "input_sha256": file_sha256(source),
        "candidate_sha256": file_sha256(destination),
        "requested_precision": requested_precision,
        "mock": False,
    }
    mismatches = [
        f"{field}: expected {expected_value!r}, observed {raw.get(field)!r}"
        for field, expected_value in expected.items()
        if raw.get(field) != expected_value
    ]
    if mismatches:
        raise ArtifactError("CoZ worker metadata mismatch: " + "; ".join(mismatches))
    return raw
