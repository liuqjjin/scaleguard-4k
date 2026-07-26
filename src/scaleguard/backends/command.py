"""Shell-free command adapters for integration testing and custom launchers."""

from __future__ import annotations

from pathlib import Path
from types import TracebackType

from scaleguard.config import CoZConfig, FourKAgentConfig, RuntimeConfig
from scaleguard.contracts import WorkerResult
from scaleguard.images import discover_single_output, inspect_image, normalize_to_png
from scaleguard.runtime.process import ProcessRunner, format_command


class CommandRestorationBackend:
    name = "4kagent_command"
    mock = False

    def __init__(self, config: FourKAgentConfig, runtime: RuntimeConfig) -> None:
        self.config = config
        self.runner = ProcessRunner(
            runtime.process_timeout_seconds,
            runtime.gpu_poll_interval_seconds,
        )

    def restore(
        self,
        source: Path,
        destination: Path,
        *,
        bridge_factor: int,
        run_dir: Path,
    ) -> WorkerResult:
        private_dir = run_dir / "workers" / "4kagent-command"
        input_dir = private_dir / "input"
        output_dir = private_dir / "output"
        private_input = input_dir / "source.png"
        normalize_to_png(source, private_input)
        output_dir.mkdir(parents=True, exist_ok=True)
        argv = format_command(
            self.config.command,
            {
                "input": str(private_input),
                "input_dir": str(input_dir),
                "output": str(output_dir / "output.png"),
                "output_dir": str(output_dir),
                "bridge_factor": str(bridge_factor),
            },
        )
        evidence = self.runner.run(
            argv,
            cwd=private_dir,
            log_dir=private_dir / "logs",
            env={"CUDA_VISIBLE_DEVICES": self.config.tool_gpu},
            label="4kagent-command",
        )
        produced = discover_single_output(output_dir)
        normalize_to_png(produced, destination)
        return WorkerResult(
            image=inspect_image(destination, mock=False, stage="4kagent_restoration"),
            metadata={"backend": self.name, "bridge_factor": bridge_factor},
            process=evidence,
        )


class CommandScaleSession:
    name = "coz_command"
    mock = False

    def __init__(
        self,
        config: CoZConfig,
        runtime: RuntimeConfig,
        run_dir: Path,
    ) -> None:
        self.config = config
        self.run_dir = run_dir
        self.runner = ProcessRunner(
            runtime.process_timeout_seconds,
            runtime.gpu_poll_interval_seconds,
        )

    def __enter__(self) -> CommandScaleSession:
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
        private_dir = self.run_dir / "workers" / "coz-command" / f"step_{step_index:02d}"
        input_dir = private_dir / "input"
        output_dir = private_dir / "output"
        private_input = input_dir / "source.png"
        normalize_to_png(source, private_input)
        output_dir.mkdir(parents=True, exist_ok=True)
        argv = format_command(
            self.config.command,
            {
                "input": str(private_input),
                "input_dir": str(input_dir),
                "output": str(output_dir / "output.png"),
                "output_dir": str(output_dir),
                "seed": str(seed),
                "step_index": str(step_index),
            },
        )
        evidence = self.runner.run(
            argv,
            cwd=private_dir,
            log_dir=private_dir / "logs",
            env={"CUDA_VISIBLE_DEVICES": self.config.visible_devices},
            label=f"coz-command-{step_index:02d}",
        )
        produced = discover_single_output(
            output_dir,
            expected_size=(
                inspect_image(private_input, mock=False, stage="worker_input").width * 4,
                inspect_image(private_input, mock=False, stage="worker_input").height * 4,
            ),
        )
        normalize_to_png(produced, destination)
        return WorkerResult(
            image=inspect_image(destination, mock=False, stage=f"coz_scale_{step_index}"),
            metadata={"backend": self.name, "seed": seed},
            process=evidence,
        )

    def accept(self, candidate: WorkerResult, *, step_index: int) -> None:
        del candidate, step_index

    def rollback(self, *, step_index: int) -> None:
        del step_index


class CommandScaleBackend:
    name = "coz_command"
    mock = False

    def __init__(self, config: CoZConfig, runtime: RuntimeConfig) -> None:
        self.config = config
        self.runtime = runtime

    def session(self, run_dir: Path) -> CommandScaleSession:
        return CommandScaleSession(self.config, self.runtime, run_dir)
