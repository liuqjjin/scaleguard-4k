"""Read-only 4KAgent checkout adapter."""

from __future__ import annotations

import os
from pathlib import Path

from scaleguard.config import FourKAgentConfig, RuntimeConfig
from scaleguard.contracts import ProcessEvidence, WorkerResult
from scaleguard.errors import ArtifactError, WorkerError
from scaleguard.images import inspect_image, normalize_to_png
from scaleguard.runtime.process import ProcessRunner, format_command, project_executable
from scaleguard.runtime.service import ManagedService, tcp_ready
from scaleguard.strict_json import StrictJSONError, loads_object


def _project_path(project_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


class FourKAgentBackend:
    name = "4kagent_upstream"
    mock = False

    def __init__(
        self,
        config: FourKAgentConfig,
        runtime: RuntimeConfig,
        *,
        project_root: Path,
    ) -> None:
        if config.checkout is None:
            raise ValueError("4KAgent checkout is required")
        self.config = config
        self.project_root = project_root.resolve()
        self.checkout = _project_path(self.project_root, config.checkout)
        self.python_executable = project_executable(self.project_root, config.python_executable)
        self.runner = ProcessRunner(
            timeout_seconds=runtime.process_timeout_seconds,
            gpu_poll_interval_seconds=runtime.gpu_poll_interval_seconds,
        )
        self.overlay = (
            self.project_root / "third_party" / "overlays" / "4kagent" / "run_native_restoration.py"
        ).resolve()

    def restore(
        self,
        source: Path,
        destination: Path,
        *,
        bridge_factor: int,
        run_dir: Path,
    ) -> WorkerResult:
        private_dir = run_dir / "workers" / "4kagent"
        private_dir.mkdir(parents=True, exist_ok=True)
        private_dir.chmod(0o700)
        outlines_cache = private_dir / "outlines-cache"
        outlines_cache.mkdir(mode=0o700)
        input_path = private_dir / "input" / "source.png"
        raw_output = private_dir / "raw_output"
        normalize_to_png(source, input_path)
        argv = [
            self.python_executable,
            str(self.overlay),
            "--checkout",
            str(self.checkout),
            "--input",
            str(input_path),
            "--output-dir",
            str(raw_output),
            "--profile",
            self.config.profile,
            "--tool-gpu",
            self.config.tool_gpu,
            "--bridge-factor",
            str(bridge_factor),
            "--runtime-view",
            str((private_dir / "runtime-view").resolve()),
            "--toolbox-root",
            str(
                _project_path(
                    self.project_root,
                    self.config.toolbox_root or Path("weights/4kagent/runtime/toolbox-root"),
                )
            ),
            "--hps-root",
            str(
                _project_path(
                    self.project_root,
                    self.config.hps_root or Path("weights/4kagent/hpsv2"),
                )
            ),
            "--quality-model-path",
            str(
                _project_path(
                    self.project_root,
                    self.config.quality_model_path
                    or Path("weights/metrics/pyiqa/musiq_koniq_ckpt-e95806b9.pth"),
                )
            ),
            "--llm-model",
            self.config.llm_model,
            "--api-key-env",
            self.config.api_key_env,
        ]
        if self.config.perception_model_path:
            model_path = Path(self.config.perception_model_path)
            resolved_model = (
                model_path
                if model_path.is_absolute()
                else (self.project_root / model_path).resolve()
            )
            argv.extend(["--perception-model-path", str(resolved_model)])
        service_evidence: dict[str, object]
        if self.config.depictqa_command:
            if self.config.depictqa_cwd is None:
                raise WorkerError("managed DepictQA service requires depictqa_cwd")
            service_argv = format_command(
                self.config.depictqa_command,
                {
                    "project_root": str(self.project_root),
                    "checkout": str(self.checkout),
                    "service_work_dir": str((private_dir / "depictqa").resolve()),
                },
            )
            service = ManagedService(
                service_argv,
                cwd=_project_path(self.project_root, self.config.depictqa_cwd),
                log_dir=private_dir / "depictqa",
                host=self.config.depictqa_host,
                port=self.config.depictqa_port,
                startup_timeout_seconds=self.config.depictqa_startup_timeout_seconds,
                env={
                    "CUDA_VISIBLE_DEVICES": self.config.depictqa_visible_devices,
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "HF_HUB_DISABLE_TELEMETRY": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
                label="depictqa-eval",
            )
            with service:
                evidence = self._run_agent(argv, private_dir)
            service_evidence = service.evidence()
        else:
            if not tcp_ready(self.config.depictqa_host, self.config.depictqa_port):
                raise WorkerError(
                    "DepictQA is not reachable and no managed depictqa_command is configured"
                )
            service_evidence = {
                "managed": False,
                "host": self.config.depictqa_host,
                "port": self.config.depictqa_port,
            }
            evidence = self._run_agent(argv, private_dir)
        evidence_path = raw_output / "scaleguard-result.json"
        try:
            private_root = private_dir.resolve()
            raw_root = raw_output.resolve()
            if not raw_root.is_relative_to(private_root):
                raise ArtifactError(
                    f"4KAgent output directory escaped its private directory: {raw_output}"
                )
            resolved_evidence = evidence_path.resolve()
            if not resolved_evidence.is_relative_to(raw_root):
                raise ArtifactError(
                    f"4KAgent result evidence escaped its private output directory: {evidence_path}"
                )
            result_data = loads_object(resolved_evidence.read_text(encoding="utf-8"))
            reported_result = Path(result_data["result"])
            upstream_result = (
                reported_result.resolve()
                if reported_result.is_absolute()
                else (raw_root / reported_result).resolve()
            )
        except ArtifactError:
            raise
        except (OSError, KeyError, TypeError, StrictJSONError) as error:
            raise ArtifactError(
                f"4KAgent did not write a valid adapter result at {evidence_path}: {error}"
            ) from error
        except RuntimeError as error:
            raise ArtifactError(
                f"4KAgent result could not be resolved safely at {evidence_path}: {error}"
            ) from error
        if not upstream_result.is_relative_to(raw_root):
            raise ArtifactError(
                f"4KAgent result escaped its private output directory: {upstream_result}"
            )
        normalize_to_png(upstream_result, destination)
        return WorkerResult(
            image=inspect_image(destination, mock=False, stage="4kagent_restoration"),
            metadata={
                "backend": self.name,
                "bridge_factor": bridge_factor,
                "execution_path": result_data.get("execution_path", {}),
                "terminal_generative_sr": False,
                "depictqa_service": service_evidence,
                "llm_model": self.config.llm_model,
            },
            process=evidence,
        )

    def _run_agent(self, argv: list[str], private_dir: Path) -> ProcessEvidence:
        hps_root = self.config.hps_root or Path("weights/4kagent/hpsv2")
        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            raise WorkerError(f"required scheduler credential is absent: {self.config.api_key_env}")
        return self.runner.run(
            argv,
            cwd=self.checkout,
            log_dir=private_dir / "logs",
            env={
                "CUDA_VISIBLE_DEVICES": self.config.tool_gpu,
                "HPS_ROOT": str(_project_path(self.project_root, hps_root)),
                "OUTLINES_CACHE_DIR": str((private_dir / "outlines-cache").resolve()),
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "TORCH_FORCE_WEIGHTS_ONLY_LOAD": "1",
                self.config.api_key_env: api_key,
            },
            label="4kagent",
        )
