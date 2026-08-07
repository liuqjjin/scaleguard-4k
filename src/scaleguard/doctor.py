"""Local readiness checks that never imply an unperformed GPU reproduction."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from scaleguard.config import PipelineConfig
from scaleguard.evaluation.calibration import verify_calibration_receipt
from scaleguard.runtime.process import format_command, minimal_subprocess_environment
from scaleguard.runtime.service import tcp_ready


@dataclass(frozen=True, slots=True)
class DoctorCheck:
    name: str
    status: str
    detail: str


def _gpu_inventory() -> tuple[list[tuple[str, int, str]], str | None]:
    if shutil.which("nvidia-smi") is None:
        return [], "nvidia-smi is not installed"
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.total,uuid",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
            env=minimal_subprocess_environment(),
        )
    except subprocess.TimeoutExpired:
        return [], "nvidia-smi timed out after 10 seconds"
    except OSError as error:
        return [], f"cannot execute nvidia-smi: {error}"
    if result.returncode != 0:
        return [], result.stderr.strip() or f"nvidia-smi exited {result.returncode}"
    inventory: list[tuple[str, int, str]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            inventory.append((parts[0], int(parts[1]), parts[2]))
        except ValueError:
            continue
    return inventory, None


def run_doctor(config: PipelineConfig, project_root: Path) -> list[DoctorCheck]:
    checks = [
        DoctorCheck(
            "python",
            "pass" if sys.version_info >= (3, 10) else "fail",
            sys.version.split()[0],
        ),
        DoctorCheck(
            "project_root",
            "pass" if (project_root / "pyproject.toml").is_file() else "fail",
            str(project_root),
        ),
    ]
    for name, mode, checkout in (
        ("4kagent_checkout", config.fourkagent.mode, config.fourkagent.checkout),
        ("coz_checkout", config.coz.mode, config.coz.checkout),
    ):
        required = mode in {"upstream", "persistent"}
        resolved = (
            checkout
            if checkout is None or checkout.is_absolute()
            else (project_root / checkout).resolve()
        )
        exists = resolved is not None and (resolved / ".git").exists()
        checks.append(
            DoctorCheck(
                name,
                "pass" if exists else ("fail" if required else "skip"),
                str(resolved) if resolved is not None else f"not required in {mode} mode",
            )
        )
    for name, path, kind in (
        ("coz_model", Path(config.coz.model_path), "directory"),
        ("coz_qwen_model", Path(config.coz.qwen_model_path), "directory"),
        ("coz_sr_lora", config.coz.sr_lora_path, "file"),
        ("coz_vae", config.coz.vae_path, "file"),
        ("coz_vlm_lora", config.coz.vlm_lora_path, "directory"),
    ):
        required = config.coz.mode in {"upstream", "persistent"} and (
            name != "coz_vlm_lora" or config.coz.prompt_type == "vlm"
        )
        resolved = path if path is None or path.is_absolute() else (project_root / path).resolve()
        exists = resolved is not None and (
            resolved.is_dir() if kind == "directory" else resolved.is_file()
        )
        checks.append(
            DoctorCheck(
                name,
                "pass" if exists else ("fail" if required else "skip"),
                (str(resolved) if required else f"not required in {config.coz.mode} mode"),
            )
        )
    if config.fourkagent.mode == "upstream":
        scheduler_host = urlsplit(config.fourkagent.llm_base_url).hostname or "invalid"
        checks.append(
            DoctorCheck(
                "remote_scheduler",
                "pass",
                (
                    f"{config.fourkagent.llm_provider}/{config.fourkagent.llm_model} "
                    f"in {config.fourkagent.llm_region} via {scheduler_host}; "
                    "network availability is checked only during a real run"
                ),
            )
        )
        fourk_paths: tuple[tuple[str, Path | None, str], ...] = (
            ("4kagent_toolbox", config.fourkagent.toolbox_root, "directory"),
            ("4kagent_hps", config.fourkagent.hps_root, "directory"),
            (
                "4kagent_quality_model",
                config.fourkagent.quality_model_path,
                "file",
            ),
            (
                "4kagent_perception_model",
                (
                    Path(config.fourkagent.perception_model_path)
                    if config.fourkagent.perception_model_path
                    else None
                ),
                "directory",
            ),
        )
        for name, path, kind in fourk_paths:
            resolved = (
                path if path is None or path.is_absolute() else (project_root / path).resolve()
            )
            exists = resolved is not None and (
                resolved.is_dir() if kind == "directory" else resolved.is_file()
            )
            checks.append(
                DoctorCheck(
                    name,
                    "pass" if exists else "fail",
                    str(resolved) if resolved is not None else "not configured",
                )
            )
        api_key_present = bool(os.environ.get(config.fourkagent.api_key_env))
        checks.append(
            DoctorCheck(
                "4kagent_api_key",
                "pass" if api_key_present else "fail",
                (
                    f"{config.fourkagent.api_key_env} is set"
                    if api_key_present
                    else f"set {config.fourkagent.api_key_env}; value is never logged"
                ),
            )
        )
        if config.fourkagent.depictqa_command:
            command = format_command(
                config.fourkagent.depictqa_command,
                {
                    "project_root": str(project_root.resolve()),
                    "checkout": str(
                        (
                            project_root
                            / (config.fourkagent.checkout or Path("third_party/checkouts/4KAgent"))
                        ).resolve()
                    ),
                    "service_work_dir": str(
                        (project_root / ".runtime" / "doctor" / "depictqa").resolve()
                    ),
                },
            )
            service_cwd = config.fourkagent.depictqa_cwd
            resolved_cwd = (
                service_cwd
                if service_cwd is None or service_cwd.is_absolute()
                else (project_root / service_cwd).resolve()
            )
            executable = Path(command[0])
            executable_exists = (
                shutil.which(command[0]) is not None
                if len(executable.parts) == 1
                else executable.is_file()
            )
            ready = resolved_cwd is not None and resolved_cwd.is_dir() and executable_exists
            checks.append(
                DoctorCheck(
                    "depictqa_service",
                    "pass" if ready else "fail",
                    (
                        f"managed {config.fourkagent.depictqa_host}:"
                        f"{config.fourkagent.depictqa_port}; cwd={resolved_cwd}; "
                        f"executable={command[0]}"
                    ),
                )
            )
        else:
            ready = tcp_ready(
                config.fourkagent.depictqa_host,
                config.fourkagent.depictqa_port,
            )
            checks.append(
                DoctorCheck(
                    "depictqa_service",
                    "pass" if ready else "fail",
                    (
                        f"external endpoint {config.fourkagent.depictqa_host}:"
                        f"{config.fourkagent.depictqa_port}"
                    ),
                )
            )
    inventory, gpu_error = _gpu_inventory()
    required_gpu_indices: list[str] = []
    if config.fourkagent.mode == "upstream":
        required_gpu_indices.extend(
            (
                config.fourkagent.tool_gpu,
                config.fourkagent.depictqa_visible_devices,
            )
        )
    if config.coz.mode in {"upstream", "persistent"}:
        required_gpu_indices.extend(config.coz.visible_devices.split(","))
    required_gpu_indices = list(dict.fromkeys(required_gpu_indices))
    gpu_required = bool(required_gpu_indices)
    if gpu_error:
        checks.append(DoctorCheck("gpu_inventory", "fail" if gpu_required else "skip", gpu_error))
    else:
        by_index = {index: (memory, uuid) for index, memory, uuid in inventory}
        missing = [index for index in required_gpu_indices if index not in by_index]
        inadequate = [
            index
            for index in required_gpu_indices
            if index in by_index and by_index[index][0] < 23_000
        ]
        adequate = not missing and not inadequate
        detail_parts = [f"GPU {index}: {memory} MiB {uuid}" for index, memory, uuid in inventory]
        if required_gpu_indices:
            detail_parts.append("selected=" + ",".join(required_gpu_indices))
        if missing:
            detail_parts.append("missing=" + ",".join(missing))
        if inadequate:
            detail_parts.append("below_23000_MiB=" + ",".join(inadequate))
        detail = "; ".join(detail_parts)
        checks.append(
            DoctorCheck(
                "gpu_inventory",
                "pass" if adequate else ("fail" if gpu_required else "warn"),
                detail or "no GPUs discovered",
            )
        )
    if config.metrics.quality_backend == "gradient_proxy":
        checks.append(
            DoctorCheck(
                "quality_gate",
                "warn",
                "gradient_proxy is CPU plumbing only; it is not a production IQA gate",
            )
        )
    else:
        receipt = config.metrics.calibration_receipt
        resolved_receipt = (
            receipt
            if receipt is None or receipt.is_absolute()
            else (project_root / receipt).resolve()
        )
        valid = False
        reasons = ["calibration_receipt_not_configured"]
        if resolved_receipt is not None:
            try:
                valid, reasons = verify_calibration_receipt(resolved_receipt, config)
            except (OSError, ValueError) as error:
                reasons = [f"unreadable:{type(error).__name__}"]
        checks.append(
            DoctorCheck(
                "quality_gate",
                "pass" if valid else "warn",
                (
                    f"verified calibration receipt: {resolved_receipt}"
                    if valid
                    else "thresholds are uncalibrated: " + ", ".join(reasons)
                ),
            )
        )
    return checks
