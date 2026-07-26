"""Strict YAML configuration for reproducible runs."""

from __future__ import annotations

import dataclasses
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar, cast

from scaleguard.errors import ConfigurationError
from scaleguard.strict_yaml import StrictYAMLError
from scaleguard.strict_yaml import loads as load_strict_yaml

_CREDENTIAL_ENV_NAME = re.compile(r"[A-Z][A-Z0-9_]*_(?:API_KEY|TOKEN|CREDENTIAL|SECRET)")


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    run_root: Path = Path("runs")
    process_timeout_seconds: float = 3600.0
    keep_temporary_files: bool = False
    gpu_poll_interval_seconds: float = 0.5


@dataclass(frozen=True, slots=True)
class FourKAgentConfig:
    mode: str = "fake"
    checkout: Path | None = None
    python_executable: str = "python"
    profile: str = "FastGen4K_P"
    tool_gpu: str = "0"
    command: tuple[str, ...] = ()
    depictqa_command: tuple[str, ...] = ()
    depictqa_cwd: Path | None = None
    depictqa_host: str = "127.0.0.1"
    depictqa_port: int = 5001
    depictqa_startup_timeout_seconds: float = 600.0
    depictqa_visible_devices: str = "1"
    perception_model_path: str = ""
    toolbox_root: Path | None = None
    hps_root: Path | None = None
    quality_model_path: Path | None = None
    llm_model: str = "gpt-4-turbo"
    api_key_env: str = "OPENAI_API_KEY"


@dataclass(frozen=True, slots=True)
class CoZConfig:
    mode: str = "fake"
    checkout: Path | None = None
    python_executable: str = "python"
    visible_devices: str = "0,1"
    command: tuple[str, ...] = ()
    model_path: str = "stabilityai/stable-diffusion-3-medium-diffusers"
    qwen_model_path: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    sr_lora_path: Path | None = None
    vae_path: Path | None = None
    vlm_lora_path: Path | None = None
    prompt_type: str = "vlm"
    seed: int = 0
    mixed_precision: str = "fp32"
    tile_size: int = 512
    tile_overlap: int = 64


@dataclass(frozen=True, slots=True)
class MetricConfig:
    quality_backend: str = "gradient_proxy"
    quality_metric: str = "musiq"
    quality_device: str = "cpu"
    quality_model_path: Path | None = None
    min_quality_gain: float = -0.02
    max_scale_nrmse: float = 0.12
    max_scale_edge_mae: float = 0.10
    measurement_enabled: bool = False
    measurement_model: str = "resize"
    measurement_parameters: dict[str, Any] = field(default_factory=dict)
    max_measurement_nrmse: float = 0.12
    calibration_receipt: Path | None = None


@dataclass(frozen=True, slots=True)
class ControllerConfig:
    target_factor: int = 4
    max_coz_steps: int = 2
    color_strategy: str = "adain"
    accept_unvalidated_quality_proxy: bool = False


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    runtime: RuntimeConfig = RuntimeConfig()
    fourkagent: FourKAgentConfig = FourKAgentConfig()
    coz: CoZConfig = CoZConfig()
    metrics: MetricConfig = MetricConfig()
    controller: ControllerConfig = ControllerConfig()

    @property
    def is_mock(self) -> bool:
        return self.fourkagent.mode == "fake" or self.coz.mode == "fake"

    def as_dict(self) -> dict[str, Any]:
        converted = _convert_paths(dataclasses.asdict(self))
        return cast(dict[str, Any], converted)


def _convert_paths(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _convert_paths(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_convert_paths(item) for item in value]
    return value


C = TypeVar(
    "C",
    RuntimeConfig,
    FourKAgentConfig,
    CoZConfig,
    MetricConfig,
    ControllerConfig,
)


def _construct(cls: type[C], raw: Any, section: str) -> C:
    constructor: Any = cls
    if raw is None:
        return cast(C, constructor())
    if not isinstance(raw, dict):
        raise ConfigurationError(f"configuration section '{section}' must be a mapping")
    if not all(isinstance(key, str) for key in raw):
        raise ConfigurationError(f"configuration section '{section}' keys must be strings")
    class_fields = dataclasses.fields(cast(Any, cls))
    allowed = {item.name for item in class_fields}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigurationError(
            f"unknown keys in configuration section '{section}': {', '.join(unknown)}"
        )
    try:
        values = dict(raw)
        for item in class_fields:
            if item.name not in values:
                continue
            if item.type in (Path, "Path") and values[item.name] is not None:
                values[item.name] = Path(values[item.name])
            elif (
                item.name
                in {
                    "checkout",
                    "depictqa_cwd",
                    "toolbox_root",
                    "hps_root",
                    "quality_model_path",
                    "sr_lora_path",
                    "vae_path",
                    "vlm_lora_path",
                    "calibration_receipt",
                    "run_root",
                }
                and values[item.name] is not None
            ):
                values[item.name] = Path(values[item.name])
            elif item.name in {"command", "depictqa_command"}:
                if not isinstance(values[item.name], list) or not all(
                    isinstance(token, str) for token in values[item.name]
                ):
                    raise ConfigurationError(f"'{section}.{item.name}' must be a list of strings")
                values[item.name] = tuple(values[item.name])
        return cast(C, constructor(**values))
    except (TypeError, ValueError) as error:
        raise ConfigurationError(f"invalid configuration section '{section}': {error}") from error


def _numeric_device_selectors(value: str, *, count: int, field_name: str) -> tuple[str, ...]:
    selectors = tuple(value.split(","))
    if (
        len(selectors) != count
        or any(re.fullmatch(r"[0-9]+", selector) is None for selector in selectors)
        or len(set(selectors)) != len(selectors)
    ):
        noun = "selector" if count == 1 else "distinct selectors"
        raise ConfigurationError(
            f"{field_name} must contain exactly {count} non-negative numeric {noun}"
        )
    return selectors


def _local_model_location(value: str) -> bool:
    path = Path(value)
    return path.is_absolute() or value.startswith(("./", "../", "weights/", "third_party/"))


def validate_config(config: PipelineConfig) -> None:
    _validate_types(config)
    if config.fourkagent.mode not in {"fake", "command", "upstream"}:
        raise ConfigurationError("fourkagent.mode must be fake, command, or upstream")
    if config.coz.mode not in {"fake", "command", "upstream", "persistent"}:
        raise ConfigurationError("coz.mode must be fake, command, upstream, or persistent")
    if config.controller.target_factor not in {1, 2, 4, 8, 16}:
        raise ConfigurationError("controller.target_factor must be one of 1, 2, 4, 8, 16")
    if not 0 <= config.controller.max_coz_steps <= 2:
        raise ConfigurationError("controller.max_coz_steps must be between 0 and 2")
    if config.controller.color_strategy not in {"none", "adain"}:
        raise ConfigurationError("controller.color_strategy must be none or adain")
    if config.metrics.quality_backend not in {"gradient_proxy", "pyiqa"}:
        raise ConfigurationError("metrics.quality_backend is not available")
    if config.metrics.quality_backend == "pyiqa" and not config.metrics.quality_metric:
        raise ConfigurationError("metrics.quality_metric is required for pyiqa")
    if config.runtime.process_timeout_seconds <= 0:
        raise ConfigurationError("runtime.process_timeout_seconds must be positive")
    if config.runtime.gpu_poll_interval_seconds <= 0:
        raise ConfigurationError("runtime.gpu_poll_interval_seconds must be positive")
    if config.fourkagent.depictqa_startup_timeout_seconds <= 0:
        raise ConfigurationError("fourkagent.depictqa_startup_timeout_seconds must be positive")
    if not 1 <= config.fourkagent.depictqa_port <= 65_535:
        raise ConfigurationError("fourkagent.depictqa_port must be between 1 and 65535")
    if config.fourkagent.mode == "upstream" and (
        config.fourkagent.depictqa_host != "127.0.0.1" or config.fourkagent.depictqa_port != 5001
    ):
        raise ConfigurationError("the audited DepictQA overlay is fixed to loopback 127.0.0.1:5001")
    if config.fourkagent.depictqa_command and config.fourkagent.depictqa_cwd is None:
        raise ConfigurationError(
            "fourkagent.depictqa_cwd is required when depictqa_command is configured"
        )
    if _CREDENTIAL_ENV_NAME.fullmatch(config.fourkagent.api_key_env) is None:
        raise ConfigurationError(
            "fourkagent.api_key_env must be an uppercase credential variable ending "
            "in _API_KEY, _TOKEN, _CREDENTIAL, or _SECRET"
        )
    if not config.fourkagent.llm_model:
        raise ConfigurationError("fourkagent.llm_model must not be empty")
    if config.coz.tile_size <= 0:
        raise ConfigurationError("coz.tile_size must be positive")
    if config.coz.tile_overlap < 0 or config.coz.tile_overlap >= config.coz.tile_size:
        raise ConfigurationError("coz.tile_overlap must be non-negative and smaller than tile_size")
    if config.fourkagent.mode != "fake" and config.fourkagent.mode == "command":
        if not config.fourkagent.command:
            raise ConfigurationError("fourkagent.command is required in command mode")
    if config.coz.mode == "command" and not config.coz.command:
        raise ConfigurationError("coz.command is required in command mode")
    if config.coz.prompt_type not in {"vlm", "vlm_base"}:
        raise ConfigurationError("coz.prompt_type must be vlm or vlm_base")
    if config.coz.mixed_precision != "fp32":
        raise ConfigurationError(
            "coz.mixed_precision must be fp32 for the audited full-image CoZ path"
        )
    if config.coz.mode in {"upstream", "persistent"}:
        _numeric_device_selectors(
            config.coz.visible_devices,
            count=2,
            field_name="coz.visible_devices",
        )
        missing = [
            name
            for name, value in {
                "checkout": config.coz.checkout,
                "sr_lora_path": config.coz.sr_lora_path,
                "vae_path": config.coz.vae_path,
            }.items()
            if value is None
        ]
        if config.coz.prompt_type == "vlm" and config.coz.vlm_lora_path is None:
            missing.append("vlm_lora_path")
        if missing:
            raise ConfigurationError(f"CoZ {config.coz.mode} mode requires: {', '.join(missing)}")
    if config.controller.target_factor == 16 and config.coz.mode == "upstream":
        raise ConfigurationError(
            "16x recursion requires coz.mode=persistent so both 4x transitions share "
            "one explicit CoZ session"
        )
    if config.fourkagent.mode == "upstream":
        tool_selectors = _numeric_device_selectors(
            config.fourkagent.tool_gpu,
            count=1,
            field_name="fourkagent.tool_gpu",
        )
        depictqa_selectors = _numeric_device_selectors(
            config.fourkagent.depictqa_visible_devices,
            count=1,
            field_name="fourkagent.depictqa_visible_devices",
        )
        if tool_selectors == depictqa_selectors:
            raise ConfigurationError(
                "fourkagent.tool_gpu and depictqa_visible_devices must select different GPUs"
            )
        required_fourkagent = {
            "checkout": config.fourkagent.checkout,
            "toolbox_root": config.fourkagent.toolbox_root,
            "hps_root": config.fourkagent.hps_root,
            "quality_model_path": config.fourkagent.quality_model_path,
            "perception_model_path": config.fourkagent.perception_model_path or None,
        }
        missing_fourkagent = [name for name, value in required_fourkagent.items() if value is None]
        if missing_fourkagent:
            raise ConfigurationError(
                "4KAgent upstream mode requires: "
                + ", ".join(f"fourkagent.{name}" for name in missing_fourkagent)
            )
        if not config.fourkagent.depictqa_command:
            raise ConfigurationError(
                "4KAgent upstream mode requires a managed fourkagent.depictqa_command "
                "so the DepictQA GPU lifetime ends before CoZ starts"
            )
    if config.metrics.quality_backend == "pyiqa" and config.metrics.quality_model_path is None:
        raise ConfigurationError("metrics.quality_model_path is required for the PyIQA backend")
    for name, value in (
        ("metrics.max_scale_nrmse", config.metrics.max_scale_nrmse),
        ("metrics.max_scale_edge_mae", config.metrics.max_scale_edge_mae),
        ("metrics.max_measurement_nrmse", config.metrics.max_measurement_nrmse),
    ):
        if value < 0:
            raise ConfigurationError(f"{name} must be non-negative")
    from scaleguard.imaging.forward_models import build_forward_model

    build_forward_model(
        config.metrics.measurement_model,
        config.metrics.measurement_parameters,
    )
    if (
        config.metrics.quality_backend == "pyiqa"
        and (config.fourkagent.mode != "fake" or config.coz.mode != "fake")
        and config.metrics.quality_device != "cpu"
    ):
        raise ConfigurationError(
            "PyIQA controller gating must use CPU while a non-mock runtime is configured; "
            "offline evaluation may use a GPU after the run"
        )
    if (
        not config.is_mock
        and config.metrics.quality_backend == "gradient_proxy"
        and not config.controller.accept_unvalidated_quality_proxy
    ):
        raise ConfigurationError(
            "gradient_proxy is not a validated production gate; set "
            "controller.accept_unvalidated_quality_proxy=true only for calibration runs"
        )
    if config.coz.mode in {"upstream", "persistent"}:
        if not _local_model_location(config.coz.model_path):
            raise ConfigurationError(
                "coz.model_path must be a local path in upstream or persistent mode"
            )
        if not _local_model_location(config.coz.qwen_model_path):
            raise ConfigurationError(
                "coz.qwen_model_path must be a local path in upstream or persistent mode"
            )


def _validate_types(config: PipelineConfig) -> None:
    def exact(value: Any, expected: type[Any], field_name: str) -> None:
        if type(value) is not expected:
            raise ConfigurationError(
                f"{field_name} must be {expected.__name__}, got {type(value).__name__}"
            )

    def text(value: Any, field_name: str) -> None:
        exact(value, str, field_name)

    def number(value: Any, field_name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigurationError(f"{field_name} must be numeric")
        if not math.isfinite(float(value)):
            raise ConfigurationError(f"{field_name} must be finite")

    if not isinstance(config.runtime.run_root, Path):
        raise ConfigurationError("runtime.run_root must be a path")
    number(config.runtime.process_timeout_seconds, "runtime.process_timeout_seconds")
    exact(config.runtime.keep_temporary_files, bool, "runtime.keep_temporary_files")
    number(config.runtime.gpu_poll_interval_seconds, "runtime.gpu_poll_interval_seconds")
    text_fields: list[tuple[str, Any]] = [
        ("fourkagent.mode", config.fourkagent.mode),
        ("fourkagent.python_executable", config.fourkagent.python_executable),
        ("fourkagent.profile", config.fourkagent.profile),
        ("fourkagent.tool_gpu", config.fourkagent.tool_gpu),
        ("fourkagent.depictqa_host", config.fourkagent.depictqa_host),
        ("fourkagent.depictqa_visible_devices", config.fourkagent.depictqa_visible_devices),
        ("fourkagent.perception_model_path", config.fourkagent.perception_model_path),
        ("fourkagent.llm_model", config.fourkagent.llm_model),
        ("fourkagent.api_key_env", config.fourkagent.api_key_env),
        ("coz.mode", config.coz.mode),
        ("coz.python_executable", config.coz.python_executable),
        ("coz.visible_devices", config.coz.visible_devices),
        ("coz.model_path", config.coz.model_path),
        ("coz.qwen_model_path", config.coz.qwen_model_path),
        ("coz.prompt_type", config.coz.prompt_type),
        ("coz.mixed_precision", config.coz.mixed_precision),
        ("metrics.quality_backend", config.metrics.quality_backend),
        ("metrics.quality_metric", config.metrics.quality_metric),
        ("metrics.quality_device", config.metrics.quality_device),
        ("metrics.measurement_model", config.metrics.measurement_model),
        ("controller.color_strategy", config.controller.color_strategy),
    ]
    for name, value in text_fields:
        text(value, name)
    path_fields: list[tuple[str, Any]] = [
        ("fourkagent.checkout", config.fourkagent.checkout),
        ("fourkagent.depictqa_cwd", config.fourkagent.depictqa_cwd),
        ("fourkagent.toolbox_root", config.fourkagent.toolbox_root),
        ("fourkagent.hps_root", config.fourkagent.hps_root),
        ("fourkagent.quality_model_path", config.fourkagent.quality_model_path),
        ("coz.checkout", config.coz.checkout),
        ("coz.sr_lora_path", config.coz.sr_lora_path),
        ("coz.vae_path", config.coz.vae_path),
        ("coz.vlm_lora_path", config.coz.vlm_lora_path),
        ("metrics.calibration_receipt", config.metrics.calibration_receipt),
        ("metrics.quality_model_path", config.metrics.quality_model_path),
    ]
    for name, value in path_fields:
        if value is not None and not isinstance(value, Path):
            raise ConfigurationError(f"{name} must be a path")
    command_fields: list[tuple[str, Any]] = [
        ("fourkagent.command", config.fourkagent.command),
        ("fourkagent.depictqa_command", config.fourkagent.depictqa_command),
        ("coz.command", config.coz.command),
    ]
    for name, value in command_fields:
        exact(value, tuple, name)
        if not all(type(token) is str for token in value):
            raise ConfigurationError(f"{name} must contain only strings")
    integer_fields: list[tuple[str, Any]] = [
        ("coz.seed", config.coz.seed),
        ("fourkagent.depictqa_port", config.fourkagent.depictqa_port),
        ("coz.tile_size", config.coz.tile_size),
        ("coz.tile_overlap", config.coz.tile_overlap),
        ("controller.target_factor", config.controller.target_factor),
        ("controller.max_coz_steps", config.controller.max_coz_steps),
    ]
    for name, value in integer_fields:
        exact(value, int, name)
    numeric_fields: list[tuple[str, Any]] = [
        ("metrics.min_quality_gain", config.metrics.min_quality_gain),
        (
            "fourkagent.depictqa_startup_timeout_seconds",
            config.fourkagent.depictqa_startup_timeout_seconds,
        ),
        ("metrics.max_scale_nrmse", config.metrics.max_scale_nrmse),
        ("metrics.max_scale_edge_mae", config.metrics.max_scale_edge_mae),
        ("metrics.max_measurement_nrmse", config.metrics.max_measurement_nrmse),
    ]
    for name, value in numeric_fields:
        number(value, name)
    exact(config.metrics.measurement_enabled, bool, "metrics.measurement_enabled")
    exact(config.metrics.measurement_parameters, dict, "metrics.measurement_parameters")
    exact(
        config.controller.accept_unvalidated_quality_proxy,
        bool,
        "controller.accept_unvalidated_quality_proxy",
    )


def load_config(path: Path) -> PipelineConfig:
    """Load a strict configuration without environment-variable interpolation."""

    try:
        raw = load_strict_yaml(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ConfigurationError(f"cannot read configuration {path}: {error}") from error
    except StrictYAMLError as error:
        raise ConfigurationError(f"invalid YAML in {path}: {error}") from error
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigurationError("configuration root must be a mapping")
    if not all(isinstance(key, str) for key in raw):
        raise ConfigurationError("configuration root keys must be strings")
    allowed = {"runtime", "fourkagent", "coz", "metrics", "controller"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigurationError(f"unknown configuration sections: {', '.join(unknown)}")
    config = PipelineConfig(
        runtime=_construct(RuntimeConfig, raw.get("runtime"), "runtime"),
        fourkagent=_construct(FourKAgentConfig, raw.get("fourkagent"), "fourkagent"),
        coz=_construct(CoZConfig, raw.get("coz"), "coz"),
        metrics=_construct(MetricConfig, raw.get("metrics"), "metrics"),
        controller=_construct(ControllerConfig, raw.get("controller"), "controller"),
    )
    validate_config(config)
    return config
