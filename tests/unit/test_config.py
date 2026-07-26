from __future__ import annotations

from pathlib import Path

import pytest

from scaleguard.config import load_config
from scaleguard.errors import ConfigurationError


def write_config(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_empty_config_uses_valid_cpu_safe_defaults(tmp_path: Path) -> None:
    config = load_config(write_config(tmp_path / "empty.yaml", "{}\n"))

    assert config.fourkagent.mode == "fake"
    assert config.coz.mode == "fake"
    assert config.coz.mixed_precision == "fp32"
    assert config.controller.target_factor == 4
    assert config.is_mock is True


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("surprise: true\n", "unknown configuration sections: surprise"),
        (
            "controller:\n  target_factor: 4\n  surprise: true\n",
            "unknown keys in configuration section 'controller': surprise",
        ),
        ("controller: fast\n", "configuration section 'controller' must be a mapping"),
        ("- controller\n- coz\n", "configuration root must be a mapping"),
        ("1: true\n", "configuration root keys must be strings"),
        (
            "controller:\n  1: true\n",
            "configuration section 'controller' keys must be strings",
        ),
        (
            "coz:\n  mode: command\n  command: python worker.py\n",
            "'coz.command' must be a list of strings",
        ),
        (
            "coz:\n  mode: command\n  command: [python, 7]\n",
            "'coz.command' must be a list of strings",
        ),
        (
            "fourkagent:\n  depictqa_command: python server.py\n",
            "'fourkagent.depictqa_command' must be a list of strings",
        ),
        (
            "runtime:\n  run_root: []\n",
            "invalid configuration section 'runtime'",
        ),
    ],
)
def test_config_rejects_unknown_or_structurally_invalid_data(
    tmp_path: Path,
    body: str,
    message: str,
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        load_config(write_config(tmp_path / "invalid.yaml", body))


@pytest.mark.parametrize(
    "body",
    [
        "fourkagent:\n  api_key_env: FIRST_KEY\n  api_key_env: SECOND_KEY\n",
        "fourkagent:\n  api_key_env: FIRST_KEY\nfourkagent:\n  api_key_env: SECOND_KEY\n",
    ],
)
def test_config_rejects_duplicate_yaml_mapping_keys(
    tmp_path: Path,
    body: str,
) -> None:
    with pytest.raises(ConfigurationError, match="duplicate mapping key"):
        load_config(write_config(tmp_path / "duplicate.yaml", body))


@pytest.mark.parametrize(
    "name",
    [
        "PATH",
        "HOME",
        "LD_PRELOAD",
        "PYTHONPATH",
        "PYTHONINSPECT",
        "BASH_ENV",
        "ENV",
        "lowercase_api_key",
    ],
)
def test_config_rejects_noncredential_or_interpreter_environment_names(
    tmp_path: Path,
    name: str,
) -> None:
    body = f"fourkagent:\n  api_key_env: {name}\n"
    with pytest.raises(ConfigurationError, match="uppercase credential variable"):
        load_config(write_config(tmp_path / "dangerous-env.yaml", body))


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("controller:\n  target_factor: 3\n", "target_factor"),
        ("controller:\n  max_coz_steps: 3\n", "max_coz_steps"),
        ("controller:\n  color_strategy: wavelet\n", "color_strategy"),
        ("runtime:\n  process_timeout_seconds: 0\n", "process_timeout_seconds"),
        ("runtime:\n  gpu_poll_interval_seconds: 0\n", "gpu_poll_interval_seconds"),
        (
            "fourkagent:\n  depictqa_startup_timeout_seconds: 0\n",
            "depictqa_startup_timeout_seconds",
        ),
        ("fourkagent:\n  depictqa_port: 0\n", "depictqa_port"),
        ("fourkagent:\n  depictqa_port: 65536\n", "depictqa_port"),
        (
            "fourkagent:\n  depictqa_command: [python, server.py]\n",
            "depictqa_cwd is required",
        ),
        ("coz:\n  tile_size: 64\n  tile_overlap: 64\n", "tile_overlap"),
        ("coz:\n  tile_size: 0\n", "tile_size"),
        ("metrics:\n  max_scale_nrmse: -0.1\n", "max_scale_nrmse"),
        ("metrics:\n  measurement_model: unknown\n", "unknown measurement model"),
        (
            "metrics:\n  measurement_model: resize\n  measurement_parameters: {sigma: 1}\n",
            "unknown measurement parameters",
        ),
        ("coz:\n  prompt_type: prose\n", "prompt_type"),
        ("coz:\n  mixed_precision: fp16\n", "mixed_precision"),
        ("fourkagent:\n  mode: command\n", "fourkagent.command"),
        ("coz:\n  mode: command\n", "coz.command"),
        ("fourkagent:\n  mode: upstream\n", "fourkagent.checkout"),
        (
            "coz:\n  mode: upstream\n  prompt_type: vlm\n",
            "checkout, sr_lora_path, vae_path, vlm_lora_path",
        ),
    ],
)
def test_config_rejects_invalid_values(tmp_path: Path, body: str, message: str) -> None:
    with pytest.raises(ConfigurationError, match=message):
        load_config(write_config(tmp_path / "invalid-value.yaml", body))


@pytest.mark.parametrize(
    "body",
    [
        "controller:\n  target_factor: true\n",
        "controller:\n  target_factor: 4.0\n",
        "runtime:\n  process_timeout_seconds: fast\n",
        "fourkagent:\n  depictqa_port: true\n",
        "fourkagent:\n  depictqa_startup_timeout_seconds: slow\n",
        "coz:\n  tile_size: many\n",
        "metrics:\n  measurement_parameters: []\n",
    ],
)
def test_config_is_strict_about_scalar_and_mapping_types(tmp_path: Path, body: str) -> None:
    with pytest.raises(ConfigurationError):
        load_config(write_config(tmp_path / "wrong-type.yaml", body))


def test_non_mock_command_backends_require_explicit_proxy_acceptance(tmp_path: Path) -> None:
    body = """
fourkagent:
  mode: command
  command: [python, fourkagent.py]
coz:
  mode: command
  command: [python, coz.py]
"""

    with pytest.raises(ConfigurationError, match="not a validated production gate"):
        load_config(write_config(tmp_path / "unsafe-proxy.yaml", body))


def test_online_pyiqa_cannot_hold_a_gpu_while_coz_owns_its_gpu_runtime(
    tmp_path: Path,
) -> None:
    body = """
coz:
  mode: persistent
  checkout: third_party/checkouts/Chain-of-Zoom
  sr_lora_path: weights/sr.pkl
  vae_path: weights/vae.pt
  vlm_lora_path: weights/vlm
metrics:
  quality_backend: pyiqa
  quality_metric: musiq
  quality_device: cuda:0
  quality_model_path: weights/musiq.pth
"""

    with pytest.raises(ConfigurationError, match="must use CPU"):
        load_config(write_config(tmp_path / "gpu-overlap.yaml", body))


def test_upstream_4kagent_requires_a_managed_depictqa_lifetime(
    tmp_path: Path,
) -> None:
    body = """
fourkagent:
  mode: upstream
  checkout: third_party/checkouts/4KAgent
  toolbox_root: weights/toolbox
  hps_root: weights/hps
  quality_model_path: weights/musiq.pth
  perception_model_path: weights/qwen
"""

    with pytest.raises(ConfigurationError, match=r"managed fourkagent\.depictqa_command"):
        load_config(write_config(tmp_path / "unmanaged-depictqa.yaml", body))


def test_sixteen_x_requires_one_persistent_coz_session(tmp_path: Path) -> None:
    body = """
coz:
  mode: upstream
  checkout: third_party/checkouts/Chain-of-Zoom
  sr_lora_path: weights/sr.pkl
  vae_path: weights/vae.pt
  vlm_lora_path: weights/vlm
controller:
  target_factor: 16
"""

    with pytest.raises(ConfigurationError, match=r"requires coz\.mode=persistent"):
        load_config(write_config(tmp_path / "one-shot-16x.yaml", body))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("visible_devices", '"0"'),
        ("visible_devices", '"0,0"'),
        ("visible_devices", '"0, gpu1"'),
    ],
)
def test_official_coz_requires_two_distinct_numeric_gpu_selectors(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    body = f"""
coz:
  mode: persistent
  {field}: {value}
  model_path: weights/models/sd3
  qwen_model_path: weights/models/qwen
  checkout: third_party/checkouts/Chain-of-Zoom
  sr_lora_path: weights/sr.pkl
  vae_path: weights/vae.pt
  vlm_lora_path: weights/vlm
controller:
  accept_unvalidated_quality_proxy: true
"""

    with pytest.raises(ConfigurationError, match=r"two|2"):
        load_config(write_config(tmp_path / "bad-gpus.yaml", body))


def test_official_coz_rejects_remote_mutable_model_identifiers(tmp_path: Path) -> None:
    body = """
coz:
  mode: persistent
  checkout: third_party/checkouts/Chain-of-Zoom
  sr_lora_path: weights/sr.pkl
  vae_path: weights/vae.pt
  vlm_lora_path: weights/vlm
"""

    with pytest.raises(ConfigurationError, match="model_path must be a local path"):
        load_config(write_config(tmp_path / "remote-model.yaml", body))


def test_paths_and_commands_are_normalized_without_interpolation(tmp_path: Path) -> None:
    body = """
runtime:
  run_root: "运行 结果"
fourkagent:
  command: [python, "{input}", "$DO_NOT_EXPAND"]
  depictqa_command: [python, server.py, "--port", "5001"]
  depictqa_cwd: "服务 目录"
"""
    config = load_config(write_config(tmp_path / "paths.yaml", body))

    assert config.runtime.run_root == Path("运行 结果")
    assert config.fourkagent.command == ("python", "{input}", "$DO_NOT_EXPAND")
    assert config.fourkagent.depictqa_command == (
        "python",
        "server.py",
        "--port",
        "5001",
    )
    assert config.fourkagent.depictqa_cwd == Path("服务 目录")
