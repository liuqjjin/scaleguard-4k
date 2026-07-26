from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scaleguard.backends.command import CommandRestorationBackend, CommandScaleBackend
from scaleguard.backends.coz import CoZBackend
from scaleguard.backends.fake import FakeRestorationBackend, FakeScaleBackend
from scaleguard.backends.fourkagent import FourKAgentBackend
from scaleguard.config import (
    ControllerConfig,
    CoZConfig,
    FourKAgentConfig,
    PipelineConfig,
)
from scaleguard.doctor import _gpu_inventory, run_doctor
from scaleguard.factory import build_backends


def test_factory_constructs_the_cpu_fake_pair(tmp_path: Path) -> None:
    restoration, scale = build_backends(PipelineConfig(), project_root=tmp_path)

    assert isinstance(restoration, FakeRestorationBackend)
    assert isinstance(scale, FakeScaleBackend)


def test_factory_constructs_command_adapters(tmp_path: Path) -> None:
    config = PipelineConfig(
        fourkagent=FourKAgentConfig(mode="command", command=("fake-fourkagent",)),
        coz=CoZConfig(mode="command", command=("fake-coz",)),
        controller=ControllerConfig(accept_unvalidated_quality_proxy=True),
    )

    restoration, scale = build_backends(config, project_root=tmp_path)

    assert isinstance(restoration, CommandRestorationBackend)
    assert isinstance(scale, CommandScaleBackend)


def test_factory_constructs_only_the_two_upstream_adapters(tmp_path: Path) -> None:
    config = PipelineConfig(
        fourkagent=FourKAgentConfig(mode="upstream", checkout=tmp_path / "4KAgent"),
        coz=CoZConfig(
            mode="persistent",
            checkout=tmp_path / "Chain-of-Zoom",
            sr_lora_path=tmp_path / "sr.pkl",
            vae_path=tmp_path / "vae.pt",
            vlm_lora_path=tmp_path / "vlm",
        ),
        controller=ControllerConfig(accept_unvalidated_quality_proxy=True),
    )

    restoration, scale = build_backends(config, project_root=tmp_path)

    assert isinstance(restoration, FourKAgentBackend)
    assert isinstance(scale, CoZBackend)


def test_gpu_inventory_skips_nvidia_smi_when_it_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("scaleguard.doctor.shutil.which", lambda _name: None)

    inventory, error = _gpu_inventory()

    assert inventory == []
    assert error == "nvidia-smi is not installed"


def test_gpu_inventory_parses_valid_rows_and_ignores_malformed_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("scaleguard.doctor.shutil.which", lambda _name: "/fake/nvidia-smi")
    completed = subprocess.CompletedProcess(
        args=["nvidia-smi"],
        returncode=0,
        stdout="0, 24564, GPU-a\nmalformed\n1, not-a-number, GPU-b\n2, 24000, GPU-c\n",
        stderr="",
    )
    monkeypatch.setattr("scaleguard.doctor.subprocess.run", lambda *_args, **_kwargs: completed)

    inventory, error = _gpu_inventory()

    assert inventory == [("0", 24564, "GPU-a"), ("2", 24000, "GPU-c")]
    assert error is None


def test_gpu_inventory_does_not_pass_ambient_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "doctor-secret")
    monkeypatch.setenv("GITHUB_TOKEN", "github-secret")
    monkeypatch.setattr("scaleguard.doctor.shutil.which", lambda _name: "/fake/nvidia-smi")
    observed_environment: dict[str, str] = {}

    def run(*_args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        environment = kwargs.get("env")
        assert isinstance(environment, dict)
        observed_environment.update(environment)
        return subprocess.CompletedProcess(
            args=["nvidia-smi"],
            returncode=0,
            stdout="0, 24564, GPU-a\n",
            stderr="",
        )

    monkeypatch.setattr("scaleguard.doctor.subprocess.run", run)

    inventory, error = _gpu_inventory()

    assert inventory == [("0", 24564, "GPU-a")]
    assert error is None
    assert "OPENAI_API_KEY" not in observed_environment
    assert "GITHUB_TOKEN" not in observed_environment


def test_gpu_inventory_preserves_command_failure_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("scaleguard.doctor.shutil.which", lambda _name: "/fake/nvidia-smi")
    completed = subprocess.CompletedProcess(
        args=["nvidia-smi"],
        returncode=9,
        stdout="",
        stderr="driver unavailable\n",
    )
    monkeypatch.setattr("scaleguard.doctor.subprocess.run", lambda *_args, **_kwargs: completed)

    inventory, error = _gpu_inventory()

    assert inventory == []
    assert error == "driver unavailable"


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (subprocess.TimeoutExpired("nvidia-smi", 10), "timed out"),
        (OSError("cannot fork"), "cannot execute nvidia-smi"),
    ],
)
def test_gpu_inventory_turns_launch_failures_into_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
    message: str,
) -> None:
    monkeypatch.setattr("scaleguard.doctor.shutil.which", lambda _name: "/fake/nvidia-smi")

    def fail(*_args: object, **_kwargs: object) -> None:
        raise error

    monkeypatch.setattr("scaleguard.doctor.subprocess.run", fail)

    inventory, detail = _gpu_inventory()

    assert inventory == []
    assert detail is not None
    assert message in detail


def test_doctor_fake_mode_is_cpu_safe_and_does_not_require_upstreams(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "pyproject.toml").touch()
    monkeypatch.setattr(
        "scaleguard.doctor._gpu_inventory",
        lambda: ([], "nvidia-smi is not installed"),
    )

    checks = {check.name: check for check in run_doctor(PipelineConfig(), tmp_path)}

    assert checks["python"].status == "pass"
    assert checks["project_root"].status == "pass"
    assert checks["4kagent_checkout"].status == "skip"
    assert checks["coz_checkout"].status == "skip"
    assert checks["coz_sr_lora"].status == "skip"
    assert checks["gpu_inventory"].status == "skip"
    assert checks["quality_gate"].status == "warn"


def test_doctor_requires_two_adequate_gpus_and_configured_upstream_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "pyproject.toml").touch()
    fourkagent = tmp_path / "4KAgent"
    coz = tmp_path / "Chain-of-Zoom"
    (fourkagent / ".git").mkdir(parents=True)
    (coz / ".git").mkdir(parents=True)
    sr_lora = tmp_path / "sr.pkl"
    vae = tmp_path / "vae.pt"
    vlm = tmp_path / "vlm"
    sr_lora.touch()
    vae.touch()
    vlm.mkdir()
    config = PipelineConfig(
        fourkagent=FourKAgentConfig(mode="upstream", checkout=fourkagent),
        coz=CoZConfig(
            mode="persistent",
            checkout=coz,
            sr_lora_path=sr_lora,
            vae_path=vae,
            vlm_lora_path=vlm,
        ),
        controller=ControllerConfig(accept_unvalidated_quality_proxy=True),
    )
    monkeypatch.setattr(
        "scaleguard.doctor._gpu_inventory",
        lambda: ([("0", 24564, "GPU-a"), ("1", 24564, "GPU-b")], None),
    )

    checks = {check.name: check for check in run_doctor(config, tmp_path)}

    assert checks["python"].status == "pass"
    assert checks["project_root"].status == "pass"
    assert checks["4kagent_checkout"].status == "pass"
    assert checks["coz_checkout"].status == "pass"
    assert checks["coz_sr_lora"].status == "pass"
    assert checks["coz_vae"].status == "pass"
    assert checks["coz_vlm_lora"].status == "pass"
    assert checks["depictqa_service"].status == "fail"
    assert checks["gpu_inventory"].status == "pass"
    assert checks["quality_gate"].status == "warn"
    assert "GPU 0: 24564 MiB GPU-a" in checks["gpu_inventory"].detail


def test_doctor_accepts_an_existing_managed_depictqa_cwd_and_executable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "pyproject.toml").touch()
    checkout = tmp_path / "third_party" / "checkouts" / "4KAgent"
    (checkout / ".git").mkdir(parents=True)
    service_cwd = checkout / "DepictQA"
    service_cwd.mkdir()
    executable = tmp_path / ".runtime" / "envs" / "depictqa" / "bin" / "python"
    executable.parent.mkdir(parents=True)
    executable.touch()
    config = PipelineConfig(
        fourkagent=FourKAgentConfig(
            mode="upstream",
            checkout=Path("third_party/checkouts/4KAgent"),
            depictqa_command=(
                "{project_root}/.runtime/envs/depictqa/bin/python",
                "src/evaluate.py",
                "--checkout",
                "{checkout}",
            ),
            depictqa_cwd=Path("third_party/checkouts/4KAgent/DepictQA"),
        ),
    )
    monkeypatch.setattr(
        "scaleguard.doctor._gpu_inventory",
        lambda: ([], "nvidia-smi is not installed"),
    )

    checks = {check.name: check for check in run_doctor(config, tmp_path)}

    assert checks["4kagent_checkout"].status == "pass"
    assert checks["depictqa_service"].status == "pass"
    assert f"cwd={service_cwd}" in checks["depictqa_service"].detail
    assert f"executable={executable}" in checks["depictqa_service"].detail
    assert checks["quality_gate"].status == "warn"


def test_doctor_fails_an_upstream_mode_with_inadequate_gpu_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "pyproject.toml").touch()
    config = PipelineConfig(
        coz=CoZConfig(mode="upstream", checkout=tmp_path / "missing"),
        controller=ControllerConfig(accept_unvalidated_quality_proxy=True),
    )
    monkeypatch.setattr(
        "scaleguard.doctor._gpu_inventory",
        lambda: ([("0", 12000, "GPU-small")], None),
    )

    checks = {check.name: check for check in run_doctor(config, tmp_path)}

    assert checks["coz_checkout"].status == "fail"
    assert checks["coz_sr_lora"].status == "fail"
    assert checks["gpu_inventory"].status == "fail"


def test_doctor_checks_the_configured_gpu_indices_not_inventory_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "pyproject.toml").touch()
    config = PipelineConfig(
        coz=CoZConfig(
            mode="persistent",
            visible_devices="2,3",
            checkout=tmp_path / "coz",
        ),
        controller=ControllerConfig(accept_unvalidated_quality_proxy=True),
    )
    monkeypatch.setattr(
        "scaleguard.doctor._gpu_inventory",
        lambda: (
            [
                ("0", 24_564, "GPU-a"),
                ("1", 24_564, "GPU-b"),
                ("2", 22_000, "GPU-c"),
                ("3", 24_564, "GPU-d"),
            ],
            None,
        ),
    )

    checks = {check.name: check for check in run_doctor(config, tmp_path)}

    assert checks["gpu_inventory"].status == "fail"
    assert "selected=2,3" in checks["gpu_inventory"].detail
    assert "below_23000_MiB=2" in checks["gpu_inventory"].detail
