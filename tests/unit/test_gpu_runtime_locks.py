from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("environment", "direct_pins"),
    [
        (
            "4kagent",
            {
                "torch==2.10.0+cu126",
                "torchvision==0.25.0+cu126",
                "torchaudio==2.10.0+cu126",
            },
        ),
        (
            "depictqa",
            {"torch==2.10.0+cu126", "torchvision==0.25.0+cu126"},
        ),
        (
            "coz",
            {
                "torch==2.10.0+cu126",
                "torchvision==0.25.0+cu126",
                "triton==3.6.0",
            },
        ),
    ],
)
def test_gpu_locks_pin_the_patched_cuda_stack(
    environment: str,
    direct_pins: set[str],
) -> None:
    lock_root = ROOT / "environments" / environment
    direct = lock_root.joinpath("requirements.lock").read_text(encoding="utf-8")
    resolved = lock_root.joinpath("requirements.resolved.lock").read_text(encoding="utf-8")

    assert direct_pins <= set(direct.splitlines())
    assert "--python-platform x86_64-manylinux_2_28" in resolved.splitlines()[1]
    assert "--torch-backend cu126" in resolved.splitlines()[1]
    assert "torch==2.10.0+cu126 \\" in resolved
    assert "torchvision==0.25.0+cu126 \\" in resolved
    assert "triton==3.6.0 \\" in resolved
    if environment == "4kagent":
        assert "torchaudio==2.10.0+cu126 \\" in resolved


def test_bootstrap_and_gpu_preflight_match_the_cuda_lock() -> None:
    bootstrap = ROOT.joinpath("scripts/bootstrap/autodl.sh").read_text(encoding="utf-8")
    gpu_check = ROOT.joinpath("scripts/autodl/check_gpu.sh").read_text(encoding="utf-8")

    assert bootstrap.count("https://download.pytorch.org/whl/cu126") == 3
    assert "download.pytorch.org/whl/cu118" not in bootstrap
    assert 'sg_min_driver="${SCALEGUARD_MIN_NVIDIA_DRIVER:-560.28.03}"' in gpu_check
