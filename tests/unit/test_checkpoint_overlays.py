from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load(relative: str, module_name: str) -> ModuleType:
    path = Path(__file__).resolve().parents[2] / relative
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    specification = importlib.util.spec_from_file_location(module_name, path)
    if specification is None or specification.loader is None:
        raise AssertionError(f"cannot load overlay: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class FakeTorch:
    def __init__(self, version: str) -> None:
        self.__version__ = version
        self.calls: list[dict[str, object]] = []

    def load(self, _path: str, **kwargs: object) -> dict[str, str]:
        self.calls.append(dict(kwargs))
        return {"state": "safe"}


@pytest.mark.parametrize(
    ("relative", "module_name"),
    [
        (
            "third_party/overlays/4kagent/run_native_restoration.py",
            "scaleguard_fourk_overlay",
        ),
        (
            "third_party/overlays/chain-of-zoom/coz_session_worker.py",
            "scaleguard_coz_overlay",
        ),
    ],
)
def test_gpu_overlays_require_fixed_torch_and_force_weights_only(
    relative: str,
    module_name: str,
) -> None:
    overlay = _load(relative, module_name)
    with pytest.raises(RuntimeError, match=r"PyTorch >=2\.10\.0"):
        overlay._install_safe_torch_load(FakeTorch("2.9.1"))

    torch = FakeTorch("2.10.0+cu126")
    overlay._install_safe_torch_load(torch)
    assert torch.load("checkpoint.pt") == {"state": "safe"}
    assert torch.calls == [{"weights_only": True}]
    with pytest.raises(RuntimeError, match=r"weights_only=False"):
        torch.load("checkpoint.pt", weights_only=False)


def test_coz_patch_cleanup_never_expands_glob_characters(tmp_path: Path) -> None:
    overlay = _load(
        "third_party/overlays/chain-of-zoom/coz_session_worker.py",
        "scaleguard_coz_cleanup_overlay",
    )
    run = tmp_path / "runs" / "*"
    sibling = tmp_path / "runs" / "victim"
    run.mkdir(parents=True)
    sibling.mkdir()
    output = run / "candidate.png"
    own_patch = run / "candidate_patch0.png"
    victim_patch = sibling / "candidate_patch0.png"
    output.write_bytes(b"result")
    own_patch.write_bytes(b"own")
    victim_patch.write_bytes(b"victim")

    overlay.remove_patch_outputs(output)

    assert not own_patch.exists()
    assert victim_patch.read_bytes() == b"victim"
