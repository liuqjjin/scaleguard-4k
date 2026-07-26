from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


def _overlay() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[2]
        / "third_party"
        / "overlays"
        / "4kagent"
        / "serve_depictqa_eval.py"
    )
    specification = importlib.util.spec_from_file_location("scaleguard_depictqa_overlay", path)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class FakeTorch:
    def __init__(self, version: str) -> None:
        self.__version__ = version
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def load(self, *args: object, **kwargs: object) -> object:
        self.calls.append((args, kwargs))
        return {"state": "safe"}


def test_depictqa_overlay_rejects_vulnerable_torch() -> None:
    with pytest.raises(RuntimeError, match=r"PyTorch >=2\.10\.0"):
        _overlay()._install_safe_torch_load(FakeTorch("2.9.1+cu126"))


def test_depictqa_overlay_forces_weights_only_and_rejects_opt_out() -> None:
    torch = FakeTorch("2.10.0+cu126")
    _overlay()._install_safe_torch_load(torch)

    assert torch.load("delta.pt", map_location="cpu") == {"state": "safe"}
    assert torch.calls == [(("delta.pt",), {"map_location": "cpu", "weights_only": True})]
    with pytest.raises(RuntimeError, match=r"weights_only=False"):
        torch.load("malicious.pt", weights_only=False)


def test_depictqa_overlay_overrides_public_debug_server_arguments() -> None:
    calls: list[dict[str, Any]] = []

    class FakeFlask:
        def run(self, **kwargs: object) -> str:
            calls.append(dict(kwargs))
            return "stopped"

    overlay = _overlay()
    overlay._install_loopback_flask(FakeFlask)

    assert FakeFlask().run(host="0.0.0.0", port=9000, debug=True, use_reloader=True) == "stopped"
    assert calls == [
        {
            "host": "127.0.0.1",
            "port": 5001,
            "debug": False,
            "use_reloader": False,
        }
    ]


def test_depictqa_overlay_bypasses_training_only_package_initializer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "src"
    model_root = source_root / "model"
    model_root.mkdir(parents=True)
    (model_root / "depictqa.py").write_text("class DepictQA: pass\n", encoding="utf-8")
    (model_root / "__init__.py").write_text(
        "raise RuntimeError('training-only initializer executed')\n",
        encoding="utf-8",
    )
    monkeypatch.delitem(sys.modules, "model", raising=False)

    try:
        _overlay()._install_inference_only_model_package(source_root)

        package = sys.modules["model"]
        assert package.__package__ == "model"
        assert package.__path__ == [str(model_root.resolve())]  # type: ignore[attr-defined]
    finally:
        sys.modules.pop("model", None)


def test_depictqa_overlay_rejects_a_symlinked_model_package(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "src"
    external_model = tmp_path / "external-model"
    external_model.mkdir(parents=True)
    (external_model / "depictqa.py").write_text("class DepictQA: pass\n", encoding="utf-8")
    source_root.mkdir()
    (source_root / "model").symlink_to(external_model, target_is_directory=True)

    with pytest.raises(RuntimeError, match="missing or unsafe"):
        _overlay()._install_inference_only_model_package(source_root)
