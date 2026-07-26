#!/usr/bin/env python3
"""Launch 4KAgent's DepictQA evaluation app from immutable, separate checkouts."""

from __future__ import annotations

import argparse
import re
import runpy
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depictqa-checkout", type=Path, required=True)
    parser.add_argument("--app-script", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--weights-root", type=Path, required=True)
    parser.add_argument("--session-dir", type=Path, required=True)
    return parser.parse_args()


def _install_safe_torch_load(torch_module: Any) -> None:
    match = re.match(r"^(\d+)\.(\d+)", str(torch_module.__version__))
    if match is None or tuple(map(int, match.groups())) < (2, 10):
        raise RuntimeError(
            "DepictQA requires PyTorch >=2.10.0 because earlier weights_only "
            "loaders are affected by CVE-2026-24747"
        )
    original_load = torch_module.load

    def safe_load(*load_args: object, **load_kwargs: object) -> object:
        if load_kwargs.get("weights_only") is False:
            raise RuntimeError("unsafe torch.load(weights_only=False) is forbidden")
        load_kwargs["weights_only"] = True
        return original_load(*load_args, **load_kwargs)

    torch_module.load = safe_load


def _install_loopback_flask(flask_class: type[Any]) -> None:
    original_run = flask_class.run

    def loopback_run(
        app: Any,
        host: str | None = None,
        port: int | None = None,
        debug: bool | None = None,
        **options: object,
    ) -> object:
        del host, port, debug
        options["use_reloader"] = False
        return original_run(
            app,
            host="127.0.0.1",
            port=5001,
            debug=False,
            **options,
        )

    flask_class.run = loopback_run


def _install_packaging_compatibility() -> None:
    try:
        __import__("pkg_resources")
    except ModuleNotFoundError as error:
        if error.name != "pkg_resources":
            raise
        compatibility = ModuleType("pkg_resources")
        compatibility.packaging = __import__("packaging")  # type: ignore[attr-defined]
        sys.modules["pkg_resources"] = compatibility


def _install_inference_only_model_package(source_root: Path) -> None:
    source_root = source_root.resolve()
    model_path = source_root / "model"
    if model_path.is_symlink() or not (model_path / "depictqa.py").is_file():
        raise RuntimeError(f"DepictQA inference package is missing or unsafe: {model_path}")
    model_root = model_path.resolve()
    if not model_root.is_relative_to(source_root):
        raise RuntimeError(f"DepictQA inference package escapes its source root: {model_root}")
    if "model" in sys.modules:
        raise RuntimeError("DepictQA model package was imported before inference isolation")
    package = ModuleType("model")
    package.__package__ = "model"
    package.__path__ = [str(model_root)]  # type: ignore[attr-defined]
    sys.modules["model"] = package


def main() -> int:
    args = parse_args()
    checkout = args.depictqa_checkout.resolve()
    app_script = args.app_script.resolve()
    base_config = args.base_config.resolve()
    weights = args.weights_root.resolve()
    for label, path in {
        "DepictQA checkout": checkout / "src" / "model" / "depictqa.py",
        "4KAgent app script": app_script,
        "4KAgent base config": base_config,
        "CLIP vision encoder": weights / "ViT-L-14.pt",
        "Vicuna model": weights / "vicuna-7b-v1.5",
        "degradation delta": weights / "delta" / "degra_eval.pt",
    }.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} is missing: {path}")

    document = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("model"), dict):
        raise ValueError(f"invalid 4KAgent DepictQA configuration: {base_config}")
    model: dict[str, Any] = document["model"]
    model["vision_encoder_path"] = str(weights / "ViT-L-14.pt")
    model["llm_path"] = str(weights / "vicuna-7b-v1.5")
    model["delta_path"] = str(weights / "delta" / "degra_eval.pt")

    args.session_dir.mkdir(parents=True, exist_ok=True)
    generated_config = args.session_dir / "depictqa-eval.yaml"
    generated_config.write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )
    import torch
    from flask import Flask

    _install_safe_torch_load(torch)
    _install_loopback_flask(Flask)
    _install_packaging_compatibility()
    source_root = checkout / "src"
    _install_inference_only_model_package(source_root)
    sys.path.insert(0, str(source_root))
    sys.argv = [str(app_script), "--cfg", str(generated_config)]
    runpy.run_path(str(app_script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
