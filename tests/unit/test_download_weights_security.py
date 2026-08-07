from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _downloader() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "scaleguard_test_download_weights_security",
        ROOT / "scripts" / "autodl" / "_download_weights.py",
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _artifact(**updates: object) -> dict[str, object]:
    artifact: dict[str, object] = {
        "id": "model",
        "provider": "huggingface",
        "repo_id": "owner/model",
        "revision": "a" * 40,
        "destination": "models/model",
    }
    artifact.update(updates)
    return artifact


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"repo_id": "-owner/model"}, "repo_id"),
        ({"repo_id": "owner/model\nforged"}, "repo_id"),
        ({"include": ["--local-dir", "/tmp/escape"]}, "unsafe Hugging Face include"),
        ({"include": ["../escape"]}, "unsafe Hugging Face include"),
        ({"exclude": ["safe\n--revision"]}, "unsafe Hugging Face exclude"),
        ({"destination": "models\nforged"}, "control character"),
    ],
)
def test_huggingface_manifest_rejects_option_and_path_injection(
    tmp_path: Path,
    updates: dict[str, object],
    message: str,
) -> None:
    downloader = _downloader()
    root = tmp_path / "weights"
    root.mkdir()

    with pytest.raises(downloader.ManifestError, match=message):
        downloader.validate_artifact(_artifact(**updates), root)


def test_huggingface_patterns_are_emitted_as_repeated_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    downloader = _downloader()
    root = tmp_path / "weights"
    root.mkdir()
    artifact = downloader.validate_artifact(
        _artifact(include=["config.json", "weights/*.safetensors"], exclude=["*.bin"]),
        root,
    )
    artifact["_root"] = root.resolve()
    observed: list[str] = []

    def run(command: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
        assert check is True
        observed.extend(command)
        destination = Path(artifact["_destination"])
        (destination / "config.json").write_text("{}\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(downloader, "huggingface_command", lambda: ["hf", "download"])
    monkeypatch.setattr(downloader.subprocess, "run", run)

    downloader.download_huggingface(artifact)

    assert observed.count("--include") == 2
    assert observed.count("--exclude") == 1
    assert observed[observed.index("--include") + 1] == "config.json"
    assert observed[observed.index("--exclude") + 1] == "*.bin"


def test_huggingface_download_rejects_destination_symlink_substitution(
    tmp_path: Path,
) -> None:
    downloader = _downloader()
    root = tmp_path / "weights"
    root.mkdir()
    artifact = downloader.validate_artifact(_artifact(), root)
    artifact["_root"] = root.resolve()
    destination = Path(artifact["_destination"])
    destination.parent.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    destination.symlink_to(outside, target_is_directory=True)

    with pytest.raises(downloader.ManifestError, match="symlink"):
        downloader.download_huggingface(artifact)


def test_checked_in_weight_lock_has_disjoint_destinations(tmp_path: Path) -> None:
    downloader = _downloader()
    root = (tmp_path / "weights").resolve()
    document = json.loads((ROOT / "weights-lock.json").read_text(encoding="utf-8"))
    artifacts = [downloader.validate_artifact(item, root) for item in document["artifacts"]]

    downloader.validate_destination_layout(artifacts)
