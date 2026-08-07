from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).parents[2]


def _load(path: Path, module_name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(module_name, path)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_fourkagent_production_logger_replaces_image_bytes_with_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    overlay_root = ROOT / "third_party" / "overlays" / "4kagent"
    monkeypatch.syspath_prepend(str(overlay_root))
    overlay = _load(
        overlay_root / "run_native_restoration.py",
        "scaleguard_test_private_fourkagent_overlay",
    )
    base_llm = SimpleNamespace(encode_img=lambda _path: "unredacted")
    overlay._install_redacted_image_logging(base_llm)

    messages: list[str] = []

    class CaptureLogger:
        def info(self, message: str) -> None:
            messages.append(message)

    private_bytes = b"private image payload"
    image_path = tmp_path / "input.png"
    image_path.write_bytes(private_bytes)
    logger = CaptureLogger()
    logger.info(f"![image]({base_llm.encode_img(image_path)})")

    log_text = "".join(messages)
    digest = hashlib.sha256(private_bytes).hexdigest()
    encoded = base64.b64encode(private_bytes).decode("ascii")
    assert "data:image/" not in log_text.casefold()
    assert encoded not in log_text
    assert f"sha256={digest}" in log_text
    assert f"bytes={len(private_bytes)}" in log_text


def test_diagnostics_sanitizer_skips_text_with_inline_image_data(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    log = source / "workers" / "4kagent" / "raw_output" / "task" / "logs" / "llm_qa.md"
    log.parent.mkdir(parents=True)
    private_bytes = b"private image payload"
    encoded = base64.b64encode(private_bytes).decode("ascii")
    log.write_text(
        f"![image](data:image/png;charset=utf-8;base64,{encoded})\n",
        encoding="utf-8",
    )
    destination = tmp_path / "staging"
    system_log = destination / "system" / "probe.log"
    system_log.parent.mkdir(parents=True)
    system_log.write_text(
        f"<img src='DATA:IMAGE/JPEG;BASE64,{encoded}'>\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "-I",
            str(ROOT / "scripts" / "autodl" / "_sanitize_diagnostics.py"),
            str(source),
            str(destination),
            str(ROOT),
            str(tmp_path / "cache"),
        ],
        check=True,
        cwd=ROOT,
    )

    copied = (
        destination / "runs" / "workers" / "4kagent" / "raw_output" / "task" / "logs" / "llm_qa.md"
    )
    assert not copied.exists()
    assert not system_log.exists()
    summary = (destination / "collection-summary.txt").read_text(encoding="utf-8")
    assert "llm_qa.md: contains embedded image data" in summary
    assert encoded not in summary


def test_diagnostics_redacts_paths_from_more_than_one_hundred_execution_receipts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    last_private_input = ""
    last_private_output = ""
    for index in range(101):
        run = source / f"run-{index:03d}"
        run.mkdir(parents=True)
        private_input = f"/mnt/private/person-{index:03d}.png"
        private_output = f"/mnt/private/result-{index:03d}.png"
        (run / "execution.json").write_text(
            json.dumps(
                {
                    "inputs": {"input_image": {"path": private_input}},
                    "outputs": [{"path": private_output}],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (run / "worker.log").write_text(
            f"input={private_input} output={private_output}\n",
            encoding="utf-8",
        )
        last_private_input = private_input
        last_private_output = private_output

    destination = tmp_path / "staging"
    subprocess.run(
        [
            sys.executable,
            "-I",
            str(ROOT / "scripts" / "autodl" / "_sanitize_diagnostics.py"),
            str(source),
            str(destination),
            str(ROOT),
            str(tmp_path / "cache"),
        ],
        check=True,
        cwd=ROOT,
    )

    copied_log = destination / "runs" / "run-100" / "worker.log"
    copied_receipt = destination / "runs" / "run-100" / "execution.json"
    for copied in (copied_log, copied_receipt):
        text = copied.read_text(encoding="utf-8")
        assert last_private_input not in text
        assert last_private_output not in text
        assert "$PRIVATE_INPUT_OR_OUTPUT" in text
