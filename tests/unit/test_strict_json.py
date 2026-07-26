from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

from scaleguard.strict_json import StrictJSONError, loads, loads_object

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_script(relative: str, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, PROJECT_ROOT / relative)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_strict_json_accepts_standard_nested_documents_and_bytes() -> None:
    assert loads(b'{"outer":{"items":[1,true,null]},"text":"ok"}') == {
        "outer": {"items": [1, True, None]},
        "text": "ok",
    }
    assert loads_object('{"status":"ok"}') == {"status": "ok"}


def test_strict_json_object_loader_rejects_non_object_root() -> None:
    with pytest.raises(StrictJSONError, match="must be an object"):
        loads_object("[]")


@pytest.mark.parametrize(
    "document",
    [
        '{"status":"failed","status":"passed"}',
        '{"outer":{"sha256":"trusted","sha256":"forged"}}',
        '{"items":[{"id":"first","id":"second"}]}',
    ],
)
def test_strict_json_rejects_duplicate_keys_at_every_depth(document: str) -> None:
    with pytest.raises(StrictJSONError, match="duplicate JSON object key"):
        loads(document)


@pytest.mark.parametrize("document", ['{"value":NaN}', '{"value":Infinity}', "{invalid"])
def test_strict_json_rejects_nonstandard_or_malformed_documents(document: str) -> None:
    with pytest.raises(StrictJSONError):
        loads(document)


def test_weight_lock_loader_rejects_nested_duplicate_identity(tmp_path: Path) -> None:
    downloader = _load_script(
        "scripts/autodl/_download_weights.py",
        "scaleguard_test_download_weights",
    )
    manifest = tmp_path / "weights-lock.json"
    manifest.write_text(
        '{"schema_version":1,"artifacts":[{"id":"trusted","id":"forged"}]}',
        encoding="utf-8",
    )

    with pytest.raises(downloader.ManifestError, match="duplicate JSON object key 'id'"):
        downloader.load_manifest(manifest)


def test_bootstrap_receipt_loader_rejects_duplicate_status(tmp_path: Path) -> None:
    validator = _load_script(
        "scripts/autodl/_validate_bootstrap_receipt.py",
        "scaleguard_test_validate_bootstrap_receipt",
    )
    receipt = tmp_path / "bootstrap.json"
    receipt.write_text('{"status":"failed","status":"passed"}', encoding="utf-8")

    with pytest.raises(validator.ReceiptError, match="duplicate JSON object key 'status'"):
        validator.load_snapshot(receipt, "bootstrap receipt")


def test_diagnostics_ignores_ambiguous_execution_receipts(tmp_path: Path) -> None:
    sanitizer = _load_script(
        "scripts/autodl/_sanitize_diagnostics.py",
        "scaleguard_test_sanitize_diagnostics",
    )
    receipt = tmp_path / "execution.json"
    receipt.write_text(
        '{"inputs":{"input_image":{"path":"private-a","path":"private-b"}}}',
        encoding="utf-8",
    )

    assert sanitizer.private_paths_from_execution(tmp_path) == []


@pytest.mark.parametrize(
    "document",
    [
        '{"status":"failed","status":"passed"}',
        '{"status":NaN}',
    ],
)
def test_autodl_shell_json_helper_rejects_ambiguous_values(
    tmp_path: Path,
    document: str,
) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text(document, encoding="utf-8")
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; sg_json_get "$2" status',
            "_",
            str(PROJECT_ROOT / "scripts" / "autodl" / "_common.sh"),
            str(receipt),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "cannot read strict JSON value" in result.stderr
