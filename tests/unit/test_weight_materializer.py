from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MATERIALIZER = PROJECT_ROOT / "scripts" / "weights" / "materialize.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _add_bytes(archive: tarfile.TarFile, name: str, content: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(content)
    archive.addfile(info, io.BytesIO(content))


def _receipt(weight_root: Path, archive: Path, dq: Path, destination: Path) -> None:
    document = {
        "schema_version": 1,
        "status": "passed",
        "git_commit": "a" * 40,
        "weight_root": str(weight_root.resolve()),
        "artifacts": [
            {
                "id": "4kagent-toolbox-archive",
                "provider": "huggingface",
                "status": "downloaded",
                "required": True,
                "destination": "4kagent/toolbox",
                "files": [
                    {
                        "path": archive.name,
                        "size_bytes": archive.stat().st_size,
                        "sha256": _sha256(archive),
                    }
                ],
            },
            {
                "id": "4kagent-depictqa-dq495k",
                "provider": "huggingface",
                "status": "downloaded",
                "required": True,
                "destination": "4kagent/depictqa/dq495k",
                "files": [
                    {
                        "path": dq.name,
                        "size_bytes": dq.stat().st_size,
                        "sha256": _sha256(dq),
                    }
                ],
            },
        ],
    }
    destination.write_text(json.dumps(document), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "weights"
    toolbox = root / "4kagent" / "toolbox"
    toolbox.mkdir(parents=True)
    archive = toolbox / "4KAgent_toolbox_pretrained_ckpts.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        _add_bytes(handle, "bundle/pretrained_ckpts/Restormer/model.pth", b"model")
        _add_bytes(
            handle,
            "bundle/executor/denoising/tools/SwinIR/model_zoo/locked.pth",
            b"swinir",
        )
    dq = root / "4kagent" / "depictqa" / "dq495k" / "ckpt.pt"
    dq.parent.mkdir(parents=True)
    dq.write_bytes(b"depictqa")
    receipt = tmp_path / "weights-receipt.json"
    _receipt(root, archive, dq, receipt)
    return root, receipt


def test_materializer_extracts_and_verifies_idempotently(tmp_path: Path) -> None:
    root, receipt = _fixture(tmp_path)
    output = tmp_path / "materialization.json"
    command = [
        sys.executable,
        str(MATERIALIZER),
        "--weights-root",
        str(root),
        "--receipt",
        str(receipt),
        "--output",
        str(output),
    ]
    subprocess.run(command, check=True)
    marker = root / ".scaleguard-materialization.json"
    assert json.loads(output.read_text()) == json.loads(marker.read_text())
    runtime_root = root / "4kagent" / "runtime" / "toolbox-root"
    assert (runtime_root / "pretrained_ckpts" / "Restormer" / "model.pth").read_bytes() == b"model"
    assert (
        root / "4kagent" / "runtime" / "depictqa" / "delta" / "DQ495K.pt"
    ).read_bytes() == b"depictqa"

    renewed = json.loads(receipt.read_text())
    renewed["completed_at_utc"] = "2026-07-27T12:00:00+00:00"
    receipt.write_text(json.dumps(renewed), encoding="utf-8")
    subprocess.run(command, check=True)
    rebound = json.loads(marker.read_text())
    assert rebound["source_weights_receipt_sha256"] == _sha256(receipt)

    verify_output = tmp_path / "verified.json"
    subprocess.run([*command, "--verify-only", "--output", str(verify_output)], check=True)
    assert verify_output.read_bytes() == marker.read_bytes()


def test_materializer_rejects_archive_links(tmp_path: Path) -> None:
    root = tmp_path / "weights"
    toolbox = root / "4kagent" / "toolbox"
    toolbox.mkdir(parents=True)
    archive = toolbox / "4KAgent_toolbox_pretrained_ckpts.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        _add_bytes(handle, "pretrained_ckpts/model.pth", b"model")
        link = tarfile.TarInfo("pretrained_ckpts/escape")
        link.type = tarfile.SYMTYPE
        link.linkname = "../../outside"
        handle.addfile(link)
    dq = root / "4kagent" / "depictqa" / "dq495k" / "ckpt.pt"
    dq.parent.mkdir(parents=True)
    dq.write_bytes(b"depictqa")
    receipt = tmp_path / "weights-receipt.json"
    _receipt(root, archive, dq, receipt)

    result = subprocess.run(
        [
            sys.executable,
            str(MATERIALIZER),
            "--weights-root",
            str(root),
            "--receipt",
            str(receipt),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "member type is not allowed" in result.stderr


def test_materializer_rejects_duplicate_download_receipt_status(tmp_path: Path) -> None:
    root, receipt = _fixture(tmp_path)
    original = receipt.read_text(encoding="utf-8").lstrip()
    receipt.write_text('{"status":"failed",' + original[1:], encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(MATERIALIZER),
            "--weights-root",
            str(root),
            "--receipt",
            str(receipt),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "duplicate JSON object key 'status'" in result.stderr
