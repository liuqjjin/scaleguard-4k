from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_writer_module() -> ModuleType:
    path = Path(__file__).parents[2] / "scripts" / "autodl" / "_write_preflight_receipt.py"
    spec = importlib.util.spec_from_file_location("scaleguard_preflight_writer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load preflight writer module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_WRITER = _load_writer_module()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_preflight_writer_does_not_follow_fixed_temporary_or_output_symlinks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "must-not-change.txt"
    target.write_text("sentinel\n", encoding="utf-8")
    fixed_temporary = tmp_path / ".runtime-preflight.json.tmp"
    fixed_temporary.symlink_to(target)
    output = tmp_path / "runtime-preflight.json"
    config = tmp_path / "runtime.yaml"
    config.write_text("runtime: {}\n", encoding="utf-8")
    document = {"schema_version": 2, "status": "passed"}
    validated: list[Path] = []

    def validate(
        receipt_path: Path,
        *,
        config_path: Path | None,
        project_root: Path,
    ) -> dict[str, object]:
        assert receipt_path.parent == tmp_path
        assert receipt_path != fixed_temporary
        assert receipt_path.is_file()
        assert not receipt_path.is_symlink()
        assert receipt_path.stat().st_mode & 0o777 == 0o600
        assert json.loads(receipt_path.read_text(encoding="utf-8")) == document
        assert config_path == config
        assert project_root == _WRITER.PROJECT_ROOT
        validated.append(receipt_path)
        return {
            "runtime_evidence_verified": True,
            "runtime_preflight_sha256": _sha256(receipt_path),
        }

    monkeypatch.setattr(_WRITER, "validate_runtime_preflight", validate)
    safe_output = _WRITER._new_output_path(output)
    _WRITER._write_validated_receipt(safe_output, document, config=config)

    assert target.read_text(encoding="utf-8") == "sentinel\n"
    assert fixed_temporary.is_symlink()
    assert validated
    assert not validated[0].exists()
    assert json.loads(output.read_text(encoding="utf-8")) == document
    assert output.stat().st_mode & 0o777 == 0o600

    linked_output = tmp_path / "linked-preflight.json"
    linked_output.symlink_to(target)
    with pytest.raises(_WRITER.RuntimePreflightError, match="already exists"):
        _WRITER._new_output_path(linked_output)
    assert target.read_text(encoding="utf-8") == "sentinel\n"


def test_preflight_writer_rejects_a_temporary_replaced_after_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "runtime-preflight.json"
    config = tmp_path / "runtime.yaml"
    config.write_text("runtime: {}\n", encoding="utf-8")

    def replace_after_validation(
        receipt_path: Path,
        **_kwargs: object,
    ) -> dict[str, object]:
        validated_digest = _sha256(receipt_path)
        replacement = receipt_path.with_name(f"{receipt_path.name}.replacement")
        replacement.write_text('{"status":"forged"}\n', encoding="utf-8")
        replacement.replace(receipt_path)
        return {"runtime_preflight_sha256": validated_digest}

    monkeypatch.setattr(
        _WRITER,
        "validate_runtime_preflight",
        replace_after_validation,
    )
    with pytest.raises(_WRITER.RuntimePreflightError, match="changed after validation"):
        _WRITER._write_validated_receipt(
            output,
            {"schema_version": 2, "status": "passed"},
            config=config,
        )

    assert not output.exists()
    assert list(tmp_path.glob(".runtime-preflight.json.*.tmp")) == []


def test_preflight_writer_publishes_only_the_validated_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "runtime-preflight.json"
    config = tmp_path / "runtime.yaml"
    config.write_text("runtime: {}\n", encoding="utf-8")
    real_link = _WRITER.os.link

    def validate(receipt_path: Path, **_kwargs: object) -> dict[str, object]:
        return {"runtime_preflight_sha256": _sha256(receipt_path)}

    def swap_then_link(
        source: Path,
        destination: Path,
        *,
        follow_symlinks: bool,
    ) -> None:
        replacement = source.with_name(f"{source.name}.replacement")
        replacement.write_text('{"status":"forged"}\n', encoding="utf-8")
        replacement.replace(source)
        real_link(source, destination, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(_WRITER, "validate_runtime_preflight", validate)
    monkeypatch.setattr(_WRITER.os, "link", swap_then_link)
    with pytest.raises(_WRITER.RuntimePreflightError, match="not the validated inode"):
        _WRITER._write_validated_receipt(
            output,
            {"schema_version": 2, "status": "passed"},
            config=config,
        )

    assert not output.exists()
    assert list(tmp_path.glob(".runtime-preflight.json.*.tmp")) == []


def test_preflight_writer_rehashes_the_validated_inode_after_linking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "runtime-preflight.json"
    config = tmp_path / "runtime.yaml"
    config.write_text("runtime: {}\n", encoding="utf-8")
    real_link = _WRITER.os.link

    def validate(receipt_path: Path, **_kwargs: object) -> dict[str, object]:
        return {"runtime_preflight_sha256": _sha256(receipt_path)}

    def mutate_then_link(
        source: Path,
        destination: Path,
        *,
        follow_symlinks: bool,
    ) -> None:
        source.write_text('{"status":"forged-in-place"}\n', encoding="utf-8")
        real_link(source, destination, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(_WRITER, "validate_runtime_preflight", validate)
    monkeypatch.setattr(_WRITER.os, "link", mutate_then_link)
    with pytest.raises(_WRITER.RuntimePreflightError, match="changed during publication"):
        _WRITER._write_validated_receipt(
            output,
            {"schema_version": 2, "status": "passed"},
            config=config,
        )

    assert not output.exists()
    assert list(tmp_path.glob(".runtime-preflight.json.*.tmp")) == []


def test_preflight_writer_rechecks_metadata_after_the_second_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "runtime-preflight.json"
    config = tmp_path / "runtime.yaml"
    config.write_text("runtime: {}\n", encoding="utf-8")
    real_lstat = _WRITER.os.lstat
    output_checks = 0

    def validate(receipt_path: Path, **_kwargs: object) -> dict[str, object]:
        return {"runtime_preflight_sha256": _sha256(receipt_path)}

    def mutate_before_final_lstat(path: Path) -> object:
        nonlocal output_checks
        if path == output:
            output_checks += 1
            if output_checks == 2:
                output.write_text('{"status":"forged-after-digest"}\n', encoding="utf-8")
        return real_lstat(path)

    monkeypatch.setattr(_WRITER, "validate_runtime_preflight", validate)
    monkeypatch.setattr(_WRITER.os, "lstat", mutate_before_final_lstat)
    with pytest.raises(
        _WRITER.RuntimePreflightError,
        match="changed during publication",
    ):
        _WRITER._write_validated_receipt(
            output,
            {"schema_version": 2, "status": "passed"},
            config=config,
        )

    assert output_checks >= 2
    assert not output.exists()
    assert list(tmp_path.glob(".runtime-preflight.json.*.tmp")) == []


def test_preflight_writer_never_clobbers_a_concurrently_created_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "runtime-preflight.json"
    config = tmp_path / "runtime.yaml"
    config.write_text("runtime: {}\n", encoding="utf-8")

    def create_output(receipt_path: Path, **_kwargs: object) -> dict[str, object]:
        output.write_text("concurrent owner\n", encoding="utf-8")
        return {"runtime_preflight_sha256": _sha256(receipt_path)}

    monkeypatch.setattr(_WRITER, "validate_runtime_preflight", create_output)
    with pytest.raises(_WRITER.RuntimePreflightError, match="appeared concurrently"):
        _WRITER._write_validated_receipt(
            output,
            {"schema_version": 2, "status": "passed"},
            config=config,
        )

    assert output.read_text(encoding="utf-8") == "concurrent owner\n"
    assert list(tmp_path.glob(".runtime-preflight.json.*.tmp")) == []


def test_preflight_writer_does_not_publish_a_receipt_that_fails_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "runtime-preflight.json"
    config = tmp_path / "runtime.yaml"
    config.write_text("runtime: {}\n", encoding="utf-8")

    def reject(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise _WRITER.RuntimePreflightError("rejected fixture")

    monkeypatch.setattr(_WRITER, "validate_runtime_preflight", reject)
    with pytest.raises(_WRITER.RuntimePreflightError, match="rejected fixture"):
        _WRITER._write_validated_receipt(
            output,
            {"schema_version": 2, "status": "passed"},
            config=config,
        )

    assert not output.exists()
    assert list(tmp_path.glob(".runtime-preflight.json.*.tmp")) == []
