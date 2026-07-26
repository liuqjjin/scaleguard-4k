from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from scaleguard.evaluation.evidence import (
    EvaluationEvidenceError,
    load_json_object,
    optional_finite_number,
    require_finite_number,
    require_text,
    sha256_file,
    verify_artifact,
    write_json_atomic,
)


def test_evidence_helpers_reject_missing_files_and_non_object_json(tmp_path: Path) -> None:
    with pytest.raises(EvaluationEvidenceError, match="cannot read evidence file"):
        sha256_file(tmp_path / "missing.json")

    document = tmp_path / "list.json"
    document.write_text("[]", encoding="utf-8")
    with pytest.raises(EvaluationEvidenceError, match="must be a JSON object"):
        load_json_object(document, kind="receipt")


def test_atomic_evidence_write_removes_a_failed_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "receipt.json"

    def fail_dump(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("serialization failed")

    monkeypatch.setattr("scaleguard.evaluation.evidence.json.dump", fail_dump)

    with pytest.raises(RuntimeError, match="serialization failed"):
        write_json_atomic(destination, {"status": "passed"})

    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: require_text({}, "name", context="sample"), "non-empty string"),
        (
            lambda: require_finite_number({"value": True}, "value", context="sample"),
            "must be numeric",
        ),
        (
            lambda: require_finite_number({"value": float("inf")}, "value", context="sample"),
            "must be finite",
        ),
    ],
)
def test_evidence_scalar_helpers_fail_closed(
    call: Callable[[], object],
    message: str,
) -> None:
    with pytest.raises(EvaluationEvidenceError, match=message):
        call()

    assert optional_finite_number({"value": None}, "value", context="sample") is None


def test_artifact_evidence_rejects_invalid_missing_and_changed_files(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    with pytest.raises(EvaluationEvidenceError, match="must be an artifact object"):
        verify_artifact(
            None,
            context="artifact",
            manifest_path=manifest,
            artifact_root=None,
        )
    with pytest.raises(EvaluationEvidenceError, match="not a lowercase SHA256 digest"):
        verify_artifact(
            {"path": "missing.bin", "sha256": "invalid"},
            context="artifact",
            manifest_path=manifest,
            artifact_root=None,
        )
    with pytest.raises(EvaluationEvidenceError, match="unavailable for hash verification"):
        verify_artifact(
            {"path": "missing.bin", "sha256": "0" * 64},
            context="artifact",
            manifest_path=manifest,
            artifact_root=None,
        )

    changed = tmp_path / "changed.bin"
    changed.write_bytes(b"changed")
    with pytest.raises(EvaluationEvidenceError, match="SHA256 mismatch"):
        verify_artifact(
            {"path": str(changed), "sha256": "0" * 64},
            context="artifact",
            manifest_path=manifest,
            artifact_root=None,
        )
