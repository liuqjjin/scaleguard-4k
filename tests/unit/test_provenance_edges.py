from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import scaleguard.provenance as provenance
from scaleguard.provenance import RuntimePreflightError

_DIGEST = "a" * 64
_COMMIT = "b" * 40
_NOW = "2026-07-27T00:00:00Z"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _inventory(path: Path) -> list[dict[str, object]]:
    return [
        {
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": provenance.sha256(path),
        }
    ]


def test_regular_evidence_rejects_missing_non_object_and_symlink_inputs(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(RuntimePreflightError, match="cannot inspect fixture"):
        provenance.load_regular_file_snapshot(missing, "fixture")

    array = tmp_path / "array.json"
    array.write_text("[]\n", encoding="utf-8")
    with pytest.raises(RuntimePreflightError, match="must be a JSON object"):
        provenance.load_evidence_snapshot(array, "fixture")

    target = tmp_path / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    alias = tmp_path / "alias.json"
    alias.symlink_to(target)
    with pytest.raises(RuntimePreflightError, match="must not be a symbolic link"):
        provenance.load_evidence_snapshot(alias, "fixture")


def test_regular_evidence_rejects_descriptor_identity_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = tmp_path / "evidence.json"
    replacement = tmp_path / "replacement.json"
    evidence.write_text("{}\n", encoding="utf-8")
    replacement.write_text('{"replacement":true}\n', encoding="utf-8")
    replacement_identity = replacement.stat()
    real_fstat = os.fstat

    def swapped_fstat(descriptor: int) -> os.stat_result:
        real_fstat(descriptor)
        return replacement_identity

    monkeypatch.setattr(provenance.os, "fstat", swapped_fstat)

    with pytest.raises(RuntimePreflightError, match="changed while it was being opened"):
        provenance.load_regular_file_snapshot(evidence, "fixture")


def test_regular_evidence_wraps_fstat_and_read_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}\n", encoding="utf-8")
    real_fstat = os.fstat

    def blocked_fstat(_descriptor: int) -> os.stat_result:
        raise OSError("fixture fstat denial")

    monkeypatch.setattr(provenance.os, "fstat", blocked_fstat)
    with pytest.raises(RuntimePreflightError, match="cannot inspect opened fixture"):
        provenance.load_regular_file_snapshot(evidence, "fixture")

    monkeypatch.setattr(provenance.os, "fstat", real_fstat)
    real_fdopen = os.fdopen

    class BrokenReader:
        def __init__(self, descriptor: int) -> None:
            self._handle = real_fdopen(descriptor, "rb")

        def __enter__(self) -> BrokenReader:
            return self

        def __exit__(self, *_args: object) -> None:
            self._handle.close()

        def read(self, _size: int) -> bytes:
            raise OSError("fixture read denial")

    monkeypatch.setattr(
        provenance.os,
        "fdopen",
        lambda descriptor, _mode: BrokenReader(descriptor),
    )
    with pytest.raises(RuntimePreflightError, match="cannot read and hash fixture"):
        provenance.load_regular_file_snapshot(evidence, "fixture")


def test_snapshot_verification_rejects_final_identity_change_and_recheck_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = tmp_path / "evidence.json"
    other = tmp_path / "other.json"
    evidence.write_text("{}\n", encoding="utf-8")
    other.write_text('{"other":true}\n', encoding="utf-8")

    with pytest.raises(RuntimePreflightError, match="changed while it was being read"):
        provenance._verify_evidence_snapshot(
            evidence,
            "fixture",
            opened=evidence.stat(),
            completed=other.stat(),
        )

    def blocked_lstat(_path: Path) -> os.stat_result:
        raise OSError("fixture recheck denial")

    monkeypatch.setattr(Path, "lstat", blocked_lstat)
    with pytest.raises(RuntimePreflightError, match="cannot recheck fixture"):
        provenance._verify_evidence_snapshot(
            evidence,
            "fixture",
            opened=other.stat(),
            completed=other.stat(),
        )


@pytest.mark.parametrize("value", [None, "", 7])
def test_evidence_path_requires_a_nonempty_string(value: object) -> None:
    with pytest.raises(RuntimePreflightError, match="has no path"):
        provenance._evidence_path(value, "fixture")


def test_evidence_path_rejects_a_symlink_before_resolution(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_text("fixture\n", encoding="utf-8")
    alias = tmp_path / "alias"
    alias.symlink_to(target)

    with pytest.raises(RuntimePreflightError, match="must not be a symbolic link"):
        provenance._evidence_path(str(alias), "fixture")


@pytest.mark.parametrize("value", [None, "", "relative/file", "/tmp/../tmp/file"])
def test_absolute_lexical_path_rejects_ambiguous_paths(value: object) -> None:
    with pytest.raises(RuntimePreflightError, match=r"has no path|lexical absolute"):
        provenance._absolute_lexical_path(value, "fixture")


@pytest.mark.parametrize(
    ("size", "digest"),
    [
        (True, _DIGEST),
        (-1, _DIGEST),
        (0, "not-a-digest"),
    ],
)
def test_current_file_identity_rejects_invalid_schema(
    tmp_path: Path,
    size: object,
    digest: object,
) -> None:
    evidence = tmp_path / "evidence"
    evidence.write_bytes(b"fixture")

    with pytest.raises(RuntimePreflightError, match="invalid file identity"):
        provenance._current_regular_identity(
            evidence,
            size=size,
            digest=digest,
            label="fixture",
        )


def test_current_file_identity_rejects_changed_bytes(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    evidence.write_bytes(b"fixture")

    with pytest.raises(RuntimePreflightError, match="changed after its environment audit"):
        provenance._current_regular_identity(
            evidence,
            size=evidence.stat().st_size,
            digest="0" * 64,
            label="fixture",
        )


def test_managed_python_aliases_accept_only_links_resolving_inside_root(
    tmp_path: Path,
) -> None:
    managed = tmp_path / "managed"
    target = managed / "base/python"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"python")
    alias = managed / "bin/python"
    alias.parent.mkdir()
    alias.symlink_to("../base/python")

    assert provenance._managed_aliases(managed, alias, label="fixture") == [
        {
            "path": "bin/python",
            "target": "../base/python",
            "resolved": str(target),
        }
    ]

    outside = tmp_path / "outside"
    outside.write_bytes(b"outside")
    alias.unlink()
    alias.symlink_to(outside)
    with pytest.raises(RuntimePreflightError, match="alias target escapes"):
        provenance._managed_aliases(managed, alias, label="fixture")


def test_managed_python_aliases_reject_missing_root_and_candidate_escape(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(RuntimePreflightError, match="root is missing or unsafe"):
        provenance._managed_aliases(missing, missing / "python", label="fixture")

    managed = tmp_path / "managed"
    managed.mkdir()
    outside = tmp_path / "outside"
    outside.write_bytes(b"outside")
    with pytest.raises(RuntimePreflightError, match="executable escapes"):
        provenance._managed_aliases(managed, outside, label="fixture")


def test_managed_python_aliases_reject_indirect_escape_and_broken_link(
    tmp_path: Path,
) -> None:
    managed = tmp_path / "managed"
    inside = managed / "inside"
    inside.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.write_bytes(b"outside")
    bridge = inside / "bridge"
    bridge.symlink_to(outside)
    alias = managed / "python"
    alias.symlink_to("inside/bridge")

    with pytest.raises(RuntimePreflightError, match="resolves outside"):
        provenance._managed_aliases(managed, alias, label="fixture")

    alias.unlink()
    alias.symlink_to("missing")
    with pytest.raises(RuntimePreflightError, match="cannot be inspected"):
        provenance._managed_aliases(managed, alias, label="fixture")


@pytest.mark.parametrize(
    "value",
    [None, "", "not-a-date", "2026-07-27T00:00:00+08:00"],
)
def test_timestamp_requires_parseable_utc(value: object) -> None:
    with pytest.raises(RuntimePreflightError, match=r"ISO-8601|must use UTC"):
        provenance._timestamp(value, "fixture")


@pytest.mark.parametrize(
    ("responses", "message"),
    [
        (
            [
                subprocess.CompletedProcess(["git"], 1, "", ""),
            ],
            "requires a committed Git HEAD",
        ),
        (
            [
                subprocess.CompletedProcess(["git"], 0, f"{_COMMIT}\n", ""),
                subprocess.CompletedProcess(["git"], 1, "", ""),
            ],
            "cannot inspect the Git worktree",
        ),
    ],
)
def test_clean_commit_check_rejects_git_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    responses: list[subprocess.CompletedProcess[str]],
    message: str,
) -> None:
    pending = iter(responses)
    monkeypatch.setattr(provenance.subprocess, "run", lambda *_args, **_kwargs: next(pending))

    with pytest.raises(RuntimePreflightError, match=message):
        provenance.require_clean_git_commit(tmp_path)


def test_current_lock_set_rejects_missing_or_symlinked_lock(tmp_path: Path) -> None:
    missing = tmp_path / "lock.txt"
    with pytest.raises(RuntimePreflightError, match="lock mismatch"):
        provenance._validate_current_lock_set(
            {"lock.txt": _DIGEST},
            expected=("lock.txt",),
            project_root=tmp_path,
            label="fixture",
        )

    target = tmp_path / "target.txt"
    target.write_text("lock\n", encoding="utf-8")
    missing.symlink_to(target)
    with pytest.raises(RuntimePreflightError, match="lock mismatch"):
        provenance._validate_current_lock_set(
            {"lock.txt": provenance.sha256(target)},
            expected=("lock.txt",),
            project_root=tmp_path,
            label="fixture",
        )


@pytest.mark.parametrize("value", [None, "", ".", "../escape", "/absolute", r"a\b", "a//b"])
def test_locked_relative_path_rejects_unsafe_forms(value: object) -> None:
    with pytest.raises(RuntimePreflightError, match="not a safe relative path"):
        provenance._locked_relative_path(value, "fixture")


def test_safe_destination_rejects_empty_traversal_and_symlink_components(
    tmp_path: Path,
) -> None:
    root = tmp_path / "weights"
    root.mkdir()
    for value in (None, "", "."):
        with pytest.raises(RuntimePreflightError, match="has no destination"):
            provenance._safe_destination(root, value, "fixture")
    for value in ("../escape", "/absolute"):
        with pytest.raises(RuntimePreflightError, match="unsafe destination"):
            provenance._safe_destination(root, value, "fixture")

    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "linked").symlink_to(outside)
    with pytest.raises(RuntimePreflightError, match="contains a symlink"):
        provenance._safe_destination(root, "linked/model.bin", "fixture")


def test_current_inventory_rejects_symlinks_missing_paths_and_nested_links(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(RuntimePreflightError, match="destination is missing"):
        provenance._current_inventory(
            missing,
            label="fixture",
            ignore_cache_metadata=False,
        )

    target = tmp_path / "target"
    target.write_bytes(b"fixture")
    alias = tmp_path / "alias"
    alias.symlink_to(target)
    with pytest.raises(RuntimePreflightError, match="destination is a symbolic link"):
        provenance._current_inventory(
            alias,
            label="fixture",
            ignore_cache_metadata=False,
        )

    directory = tmp_path / "directory"
    directory.mkdir()
    (directory / "alias").symlink_to(target)
    with pytest.raises(RuntimePreflightError, match="inventory contains a symlink"):
        provenance._current_inventory(
            directory,
            label="fixture",
            ignore_cache_metadata=False,
        )


def test_current_inventory_ignores_only_cache_metadata(tmp_path: Path) -> None:
    destination = tmp_path / "model"
    model = destination / "weights.bin"
    cache = destination / ".cache/metadata.json"
    git = destination / ".git/config"
    model.parent.mkdir()
    cache.parent.mkdir()
    git.parent.mkdir()
    model.write_bytes(b"model")
    cache.write_bytes(b"cache")
    git.write_bytes(b"git")

    inventory = provenance._current_inventory(
        destination,
        label="fixture",
        ignore_cache_metadata=True,
    )

    assert [entry["path"] for entry in inventory] == ["weights.bin"]


@pytest.mark.parametrize("value", [None, "", "relative/path"])
def test_weights_root_requires_an_absolute_existing_directory(
    tmp_path: Path,
    value: object,
) -> None:
    with pytest.raises(RuntimePreflightError, match=r"no weight root|unsafe weight root"):
        provenance._weights_root(value)

    if value == "relative/path":
        missing = tmp_path / "missing"
        with pytest.raises(RuntimePreflightError, match="weight root is missing"):
            provenance._weights_root(str(missing.resolve()))


def test_weights_root_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target)

    with pytest.raises(RuntimePreflightError, match="unsafe weight root"):
        provenance._weights_root(str(alias))


def test_locked_file_inventory_accepts_mixed_digest_declarations() -> None:
    paths, hashes = provenance._locked_file_inventory(
        {
            "files": [
                "plain.bin",
                {"path": "verified.bin", "sha256": _DIGEST},
            ]
        },
        "fixture",
    )

    assert paths == ["plain.bin", "verified.bin"]
    assert hashes == {"verified.bin": _DIGEST}


@pytest.mark.parametrize(
    ("files", "message"),
    [
        (None, "has no declared files"),
        ([], "has no declared files"),
        ([17], "is malformed"),
        ([{"path": "model.bin", "sha256": "bad"}], "invalid SHA-256"),
        (["model.bin", "model.bin"], "duplicate file paths"),
    ],
)
def test_locked_file_inventory_rejects_malformed_declarations(
    files: object,
    message: str,
) -> None:
    with pytest.raises(RuntimePreflightError, match=message):
        provenance._locked_file_inventory({"files": files}, "fixture")


@pytest.mark.parametrize(
    ("artifact", "message"),
    [
        (
            {"files": ["model.bin"]},
            "has no known_sha256 declaration",
        ),
        (
            {
                "files": ["one.bin", "two.bin"],
                "known_sha256": _DIGEST,
            },
            "invalid known_sha256",
        ),
        (
            {
                "files": [{"path": "model.bin", "sha256": _DIGEST}],
                "known_sha256": "b" * 64,
            },
            "conflicting locked SHA-256",
        ),
        (
            {
                "files": ["model.bin"],
                "known_sha256": {"model.bin": "bad"},
            },
            "invalid known_sha256",
        ),
        (
            {
                "files": [{"path": "model.bin", "sha256": _DIGEST}],
                "known_sha256": {"model.bin": "b" * 64},
            },
            "conflicting locked SHA-256",
        ),
        (
            {
                "files": ["model.bin"],
                "known_sha256": {"undeclared.bin": _DIGEST},
            },
            "outside its declared inventory",
        ),
        (
            {
                "files": ["model.bin"],
                "known_sha256": 42,
            },
            "invalid known_sha256",
        ),
    ],
)
def test_locked_expected_hashes_rejects_ambiguous_or_conflicting_hashes(
    artifact: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(RuntimePreflightError, match=message):
        provenance._locked_expected_hashes(artifact, "fixture")


def _manual_lock(
    *,
    artifact_id: str = "manual-model",
    destination: str = "models/model.bin",
    required: bool = True,
) -> dict[str, object]:
    return {
        "id": artifact_id,
        "provider": "manual",
        "required": required,
        "destination": destination,
        "files": ["model.bin"],
        "known_sha256": _DIGEST,
        "verify_on_download": True,
    }


def _https_lock(**updates: object) -> dict[str, object]:
    artifact: dict[str, object] = {
        "id": "https-model",
        "provider": "https",
        "required": True,
        "destination": "models/model.bin",
        "files": ["model.bin"],
        "known_sha256": _DIGEST,
        "verify_on_download": True,
        "url": "https://weights.invalid/model.bin",
        "sha256": _DIGEST,
    }
    artifact.update(updates)
    return artifact


@pytest.mark.parametrize(
    "document",
    [
        {},
        {"schema_version": 2, "artifacts": [_manual_lock()]},
        {"schema_version": 1, "artifacts": []},
        {"schema_version": 1, "artifacts": ["malformed"]},
        {
            "schema_version": 1,
            "artifacts": [{**_manual_lock(), "id": "contains spaces"}],
        },
        {
            "schema_version": 1,
            "artifacts": [{**_manual_lock(), "provider": "ftp"}],
        },
        {
            "schema_version": 1,
            "artifacts": [{**_manual_lock(), "required": 1}],
        },
        {
            "schema_version": 1,
            "artifacts": [{**_manual_lock(), "verify_on_download": False}],
        },
    ],
)
def test_weights_lock_rejects_invalid_schema_and_artifact_contracts(
    document: dict[str, object],
) -> None:
    with pytest.raises(RuntimePreflightError):
        provenance._locked_weight_artifacts(document)


def test_weights_lock_rejects_duplicate_and_nested_destinations() -> None:
    duplicate = [
        _manual_lock(artifact_id="one"),
        _manual_lock(artifact_id="two"),
    ]
    with pytest.raises(RuntimePreflightError, match="duplicates another artifact destination"):
        provenance._locked_weight_artifacts({"schema_version": 1, "artifacts": duplicate})

    nested = [
        _manual_lock(artifact_id="one", destination="models"),
        _manual_lock(artifact_id="two", destination="models/model.bin"),
    ]
    with pytest.raises(RuntimePreflightError, match="destination is nested"):
        provenance._locked_weight_artifacts({"schema_version": 1, "artifacts": nested})


@pytest.mark.parametrize(
    "updates",
    [
        {"url": "http://weights.invalid/model.bin"},
        {"url": "https://user:secret@weights.invalid/model.bin"},
        {"url": "https://weights.invalid/model.bin#mutable"},
        {"url": "https://[invalid/model.bin"},
        {"destination": "models/model", "files": ["model"]},
        {
            "files": ["other.bin"],
            "known_sha256": _DIGEST,
        },
        {"sha256": "b" * 64},
    ],
)
def test_https_weight_lock_requires_one_immutable_file_identity(
    updates: dict[str, object],
) -> None:
    with pytest.raises(RuntimePreflightError, match="immutable HTTPS identity"):
        provenance._locked_weight_artifacts(
            {
                "schema_version": 1,
                "artifacts": [_https_lock(**updates)],
            }
        )


def test_non_huggingface_weight_lock_rejects_mutable_revision() -> None:
    with pytest.raises(RuntimePreflightError, match="invalid source revision"):
        provenance._locked_weight_artifacts(
            {
                "schema_version": 1,
                "artifacts": [{**_manual_lock(), "revision": "main"}],
            }
        )


@pytest.mark.parametrize(
    "inventory",
    [
        None,
        [],
        ["malformed"],
        [{"path": "", "size_bytes": 1, "sha256": _DIGEST}],
        [{"path": "file", "size_bytes": True, "sha256": _DIGEST}],
        [{"path": "file", "size_bytes": -1, "sha256": _DIGEST}],
        [{"path": "file", "size_bytes": 1, "sha256": "bad"}],
        [
            {"path": "file", "size_bytes": 1, "sha256": _DIGEST},
            {"path": "file", "size_bytes": 1, "sha256": _DIGEST},
        ],
    ],
)
def test_inventory_hashes_rejects_noncanonical_records(inventory: object) -> None:
    with pytest.raises(RuntimePreflightError, match="file inventory"):
        provenance._inventory_hashes(inventory, "fixture")


def test_manual_locked_hashes_require_authenticated_upstream_digest() -> None:
    inventory = [{"path": "model.bin", "size_bytes": 1, "sha256": _DIGEST}]
    with pytest.raises(RuntimePreflightError, match="authentication claim"):
        provenance._validate_locked_artifact_hashes(
            locked=_manual_lock(),
            artifact={
                "known_hashes_verified": ["model.bin"],
                "verify_on_download": True,
                "upstream_digest_authenticated": False,
            },
            recorded_files=inventory,
            current_files=inventory,
            context="fixture",
        )


@dataclass(frozen=True)
class MaterializationEvidence:
    project_root: Path
    weights_root: Path
    materialization: Path
    marker: Path
    weights_receipt: Path
    weights_lock: dict[str, Any]
    weights_lock_digest: str
    records: tuple[dict[str, str], dict[str, str], dict[str, str]]


def _materialization_evidence(tmp_path: Path) -> MaterializationEvidence:
    project_root = tmp_path / "project"
    project_root.mkdir()
    weights_root = tmp_path / "weights"
    source = weights_root / "models/model.bin"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"model")
    source_digest = provenance.sha256(source)
    layout_file = weights_root / "runtime/model.bin"
    layout_file.parent.mkdir(parents=True)
    layout_file.write_bytes(b"layout")

    weights_lock: dict[str, Any] = {
        "schema_version": 1,
        "artifacts": [
            {
                **_manual_lock(),
                "known_sha256": source_digest,
            }
        ],
    }
    weights_lock_digest = "c" * 64
    weights_receipt = tmp_path / "artifacts/weights-receipt.json"
    _write_json(
        weights_receipt,
        {
            "schema_version": 1,
            "status": "passed",
            "completed_at_utc": _NOW,
            "source_manifest": "weights-lock.json",
            "source_manifest_sha256": weights_lock_digest,
            "git_commit": _COMMIT,
            "weight_root": str(weights_root.resolve()),
            "optional_artifacts_requested": False,
            "artifacts": [
                {
                    "id": "manual-model",
                    "provider": "manual",
                    "status": "recorded_manual",
                    "required": True,
                    "destination": "models/model.bin",
                    "files": _inventory(source),
                    "known_hashes_verified": ["model.bin"],
                    "verify_on_download": True,
                    "upstream_digest_authenticated": True,
                }
            ],
            "manual_gates": [],
        },
    )
    marker_payload = {
        "schema_version": 1,
        "status": "passed",
        "completed_at_utc": _NOW,
        "source_weights_receipt_sha256": provenance.sha256(weights_receipt),
        "weights_root": str(weights_root.resolve()),
        "source_git_commit": _COMMIT,
        "layouts": [
            {
                "id": "runtime-model",
                "source_artifact_id": "manual-model",
                "destination": "runtime/model.bin",
                "files": _inventory(layout_file),
            }
        ],
        "checkout_mutations": False,
        "errors": [],
    }
    marker = weights_root / ".scaleguard-materialization.json"
    materialization = tmp_path / "receipts/materialization.json"
    _write_json(marker, marker_payload)
    _write_json(materialization, marker_payload)
    records = (
        {"path": str(materialization), "sha256": provenance.sha256(materialization)},
        {"path": str(marker), "sha256": provenance.sha256(marker)},
        {"path": str(weights_receipt), "sha256": provenance.sha256(weights_receipt)},
    )
    return MaterializationEvidence(
        project_root=project_root,
        weights_root=weights_root,
        materialization=materialization,
        marker=marker,
        weights_receipt=weights_receipt,
        weights_lock=weights_lock,
        weights_lock_digest=weights_lock_digest,
        records=records,
    )


def _validate_materialization(evidence: MaterializationEvidence) -> None:
    provenance._validate_materialization(
        *evidence.records,
        project_root=evidence.project_root,
        commit=_COMMIT,
        weights_lock=evidence.weights_lock,
        weights_lock_digest=evidence.weights_lock_digest,
    )


def _rebind_materialization(evidence: MaterializationEvidence) -> MaterializationEvidence:
    marker = json.loads(evidence.marker.read_text(encoding="utf-8"))
    marker["source_weights_receipt_sha256"] = provenance.sha256(evidence.weights_receipt)
    _write_json(evidence.marker, marker)
    _write_json(evidence.materialization, marker)
    records = (
        {
            "path": str(evidence.materialization),
            "sha256": provenance.sha256(evidence.materialization),
        },
        {"path": str(evidence.marker), "sha256": provenance.sha256(evidence.marker)},
        {
            "path": str(evidence.weights_receipt),
            "sha256": provenance.sha256(evidence.weights_receipt),
        },
    )
    return MaterializationEvidence(
        project_root=evidence.project_root,
        weights_root=evidence.weights_root,
        materialization=evidence.materialization,
        marker=evidence.marker,
        weights_receipt=evidence.weights_receipt,
        weights_lock=evidence.weights_lock,
        weights_lock_digest=evidence.weights_lock_digest,
        records=records,
    )


def test_manual_materialization_accepts_authenticated_locked_bytes(tmp_path: Path) -> None:
    evidence = _materialization_evidence(tmp_path)

    result = provenance._validate_materialization(
        *evidence.records,
        project_root=evidence.project_root,
        commit=_COMMIT,
        weights_lock=evidence.weights_lock,
        weights_lock_digest=evidence.weights_lock_digest,
    )

    assert result[0] == evidence.weights_root.resolve()
    assert result[4]["artifacts"] == {
        "manual-model": str(evidence.weights_root / "models/model.bin")
    }


def test_materialization_requires_all_three_evidence_records(tmp_path: Path) -> None:
    evidence = _materialization_evidence(tmp_path)

    with pytest.raises(RuntimePreflightError, match="missing weight evidence"):
        provenance._validate_materialization(
            None,
            evidence.records[1],
            evidence.records[2],
            project_root=evidence.project_root,
            commit=_COMMIT,
            weights_lock=evidence.weights_lock,
            weights_lock_digest=evidence.weights_lock_digest,
        )


def test_materialization_rejects_stale_attempt_digest(tmp_path: Path) -> None:
    evidence = _materialization_evidence(tmp_path)
    records = (
        {**evidence.records[0], "sha256": "0" * 64},
        evidence.records[1],
        evidence.records[2],
    )

    with pytest.raises(RuntimePreflightError, match="verification receipt is stale"):
        provenance._validate_materialization(
            *records,
            project_root=evidence.project_root,
            commit=_COMMIT,
            weights_lock=evidence.weights_lock,
            weights_lock_digest=evidence.weights_lock_digest,
        )


def test_materialization_rejects_marker_outside_fixed_weight_root(tmp_path: Path) -> None:
    evidence = _materialization_evidence(tmp_path)
    alternate = tmp_path / "alternate-marker.json"
    alternate.write_bytes(evidence.marker.read_bytes())
    records = (
        evidence.records[0],
        {"path": str(alternate), "sha256": provenance.sha256(alternate)},
        evidence.records[2],
    )

    with pytest.raises(RuntimePreflightError, match="unexpected fixed marker"):
        provenance._validate_materialization(
            *records,
            project_root=evidence.project_root,
            commit=_COMMIT,
            weights_lock=evidence.weights_lock,
            weights_lock_digest=evidence.weights_lock_digest,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("status", "failed", "did not pass"),
        ("layouts", [], "no verified layouts"),
        ("layouts", ["malformed"], "layout 0 is malformed"),
    ],
)
def test_materialization_rejects_invalid_marker_semantics(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    evidence = _materialization_evidence(tmp_path)
    marker = json.loads(evidence.marker.read_text(encoding="utf-8"))
    marker[field] = value
    _write_json(evidence.marker, marker)
    _write_json(evidence.materialization, marker)
    evidence = _rebind_materialization(evidence)

    with pytest.raises(RuntimePreflightError, match=message):
        _validate_materialization(evidence)


def test_materialization_rejects_duplicate_layout_identity(tmp_path: Path) -> None:
    evidence = _materialization_evidence(tmp_path)
    marker = json.loads(evidence.marker.read_text(encoding="utf-8"))
    marker["layouts"].append(dict(marker["layouts"][0]))
    _write_json(evidence.marker, marker)
    _write_json(evidence.materialization, marker)
    evidence = _rebind_materialization(evidence)

    with pytest.raises(RuntimePreflightError, match="layout 1 is malformed"):
        _validate_materialization(evidence)


def test_materialization_requires_a_recorded_layout_inventory(tmp_path: Path) -> None:
    evidence = _materialization_evidence(tmp_path)
    marker = json.loads(evidence.marker.read_text(encoding="utf-8"))
    marker["layouts"][0]["files"] = []
    _write_json(evidence.marker, marker)
    _write_json(evidence.materialization, marker)
    evidence = _rebind_materialization(evidence)

    with pytest.raises(RuntimePreflightError, match="has no file inventory"):
        _validate_materialization(evidence)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_manifest", "other.json", "invalid manifest binding"),
        ("artifacts", [], "no artifact records"),
        ("artifacts", ["malformed"], "artifact 0 is malformed"),
    ],
)
def test_materialization_rejects_invalid_source_receipt_structure(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    evidence = _materialization_evidence(tmp_path)
    receipt = json.loads(evidence.weights_receipt.read_text(encoding="utf-8"))
    receipt[field] = value
    _write_json(evidence.weights_receipt, receipt)
    evidence = _rebind_materialization(evidence)

    with pytest.raises(RuntimePreflightError, match=message):
        _validate_materialization(evidence)


def test_materialization_rejects_destination_claim_on_skipped_optional_artifact(
    tmp_path: Path,
) -> None:
    evidence = _materialization_evidence(tmp_path)
    optional = _manual_lock(
        artifact_id="optional-model",
        destination="optional/model.bin",
        required=False,
    )
    evidence.weights_lock["artifacts"].append(optional)
    receipt = json.loads(evidence.weights_receipt.read_text(encoding="utf-8"))
    receipt["artifacts"].append(
        {
            "id": "optional-model",
            "provider": "manual",
            "required": False,
            "status": "skipped",
            "destination": "forged/model.bin",
        }
    )
    _write_json(evidence.weights_receipt, receipt)
    evidence = _rebind_materialization(evidence)

    with pytest.raises(RuntimePreflightError, match="identity disagrees with weights lock"):
        _validate_materialization(evidence)


def _source_resolution_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    weights_root = tmp_path / "weights"
    weights_root.mkdir()
    receipt = tmp_path / "artifacts/weight-download/run/weights-receipt.json"
    _write_json(receipt, {"status": "passed"})
    marker = {
        "weights_root": str(weights_root.resolve()),
        "source_weights_receipt_sha256": provenance.sha256(receipt),
    }
    marker_path = weights_root / ".scaleguard-materialization.json"
    materialization = tmp_path / "materialization.json"
    _write_json(marker_path, marker)
    _write_json(materialization, marker)
    return materialization, marker_path, receipt


def test_materialization_source_resolution_rejects_marker_divergence(
    tmp_path: Path,
) -> None:
    materialization, marker, _receipt = _source_resolution_fixture(tmp_path)
    marker_payload = json.loads(marker.read_text(encoding="utf-8"))
    marker_payload["unexpected"] = True
    _write_json(marker, marker_payload)

    with pytest.raises(RuntimePreflightError, match="differs from the fixed marker"):
        provenance.resolve_materialization_sources(
            materialization,
            artifact_root=tmp_path / "artifacts",
        )


def test_materialization_source_resolution_requires_a_digest_bound_receipt(
    tmp_path: Path,
) -> None:
    materialization, marker, _receipt = _source_resolution_fixture(tmp_path)
    marker_payload = json.loads(marker.read_text(encoding="utf-8"))
    marker_payload["source_weights_receipt_sha256"] = None
    _write_json(marker, marker_payload)
    _write_json(materialization, marker_payload)

    with pytest.raises(RuntimePreflightError, match="has no source receipt digest"):
        provenance.resolve_materialization_sources(
            materialization,
            artifact_root=tmp_path / "artifacts",
        )


def test_materialization_source_resolution_skips_unsafe_and_nonmatching_candidates(
    tmp_path: Path,
) -> None:
    materialization, _marker, receipt = _source_resolution_fixture(tmp_path)
    receipt.write_text('{"status":"different"}\n', encoding="utf-8")
    directory_candidate = tmp_path / "artifacts/weight-download/newer/weights-receipt.json"
    directory_candidate.mkdir(parents=True)
    symlink_candidate = tmp_path / "artifacts/weight-download/latest/weights-receipt.json"
    symlink_candidate.parent.mkdir(parents=True)
    symlink_candidate.symlink_to(receipt)

    with pytest.raises(RuntimePreflightError, match="no weight download receipt matches"):
        provenance.resolve_materialization_sources(
            materialization,
            artifact_root=tmp_path / "artifacts",
        )


def test_materialization_source_resolution_strictly_decodes_matching_receipt(
    tmp_path: Path,
) -> None:
    materialization, marker, receipt = _source_resolution_fixture(tmp_path)
    receipt.write_text('{"duplicate":1,"duplicate":2}\n', encoding="utf-8")
    marker_payload = json.loads(marker.read_text(encoding="utf-8"))
    marker_payload["source_weights_receipt_sha256"] = provenance.sha256(receipt)
    _write_json(marker, marker_payload)
    _write_json(materialization, marker_payload)

    with pytest.raises(RuntimePreflightError, match="duplicate JSON object key"):
        provenance.resolve_materialization_sources(
            materialization,
            artifact_root=tmp_path / "artifacts",
        )
