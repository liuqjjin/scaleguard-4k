from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from scaleguard.provenance import (
    BOOTSTRAP_LOCK_PATHS,
    ENVIRONMENT_LOCK_PATHS,
    ENVIRONMENT_RUNTIME_IMPORTS,
    FOURKAGENT_AUDITED_OVERRIDES,
    LOCK_PATHS,
    RuntimePreflightError,
    require_clean_git_commit,
    resolve_materialization_sources,
    sha256,
    validate_runtime_preflight,
)

_NOW = "2026-07-27T00:00:00Z"


@dataclass(frozen=True)
class RuntimeEvidence:
    root: Path
    config: Path
    bootstrap: Path
    materialization: Path
    marker: Path
    weights_receipt: Path
    weights_root: Path
    artifact_root: Path
    source_file: Path
    layout_file: Path
    preflight: Path
    commit: str


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_clean_git_check_does_not_pass_ambient_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "promotion-secret")
    monkeypatch.setenv("GITHUB_TOKEN", "github-secret")
    observed_environments: list[dict[str, str]] = []
    commit = "a" * 40

    def run(
        args: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        environment = kwargs.get("env")
        assert isinstance(environment, dict)
        observed_environments.append(environment)
        stdout = f"{commit}\n" if "rev-parse" in args else ""
        return subprocess.CompletedProcess(args=args, returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr("scaleguard.provenance.subprocess.run", run)

    assert require_clean_git_commit(tmp_path) == commit
    assert len(observed_environments) == 2
    assert all("OPENAI_API_KEY" not in env for env in observed_environments)
    assert all("GITHUB_TOKEN" not in env for env in observed_environments)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _file_inventory(path: Path) -> list[dict[str, object]]:
    return [
        {
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    ]


@pytest.fixture
def runtime_evidence(tmp_path: Path) -> RuntimeEvidence:
    root = tmp_path / "project"
    root.mkdir()
    _git(root, "init", "-q")
    (root / ".gitignore").write_text(".runtime/\n.venv/\n", encoding="utf-8")

    config = root / "configs" / "runtime.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("runtime:\n  device: cuda:0\n", encoding="utf-8")
    all_locks = dict.fromkeys((*LOCK_PATHS, *BOOTSTRAP_LOCK_PATHS))
    for relative in all_locks:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        content = (
            "0.11.16\n"
            if relative == "environments/uv.version"
            else (f"fixture lock: {relative}\n")
        )
        path.write_text(content, encoding="utf-8")
    _write_json(
        root / "weights-lock.json",
        {
            "schema_version": 1,
            "artifacts": [
                {
                    "id": "model-https",
                    "provider": "https",
                    "required": True,
                    "destination": "downloads/model.bin",
                    "url": "https://weights.invalid/model.bin",
                },
                {
                    "id": "model-hf",
                    "provider": "huggingface",
                    "required": False,
                    "destination": "downloads/hf",
                    "repo_id": "scaleguard/fixture-model",
                    "revision": "1" * 40,
                },
            ],
        },
    )

    _git(root, "add", ".gitignore", "configs/runtime.yaml", *all_locks)
    _git(
        root,
        "-c",
        "user.name=ScaleGuard Tests",
        "-c",
        "user.email=tests@scaleguard.invalid",
        "-c",
        "commit.gpgSign=false",
        "commit",
        "-qm",
        "fixture",
    )
    commit = _git(root, "rev-parse", "HEAD")

    python_executable = root / ".runtime" / "python" / "python3.10"
    python_executable.parent.mkdir(parents=True)
    python_executable.write_text("fixture interpreter\n", encoding="utf-8")
    environment_records: dict[str, dict[str, str]] = {}
    for name, lock_paths in ENVIRONMENT_LOCK_PATHS.items():
        path = root / ".runtime" / "receipts" / f"{name}.json"
        package = f"{name}-fixture"
        status = "passed_with_audited_override" if name == "4kagent" else "passed"
        _write_json(
            path,
            {
                "schema_version": 1,
                "name": name,
                "status": status,
                "created_at_utc": _NOW,
                "python": {
                    "executable": str(python_executable.resolve()),
                    "version": "3.10.18",
                    "implementation": "CPython",
                    "platform": "Linux-6.8.0-x86_64",
                },
                "locks": [
                    {
                        "path": str((root / relative).resolve()),
                        "sha256": sha256(root / relative),
                        "pinned_packages": 1,
                    }
                    for relative in lock_paths
                ],
                "expected_packages": {package: "1.0"},
                "packages": {package: "1.0"},
                "runtime_imports": [
                    {"module": module, "symbols": list(symbols)}
                    for module, symbols in ENVIRONMENT_RUNTIME_IMPORTS[name]
                ],
                "audited_overrides": (
                    list(FOURKAGENT_AUDITED_OVERRIDES) if name == "4kagent" else []
                ),
                "issues": [],
            },
        )
        environment_records[name] = {
            "path": f".runtime/receipts/{name}.json",
            "sha256": sha256(path),
            "status": status,
        }

    bootstrap = root / ".runtime" / "receipts" / "bootstrap.json"
    _write_json(
        bootstrap,
        {
            "schema_version": 1,
            "status": "passed",
            "created_at_utc": _NOW,
            "project_commit": commit,
            "python_version": "3.10.18",
            "uv_version": "0.11.16",
            "platform": {
                "system": "Linux",
                "machine": "x86_64",
                "glibc": "2.35",
            },
            "locks": {relative: sha256(root / relative) for relative in BOOTSTRAP_LOCK_PATHS},
            "environments": environment_records,
        },
    )

    weights_root = tmp_path / "weights"
    source_file = weights_root / "downloads" / "model.bin"
    source_file.parent.mkdir(parents=True)
    source_file.write_bytes(b"source weights")
    hf_source_file = weights_root / "downloads" / "hf" / "model.bin"
    hf_source_file.parent.mkdir(parents=True)
    hf_source_file.write_bytes(b"optional Hugging Face weights")
    layout_file = weights_root / "runtime" / "model.bin"
    layout_file.parent.mkdir(parents=True)
    layout_file.write_bytes(b"materialized weights")
    artifact_root = tmp_path / "artifacts" / "autodl"
    weights_receipt = (
        artifact_root / "weight-download" / "20260727T000000Z" / "weights-receipt.json"
    )
    _write_json(
        weights_receipt,
        {
            "schema_version": 1,
            "status": "passed",
            "completed_at_utc": _NOW,
            "source_manifest": "weights-lock.json",
            "source_manifest_sha256": sha256(root / "weights-lock.json"),
            "git_commit": commit,
            "weight_root": str(weights_root.resolve()),
            "optional_artifacts_requested": True,
            "artifacts": [
                {
                    "id": "model-https",
                    "provider": "https",
                    "status": "downloaded",
                    "required": True,
                    "destination": "downloads/model.bin",
                    "files": _file_inventory(source_file),
                    "url": "https://weights.invalid/model.bin",
                },
                {
                    "id": "model-hf",
                    "provider": "huggingface",
                    "status": "downloaded",
                    "required": False,
                    "destination": "downloads/hf",
                    "files": _file_inventory(hf_source_file),
                    "repo_id": "scaleguard/fixture-model",
                    "revision": "1" * 40,
                },
            ],
            "manual_gates": [],
        },
    )

    marker = weights_root / ".scaleguard-materialization.json"
    marker_document = {
        "schema_version": 1,
        "status": "passed",
        "completed_at_utc": _NOW,
        "source_weights_receipt_sha256": sha256(weights_receipt),
        "weights_root": str(weights_root.resolve()),
        "source_git_commit": commit,
        "layouts": [
            {
                "id": "runtime-model",
                "source_artifact_id": "model-https",
                "destination": "runtime/model.bin",
                "files": _file_inventory(layout_file),
            }
        ],
        "checkout_mutations": False,
        "errors": [],
    }
    _write_json(marker, marker_document)
    materialization = root / ".runtime" / "receipts" / "materialization.json"
    _write_json(materialization, marker_document)

    preflight = root / ".runtime" / "receipts" / "runtime-preflight.json"
    _write_json(
        preflight,
        {
            "schema_version": 1,
            "status": "passed",
            "created_at_utc": _NOW,
            "project_commit": commit,
            "config": {
                "path": str(config.resolve()),
                "sha256": sha256(config),
            },
            "locks": {relative: sha256(root / relative) for relative in LOCK_PATHS},
            "bootstrap": {
                "path": str(bootstrap.resolve()),
                "sha256": sha256(bootstrap),
            },
            "materialization": {
                "path": str(materialization.resolve()),
                "sha256": sha256(materialization),
            },
            "materialization_marker": {
                "path": str(marker.resolve()),
                "sha256": sha256(marker),
            },
            "source_weights_receipt": {
                "path": str(weights_receipt.resolve()),
                "sha256": sha256(weights_receipt),
            },
        },
    )
    return RuntimeEvidence(
        root=root,
        config=config,
        bootstrap=bootstrap,
        materialization=materialization,
        marker=marker,
        weights_receipt=weights_receipt,
        weights_root=weights_root,
        artifact_root=artifact_root,
        source_file=source_file,
        layout_file=layout_file,
        preflight=preflight,
        commit=commit,
    )


def _validate(evidence: RuntimeEvidence) -> dict[str, Any]:
    return validate_runtime_preflight(
        evidence.preflight,
        config_path=evidence.config,
        project_root=evidence.root,
    )


def _refresh_preflight_record(
    evidence: RuntimeEvidence,
    record_name: str,
    path: Path,
) -> None:
    preflight = _read_json(evidence.preflight)
    preflight[record_name]["sha256"] = sha256(path)
    _write_json(evidence.preflight, preflight)


def _refresh_bootstrap(evidence: RuntimeEvidence) -> None:
    _refresh_preflight_record(evidence, "bootstrap", evidence.bootstrap)


def _rebind_weight_chain(evidence: RuntimeEvidence) -> None:
    marker = _read_json(evidence.marker)
    marker["source_weights_receipt_sha256"] = sha256(evidence.weights_receipt)
    _write_json(evidence.marker, marker)
    _write_json(evidence.materialization, marker)
    _refresh_preflight_record(
        evidence,
        "source_weights_receipt",
        evidence.weights_receipt,
    )
    _refresh_preflight_record(
        evidence,
        "materialization_marker",
        evidence.marker,
    )
    _refresh_preflight_record(
        evidence,
        "materialization",
        evidence.materialization,
    )


def test_runtime_preflight_accepts_a_complete_current_evidence_chain(
    runtime_evidence: RuntimeEvidence,
) -> None:
    result = _validate(runtime_evidence)

    assert result == {
        "runtime_evidence_verified": True,
        "runtime_preflight_receipt": str(runtime_evidence.preflight.resolve()),
        "runtime_preflight_sha256": sha256(runtime_evidence.preflight),
        "bootstrap_receipt_sha256": sha256(runtime_evidence.bootstrap),
        "materialization_receipt_sha256": sha256(runtime_evidence.materialization),
        "materialization_marker_sha256": sha256(runtime_evidence.marker),
        "source_weights_receipt_sha256": sha256(runtime_evidence.weights_receipt),
        "weights_root": str(runtime_evidence.weights_root.resolve()),
        "project_commit": runtime_evidence.commit,
        "project_root": str(runtime_evidence.root.resolve()),
        "runtime_config_path": str(runtime_evidence.config.resolve()),
    }


def test_runtime_preflight_rejects_duplicate_receipt_keys(
    runtime_evidence: RuntimeEvidence,
) -> None:
    original = runtime_evidence.preflight.read_text(encoding="utf-8").lstrip()
    runtime_evidence.preflight.write_text(
        '{"status":"failed",' + original[1:],
        encoding="utf-8",
    )

    with pytest.raises(RuntimePreflightError, match="duplicate JSON object key 'status'"):
        _validate(runtime_evidence)


def test_materialization_source_resolution_supports_search_and_override(
    runtime_evidence: RuntimeEvidence,
) -> None:
    searched = resolve_materialization_sources(
        runtime_evidence.materialization,
        artifact_root=runtime_evidence.artifact_root,
    )
    overridden = resolve_materialization_sources(
        runtime_evidence.materialization,
        artifact_root=runtime_evidence.artifact_root / "unused",
        weight_receipt_override=runtime_evidence.weights_receipt,
    )

    expected = (
        runtime_evidence.marker.resolve(),
        runtime_evidence.weights_receipt.resolve(),
    )
    assert searched == expected
    assert overridden == expected


def test_runtime_preflight_requires_a_clean_git_worktree(
    runtime_evidence: RuntimeEvidence,
) -> None:
    (runtime_evidence.root / "untracked.txt").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(RuntimePreflightError, match="requires a clean Git worktree"):
        _validate(runtime_evidence)


@pytest.mark.parametrize(
    ("relative", "replacement", "message"),
    [
        ("configs/runtime.yaml", "runtime:\n  device: cpu\n", "runtime config changed"),
        ("uv.lock", "tampered\n", r"runtime preflight receipt lock mismatch: uv\.lock"),
    ],
)
def test_runtime_preflight_rechecks_tracked_inputs_even_if_git_hides_the_change(
    runtime_evidence: RuntimeEvidence,
    relative: str,
    replacement: str,
    message: str,
) -> None:
    _git(runtime_evidence.root, "update-index", "--assume-unchanged", relative)
    (runtime_evidence.root / relative).write_text(replacement, encoding="utf-8")

    with pytest.raises(RuntimePreflightError, match=message):
        _validate(runtime_evidence)


def test_runtime_preflight_rejects_stale_bootstrap_digest(
    runtime_evidence: RuntimeEvidence,
) -> None:
    bootstrap = _read_json(runtime_evidence.bootstrap)
    bootstrap["unexpected"] = "mutation"
    _write_json(runtime_evidence.bootstrap, bootstrap)

    with pytest.raises(RuntimePreflightError, match="bootstrap receipt is stale"):
        _validate(runtime_evidence)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 2, "bootstrap receipt is stale"),
        ("python_version", "3.11.9", "unexpected Python version"),
        ("uv_version", "latest", "unexpected uv version"),
        ("platform", {"system": "Darwin"}, "unexpected platform"),
    ],
)
def test_runtime_preflight_deeply_validates_bootstrap_identity(
    runtime_evidence: RuntimeEvidence,
    field: str,
    value: object,
    message: str,
) -> None:
    bootstrap = _read_json(runtime_evidence.bootstrap)
    bootstrap[field] = value
    _write_json(runtime_evidence.bootstrap, bootstrap)
    _refresh_bootstrap(runtime_evidence)

    with pytest.raises(RuntimePreflightError, match=message):
        _validate(runtime_evidence)


def test_runtime_preflight_requires_the_exact_bootstrap_lock_set(
    runtime_evidence: RuntimeEvidence,
) -> None:
    bootstrap = _read_json(runtime_evidence.bootstrap)
    bootstrap["locks"].pop("environments/bootstrap/uv.lock")
    _write_json(runtime_evidence.bootstrap, bootstrap)
    _refresh_bootstrap(runtime_evidence)

    with pytest.raises(RuntimePreflightError, match="unexpected lock set"):
        _validate(runtime_evidence)


def test_runtime_preflight_reloads_environment_receipt_content(
    runtime_evidence: RuntimeEvidence,
) -> None:
    environment = runtime_evidence.root / ".runtime" / "receipts" / "coz.json"
    document = _read_json(environment)
    document["issues"] = [{"issue": "tampered"}]
    _write_json(environment, document)
    bootstrap = _read_json(runtime_evidence.bootstrap)
    bootstrap["environments"]["coz"]["sha256"] = sha256(environment)
    _write_json(runtime_evidence.bootstrap, bootstrap)
    _refresh_bootstrap(runtime_evidence)

    with pytest.raises(RuntimePreflightError, match="environment coz receipt content mismatch"):
        _validate(runtime_evidence)


def test_runtime_preflight_rejects_stale_fixed_marker_digest(
    runtime_evidence: RuntimeEvidence,
) -> None:
    marker = _read_json(runtime_evidence.marker)
    marker["unexpected"] = "mutation"
    _write_json(runtime_evidence.marker, marker)

    with pytest.raises(RuntimePreflightError, match="fixed materialization marker is stale"):
        _validate(runtime_evidence)


def test_runtime_preflight_requires_attempt_receipt_to_equal_fixed_marker(
    runtime_evidence: RuntimeEvidence,
) -> None:
    materialization = _read_json(runtime_evidence.materialization)
    materialization["unexpected"] = "mutation"
    _write_json(runtime_evidence.materialization, materialization)
    _refresh_preflight_record(
        runtime_evidence,
        "materialization",
        runtime_evidence.materialization,
    )

    with pytest.raises(RuntimePreflightError, match="differs from the fixed marker"):
        _validate(runtime_evidence)


def test_runtime_preflight_rechecks_materialized_layout_inventory(
    runtime_evidence: RuntimeEvidence,
) -> None:
    runtime_evidence.layout_file.write_bytes(b"mutated materialized weights")

    with pytest.raises(RuntimePreflightError, match="layout 0 no longer matches"):
        _validate(runtime_evidence)


def test_runtime_preflight_rejects_stale_source_weights_digest(
    runtime_evidence: RuntimeEvidence,
) -> None:
    receipt = _read_json(runtime_evidence.weights_receipt)
    receipt["unexpected"] = "mutation"
    _write_json(runtime_evidence.weights_receipt, receipt)

    with pytest.raises(RuntimePreflightError, match="source weights receipt is stale"):
        _validate(runtime_evidence)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("status", "failed", "source weights receipt did not pass"),
        ("weight_root", "/wrong/root", "bound to another weight root"),
        ("git_commit", "0" * 40, "source weights receipt did not pass"),
        ("manual_gates", [{"id": "gated-model"}], "source weights receipt did not pass"),
    ],
)
def test_runtime_preflight_validates_source_weights_receipt_semantics(
    runtime_evidence: RuntimeEvidence,
    field: str,
    value: object,
    message: str,
) -> None:
    receipt = _read_json(runtime_evidence.weights_receipt)
    receipt[field] = value
    _write_json(runtime_evidence.weights_receipt, receipt)
    _rebind_weight_chain(runtime_evidence)

    with pytest.raises(RuntimePreflightError, match=message):
        _validate(runtime_evidence)


def test_runtime_preflight_rejects_artifact_omitted_from_source_receipt(
    runtime_evidence: RuntimeEvidence,
) -> None:
    receipt = _read_json(runtime_evidence.weights_receipt)
    receipt["artifacts"].pop()
    _write_json(runtime_evidence.weights_receipt, receipt)
    _rebind_weight_chain(runtime_evidence)

    with pytest.raises(RuntimePreflightError, match="artifact IDs disagree with weights lock"):
        _validate(runtime_evidence)


@pytest.mark.parametrize(
    ("artifact_index", "field", "value"),
    [
        (0, "provider", "manual"),
        (0, "required", False),
        (0, "destination", "downloads/forged.bin"),
        (0, "url", "https://attacker.invalid/model.bin"),
        (1, "repo_id", "attacker/forged-model"),
        (1, "revision", "0" * 40),
    ],
)
def test_runtime_preflight_rejects_locked_artifact_identity_forgery(
    runtime_evidence: RuntimeEvidence,
    artifact_index: int,
    field: str,
    value: object,
) -> None:
    receipt = _read_json(runtime_evidence.weights_receipt)
    receipt["artifacts"][artifact_index][field] = value
    _write_json(runtime_evidence.weights_receipt, receipt)
    _rebind_weight_chain(runtime_evidence)

    with pytest.raises(RuntimePreflightError, match=r"identity.*weights lock"):
        _validate(runtime_evidence)


def test_runtime_preflight_requires_every_locked_required_artifact_complete(
    runtime_evidence: RuntimeEvidence,
) -> None:
    receipt = _read_json(runtime_evidence.weights_receipt)
    receipt["artifacts"][0] = {
        "id": "model-https",
        "provider": "https",
        "status": "skipped",
        "required": True,
        "reason": "forged skip",
    }
    _write_json(runtime_evidence.weights_receipt, receipt)
    _rebind_weight_chain(runtime_evidence)

    with pytest.raises(RuntimePreflightError, match=r"completion disagrees with weights lock"):
        _validate(runtime_evidence)


def test_runtime_preflight_requires_layout_sources_to_be_complete(
    runtime_evidence: RuntimeEvidence,
) -> None:
    receipt = _read_json(runtime_evidence.weights_receipt)
    receipt["artifacts"][1] = {
        "id": "model-hf",
        "provider": "huggingface",
        "status": "skipped",
        "required": False,
        "reason": "optional artifact was not requested",
    }
    _write_json(runtime_evidence.weights_receipt, receipt)
    marker = _read_json(runtime_evidence.marker)
    marker["layouts"][0]["source_artifact_id"] = "model-hf"
    _write_json(runtime_evidence.marker, marker)
    _rebind_weight_chain(runtime_evidence)

    with pytest.raises(RuntimePreflightError, match="incomplete source artifacts"):
        _validate(runtime_evidence)


def test_runtime_preflight_rechecks_source_artifact_inventory(
    runtime_evidence: RuntimeEvidence,
) -> None:
    runtime_evidence.source_file.write_bytes(b"mutated source weights")

    with pytest.raises(RuntimePreflightError, match="no longer matches its inventory"):
        _validate(runtime_evidence)
