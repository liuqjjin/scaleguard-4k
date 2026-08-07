from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any

import pytest

import scaleguard.provenance as provenance
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
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
    gpu_preflight: Path
    runtime_environments: dict[str, Path]
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


def test_independent_environment_reaudit_rejects_drift_without_forwarding_credentials(
    runtime_evidence: RuntimeEvidence,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auditor = runtime_evidence.root / "scripts/bootstrap/audit_environment.py"
    auditor.parent.mkdir(parents=True)
    auditor.write_text("# fixture auditor\n", encoding="utf-8")
    expected = {
        name: _read_json(path) for name, path in runtime_evidence.runtime_environments.items()
    }
    observed_environments: list[dict[str, str]] = []

    def run(
        args: tuple[str, ...],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        environment = kwargs.get("env")
        assert isinstance(environment, dict)
        observed_environments.append(environment)
        output = Path(args[args.index("--output") + 1])
        _write_json(output, {"name": args[args.index("--name") + 1]})
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    def validate(
        name: str,
        record: object,
        **_kwargs: object,
    ) -> tuple[Path, dict[str, Any], str]:
        assert isinstance(record, dict)
        path = Path(str(record["path"]))
        document = json.loads(json.dumps(expected[name]))
        if name == "coz":
            document["packages"]["coz-fixture"] = "9.9"
        return path, document, sha256(path)

    monkeypatch.setenv("OPENAI_API_KEY", "promotion-secret")
    monkeypatch.setenv("GITHUB_TOKEN", "github-secret")
    monkeypatch.setattr(provenance.subprocess, "run", run)
    monkeypatch.setattr(provenance, "validate_environment_receipt", validate)

    with pytest.raises(RuntimePreflightError, match="coz differs from independent re-audit"):
        provenance._reaudit_runtime_environments(
            expected,
            project_root=runtime_evidence.root,
            receipt_parent=runtime_evidence.preflight.parent,
        )

    assert len(observed_environments) == len(ENVIRONMENT_LOCK_PATHS)
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


def _runtime_import_origin(root: Path, environment: Path, name: str, module: str) -> Path:
    from_checkout = (
        name == "scaleguard"
        or (name == "4kagent" and module.startswith(("llm.", "pipeline.", "entrypoint:")))
        or (name == "depictqa" and module.startswith("model."))
        or (name == "coz" and module == "osediff_sd3")
    )
    if not from_checkout:
        origin = environment / "lib/python3.10/site-packages/runtime_fixture.py"
    elif name == "scaleguard":
        origin = root / "src" / (module.replace(".", "/") + ".py")
    elif name == "4kagent":
        checkout = root / "third_party/checkouts/4KAgent"
        origin = (
            checkout / module.removeprefix("entrypoint:")
            if module.startswith("entrypoint:")
            else checkout / (module.replace(".", "/") + ".py")
        )
    elif name == "depictqa":
        origin = root / "third_party/dependencies/DepictQA/src" / (module.replace(".", "/") + ".py")
    else:
        origin = root / "third_party/checkouts/Chain-of-Zoom/osediff_sd3.py"
    origin.parent.mkdir(parents=True, exist_ok=True)
    origin.write_text("FIXTURE = True\n", encoding="utf-8")
    return origin.resolve()


def _environment_document(
    root: Path,
    *,
    name: str,
    lock_paths: tuple[str, ...],
    status: str,
    base_prefix: Path,
    base_executable: Path,
    stdlib_root: Path,
) -> dict[str, Any]:
    prefix = root / (".venv" if name == "scaleguard" else f".runtime/envs/{name}")
    executable = prefix / "bin/python"
    executable.parent.mkdir(parents=True)
    executable.symlink_to(base_executable)
    pyvenv = prefix / "pyvenv.cfg"
    pyvenv.write_text(f"home = {base_prefix}\n", encoding="utf-8")
    package = f"{name}-fixture"
    packages = {package: "1.0"}
    distribution = {
        "name": package,
        "version": "1.0",
        "record_path": (f"lib/python3.10/site-packages/{package}-1.0.dist-info/RECORD"),
        "file_count": 1,
        "merkle_root": hashlib.sha256(f"{name}-distribution".encode()).hexdigest(),
    }
    venv_metadata = {
        "file_count": 1,
        "merkle_root": hashlib.sha256(f"{name}-venv".encode()).hexdigest(),
    }
    interpreter = {
        "realpath": str(base_executable),
        "size_bytes": base_executable.stat().st_size,
        "sha256": sha256(base_executable),
        "pyvenv_config_path": str(pyvenv),
        "pyvenv_config_size_bytes": pyvenv.stat().st_size,
        "pyvenv_config_sha256": sha256(pyvenv),
    }
    base_runtime = {
        "prefix": str(base_prefix),
        "executable": str(base_executable),
        "executable_realpath": str(base_executable),
        "executable_size_bytes": base_executable.stat().st_size,
        "executable_sha256": sha256(base_executable),
        "executable_alias_count": 0,
        "executable_alias_merkle_root": provenance._merkle_root([]),
        "executable_aliases": [],
        "stdlib_root": str(stdlib_root),
        "stdlib_file_count": 1,
        "stdlib_merkle_root": hashlib.sha256(b"fixture-stdlib").hexdigest(),
    }
    inventory_root, inventory_count, runtime_root, runtime_count = (
        provenance._installation_merkle_roots(
            [distribution],
            venv_metadata,
            interpreter,
            base_runtime,
        )
    )
    installation = {
        "algorithm": "sha256-merkle-v1",
        "environment_root": str(prefix),
        "distribution_count": 1,
        "distribution_file_count": 1,
        "file_count": inventory_count + runtime_count,
        "merkle_root": provenance._merkle_root(
            [
                provenance._merkle_payload(
                    {
                        "kind": "venv-installation",
                        "file_count": inventory_count,
                        "merkle_root": inventory_root,
                    }
                ),
                provenance._merkle_payload(
                    {
                        "kind": "python-runtime",
                        "file_count": runtime_count,
                        "merkle_root": runtime_root,
                    }
                ),
            ]
        ),
        "distributions": [distribution],
        "venv_metadata": venv_metadata,
        "interpreter": interpreter,
        "base_runtime": base_runtime,
    }
    return {
        "schema_version": 2,
        "name": name,
        "status": status,
        "created_at_utc": _NOW,
        "python": {
            "executable": str(executable),
            "executable_realpath": str(base_executable),
            "prefix": str(prefix),
            "base_prefix": str(base_prefix),
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
        "expected_packages": packages,
        "packages": packages,
        "installation_files": installation,
        "runtime_imports": [
            {
                "module": module,
                "symbols": list(symbols),
                "origin": str(_runtime_import_origin(root, prefix, name, module)),
            }
            for module, symbols in ENVIRONMENT_RUNTIME_IMPORTS[name]
        ],
        "audited_overrides": (list(FOURKAGENT_AUDITED_OVERRIDES) if name == "4kagent" else []),
        "issues": [],
    }


@pytest.fixture
def runtime_evidence(tmp_path: Path) -> RuntimeEvidence:
    root = tmp_path / "project"
    root.mkdir()
    _git(root, "init", "-q")
    (root / ".gitignore").write_text(
        ".runtime/\n.venv/\nthird_party/\nsrc/\n",
        encoding="utf-8",
    )

    config = root / "configs" / "runtime.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("runtime:\n  device: cuda:0\n", encoding="utf-8")
    all_locks = dict.fromkeys((*LOCK_PATHS, *BOOTSTRAP_LOCK_PATHS))
    for relative in all_locks:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == "environments/uv.version":
            content = "0.11.16\n"
        elif relative == "environments/bootstrap/uv-binary.sha256":
            content = f"{'f' * 64}\n"
        elif relative == "environments/python-downloads.json":
            content = json.dumps(
                {
                    "cpython-3.10.18-linux-x86_64-gnu": {
                        "name": "cpython",
                        "arch": {"family": "x86_64", "variant": None},
                        "os": "linux",
                        "libc": "gnu",
                        "major": 3,
                        "minor": 10,
                        "patch": 18,
                        "prerelease": "",
                        "url": (
                            "https://github.com/astral-sh/"
                            "python-build-standalone/releases/download/fixture/python.tar.gz"
                        ),
                        "sha256": "e" * 64,
                        "variant": None,
                        "build": "fixture",
                    }
                }
            )
        else:
            content = f"fixture lock: {relative}\n"
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
                    "files": ["model.bin"],
                    "known_sha256": hashlib.sha256(b"source weights").hexdigest(),
                    "verify_on_download": True,
                    "url": "https://weights.invalid/model.bin",
                    "sha256": hashlib.sha256(b"source weights").hexdigest(),
                },
                {
                    "id": "model-hf",
                    "provider": "huggingface",
                    "required": False,
                    "destination": "downloads/hf",
                    "files": ["**/*"],
                    "known_sha256": None,
                    "verify_on_download": True,
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

    base_prefix = root / ".runtime/python/cpython-3.10.18"
    base_executable = base_prefix / "bin/python3.10"
    base_executable.parent.mkdir(parents=True)
    base_executable.write_text("fixture interpreter\n", encoding="utf-8")
    stdlib_root = base_prefix / "lib/python3.10"
    stdlib_root.mkdir(parents=True)
    (stdlib_root / "fixture.py").write_text("FIXTURE = True\n", encoding="utf-8")
    environment_records: dict[str, dict[str, str]] = {}
    for name, lock_paths in ENVIRONMENT_LOCK_PATHS.items():
        path = root / ".runtime" / "receipts" / f"{name}.json"
        status = "passed_with_audited_override" if name == "4kagent" else "passed"
        _write_json(
            path,
            _environment_document(
                root,
                name=name,
                lock_paths=lock_paths,
                status=status,
                base_prefix=base_prefix,
                base_executable=base_executable,
                stdlib_root=stdlib_root,
            ),
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
            "uv_binary_sha256": "f" * 64,
            "python_distribution": {
                "key": "cpython-3.10.18-linux-x86_64-gnu",
                "build": "fixture",
                "url": (
                    "https://github.com/astral-sh/"
                    "python-build-standalone/releases/download/fixture/python.tar.gz"
                ),
                "archive_sha256": "e" * 64,
            },
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
                    "known_hashes_verified": ["model.bin"],
                    "verify_on_download": True,
                    "url": "https://weights.invalid/model.bin",
                },
                {
                    "id": "model-hf",
                    "provider": "huggingface",
                    "status": "downloaded",
                    "required": False,
                    "destination": "downloads/hf",
                    "files": _file_inventory(hf_source_file),
                    "known_hashes_verified": [],
                    "verify_on_download": True,
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
    gpu_preflight = preflight.parent / "gpu-preflight" / "gpu_check.json"
    selected_gpus = [
        {
            "logical_index": index,
            "physical_index": str(index),
            "uuid": f"GPU-fixture-{index}",
            "name": "NVIDIA GeForce RTX 4090",
            "memory_total_mib": 24564,
            "driver_version": "595.71.05",
        }
        for index in range(2)
    ]
    _write_json(
        gpu_preflight,
        {
            "schema_version": 1,
            "checked_at_utc": _NOW,
            "status": "passed",
            "git_commit": commit,
            "requirements": {"minimum_gpu_count": 2},
            "cuda_visible_devices": "0,1",
            "selected_gpus": selected_gpus,
        },
    )
    runtime_environment_root = preflight.parent / "runtime-environments"
    runtime_environments: dict[str, Path] = {}
    runtime_environment_records: dict[str, dict[str, str]] = {}
    for name in ENVIRONMENT_LOCK_PATHS:
        baseline = root / ".runtime" / "receipts" / f"{name}.json"
        current = runtime_environment_root / f"{name}.json"
        _write_json(current, _read_json(baseline))
        runtime_environments[name] = current
        runtime_environment_records[name] = {
            "path": str(current.resolve()),
            "sha256": sha256(current),
            "status": _read_json(current)["status"],
        }
    _write_json(
        preflight,
        {
            "schema_version": 2,
            "status": "passed",
            "created_at_utc": _NOW,
            "stage_started_at_utc": _NOW,
            "project_commit": commit,
            "gpu_preflight": {
                "path": str(gpu_preflight.resolve()),
                "sha256": sha256(gpu_preflight),
                "cuda_visible_devices": "0,1",
                "selected_gpus": selected_gpus,
            },
            "config": {
                "path": str(config.resolve()),
                "sha256": sha256(config),
            },
            "locks": {relative: sha256(root / relative) for relative in LOCK_PATHS},
            "bootstrap": {
                "path": str(bootstrap.resolve()),
                "sha256": sha256(bootstrap),
            },
            "runtime_environments": runtime_environment_records,
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
        gpu_preflight=gpu_preflight,
        runtime_environments=runtime_environments,
        commit=commit,
    )


def _validate(evidence: RuntimeEvidence) -> dict[str, Any]:
    return validate_runtime_preflight(
        evidence.preflight,
        config_path=evidence.config,
        project_root=evidence.root,
        require_runtime_profile=False,
    )


def _refresh_preflight_record(
    evidence: RuntimeEvidence,
    record_name: str,
    path: Path,
) -> None:
    preflight = _read_json(evidence.preflight)
    preflight[record_name]["sha256"] = sha256(path)
    _write_json(evidence.preflight, preflight)


def _refresh_gpu_preflight_binding(evidence: RuntimeEvidence) -> None:
    gpu_document = _read_json(evidence.gpu_preflight)
    preflight = _read_json(evidence.preflight)
    preflight["gpu_preflight"] = {
        "path": str(evidence.gpu_preflight.resolve()),
        "sha256": sha256(evidence.gpu_preflight),
        "cuda_visible_devices": gpu_document.get("cuda_visible_devices"),
        "selected_gpus": gpu_document.get("selected_gpus"),
    }
    _write_json(evidence.preflight, preflight)


def _refresh_bootstrap(evidence: RuntimeEvidence) -> None:
    _refresh_preflight_record(evidence, "bootstrap", evidence.bootstrap)


def _refresh_runtime_environment(
    evidence: RuntimeEvidence,
    name: str,
) -> None:
    path = evidence.runtime_environments[name]
    document = _read_json(path)
    preflight = _read_json(evidence.preflight)
    preflight["runtime_environments"][name] = {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "status": document.get("status"),
    }
    _write_json(evidence.preflight, preflight)


def _freeze_now(
    monkeypatch: pytest.MonkeyPatch,
    value: datetime,
) -> None:
    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> datetime:
            return value if tz is not None else value.replace(tzinfo=None)

    monkeypatch.setattr("scaleguard.provenance.datetime", FrozenDateTime)


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
    gpu_document = _read_json(runtime_evidence.gpu_preflight)
    gpu_binding = {
        "schema_version": 1,
        "receipt_path": str(runtime_evidence.gpu_preflight.resolve()),
        "receipt_sha256": sha256(runtime_evidence.gpu_preflight),
        "cuda_visible_devices": "0,1",
        "selectors": ["0", "1"],
        "selected_gpus": gpu_document["selected_gpus"],
    }

    assert result == {
        "runtime_evidence_verified": True,
        "runtime_preflight_receipt": str(runtime_evidence.preflight.resolve()),
        "runtime_preflight_sha256": sha256(runtime_evidence.preflight),
        "gpu_preflight_receipt_sha256": sha256(runtime_evidence.gpu_preflight),
        "gpu_preflight_binding": gpu_binding,
        "bootstrap_receipt_sha256": sha256(runtime_evidence.bootstrap),
        "runtime_environment_receipt_sha256": {
            name: sha256(path) for name, path in runtime_evidence.runtime_environments.items()
        },
        "materialization_receipt_sha256": sha256(runtime_evidence.materialization),
        "materialization_marker_sha256": sha256(runtime_evidence.marker),
        "source_weights_receipt_sha256": sha256(runtime_evidence.weights_receipt),
        "weights_root": str(runtime_evidence.weights_root.resolve()),
        "project_commit": runtime_evidence.commit,
        "project_root": str(runtime_evidence.root.resolve()),
        "runtime_config_path": str(runtime_evidence.config.resolve()),
        "runtime_config_sha256": sha256(runtime_evidence.config),
        "runtime_stage_started_at": "2026-07-27T00:00:00+00:00",
    }


def test_runtime_preflight_rejects_a_deleted_gpu_receipt(
    runtime_evidence: RuntimeEvidence,
) -> None:
    runtime_evidence.gpu_preflight.unlink()

    with pytest.raises(RuntimePreflightError, match="GPU preflight receipt"):
        _validate(runtime_evidence)


def test_runtime_preflight_rejects_gpu_receipt_tampering(
    runtime_evidence: RuntimeEvidence,
) -> None:
    gpu_document = _read_json(runtime_evidence.gpu_preflight)
    gpu_document["status"] = "failed"
    _write_json(runtime_evidence.gpu_preflight, gpu_document)

    with pytest.raises(RuntimePreflightError, match="digest mismatch"):
        _validate(runtime_evidence)


def test_runtime_preflight_rejects_a_rebound_gpu_topology(
    runtime_evidence: RuntimeEvidence,
) -> None:
    gpu_document = _read_json(runtime_evidence.gpu_preflight)
    selected = gpu_document["selected_gpus"]
    assert isinstance(selected, list)
    assert isinstance(selected[1], dict)
    selected[1]["physical_index"] = "2"
    gpu_document["cuda_visible_devices"] = "0,2"
    _write_json(runtime_evidence.gpu_preflight, gpu_document)
    _refresh_gpu_preflight_binding(runtime_evidence)

    with pytest.raises(RuntimePreflightError, match="logical and physical 0,1 topology"):
        _validate(runtime_evidence)


def test_runtime_execution_binding_contains_the_normalized_gpu_identity(
    runtime_evidence: RuntimeEvidence,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        provenance,
        "_runtime_profile_binding",
        lambda **_kwargs: ({"schema_version": 1}, object()),
    )

    result = validate_runtime_preflight(
        runtime_evidence.preflight,
        config_path=runtime_evidence.config,
        project_root=runtime_evidence.root,
        require_runtime_profile=True,
    )

    gpu_binding = result["gpu_preflight_binding"]
    assert result["runtime_execution_binding"]["gpu_preflight"] == gpu_binding
    assert result["runtime_execution_binding_sha256"] == provenance._canonical_sha256(
        result["runtime_execution_binding"]
    )


def test_runtime_preflight_opens_each_json_receipt_once(
    runtime_evidence: RuntimeEvidence,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_paths = {
        runtime_evidence.preflight.resolve(),
        runtime_evidence.gpu_preflight.resolve(),
        runtime_evidence.bootstrap.resolve(),
        runtime_evidence.materialization.resolve(),
        runtime_evidence.marker.resolve(),
        runtime_evidence.weights_receipt.resolve(),
        (runtime_evidence.root / "weights-lock.json").resolve(),
        (runtime_evidence.root / "environments" / "uv.version").resolve(),
        *(
            runtime_evidence.root / ".runtime" / "receipts" / f"{name}.json"
            for name in ENVIRONMENT_LOCK_PATHS
        ),
        *(path.resolve() for path in runtime_evidence.runtime_environments.values()),
    }
    open_counts = dict.fromkeys(receipt_paths, 0)
    real_open = os.open

    def count_receipt_open(path: os.PathLike[str] | str, flags: int) -> int:
        resolved = Path(path).resolve()
        if resolved in open_counts:
            open_counts[resolved] += 1
        return real_open(path, flags)

    monkeypatch.setattr("scaleguard.provenance.os.open", count_receipt_open)

    assert _validate(runtime_evidence)["runtime_evidence_verified"] is True
    assert open_counts
    assert set(open_counts.values()) == {1}


def test_runtime_preflight_carries_one_atomic_receipt_snapshot_to_its_result(
    runtime_evidence: RuntimeEvidence,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_digest = sha256(runtime_evidence.preflight)
    real_snapshot = provenance._load_snapshot
    replaced = False

    def replace_after_snapshot(path: Path, label: str) -> tuple[dict[str, Any], str]:
        nonlocal replaced
        document, digest = real_snapshot(path, label)
        if path == runtime_evidence.preflight.resolve() and not replaced:
            replacement = dict(document)
            replacement["status"] = "failed"
            replacement_path = path.with_name(".runtime-preflight.replacement.json")
            _write_json(replacement_path, replacement)
            replacement_path.replace(path)
            replaced = True
        return document, digest

    monkeypatch.setattr(provenance, "_load_snapshot", replace_after_snapshot)

    result = _validate(runtime_evidence)

    assert replaced is True
    assert result["runtime_evidence_verified"] is True
    assert result["runtime_preflight_sha256"] == original_digest
    assert sha256(runtime_evidence.preflight) != original_digest
    assert _read_json(runtime_evidence.preflight)["status"] == "failed"


def test_runtime_preflight_rejects_atomic_replacement_during_a_receipt_snapshot(
    runtime_evidence: RuntimeEvidence,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = runtime_evidence.runtime_environments["coz"].resolve()
    replacement = runtime_evidence.preflight.parent / "coz-atomic-replacement.json"
    replacement_document = _read_json(target)
    replacement_document["issues"] = [{"issue": "atomic replacement"}]
    _write_json(replacement, replacement_document)
    real_open = os.open
    replaced = False

    def replace_after_open(path: os.PathLike[str] | str, flags: int) -> int:
        nonlocal replaced
        descriptor = real_open(path, flags)
        if Path(path) == target and not replaced:
            replacement.replace(target)
            replaced = True
        return descriptor

    monkeypatch.setattr("scaleguard.provenance.os.open", replace_after_open)

    with pytest.raises(RuntimePreflightError, match="changed while it was being read"):
        _validate(runtime_evidence)
    assert replaced is True


def test_regular_snapshot_rejects_in_place_mutation_after_the_final_fstat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = tmp_path / "evidence.json"
    evidence.write_bytes(b'{"status":"passed"}\n')
    identity = evidence.stat()
    real_fstat = os.fstat
    observations = 0

    def mutate_after_completed_read(descriptor: int) -> os.stat_result:
        nonlocal observations
        current = real_fstat(descriptor)
        if (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino):
            observations += 1
            if observations == 2:
                evidence.write_bytes(b'{"status":"forged-after-fstat"}\n')
        return current

    monkeypatch.setattr("scaleguard.provenance.os.fstat", mutate_after_completed_read)

    with pytest.raises(RuntimePreflightError, match="changed while it was being read"):
        provenance.load_regular_file_snapshot(evidence, "test evidence")
    assert observations == 2


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


def test_runtime_preflight_requires_exactly_four_fresh_environment_receipts(
    runtime_evidence: RuntimeEvidence,
) -> None:
    preflight = _read_json(runtime_evidence.preflight)
    preflight["runtime_environments"].pop("coz")
    _write_json(runtime_evidence.preflight, preflight)

    with pytest.raises(RuntimePreflightError, match="unexpected runtime environment set"):
        _validate(runtime_evidence)


def test_runtime_preflight_rejects_a_fresh_receipt_outside_the_attempt_directory(
    runtime_evidence: RuntimeEvidence,
) -> None:
    outside = runtime_evidence.preflight.parent / "coz-current.json"
    _write_json(outside, _read_json(runtime_evidence.runtime_environments["coz"]))
    preflight = _read_json(runtime_evidence.preflight)
    preflight["runtime_environments"]["coz"].update(
        {
            "path": str(outside.resolve()),
            "sha256": sha256(outside),
        }
    )
    _write_json(runtime_evidence.preflight, preflight)

    with pytest.raises(RuntimePreflightError, match="references an unexpected receipt"):
        _validate(runtime_evidence)


def test_runtime_preflight_rejects_a_symlinked_fresh_environment_receipt(
    runtime_evidence: RuntimeEvidence,
) -> None:
    fresh = runtime_evidence.runtime_environments["coz"]
    fresh.unlink()
    fresh.symlink_to(runtime_evidence.root / ".runtime" / "receipts" / "coz.json")

    with pytest.raises(RuntimePreflightError, match="must not be a symbolic link"):
        _validate(runtime_evidence)


@pytest.mark.parametrize("kind", ["directory", "fifo"])
def test_runtime_preflight_rejects_non_regular_fresh_environment_receipts(
    runtime_evidence: RuntimeEvidence,
    kind: str,
) -> None:
    fresh = runtime_evidence.runtime_environments["coz"]
    fresh.unlink()
    if kind == "directory":
        fresh.mkdir()
    else:
        os.mkfifo(fresh)

    with pytest.raises(RuntimePreflightError, match="is not a regular file"):
        _validate(runtime_evidence)


def test_runtime_preflight_wraps_evidence_open_errors(
    runtime_evidence: RuntimeEvidence,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_open = os.open

    def fail_preflight_open(path: os.PathLike[str] | str, flags: int) -> int:
        if Path(path) == runtime_evidence.preflight:
            raise PermissionError("blocked by fixture")
        return real_open(path, flags)

    monkeypatch.setattr("scaleguard.provenance.os.open", fail_preflight_open)

    with pytest.raises(RuntimePreflightError, match="cannot open runtime preflight receipt"):
        _validate(runtime_evidence)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("status", "has an invalid status"),
        ("python", "python differs from the bootstrap baseline"),
        ("locks", "locks differs from the bootstrap baseline"),
        ("expected_packages", "expected_packages differs from the bootstrap baseline"),
        ("packages", "packages differs from the bootstrap baseline"),
        ("runtime_imports", "unexpected runtime import set"),
        ("audited_overrides", "package audit is inconsistent"),
        ("issues", "receipt content mismatch"),
        ("unexpected", "receipt content mismatch"),
    ],
)
def test_runtime_preflight_rejects_fresh_environment_identity_drift(
    runtime_evidence: RuntimeEvidence,
    case: str,
    message: str,
) -> None:
    name = "scaleguard"
    path = runtime_evidence.runtime_environments[name]
    document = _read_json(path)
    if case == "status":
        document["status"] = "failed"
    elif case == "python":
        document["python"]["platform"] = "Linux-6.9.0-x86_64"
    elif case == "locks":
        document["locks"][0]["pinned_packages"] = 2
    elif case == "expected_packages":
        document["expected_packages"]["extra-package"] = "9.0"
        document["packages"]["extra-package"] = "9.0"
    elif case == "packages":
        document["packages"]["extra-package"] = "9.0"
    elif case == "runtime_imports":
        document["runtime_imports"].append({"module": "forged", "symbols": []})
    elif case == "audited_overrides":
        document["audited_overrides"] = [{"package": "forged"}]
    elif case == "issues":
        document["issues"] = [{"issue": "forged"}]
    elif case == "unexpected":
        document["unexpected"] = "forged"
    else:  # pragma: no cover - the parameter table is exhaustive
        raise AssertionError(case)
    _write_json(path, document)
    _refresh_runtime_environment(runtime_evidence, name)

    with pytest.raises(RuntimePreflightError, match=message):
        _validate(runtime_evidence)


def test_runtime_preflight_rejects_extra_attempt_environment_files(
    runtime_evidence: RuntimeEvidence,
) -> None:
    extra = runtime_evidence.preflight.parent / "runtime-environments" / "stale.json"
    _write_json(extra, {"status": "stale"})

    with pytest.raises(RuntimePreflightError, match="unexpected file set"):
        _validate(runtime_evidence)


def test_runtime_preflight_binds_fresh_receipts_to_the_stage_window(
    runtime_evidence: RuntimeEvidence,
) -> None:
    path = runtime_evidence.runtime_environments["coz"]
    document = _read_json(path)
    document["created_at_utc"] = "2026-07-26T23:59:59Z"
    _write_json(path, document)
    _refresh_runtime_environment(runtime_evidence, "coz")

    with pytest.raises(RuntimePreflightError, match="not created during this preflight stage"):
        _validate(runtime_evidence)


def test_runtime_preflight_rejects_a_stage_start_after_receipt_creation(
    runtime_evidence: RuntimeEvidence,
) -> None:
    preflight = _read_json(runtime_evidence.preflight)
    preflight["stage_started_at_utc"] = "2026-07-27T00:00:01Z"
    _write_json(runtime_evidence.preflight, preflight)

    with pytest.raises(RuntimePreflightError, match="predates its stage start"):
        _validate(runtime_evidence)


def test_runtime_preflight_rejects_old_environment_receipts_copied_into_a_new_attempt(
    runtime_evidence: RuntimeEvidence,
) -> None:
    preflight = _read_json(runtime_evidence.preflight)
    preflight["stage_started_at_utc"] = "2020-01-01T00:00:00Z"
    for name, path in runtime_evidence.runtime_environments.items():
        receipt = _read_json(path)
        receipt["created_at_utc"] = "2020-01-02T00:00:00Z"
        _write_json(path, receipt)
        preflight["runtime_environments"][name]["sha256"] = sha256(path)
    _write_json(runtime_evidence.preflight, preflight)

    with pytest.raises(RuntimePreflightError, match="exceeded its maximum evidence window"):
        _validate(runtime_evidence)


@pytest.mark.parametrize(
    ("clock_offset", "accepted"),
    [
        (timedelta(minutes=10), True),
        (timedelta(minutes=16), False),
        (-timedelta(minutes=2), False),
    ],
)
def test_runtime_preflight_only_applies_age_limits_to_real_run_validation(
    runtime_evidence: RuntimeEvidence,
    monkeypatch: pytest.MonkeyPatch,
    clock_offset: timedelta,
    accepted: bool,
) -> None:
    receipt_time = datetime(2026, 7, 27, tzinfo=timezone.utc)
    _freeze_now(monkeypatch, receipt_time + clock_offset)

    assert _validate(runtime_evidence)["runtime_evidence_verified"] is True
    if accepted:
        assert (
            validate_runtime_preflight(
                runtime_evidence.preflight,
                config_path=runtime_evidence.config,
                project_root=runtime_evidence.root,
                require_recent=True,
                require_runtime_profile=False,
            )["runtime_evidence_verified"]
            is True
        )
    else:
        with pytest.raises(RuntimePreflightError, match="not recent enough"):
            validate_runtime_preflight(
                runtime_evidence.preflight,
                config_path=runtime_evidence.config,
                project_root=runtime_evidence.root,
                require_recent=True,
                require_runtime_profile=False,
            )


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
        ("uv_binary_sha256", "0" * 64, "unexpected uv binary identity"),
        (
            "python_distribution",
            {"key": "unlocked"},
            "unexpected managed Python distribution",
        ),
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


def test_runtime_preflight_rejects_rebound_forged_locked_weight(
    runtime_evidence: RuntimeEvidence,
) -> None:
    runtime_evidence.source_file.write_bytes(b"self-consistent forged weights")
    receipt = _read_json(runtime_evidence.weights_receipt)
    receipt["artifacts"][0]["files"] = _file_inventory(runtime_evidence.source_file)
    receipt["artifacts"][0]["known_hashes_verified"] = ["model.bin"]
    _write_json(runtime_evidence.weights_receipt, receipt)
    _rebind_weight_chain(runtime_evidence)

    with pytest.raises(RuntimePreflightError, match="locked SHA-256"):
        _validate(runtime_evidence)


def test_runtime_preflight_rejects_forged_known_hash_verification_claim(
    runtime_evidence: RuntimeEvidence,
) -> None:
    receipt = _read_json(runtime_evidence.weights_receipt)
    receipt["artifacts"][0]["known_hashes_verified"] = []
    _write_json(runtime_evidence.weights_receipt, receipt)
    _rebind_weight_chain(runtime_evidence)

    with pytest.raises(RuntimePreflightError, match="hash verification disagrees"):
        _validate(runtime_evidence)


@pytest.mark.parametrize(
    ("provider_fields", "message"),
    [
        (
            {
                "provider": "https",
                "url": "https://weights.invalid/model.bin",
                "sha256": None,
                "known_sha256": None,
            },
            "immutable HTTPS identity",
        ),
        (
            {
                "provider": "https",
                "url": "https://weights.invalid/model.bin?mutable=1",
                "sha256": "a" * 64,
                "known_sha256": "a" * 64,
            },
            "immutable HTTPS identity",
        ),
        (
            {
                "provider": "huggingface",
                "repo_id": "scaleguard/model",
                "revision": "main",
                "known_sha256": None,
            },
            "immutable Hugging Face identity",
        ),
    ],
)
def test_locked_weight_artifact_requires_immutable_provider_identity(
    provider_fields: dict[str, object],
    message: str,
) -> None:
    artifact: dict[str, object] = {
        "id": "fixture",
        "required": True,
        "destination": "models/model.bin",
        "files": ["model.bin"],
        "verify_on_download": True,
        **provider_fields,
    }

    with pytest.raises(RuntimePreflightError, match=message):
        provenance._locked_weight_artifacts(
            {
                "schema_version": 1,
                "artifacts": [artifact],
            }
        )


def _create_profile_files(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, str], dict[str, dict[str, str]]]:
    root = tmp_path / "profile-project"
    config_path = root / "configs/runtime/autodl-2x4090.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_bytes((_PROJECT_ROOT / "configs/runtime/autodl-2x4090.yaml").read_bytes())

    for relative in (
        "third_party/checkouts/4KAgent",
        "third_party/checkouts/Chain-of-Zoom",
        "third_party/dependencies/DepictQA",
    ):
        (root / relative).mkdir(parents=True)
    for relative in (
        "third_party/overlays/4kagent/run_native_restoration.py",
        "third_party/overlays/4kagent/scheduler_client.py",
        "third_party/overlays/4kagent/serve_depictqa_eval.py",
        "third_party/overlays/chain-of-zoom/coz_session_worker.py",
    ):
        overlay = root / relative
        overlay.parent.mkdir(parents=True, exist_ok=True)
        overlay.write_text("# profile fixture\n", encoding="utf-8")

    directory_assets = (
        "weights/models/stabilityai/stable-diffusion-3-medium-diffusers",
        "weights/models/Qwen/Qwen2.5-VL-3B-Instruct",
        "weights/4kagent/models/Qwen2.5-VL-7B-Instruct",
        "weights/4kagent/runtime/toolbox-root",
        "weights/4kagent/hpsv2",
        "weights/4kagent/depictqa",
        "weights/chain-of-zoom/ckpt/VLM_LoRA/checkpoint-10000",
    )
    for relative in directory_assets:
        (root / relative).mkdir(parents=True, exist_ok=True)
    for relative in (
        "weights/chain-of-zoom/ckpt/SR_LoRA/model_20001.pkl",
        "weights/chain-of-zoom/ckpt/SR_VAE/vae_encoder_20001.pt",
        "weights/metrics/pyiqa/musiq_koniq_ckpt-e95806b9.pth",
    ):
        asset = root / relative
        asset.parent.mkdir(parents=True, exist_ok=True)
        asset.write_bytes(b"profile fixture\n")

    environment_bindings: dict[str, dict[str, str]] = {}
    for name, relative in {
        "scaleguard": ".venv/bin/python",
        "4kagent": ".runtime/envs/4kagent/bin/python",
        "depictqa": ".runtime/envs/depictqa/bin/python",
        "coz": ".runtime/envs/coz/bin/python",
    }.items():
        executable = root / relative
        executable.parent.mkdir(parents=True, exist_ok=True)
        executable.write_text("#!/bin/sh\n", encoding="utf-8")
        environment_bindings[name] = {"executable": str(executable)}

    assets = {
        "coz_sd3": str(root / "weights/models/stabilityai/stable-diffusion-3-medium-diffusers"),
        "coz_qwen": str(root / "weights/models/Qwen/Qwen2.5-VL-3B-Instruct"),
        "fourkagent_qwen": str(root / "weights/4kagent/models/Qwen2.5-VL-7B-Instruct"),
        "coz_sr_lora": str(root / "weights/chain-of-zoom/ckpt/SR_LoRA/model_20001.pkl"),
        "coz_vae": str(root / "weights/chain-of-zoom/ckpt/SR_VAE/vae_encoder_20001.pt"),
        "coz_vlm_lora": str(root / "weights/chain-of-zoom/ckpt/VLM_LoRA/checkpoint-10000"),
        "fourkagent_hps": str(root / "weights/4kagent/hpsv2"),
        "fourkagent_toolbox": str(root / "weights/4kagent/runtime/toolbox-root"),
        "depictqa_root": str(root / "weights/4kagent/depictqa"),
        "quality_musiq": str(root / "weights/metrics/pyiqa/musiq_koniq_ckpt-e95806b9.pth"),
    }
    return root, config_path, assets, environment_bindings


def test_runtime_profile_binding_normalizes_every_audited_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, config_path, assets, environments = _create_profile_files(tmp_path)
    upstream_lock = {
        "schema_version": 1,
        "repositories": {
            "fourkagent": {
                "checkout": "third_party/checkouts/4KAgent",
                "commit": "1" * 40,
                "tree": "2" * 40,
            },
            "chain_of_zoom": {
                "checkout": "third_party/checkouts/Chain-of-Zoom",
                "commit": "3" * 40,
                "tree": "4" * 40,
            },
        },
    }
    dependency_lock = {
        "schema_version": 1,
        "dependencies": {
            "depictqa": {
                "checkout": "third_party/dependencies/DepictQA",
                "commit": "5" * 40,
                "tree": "6" * 40,
                "role": "4kagent_transitive_perception_service",
                "parent": "fourkagent",
            }
        },
    }
    monkeypatch.setattr(
        provenance,
        "_require_verified_upstreams",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        provenance,
        "_environment_binding",
        lambda *_args, **_kwargs: environments,
    )
    monkeypatch.setattr(
        provenance,
        "_required_runtime_paths",
        lambda **_kwargs: assets,
    )

    binding, config = provenance._runtime_profile_binding(
        config_payload=config_path.read_bytes(),
        config_path=config_path,
        project_root=root,
        weights_root=root / "weights",
        weights_lock={"schema_version": 1, "artifacts": []},
        upstream_lock=upstream_lock,
        dependency_lock=dependency_lock,
        runtime_environments={},
        materialized_paths={},
        lock_digests={
            "upstream-lock.yaml": "7" * 64,
            "runtime-dependencies.yaml": "8" * 64,
        },
        require_current_scaleguard=False,
    )

    assert set(binding["checkouts"]) == {
        "fourkagent",
        "chain_of_zoom",
        "depictqa",
    }
    assert set(binding["upstreams"]) == {
        "fourkagent",
        "chain_of_zoom",
        "depictqa",
    }
    assert config.fourkagent.checkout == root / "third_party/checkouts/4KAgent"
    assert config.fourkagent.python_executable == environments["4kagent"]["executable"]
    assert config.fourkagent.perception_model_path == assets["fourkagent_qwen"]
    assert config.coz.checkout == root / "third_party/checkouts/Chain-of-Zoom"
    assert config.coz.python_executable == environments["coz"]["executable"]
    assert config.coz.vlm_lora_path == Path(assets["coz_vlm_lora"])
    assert config.metrics.quality_model_path == Path(assets["quality_musiq"])


def test_runtime_environment_binding_captures_python_byte_identity(
    runtime_evidence: RuntimeEvidence,
) -> None:
    receipts = {
        name: _read_json(path) for name, path in runtime_evidence.runtime_environments.items()
    }

    bindings = provenance._environment_binding(
        receipts,
        project_root=runtime_evidence.root,
        require_current_scaleguard=False,
    )

    assert set(bindings) == set(ENVIRONMENT_LOCK_PATHS)
    for binding in bindings.values():
        assert binding["installation_file_count"] > 0
        assert len(binding["installation_merkle_root"]) == 64
        assert len(binding["interpreter_sha256"]) == 64
        assert len(binding["base_stdlib_merkle_root"]) == 64


def test_required_runtime_paths_cover_every_locked_runtime_asset(
    tmp_path: Path,
) -> None:
    weights_root = tmp_path / "weights"
    artifact_paths: dict[str, str] = {}
    locked_artifacts: list[dict[str, object]] = []
    for _role, (artifact_id, relative) in provenance._RUNTIME_WEIGHT_ARTIFACTS.items():
        path = weights_root / relative
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"locked fixture\n")
        else:
            path.mkdir(parents=True, exist_ok=True)
        artifact_paths[artifact_id] = str(path)
        locked_artifacts.append(
            {
                "id": artifact_id,
                "provider": "manual",
                "required": True,
                "destination": relative,
                "files": [path.name] if path.is_file() else ["**/*"],
                "known_sha256": None,
                "verify_on_download": True,
            }
        )
    quality_checkpoint = weights_root / "metrics/pyiqa/musiq_koniq_ckpt-e95806b9.pth"
    quality_checkpoint.write_bytes(b"quality fixture\n")
    layout_paths: dict[str, str] = {}
    for _role, (layout_id, relative) in provenance._RUNTIME_WEIGHT_LAYOUTS.items():
        path = weights_root / relative
        path.mkdir(parents=True, exist_ok=True)
        layout_paths[layout_id] = str(path)

    paths = provenance._required_runtime_paths(
        weights_root=weights_root,
        weights_lock={
            "schema_version": 1,
            "artifacts": locked_artifacts,
        },
        materialized_paths={
            "artifacts": artifact_paths,
            "layouts": layout_paths,
        },
    )

    assert paths["quality_musiq"] == str(quality_checkpoint)
    assert paths["coz_vlm_lora"].endswith("VLM_LoRA/checkpoint-10000")
    assert paths["depictqa_root"] == str(weights_root / "4kagent/depictqa")
    assert paths["fourkagent_toolbox"] == str(weights_root / "4kagent/runtime/toolbox-root")
