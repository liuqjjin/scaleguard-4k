from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

import scaleguard.experiments as experiments
from scaleguard.config import EXPERIMENT_GROUP_SEMANTICS, EXPERIMENT_GROUPS
from scaleguard.experiments import (
    ExperimentProtocolError,
    load_ablation_protocol,
    run_ablation_suite,
)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs" / "experiments" / "ablation.yaml"


def _write_base_config(path: Path, *, maximum_steps: int = 2) -> Path:
    payload = yaml.safe_load(
        (ROOT / "configs/runtime/autodl-2x4090.yaml").read_text(encoding="utf-8")
    )
    payload["controller"]["max_coz_steps"] = maximum_steps
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _make_clean_project(path: Path) -> tuple[Path, str]:
    runner = path / experiments.INTEGRATION_RUNNER
    runner.parent.mkdir(parents=True)
    runner.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    runner.chmod(0o755)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "tests@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "ScaleGuard Tests"],
        check=True,
    )
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(path), "commit", "-q", "-m", "fixture"],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return runner, commit


def _receipt_digest(receipt: dict[str, Any]) -> str:
    unsigned = copy.deepcopy(receipt)
    expected = unsigned.pop("receipt_sha256")
    observed = hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    assert observed == expected
    return observed


def _write_bound_manifest(
    argv: list[str],
    *,
    project_commit: str,
    experiment_group: str | None = None,
) -> Path:
    config_path = Path(argv[argv.index("--config") + 1])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if experiment_group is not None:
        config["runtime"]["experiment_group"] = experiment_group
    declared_group = config["runtime"]["experiment_group"]
    fresh_digest = hashlib.sha256(declared_group.encode()).hexdigest()
    run_root = Path(config["runtime"]["run_root"])
    manifest_path = run_root / "fixture-run" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    provenance = {
        "runtime_evidence_verified": True,
        "runtime_profile_bound": True,
        "runtime_preflight_receipt": "/evidence/runtime-preflight.json",
        "runtime_preflight_sha256": fresh_digest,
        "bootstrap_receipt_sha256": "1" * 64,
        "runtime_environment_receipt_sha256": {
            role: hashlib.sha256(f"{declared_group}:{role}".encode()).hexdigest()
            for role in ("scaleguard", "4kagent", "depictqa", "coz")
        },
        "materialization_receipt_sha256": hashlib.sha256(
            f"{declared_group}:materialization".encode()
        ).hexdigest(),
        "materialization_marker_sha256": "7" * 64,
        "source_weights_receipt_sha256": "8" * 64,
        "weights_root": "/weights",
        "project_commit": project_commit,
        "project_root": "/project",
        "restoration_backend": (
            "scaleguard_identity_observation" if declared_group == "B-only" else "4kagent_upstream"
        ),
        "scale_backend": "chain_of_zoom",
        "runtime_execution_binding": {"schema_version": 1, "assets": {"fixed": True}},
        "runtime_execution_binding_sha256": "9" * 64,
        "runtime_config_path": str(config_path),
        "runtime_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "runtime_stage_started_at": "2026-07-27T00:00:00+00:00",
    }
    payload = {
        "mock": False,
        "status": "succeeded",
        "completion_level": ("AB_INTEGRATED" if declared_group == "ScaleGuard" else "STATIC_READY"),
        "target_reached": True,
        "config": config,
        "provenance": provenance,
        "input_image": {
            "sha256": hashlib.sha256(Path(argv[argv.index("--input") + 1]).read_bytes()).hexdigest()
        },
        "restoration_metadata": (
            {
                "backend": "scaleguard_identity_observation",
                "algorithmic_restoration": False,
            }
            if declared_group == "B-only"
            else {"backend": "4kagent_upstream"}
        ),
        "restoration_process": (None if declared_group == "B-only" else {"returncode": 0}),
        "scale_session_process": (None if declared_group == "A-only" else {"returncode": 0}),
        "steps": (
            []
            if declared_group == "A-only"
            else [
                {
                    "accepted": True,
                    "decision": "stop",
                    "candidate": {"mock": False, "sha256": "c" * 64},
                    "worker_metadata": {
                        "backend": "chain_of_zoom_persistent",
                        "candidate_sha256": "c" * 64,
                    },
                }
            ]
        ),
        "events": [],
        "final_metrics": {
            "selected_scale": 1 if declared_group == "A-only" else 4,
        },
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _file_entry(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _canonical_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _write_files_inventory(attempt_dir: Path) -> Path:
    destination = attempt_dir / "files.json"
    files = []
    for path in sorted(attempt_dir.rglob("*")):
        if path.is_file() and path != destination:
            entry = _file_entry(path)
            entry["path"] = path.relative_to(attempt_dir).as_posix()
            files.append(entry)
    destination.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "root": attempt_dir.name,
                "files": files,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def _write_wrapper_attempt(
    argv: list[str],
    manifest_path: Path,
    *,
    project_commit: str,
    gpu_name: str = "NVIDIA GeForce RTX 4090",
    gpu_uuid_prefix: str = "GPU-fixture",
) -> Path:
    config_path = Path(argv[argv.index("--config") + 1])
    input_path = Path(argv[argv.index("--input") + 1])
    output_path = Path(argv[argv.index("--output") + 1])
    pointer_path = Path(argv[argv.index("--evidence-output") + 1])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    group = config["runtime"]["experiment_group"]
    sample_id = config["runtime"]["experiment_sample_id"]
    attempt_id = f"attempt-{group.lower()}"
    attempt_dir = pointer_path.parent / attempt_id
    gpu_preflight_dir = attempt_dir / "gpu-preflight"
    gpu_preflight_dir.mkdir(parents=True)

    selected_gpus = [
        {
            "logical_index": 0,
            "physical_index": "0",
            "uuid": f"{gpu_uuid_prefix}-0",
            "name": gpu_name,
            "memory_total_mib": 24564,
            "driver_version": "560.35.03",
        },
        {
            "logical_index": 1,
            "physical_index": "1",
            "uuid": f"{gpu_uuid_prefix}-1",
            "name": gpu_name,
            "memory_total_mib": 24564,
            "driver_version": "560.35.03",
        },
    ]
    gpu_inventory = gpu_preflight_dir / "gpu_inventory.csv"
    with gpu_inventory.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for gpu in selected_gpus:
            writer.writerow(
                [
                    gpu["physical_index"],
                    gpu["uuid"],
                    gpu["name"],
                    gpu["memory_total_mib"],
                    gpu["driver_version"],
                ]
            )
    gpu_preflight = gpu_preflight_dir / "gpu_check.json"
    gpu_preflight.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "passed",
                "git_commit": project_commit,
                "requirements": {"minimum_gpu_count": 2},
                "cuda_visible_devices": None,
                "selected_gpus": selected_gpus,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    runtime_preflight = attempt_dir / "runtime-preflight.json"
    runtime_preflight.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "passed",
                "project_commit": project_commit,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    copied_manifest = attempt_dir / "scaleguard-run-manifest.json"
    shutil.copy2(manifest_path, copied_manifest)
    output_path.write_bytes(f"fixture output for {group}\n".encode())
    output_evidence = attempt_dir / "output-evidence.png"
    shutil.copy2(output_path, output_evidence)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
    model_evidence_summary = {
        "status": "passed",
        "stage": "experiment",
        "mock": False,
        "manifest_status": manifest["status"],
        "completion_level": manifest["completion_level"],
        "restoration_backend": (
            "scaleguard_identity_observation" if group == "B-only" else "4kagent_upstream"
        ),
        "scale_backend": "chain_of_zoom",
        "successful_coz_candidates": 0 if group == "A-only" else 1,
        "experiment_group": group,
        "experiment_sample_id": sample_id,
        "runtime_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "runtime_preflight_sha256": hashlib.sha256(runtime_preflight.read_bytes()).hexdigest(),
        "source_manifest": str(manifest_path.resolve()),
        "manifest_sha256": hashlib.sha256(copied_manifest.read_bytes()).hexdigest(),
        "runtime_preflight_path": str(runtime_preflight.resolve()),
        "invoked_input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "invoked_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "final_output_sha256": output_sha256,
        "output_evidence_path": str(output_evidence.resolve()),
        "output_evidence_sha256": output_sha256,
    }
    model_evidence = attempt_dir / "model-evidence.json"
    model_evidence.write_text(
        json.dumps(model_evidence_summary) + "\n",
        encoding="utf-8",
    )
    execution = attempt_dir / "execution.json"
    execution.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage": "experiment",
                "status": "passed",
                "return_code": 0,
                "scaleguard_command_return_code": 0,
                "git_commit": project_commit,
                "inputs": {
                    "runtime_config": _file_entry(config_path),
                    "input_image": _file_entry(input_path),
                    "runtime_preflight": _file_entry(runtime_preflight),
                },
                "outputs": [_file_entry(output_path)],
                "model_evidence": {
                    "complete": True,
                    "helper_completed": True,
                    "hashes_consistent": True,
                    "output_snapshot": _file_entry(output_evidence),
                    "run_manifest_snapshot": _file_entry(copied_manifest),
                    "summary": model_evidence_summary,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    simple_files = {
        "experiment.log": "redacted experiment log\n",
        "gpu-samples.csv": "timestamp,index,memory_used_mib\n0,0,1024\n",
        "nvidia-smi-before.txt": "fixture before\n",
        "nvidia-smi-after.txt": "fixture after\n",
    }
    for name, content in simple_files.items():
        (attempt_dir / name).write_text(content, encoding="utf-8")
    inventory = _write_files_inventory(attempt_dir)

    role_paths = {
        "execution": execution,
        "run_manifest": copied_manifest,
        "model_evidence": model_evidence,
        "raw_log": attempt_dir / "experiment.log",
        "gpu_samples": attempt_dir / "gpu-samples.csv",
        "nvidia_smi_before": attempt_dir / "nvidia-smi-before.txt",
        "nvidia_smi_after": attempt_dir / "nvidia-smi-after.txt",
        "gpu_inventory": gpu_inventory,
        "gpu_preflight": gpu_preflight,
        "files_inventory": inventory,
        "runtime_preflight": runtime_preflight,
    }
    identity_payload = {
        "cuda_visible_devices": None,
        "selected_gpus": selected_gpus,
    }
    class_payload = {
        "selected_gpus": [
            {
                "logical_index": gpu["logical_index"],
                "name": gpu["name"],
                "memory_total_mib": gpu["memory_total_mib"],
                "driver_version": gpu["driver_version"],
            }
            for gpu in selected_gpus
        ]
    }
    pointer_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "succeeded",
                "stage": "experiment",
                "attempt_id": attempt_id,
                "attempt_dir": str(attempt_dir.resolve()),
                "started_at_utc": "2026-07-27T00:00:00Z",
                "completed_at_utc": "2026-07-27T00:01:00Z",
                "experiment_group": group,
                "experiment_sample_id": sample_id,
                "files": {role: _file_entry(path) for role, path in role_paths.items()},
                "hardware": {
                    "identity_sha256": _canonical_digest(identity_payload),
                    "class_sha256": _canonical_digest(class_payload),
                    "selected_gpu_count": 2,
                    "cuda_visible_devices": None,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return pointer_path


def _write_failed_wrapper_pointer(argv: list[str]) -> Path:
    config_path = Path(argv[argv.index("--config") + 1])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    pointer_path = Path(argv[argv.index("--evidence-output") + 1])
    attempt_id = f"attempt-{config['runtime']['experiment_group'].lower()}"
    attempt_dir = pointer_path.parent / attempt_id
    attempt_dir.mkdir()
    pointer_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "failed",
                "stage": "experiment",
                "attempt_id": attempt_id,
                "attempt_dir": str(attempt_dir.resolve()),
                "started_at_utc": "2026-07-27T00:00:00Z",
                "completed_at_utc": "2026-07-27T00:00:01Z",
                "experiment_group": config["runtime"]["experiment_group"],
                "experiment_sample_id": config["runtime"]["experiment_sample_id"],
                "files": {},
                "hardware": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return pointer_path


def test_attempt_json_and_csv_parsers_use_the_recorded_byte_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt_dir = tmp_path / "attempt"
    attempt_dir.mkdir()
    snapshots = {
        "execution": b'{"version":"snapshot"}\n',
        "gpu_inventory": b"0,GPU-snapshot,RTX 4090,24564,560.35.03\n",
    }
    current = {
        "execution": b'{"version":"replaced"}\n',
        "gpu_inventory": b"0,GPU-replaced,RTX 5090,32768,999.0\n",
    }
    paths = {
        role: attempt_dir / experiments._ATTEMPT_FILE_RELATIVE_PATHS[role] for role in snapshots
    }
    for role, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(current[role])

    original_loader = experiments.load_regular_file_snapshot

    def load(path: Path, label: str) -> tuple[bytes, str]:
        for role, candidate in paths.items():
            if path == candidate:
                payload = snapshots[role]
                return payload, hashlib.sha256(payload).hexdigest()
        return original_loader(path, label)

    monkeypatch.setattr(experiments, "load_regular_file_snapshot", load)

    for role, path in paths.items():
        payload = snapshots[role]
        entry, observed, issues = experiments._attempt_file_entry(
            {
                "path": str(path.resolve()),
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
            role=role,
            attempt_dir=attempt_dir,
        )
        assert issues == []
        assert entry is not None
        assert observed == payload

    parsed = json.loads(snapshots["execution"])
    assert parsed["version"] == "snapshot"

    selected_gpus = [
        {
            "logical_index": 0,
            "physical_index": "0",
            "uuid": "GPU-snapshot",
            "name": "RTX 4090",
            "memory_total_mib": 24564,
            "driver_version": "560.35.03",
        }
    ]
    project_commit = "1" * 40
    identity = {
        "cuda_visible_devices": None,
        "selected_gpus": selected_gpus,
    }
    hardware, hardware_issues = experiments._validate_attempt_hardware(
        {
            "identity_sha256": _canonical_digest(identity),
            "class_sha256": _canonical_digest(
                {
                    "selected_gpus": [
                        {
                            key: gpu[key]
                            for key in (
                                "logical_index",
                                "name",
                                "memory_total_mib",
                                "driver_version",
                            )
                        }
                        for gpu in selected_gpus
                    ]
                }
            ),
            "selected_gpu_count": 1,
            "cuda_visible_devices": None,
        },
        gpu_preflight_payload=(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "passed",
                    "git_commit": project_commit,
                    "requirements": {"minimum_gpu_count": 1},
                    "cuda_visible_devices": None,
                    "selected_gpus": selected_gpus,
                }
            ).encode()
        ),
        gpu_inventory_payload=snapshots["gpu_inventory"],
        project_commit=project_commit,
    )
    assert hardware_issues == []
    assert hardware is not None
    assert hardware["selected_gpus"][0]["uuid"] == "GPU-snapshot"


def test_manifest_inspection_rejects_validation_of_different_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "runs"
    manifest_path = run_root / "run" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text('{"snapshot":"first"}\n', encoding="utf-8")
    monkeypatch.setattr(
        experiments,
        "validate_run_manifest",
        lambda _path: {"snapshot": "second"},
    )
    group = load_ablation_protocol(PROTOCOL).groups[0]

    record, issues, manifest = experiments._inspect_manifest(
        run_root,
        group=group,
        sample_id="sample",
        seed=7,
        project_commit="0" * 40,
    )

    assert record is None
    assert manifest is None
    assert any("run manifest changed while it was validated" in issue for issue in issues)


def test_executable_protocol_is_strict_and_uses_canonical_semantics(
    tmp_path: Path,
) -> None:
    protocol = load_ablation_protocol(PROTOCOL)

    assert protocol.name == "core-ablation"
    assert tuple(group.id for group in protocol.groups) == EXPERIMENT_GROUPS
    for group in protocol.groups:
        assert (
            group.fourkagent_mode,
            group.coz_mode,
            group.target_factor,
            group.max_coz_steps,
            group.acceptance_policy,
        ) == EXPERIMENT_GROUP_SEMANTICS[group.id]
    assert protocol.groups[0].comparison_resolution == "restoration_native"

    raw = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    raw["unexpected"] = True
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(ExperimentProtocolError, match="unknown unexpected"):
        load_ablation_protocol(invalid)

    raw.pop("unexpected")
    raw["status"] = "protocol_only"
    invalid.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(ExperimentProtocolError, match="status must be 'executable'"):
        load_ablation_protocol(invalid)


def test_plan_materializes_complete_hash_paired_matrix(tmp_path: Path) -> None:
    project = tmp_path / "project"
    runner, commit = _make_clean_project(project)
    base = _write_base_config(tmp_path / "base.yaml")
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first.write_bytes(b"first-authorized-input")
    second.write_bytes(b"second-authorized-input")
    output = tmp_path / "suite"

    receipt = run_ablation_suite(
        protocol_path=PROTOCOL,
        base_config_path=base,
        inputs=[first, second],
        seeds=[17, 23],
        output_directory=output,
        project_root=project,
        plan_only=True,
    )

    assert receipt["status"] == "planned"
    assert receipt["project_commit"] == commit
    assert receipt["integration_runner"]["path"] == str(runner)
    assert receipt["counts"] == {
        "total": 16,
        "planned": 16,
        "running": 0,
        "passed": 0,
        "failed": 0,
    }
    assert len({job["sample_id"] for job in receipt["jobs"]}) == 4
    for job in receipt["jobs"]:
        assert job["sample_id"] == (f"{job['input']['sha256'][:16]}-s{job['seed']}")
        assert job["argv"] == [
            str(runner),
            "--config",
            job["config"]["path"],
            "--input",
            job["input"]["path"],
            "--output",
            job["output_path"],
            "--evidence-output",
            job["wrapper_evidence_pointer"],
        ]
        config = yaml.safe_load(Path(job["config"]["path"]).read_text(encoding="utf-8"))
        semantics = EXPERIMENT_GROUP_SEMANTICS[job["group"]]
        assert config["runtime"]["experiment_group"] == job["group"]
        assert config["runtime"]["experiment_sample_id"] == job["sample_id"]
        assert config["runtime"]["run_root"] == job["run_root"]
        assert (
            config["fourkagent"]["mode"],
            config["coz"]["mode"],
            config["controller"]["target_factor"],
            config["controller"]["max_coz_steps"],
            config["controller"]["acceptance_policy"],
        ) == semantics
        assert config["coz"]["seed"] == job["seed"]

    for sample_id in {job["sample_id"] for job in receipt["jobs"]}:
        paired = [job for job in receipt["jobs"] if job["sample_id"] == sample_id]
        assert {job["group"] for job in paired} == set(EXPERIMENT_GROUPS)
        assert len({job["input"]["path"] for job in paired}) == 1
        assert len({job["input"]["sha256"] for job in paired}) == 1
        assert len({job["seed"] for job in paired}) == 1

    on_disk = json.loads((output / "suite-receipt.json").read_text(encoding="utf-8"))
    _receipt_digest(on_disk)
    assert not list(output.rglob("*.tmp"))


def test_suite_rejects_duplicate_samples_and_unsafe_output_roots(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    _make_clean_project(project)
    base = _write_base_config(tmp_path / "base.yaml")
    first = tmp_path / "first.png"
    duplicate = tmp_path / "duplicate.png"
    first.write_bytes(b"identical")
    duplicate.write_bytes(b"identical")

    with pytest.raises(ExperimentProtocolError, match="duplicate sample ID"):
        run_ablation_suite(
            protocol_path=PROTOCOL,
            base_config_path=base,
            inputs=[first, duplicate],
            seeds=[7],
            output_directory=tmp_path / "unused",
            project_root=project,
            plan_only=True,
        )
    with pytest.raises(ExperimentProtocolError, match="duplicate seeds"):
        run_ablation_suite(
            protocol_path=PROTOCOL,
            base_config_path=base,
            inputs=[first],
            seeds=[7, 7],
            output_directory=tmp_path / "unused",
            project_root=project,
            plan_only=True,
        )
    with pytest.raises(ExperimentProtocolError, match="dangerous output directory"):
        run_ablation_suite(
            protocol_path=PROTOCOL,
            base_config_path=base,
            inputs=[first],
            seeds=[7],
            output_directory=project,
            project_root=project,
            plan_only=True,
        )
    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ExperimentProtocolError, match="must be empty"):
        run_ablation_suite(
            protocol_path=PROTOCOL,
            base_config_path=base,
            inputs=[first],
            seeds=[7],
            output_directory=nonempty,
            project_root=project,
            plan_only=True,
        )
    assert (nonempty / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_execution_continues_after_failures_and_checks_manifest_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    runner, commit = _make_clean_project(project)
    base = _write_base_config(tmp_path / "base.yaml")
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"authorized-input")
    calls: list[list[str]] = []
    validated: list[Path] = []
    secret = "must-not-appear-in-suite-evidence"
    monkeypatch.setenv("OPENAI_API_KEY", secret)

    def validate(path: Path) -> dict[str, Any]:
        validated.append(path)
        return json.loads(path.read_text(encoding="utf-8"))

    monkeypatch.setattr(experiments, "validate_run_manifest", validate)

    def invoke(argv: Any, cwd: Path) -> int:
        tokens = list(argv)
        calls.append(tokens)
        assert cwd == project
        assert tokens[0] == str(runner)
        assert tokens[1::2] == [
            "--config",
            "--input",
            "--output",
            "--evidence-output",
        ]
        assert secret not in json.dumps(tokens)
        config = yaml.safe_load(Path(tokens[2]).read_text(encoding="utf-8"))
        group = config["runtime"]["experiment_group"]
        if group == "A-only":
            _write_failed_wrapper_pointer(tokens)
            return 17
        manifest = _write_bound_manifest(
            tokens,
            project_commit=commit,
            experiment_group="A-only" if group == "B-only" else None,
        )
        _write_wrapper_attempt(tokens, manifest, project_commit=commit)
        return 0

    output = tmp_path / "suite"
    receipt = run_ablation_suite(
        protocol_path=PROTOCOL,
        base_config_path=base,
        inputs=[input_path],
        seeds=[31],
        output_directory=output,
        project_root=project,
        command_runner=invoke,
    )

    assert len(calls) == 4
    assert receipt["status"] == "completed_with_failures"
    by_group = {job["group"]: job for job in receipt["jobs"]}
    assert by_group["A-only"]["returncode"] == 17
    assert "manifest_missing" in by_group["A-only"]["issues"]
    assert by_group["B-only"]["returncode"] == 0
    assert "manifest_experiment_group_mismatch" in by_group["B-only"]["issues"]
    assert by_group["AB-fixed"]["status"] == "passed"
    assert by_group["ScaleGuard"]["status"] == "passed"
    assert by_group["AB-fixed"]["manifest"]["sha256"]
    assert by_group["AB-fixed"]["runtime_evidence"]["runtime_environment_receipt_sha256"] == {
        role: hashlib.sha256(f"AB-fixed:{role}".encode()).hexdigest()
        for role in ("scaleguard", "4kagent", "depictqa", "coz")
    }
    assert len(validated) >= 6
    serialized = (output / "suite-receipt.json").read_text(encoding="utf-8")
    assert secret not in serialized
    _receipt_digest(json.loads(serialized))


def test_passed_suite_receipt_reader_revalidates_bound_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    _runner, commit = _make_clean_project(project)
    base = _write_base_config(tmp_path / "base.yaml")
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"authorized-input")
    monkeypatch.setattr(
        experiments,
        "validate_run_manifest",
        lambda path: json.loads(path.read_text(encoding="utf-8")),
    )

    def invoke(argv: Any, _cwd: Path) -> int:
        tokens = list(argv)
        manifest = _write_bound_manifest(tokens, project_commit=commit)
        _write_wrapper_attempt(tokens, manifest, project_commit=commit)
        return 0

    output = tmp_path / "suite"
    receipt = run_ablation_suite(
        protocol_path=PROTOCOL,
        base_config_path=base,
        inputs=[input_path],
        seeds=[37],
        output_directory=output,
        project_root=project,
        command_runner=invoke,
    )

    assert receipt["status"] == "passed"
    receipt_path = output / "suite-receipt.json"
    validated = experiments.validate_ablation_suite_receipt(receipt_path)
    assert validated["project_commit"] == commit
    assert validated["path"] == str(receipt_path.resolve())
    assert validated["size_bytes"] == receipt_path.stat().st_size
    assert len(validated["jobs"]) == len(EXPERIMENT_GROUPS)
    assert {job["group"] for job in validated["jobs"]} == set(EXPERIMENT_GROUPS)
    assert len({job["hardware"]["identity_sha256"] for job in validated["jobs"]}) == 1

    pointer = Path(receipt["jobs"][0]["wrapper_evidence_pointer"])
    attempt = json.loads(pointer.read_text(encoding="utf-8"))
    (Path(attempt["attempt_dir"]) / "experiment.log").write_text(
        "tampered after suite validation\n",
        encoding="utf-8",
    )
    with pytest.raises(ExperimentProtocolError, match="wrapper attempt revalidation failed"):
        experiments.validate_ablation_suite_receipt(receipt_path)


def test_post_wrapper_manifest_tampering_fails_the_suite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    _runner, commit = _make_clean_project(project)
    base = _write_base_config(tmp_path / "base.yaml")
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"authorized-input")
    first_manifest: Path | None = None
    first_pointer: Path | None = None

    monkeypatch.setattr(
        experiments,
        "validate_run_manifest",
        lambda path: json.loads(path.read_text(encoding="utf-8")),
    )

    def invoke(argv: Any, _cwd: Path) -> int:
        nonlocal first_manifest, first_pointer
        tokens = list(argv)
        manifest = _write_bound_manifest(tokens, project_commit=commit)
        pointer = _write_wrapper_attempt(tokens, manifest, project_commit=commit)
        config = yaml.safe_load(Path(tokens[2]).read_text(encoding="utf-8"))
        if config["runtime"]["experiment_group"] == "A-only":
            first_manifest = manifest
            first_pointer = pointer
        if config["runtime"]["experiment_group"] == "ScaleGuard":
            assert first_manifest is not None
            assert first_pointer is not None
            payload = json.loads(first_manifest.read_text(encoding="utf-8"))
            payload["post_wrapper_tamper"] = True
            first_manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            attempt = json.loads(first_pointer.read_text(encoding="utf-8"))
            (Path(attempt["attempt_dir"]) / "experiment.log").write_text(
                "tampered raw log\n",
                encoding="utf-8",
            )
        return 0

    receipt = run_ablation_suite(
        protocol_path=PROTOCOL,
        base_config_path=base,
        inputs=[input_path],
        seeds=[41],
        output_directory=tmp_path / "suite",
        project_root=project,
        command_runner=invoke,
    )

    by_group = {job["group"]: job for job in receipt["jobs"]}
    assert receipt["status"] == "completed_with_failures"
    assert by_group["A-only"]["status"] == "failed"
    assert any(
        "post_suite_manifest_or_artifact_inventory_changed" in issue
        for issue in by_group["A-only"]["issues"]
    )
    assert any(
        "wrapper_attempt_file_invalid:raw_log" in issue for issue in by_group["A-only"]["issues"]
    )
    assert by_group["B-only"]["status"] == "passed"
    assert by_group["AB-fixed"]["status"] == "passed"
    assert by_group["ScaleGuard"]["status"] == "passed"


def test_hardware_identity_and_class_must_match_within_a_sample(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    _runner, commit = _make_clean_project(project)
    base = _write_base_config(tmp_path / "base.yaml")
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"authorized-input")
    monkeypatch.setattr(
        experiments,
        "validate_run_manifest",
        lambda path: json.loads(path.read_text(encoding="utf-8")),
    )

    def invoke(argv: Any, _cwd: Path) -> int:
        tokens = list(argv)
        manifest = _write_bound_manifest(tokens, project_commit=commit)
        config = yaml.safe_load(Path(tokens[2]).read_text(encoding="utf-8"))
        changed = config["runtime"]["experiment_group"] == "ScaleGuard"
        _write_wrapper_attempt(
            tokens,
            manifest,
            project_commit=commit,
            gpu_name=("NVIDIA GeForce RTX 5090" if changed else "NVIDIA GeForce RTX 4090"),
            gpu_uuid_prefix="GPU-other" if changed else "GPU-fixture",
        )
        return 0

    receipt = run_ablation_suite(
        protocol_path=PROTOCOL,
        base_config_path=base,
        inputs=[input_path],
        seeds=[43],
        output_directory=tmp_path / "suite",
        project_root=project,
        command_runner=invoke,
    )

    assert receipt["status"] == "completed_with_failures"
    assert all(job["status"] == "failed" for job in receipt["jobs"])
    assert all(
        "sample_pairing_mismatch:hardware_identity_sha256" in job["issues"]
        and "sample_pairing_mismatch:hardware_class_sha256" in job["issues"]
        for job in receipt["jobs"]
    )


@pytest.mark.parametrize(
    ("accepted", "decision", "final_gate"),
    [
        (False, "stop", False),
        (False, "rollback", False),
        (True, "stop", True),
    ],
)
def test_scaleguard_rollback_contract_accepts_both_controller_paths(
    accepted: bool,
    decision: str,
    final_gate: bool,
) -> None:
    scaleguard = load_ablation_protocol(PROTOCOL).groups[-1]
    manifest: dict[str, Any] = {
        "mock": False,
        "status": "succeeded_with_rollback",
        "completion_level": "COMPONENT_REPRODUCED",
        "target_reached": False,
        "provenance": {
            "restoration_backend": "4kagent_upstream",
            "scale_backend": "chain_of_zoom",
        },
        "restoration_metadata": {"backend": "4kagent_upstream"},
        "restoration_process": {"returncode": 0},
        "scale_session_process": {"returncode": 0},
        "steps": [
            {
                "accepted": accepted,
                "decision": decision,
                "candidate": {"mock": False, "sha256": "c" * 64},
                "worker_metadata": {
                    "backend": "chain_of_zoom_persistent",
                    "candidate_sha256": "c" * 64,
                },
            }
        ],
        "events": ([{"event": "final_gate_rollback"}] if final_gate else []),
        "final_metrics": {"selected_scale": 2},
    }

    assert experiments.manifest_experiment_issues(manifest, scaleguard.id) == []


def test_scaleguard_rollback_rejects_an_unbound_accepted_stop() -> None:
    scaleguard = load_ablation_protocol(PROTOCOL).groups[-1]
    manifest = {
        "mock": False,
        "status": "succeeded_with_rollback",
        "completion_level": "COMPONENT_REPRODUCED",
        "target_reached": False,
        "provenance": {
            "restoration_backend": "4kagent_upstream",
            "scale_backend": "chain_of_zoom",
        },
        "restoration_metadata": {"backend": "4kagent_upstream"},
        "restoration_process": {"returncode": 0},
        "scale_session_process": {"returncode": 0},
        "steps": [
            {
                "accepted": True,
                "decision": "stop",
                "candidate": {"mock": False, "sha256": "c" * 64},
                "worker_metadata": {
                    "backend": "chain_of_zoom_persistent",
                    "candidate_sha256": "c" * 64,
                },
            }
        ],
        "events": [],
        "final_metrics": {"selected_scale": 2},
    }

    assert experiments.manifest_experiment_issues(
        manifest,
        scaleguard.id,
    ) == ["manifest_scaleguard_outcome_invalid"]


def test_default_runner_uses_fixed_shell_and_minimal_phase_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}
    config = _write_base_config(tmp_path / "config.yaml")
    poison = tmp_path / "poison.sh"
    poison.write_text("exit 97\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "scheduler-secret")
    monkeypatch.setenv("UNRELATED_SERVICE_TOKEN", "must-not-cross")
    monkeypatch.setenv("PIP_INDEX_URL", "https://user:pass@example.invalid/simple")
    monkeypatch.setenv("BASH_ENV", str(poison))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "poison-python"))
    monkeypatch.setenv("LD_PRELOAD", str(tmp_path / "poison-loader.so"))
    monkeypatch.setenv("TMPDIR", str(tmp_path / "poison-tmp"))

    def run(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        observed["argv"] = argv
        observed["kwargs"] = kwargs
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(experiments.subprocess, "run", run)
    returncode = experiments._default_command_runner(
        [
            "/project/scripts/autodl/run_experiment.sh",
            "--config",
            str(config),
        ],
        tmp_path,
    )

    assert returncode == 0
    assert observed["argv"][:3] == [
        "/bin/bash",
        "-p",
        "/project/scripts/autodl/run_experiment.sh",
    ]
    environment = observed["kwargs"].pop("env")
    assert observed["kwargs"] == {"cwd": tmp_path, "check": False}
    assert environment["OPENAI_API_KEY"] == "scheduler-secret"
    assert environment["PATH"] == experiments._FIXED_SYSTEM_PATH
    assert environment["TMPDIR"] == "/tmp"
    isolated_home = Path(environment["HOME"])
    assert isolated_home.parent == (tmp_path / ".runtime").resolve()
    assert isolated_home.name.startswith("experiment-home-")
    assert not isolated_home.is_symlink()
    assert isolated_home.stat().st_mode & 0o777 == 0o700
    assert list(isolated_home.iterdir()) == []
    assert environment["GIT_CONFIG_GLOBAL"] == os.devnull
    assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
    assert environment["GIT_CONFIG_SYSTEM"] == os.devnull
    assert "UNRELATED_SERVICE_TOKEN" not in environment
    assert "PIP_INDEX_URL" not in environment
    assert "BASH_ENV" not in environment
    assert "PYTHONPATH" not in environment
    assert "LD_PRELOAD" not in environment

    identity_payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    identity_payload["runtime"]["experiment_group"] = "B-only"
    identity_payload["runtime"]["experiment_sample_id"] = "sample-1"
    identity_payload["fourkagent"]["mode"] = "identity"
    identity_payload["controller"].update(
        {"target_factor": 4, "max_coz_steps": 1, "acceptance_policy": "fixed"}
    )
    identity = tmp_path / "identity.yaml"
    identity.write_text(yaml.safe_dump(identity_payload, sort_keys=False), encoding="utf-8")
    observed.clear()
    assert (
        experiments._default_command_runner(
            [
                "/project/scripts/autodl/run_experiment.sh",
                "--config",
                str(identity),
            ],
            tmp_path,
        )
        == 0
    )
    identity_environment = observed["kwargs"]["env"]
    assert "OPENAI_API_KEY" not in identity_environment


def test_default_runner_rejects_a_symlinked_runtime_root(tmp_path: Path) -> None:
    config = _write_base_config(tmp_path / "config.yaml")
    attacker = tmp_path / "attacker"
    attacker.mkdir()
    (tmp_path / ".runtime").symlink_to(attacker, target_is_directory=True)

    with pytest.raises(ExperimentProtocolError, match="runtime root must not be a symlink"):
        experiments._default_command_runner(
            [
                "/project/scripts/autodl/run_experiment.sh",
                "--config",
                str(config),
            ],
            tmp_path,
        )
