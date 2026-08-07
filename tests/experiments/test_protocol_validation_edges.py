from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

import scaleguard.experiments as experiments
from scaleguard.config import EXPERIMENT_GROUPS
from scaleguard.experiments import ExperimentProtocolError

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs" / "experiments" / "ablation.yaml"
BASE_CONFIG = ROOT / "configs" / "runtime" / "autodl-2x4090.yaml"
COMMIT = "a" * 40


def _write_yaml(path: Path, payload: Any) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _file_entry(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _protocol_document() -> dict[str, Any]:
    loaded = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda root: root.update(schema_version=2), "schema_version"),
        (lambda root: root.update(name="renamed-ablation"), "fixed protocol"),
        (lambda root: root.update(integration_runner="other.sh"), "integration_runner"),
        (lambda root: root.update(base_requirements=[]), "base_requirements must be"),
        (
            lambda root: root["base_requirements"].update(controller_target_factor=8),
            "base_requirements disagrees",
        ),
        (lambda root: root.update(groups=[]), "groups must contain"),
        (lambda root: root["groups"].__setitem__(0, []), r"groups\[0\] must be"),
        (lambda root: root["groups"][0].update(description=""), "description"),
        (lambda root: root["groups"][0].update(target_factor=4.0), "target_factor must be"),
        (lambda root: root.update(paired_requirements=[]), "paired_requirements must be"),
        (
            lambda root: root["paired_requirements"].update(same_input_snapshot=False),
            "paired_requirements value must be true",
        ),
        (lambda root: root.update(metrics=[]), "metrics must be"),
        (
            lambda root: root["metrics"].update(full_reference=["psnr", "psnr"]),
            "non-empty unique string list",
        ),
        (
            lambda root: root["metrics"].update(full_reference=["psnr", "ssim"]),
            "executable metric contract",
        ),
        (lambda root: root.update(notes=[]), "notes must be"),
    ],
)
def test_protocol_rejects_semantic_drift(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], Any],
    message: str,
) -> None:
    document = _protocol_document()
    mutate(document)
    path = _write_yaml(tmp_path / "protocol.yaml", document)

    with pytest.raises(ExperimentProtocolError, match=message):
        experiments.load_ablation_protocol(path)


def test_protocol_rejects_non_mapping_and_invalid_yaml(tmp_path: Path) -> None:
    sequence = _write_yaml(tmp_path / "sequence.yaml", ["not", "a", "mapping"])
    with pytest.raises(ExperimentProtocolError, match="must be a string-keyed mapping"):
        experiments.load_ablation_protocol(sequence)

    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("groups: [\n", encoding="utf-8")
    with pytest.raises(ExperimentProtocolError, match="invalid ablation protocol YAML"):
        experiments.load_ablation_protocol(malformed)

    with pytest.raises(ExperimentProtocolError, match="cannot read ablation protocol"):
        experiments.load_ablation_protocol(tmp_path / "missing.yaml")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda root: root["controller"].update(target_factor=2),
            "target_factor must be 4",
        ),
        (
            lambda root: root["controller"].update(max_coz_steps=0),
            "max_coz_steps must be at least 1",
        ),
        (
            lambda root: root["controller"].update(max_coz_steps=True),
            "max_coz_steps must be an integer",
        ),
        (
            lambda root: root["fourkagent"].update(mode="fake"),
            "fourkagent.mode must be 'upstream'",
        ),
        (
            lambda root: root["coz"].update(mode="command"),
            "coz.mode must be 'persistent'",
        ),
        (
            lambda root: root.update(metrics=[]),
            "base config.metrics must be a string-keyed mapping",
        ),
    ],
)
def test_base_config_rejects_non_paired_semantics(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], Any],
    message: str,
) -> None:
    document = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    mutate(document)

    with pytest.raises(ExperimentProtocolError, match=message):
        experiments._load_base_config(_write_yaml(tmp_path / "base.yaml", document))


def test_input_seed_and_sample_boundaries(tmp_path: Path) -> None:
    with pytest.raises(ExperimentProtocolError, match="at least one --input"):
        experiments._load_inputs([])

    unusual = tmp_path / "authorized.EXTENSION-TOO-LONG"
    unusual.write_bytes(b"authorized")
    source = experiments._load_inputs([unusual])[0]
    assert source.suffix == ".input"

    base = {"coz": {"seed": 11}}
    assert experiments._validate_seeds(None, base) == (11,)
    for invalid in ([True], [-1], [2**63]):
        with pytest.raises(ExperimentProtocolError, match="between 0 and"):
            experiments._validate_seeds(invalid, base)
    with pytest.raises(ExperimentProtocolError, match="duplicate seeds"):
        experiments._validate_seeds([3, 3], base)
    with pytest.raises(ExperimentProtocolError, match="unsafe sample ID"):
        experiments._sample_id("/" * 64, 1)

    duplicate = experiments._InputSource(
        requested_path=str(unusual),
        path=unusual,
        size_bytes=source.size_bytes,
        sha256=source.sha256,
        suffix=source.suffix,
    )
    with pytest.raises(ExperimentProtocolError, match="duplicate sample ID"):
        experiments._validate_sample_ids([source, duplicate], [11])


def test_tree_and_output_boundaries_reject_unsafe_filesystem_state(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(ExperimentProtocolError, match="run root is missing or unsafe"):
        experiments._tree_inventory(missing)

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ExperimentProtocolError, match="contains no artifacts"):
        experiments._tree_inventory(empty)

    tree = tmp_path / "tree"
    tree.mkdir()
    target = tmp_path / "target"
    target.write_text("target", encoding="utf-8")
    (tree / "escape").symlink_to(target)
    with pytest.raises(ExperimentProtocolError, match="must not be a symlink"):
        experiments._tree_inventory(tree)

    project = tmp_path / "project"
    project.mkdir()
    output_target = tmp_path / "output-target"
    output_target.mkdir()
    output_link = tmp_path / "output-link"
    output_link.symlink_to(output_target, target_is_directory=True)
    with pytest.raises(ExperimentProtocolError, match="must not be a symlink"):
        experiments._safe_output_directory(output_link, project)

    file_output = tmp_path / "file-output"
    file_output.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ExperimentProtocolError, match="not a directory"):
        experiments._safe_output_directory(file_output, project)


def test_snapshot_copy_and_file_evidence_detect_byte_drift(tmp_path: Path) -> None:
    source_path = tmp_path / "source.png"
    source_path.write_bytes(b"current")
    source = experiments._InputSource(
        requested_path=str(source_path),
        path=source_path,
        size_bytes=len(b"original"),
        sha256=hashlib.sha256(b"original").hexdigest(),
        suffix=".png",
    )
    with pytest.raises(ExperimentProtocolError, match="input changed before snapshotting"):
        experiments._copy_input_snapshot(source, tmp_path / "copy.png")

    evidence = {
        "path": str(source_path),
        "size_bytes": 1,
        "sha256": "0" * 64,
    }
    assert experiments._verify_file_evidence(evidence, "input") == [
        "input_size_changed",
        "input_sha256_changed",
    ]
    assert experiments._verify_file_evidence({}, "input")[0].startswith("input_unreadable:")

    with pytest.raises(ExperimentProtocolError, match="cannot read directory"):
        experiments._hash_regular_file(tmp_path, "directory")


def _inventory_fixture(tmp_path: Path) -> tuple[Path, dict[str, Any], bytes]:
    attempt = tmp_path / "attempt"
    attempt.mkdir(parents=True)
    artifact = attempt / "artifact.bin"
    artifact.write_bytes(b"artifact")
    record = _file_entry(artifact)
    record["path"] = artifact.name
    inventory = attempt / "files.json"
    document = {
        "schema_version": 1,
        "root": attempt.name,
        "files": [record],
    }
    payload = json.dumps(document).encode()
    inventory.write_bytes(payload)
    return attempt, _file_entry(inventory), payload


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda document: document.update(schema_version=2), "wrong schema or root"),
        (lambda document: document.update(files={}), "files must be a list"),
        (
            lambda document: document["files"][0].update(path="../artifact.bin"),
            "unsafe or duplicated",
        ),
        (
            lambda document: document["files"][0].update(size_bytes=-1),
            "invalid byte identity",
        ),
        (
            lambda document: document["files"][0].update(sha256="0" * 64),
            "byte identity changed",
        ),
        (lambda document: document.update(files=[]), "coverage mismatch"),
    ],
)
def test_attempt_inventory_rejects_incomplete_or_forged_records(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], Any],
    message: str,
) -> None:
    attempt, entry, payload = _inventory_fixture(tmp_path)
    document = json.loads(payload)
    mutate(document)

    digest, issues = experiments._validate_attempt_inventory(
        attempt,
        entry,
        json.dumps(document).encode(),
    )

    assert digest is None
    assert len(issues) == 1
    assert message in issues[0]


def test_attempt_inventory_accepts_complete_records_and_rejects_symlinks(tmp_path: Path) -> None:
    attempt, entry, payload = _inventory_fixture(tmp_path / "complete")
    digest, issues = experiments._validate_attempt_inventory(attempt, entry, payload)
    assert issues == []
    assert digest == experiments._canonical_sha256({"files": json.loads(payload)["files"]})

    other_attempt, other_entry, other_payload = _inventory_fixture(tmp_path / "linked")
    outside = tmp_path / "outside"
    outside.write_text("outside", encoding="utf-8")
    (other_attempt / "link").symlink_to(outside)
    digest, issues = experiments._validate_attempt_inventory(
        other_attempt,
        other_entry,
        other_payload,
    )
    assert digest is None
    assert "contains a symlink" in issues[0]


def _hardware_fixture() -> tuple[dict[str, Any], dict[str, Any], bytes]:
    gpu = {
        "logical_index": 0,
        "physical_index": "2",
        "uuid": "GPU-authorized",
        "name": "NVIDIA GeForce RTX 4090",
        "memory_total_mib": 24564,
        "driver_version": "560.35.03",
    }
    identity = {"cuda_visible_devices": "2", "selected_gpus": [gpu]}
    gpu_class = {
        "selected_gpus": [
            {
                "logical_index": 0,
                "name": gpu["name"],
                "memory_total_mib": gpu["memory_total_mib"],
                "driver_version": gpu["driver_version"],
            }
        ]
    }
    hardware = {
        "identity_sha256": experiments._wrapper_canonical_sha256(identity),
        "class_sha256": experiments._wrapper_canonical_sha256(gpu_class),
        "selected_gpu_count": 1,
        "cuda_visible_devices": "2",
    }
    preflight = {
        "schema_version": 1,
        "status": "passed",
        "git_commit": COMMIT,
        "requirements": {"minimum_gpu_count": 1},
        "cuda_visible_devices": "2",
        "selected_gpus": [gpu],
    }
    inventory = b"2,GPU-authorized,NVIDIA GeForce RTX 4090,24564,560.35.03\n"
    return hardware, preflight, inventory


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda hardware, _preflight, _inventory: hardware.update(identity_sha256="INVALID"),
            "digests must be lowercase",
        ),
        (
            lambda hardware, _preflight, _inventory: hardware.update(cuda_visible_devices=False),
            "cuda_visible_devices must be string or null",
        ),
        (
            lambda _hardware, preflight, _inventory: preflight.update(status="failed"),
            "not passed or commit-bound",
        ),
        (
            lambda hardware, _preflight, _inventory: hardware.update(selected_gpu_count=2),
            "selected GPU count is inconsistent",
        ),
        (
            lambda _hardware, preflight, _inventory: preflight["selected_gpus"][0].update(
                memory_total_mib=0
            ),
            "invalid hardware identity",
        ),
        (
            lambda _hardware, preflight, _inventory: preflight["requirements"].update(
                minimum_gpu_count=2
            ),
            "does not meet its GPU-count requirement",
        ),
        (
            lambda _hardware, _preflight, inventory: inventory.__setitem__(0, b"malformed,row\n"),
            "malformed row",
        ),
        (
            lambda _hardware, _preflight, inventory: inventory.__setitem__(
                0,
                (
                    b"2,GPU-authorized,NVIDIA GeForce RTX 4090,24564,560.35.03\n"
                    b"2,GPU-copy,NVIDIA GeForce RTX 4090,24564,560.35.03\n"
                ),
            ),
            "duplicates a physical index",
        ),
        (
            lambda _hardware, _preflight, inventory: inventory.__setitem__(
                0, b"2,GPU-authorized,NVIDIA GeForce RTX 4090,invalid,560.35.03\n"
            ),
            "invalid memory",
        ),
        (
            lambda _hardware, _preflight, inventory: inventory.__setitem__(
                0, b"2,GPU-other,NVIDIA GeForce RTX 4090,24564,560.35.03\n"
            ),
            "disagrees with raw inventory",
        ),
        (
            lambda hardware, _preflight, _inventory: hardware.update(class_sha256="0" * 64),
            "digest disagrees",
        ),
    ],
)
def test_attempt_hardware_rejects_unverifiable_identity(
    mutate: Callable[[dict[str, Any], dict[str, Any], list[bytes]], Any],
    message: str,
) -> None:
    hardware, preflight, raw_inventory = _hardware_fixture()
    inventory = [raw_inventory]
    mutate(hardware, preflight, inventory)

    verified, issues = experiments._validate_attempt_hardware(
        hardware,
        gpu_preflight_payload=json.dumps(preflight).encode(),
        gpu_inventory_payload=inventory[0],
        project_commit=COMMIT,
    )

    assert verified is None
    assert len(issues) == 1
    assert message in issues[0]


def test_attempt_hardware_accepts_byte_bound_inventory() -> None:
    hardware, preflight, inventory = _hardware_fixture()
    verified, issues = experiments._validate_attempt_hardware(
        hardware,
        gpu_preflight_payload=json.dumps(preflight).encode(),
        gpu_inventory_payload=inventory,
        project_commit=COMMIT,
    )
    assert issues == []
    assert verified is not None
    assert verified["selected_gpus"] == preflight["selected_gpus"]


def _execution_fixture(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], bytes]:
    config = tmp_path / "config.yaml"
    source = tmp_path / "input.png"
    output = tmp_path / "output.png"
    preflight = tmp_path / "runtime-preflight.json"
    config.write_text("runtime: {}\n", encoding="utf-8")
    source.write_bytes(b"input")
    output.write_bytes(b"output")
    preflight_payload = json.dumps(
        {"schema_version": 2, "status": "passed", "project_commit": COMMIT}
    ).encode()
    preflight.write_bytes(preflight_payload)
    job = {
        "config": _file_entry(config),
        "input": _file_entry(source),
        "output_path": str(output.resolve()),
    }
    preflight_entry = _file_entry(preflight)
    execution = {
        "schema_version": 1,
        "stage": "experiment",
        "status": "passed",
        "return_code": 0,
        "scaleguard_command_return_code": 0,
        "git_commit": COMMIT,
        "inputs": {
            "runtime_config": _file_entry(config),
            "input_image": _file_entry(source),
            "runtime_preflight": dict(preflight_entry),
        },
        "outputs": [_file_entry(output)],
        "model_evidence": {
            "complete": True,
            "helper_completed": True,
            "hashes_consistent": True,
        },
    }
    return execution, preflight_entry, job, preflight_payload


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda execution, _preflight: execution.update(status="failed"),
            "not passed or job-bound",
        ),
        (
            lambda execution, _preflight: execution["inputs"]["runtime_config"].update(
                path="relative.yaml"
            ),
            "input paths are not job-bound",
        ),
        (
            lambda execution, _preflight: execution["inputs"]["runtime_preflight"].update(
                sha256="0" * 64
            ),
            "runtime preflight is not pointer-bound",
        ),
        (
            lambda execution, _preflight: execution["outputs"][0].update(path="relative.png"),
            "output path is unsafe",
        ),
        (
            lambda execution, _preflight: execution["outputs"][0].update(sha256="0" * 64),
            "output is not job-bound",
        ),
        (
            lambda _execution, preflight: preflight.update(status="failed"),
            "runtime preflight is not passed",
        ),
    ],
)
def test_attempt_execution_rejects_unbound_claims(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any], dict[str, Any]], Any],
    message: str,
) -> None:
    execution, preflight_entry, job, preflight_payload = _execution_fixture(tmp_path)
    preflight = json.loads(preflight_payload)
    mutate(execution, preflight)

    issues = experiments._validate_attempt_execution(
        json.dumps(execution).encode(),
        preflight_entry,
        json.dumps(preflight).encode(),
        job=job,
        project_commit=COMMIT,
    )

    assert len(issues) == 1
    assert message in issues[0]


def test_attempt_execution_accepts_all_bound_bytes(tmp_path: Path) -> None:
    execution, preflight_entry, job, preflight_payload = _execution_fixture(tmp_path)
    assert (
        experiments._validate_attempt_execution(
            json.dumps(execution).encode(),
            preflight_entry,
            preflight_payload,
            job=job,
            project_commit=COMMIT,
        )
        == []
    )


def _system_evidence_fixture() -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    selected = [
        {
            "logical_index": index,
            "physical_index": str(index),
            "uuid": f"GPU-{index}",
            "name": "NVIDIA GeForce RTX 4090",
            "memory_total_mib": 24564,
            "driver_version": "560.35.03",
        }
        for index in range(2)
    ]
    hardware = {"selected_gpu_count": 2, "selected_gpus": selected}
    peaks = {
        str(index): {
            "uuid": f"GPU-{index}",
            "name": "NVIDIA GeForce RTX 4090",
            "memory_total_mib": 24564,
            "peak_memory_used_mib": 2048 + index * 1024,
            "peak_utilization_percent": 50 + index * 10,
        }
        for index in range(2)
    }
    execution = {
        "started_at_utc": "2026-08-08T00:00:00Z",
        "completed_at_utc": "2026-08-08T00:00:42Z",
        "duration_seconds": 42,
        "gpu_sampling": {
            "sample_count": 4,
            "sample_interval_seconds": 1.0,
            "window_started_at_utc": "2026-08-08T00:00:00Z",
            "window_completed_at_utc": "2026-08-08T00:00:01Z",
            "window_duration_seconds": 1.0,
            "boundary_tolerance_seconds": 5.0,
            "maximum_gap_tolerance_seconds": 2.0,
            "maximum_observed_gap_seconds": 1.0,
            "temporal_coverage_complete": True,
            "minimum_gpu_count": 2,
            "preflight_receipt_bound": True,
            "inventory_binding_complete": True,
            "workload_sampling_complete": True,
            "workload_observed_by_uuid": {"GPU-0": True, "GPU-1": True},
            "workload_samples_by_uuid": {"GPU-0": 1, "GPU-1": 1},
            "attribution_scope": "physical_gpu_host_level_not_process_attributed",
            "evidence_complete": True,
            "peak_by_physical_index": peaks,
            "raw_csv": "gpu-samples.csv",
        },
    }
    samples = (
        b"timestamp_utc,sample_kind,index,uuid,name,memory_used_mib,"
        b"memory_total_mib,utilization_gpu_percent\n"
        b"2026-08-08T00:00:00Z,inventory,0,GPU-0,NVIDIA GeForce RTX 4090,"
        b"1024,24564,0\n"
        b"2026-08-08T00:00:00Z,inventory,1,GPU-1,NVIDIA GeForce RTX 4090,"
        b"2048,24564,0\n"
        b"2026-08-08T00:00:01Z,workload,0,GPU-0,NVIDIA GeForce RTX 4090,"
        b"2048,24564,50\n"
        b"2026-08-08T00:00:01Z,workload,1,GPU-1,NVIDIA GeForce RTX 4090,"
        b"3072,24564,60\n"
    )
    return execution, samples, hardware


def test_attempt_system_evidence_replays_physical_gpu_samples() -> None:
    execution, samples, hardware = _system_evidence_fixture()

    verified, issues = experiments._validate_attempt_system_evidence(
        json.dumps(execution).encode(),
        samples,
        hardware,
    )

    assert issues == []
    assert verified is not None
    assert verified["duration_seconds"] == 42
    normalized = verified["gpu_sampling"]["peak_by_physical_index"]
    assert normalized["0"]["uuid_sha256"] == hashlib.sha256(b"GPU-0").hexdigest()
    assert "uuid" not in normalized["0"]

    tampered = copy.deepcopy(execution)
    tampered["gpu_sampling"]["peak_by_physical_index"]["0"]["peak_memory_used_mib"] = 9999
    rejected, rejected_issues = experiments._validate_attempt_system_evidence(
        json.dumps(tampered).encode(),
        samples,
        hardware,
    )
    assert rejected is None
    assert rejected_issues
    assert "does not replay" in rejected_issues[0]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda execution, _samples, _hardware: execution.update(duration_seconds=-1),
            "non-negative",
        ),
        (
            lambda execution, _samples, _hardware: execution["gpu_sampling"].update(
                sample_interval_seconds=0.0
            ),
            "interval",
        ),
        (
            lambda execution, _samples, _hardware: execution["gpu_sampling"].update(
                evidence_complete=False
            ),
            "must be true",
        ),
        (
            lambda execution, _samples, _hardware: execution["gpu_sampling"].update(
                temporal_coverage_complete=False
            ),
            "must be true",
        ),
        (
            lambda execution, _samples, _hardware: execution["gpu_sampling"].update(
                minimum_gpu_count=1
            ),
            "selected topology",
        ),
        (
            lambda execution, _samples, _hardware: execution["gpu_sampling"][
                "workload_observed_by_uuid"
            ].pop("GPU-1"),
            "workload maps",
        ),
        (
            lambda execution, _samples, _hardware: execution["gpu_sampling"][
                "workload_observed_by_uuid"
            ].update({"GPU-1": False}),
            "incomplete GPU workload",
        ),
        (
            lambda execution, _samples, _hardware: execution["gpu_sampling"][
                "peak_by_physical_index"
            ].pop("1"),
            "do not cover",
        ),
        (
            lambda _execution, samples, _hardware: samples.__setitem__(0, b"bad,header\n"),
            "header",
        ),
    ],
)
def test_attempt_system_evidence_rejects_incomplete_or_unbound_samples(
    mutate: Callable[[dict[str, Any], list[bytes], dict[str, Any]], Any],
    message: str,
) -> None:
    execution, sample_payload, hardware = _system_evidence_fixture()
    samples = [sample_payload]
    mutate(execution, samples, hardware)

    verified, issues = experiments._validate_attempt_system_evidence(
        json.dumps(execution).encode(),
        samples[0],
        hardware,
    )

    assert verified is None
    assert len(issues) == 1
    assert message in issues[0]


def test_attempt_system_evidence_rejects_short_csv_for_long_sampling_window() -> None:
    execution, samples, hardware = _system_evidence_fixture()
    execution["gpu_sampling"].update(
        window_completed_at_utc="2026-08-08T00:00:42Z",
        window_duration_seconds=42.0,
    )

    verified, issues = experiments._validate_attempt_system_evidence(
        json.dumps(execution).encode(),
        samples,
        hardware,
    )

    assert verified is None
    assert len(issues) == 1
    assert "do not cover the sampling window" in issues[0]


def test_attempt_system_evidence_rejects_timestamp_drift_inside_window() -> None:
    execution, samples, hardware = _system_evidence_fixture()
    drifted = samples.replace(
        b"2026-08-08T00:00:01Z,workload,1",
        b"2026-08-07T23:59:59Z,workload,1",
    )

    verified, issues = experiments._validate_attempt_system_evidence(
        json.dumps(execution).encode(),
        drifted,
        hardware,
    )

    assert verified is None
    assert len(issues) == 1
    assert "not monotonic" in issues[0]


def test_attempt_system_evidence_rejects_an_unobserved_sampling_gap() -> None:
    execution, samples, hardware = _system_evidence_fixture()
    execution["gpu_sampling"].update(
        window_completed_at_utc="2026-08-08T00:00:04Z",
        window_duration_seconds=4.0,
        maximum_observed_gap_seconds=4.0,
    )
    sparse = samples.replace(b"2026-08-08T00:00:01Z", b"2026-08-08T00:00:04Z")

    verified, issues = experiments._validate_attempt_system_evidence(
        json.dumps(execution).encode(),
        sparse,
        hardware,
    )

    assert verified is None
    assert len(issues) == 1
    assert "exceeds the maximum allowed gap" in issues[0]


def _model_evidence_fixture(
    tmp_path: Path,
    *,
    group: str = "ScaleGuard",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], bytes]:
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    source_manifest = tmp_path / "source-manifest.json"
    copied_manifest = attempt / "scaleguard-run-manifest.json"
    preflight = attempt / "runtime-preflight.json"
    output_evidence = attempt / "output-evidence.png"
    manifest = {"status": "succeeded", "completion_level": "AB_INTEGRATED"}
    source_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_payload = json.dumps(manifest).encode()
    copied_manifest.write_bytes(manifest_payload)
    preflight.write_text("preflight", encoding="utf-8")
    output_evidence.write_bytes(b"output")
    config_digest = "1" * 64
    input_digest = "2" * 64
    preflight_entry = _file_entry(preflight)
    job = {
        "group": group,
        "sample_id": "sample",
        "manifest": _file_entry(source_manifest),
        "input": {"sha256": input_digest},
        "config": {"sha256": config_digest},
    }
    output_digest = hashlib.sha256(output_evidence.read_bytes()).hexdigest()
    summary = {
        "status": "passed",
        "stage": "experiment",
        "mock": False,
        "experiment_group": group,
        "experiment_sample_id": "sample",
        "successful_coz_candidates": 0 if group == "A-only" else 1,
        "invoked_input_sha256": input_digest,
        "invoked_config_sha256": config_digest,
        "runtime_config_sha256": config_digest,
        "runtime_preflight_sha256": preflight_entry["sha256"],
        "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "manifest_status": manifest["status"],
        "completion_level": manifest["completion_level"],
        "source_manifest": str(source_manifest.resolve()),
        "runtime_preflight_path": str(preflight.resolve()),
        "output_evidence_path": str(output_evidence.resolve()),
        "output_evidence_sha256": output_digest,
        "final_output_sha256": output_digest,
        "restoration_backend": (
            "scaleguard_identity_observation" if group == "B-only" else "4kagent_upstream"
        ),
        "scale_backend": "chain_of_zoom",
    }
    return summary, preflight_entry, job, manifest_payload


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("manifest_missing", "no validated source manifest"),
        ("path", "paths are not attempt-bound"),
        ("summary", "incomplete or inconsistent"),
        ("backend", "wrong real backends"),
    ],
)
def test_attempt_model_evidence_rejects_unbound_claims(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    summary, preflight_entry, job, manifest_payload = _model_evidence_fixture(tmp_path)
    attempt = tmp_path / "attempt"
    if case == "manifest_missing":
        job["manifest"] = None
    elif case == "path":
        summary["source_manifest"] = "relative.json"
    elif case == "summary":
        summary["successful_coz_candidates"] = 99
    elif case == "backend":
        summary["scale_backend"] = "third-runtime"

    issues = experiments._validate_attempt_model_evidence(
        json.dumps(summary).encode(),
        preflight_entry,
        manifest_payload,
        attempt_dir=attempt,
        job=job,
    )

    assert len(issues) == 1
    assert message in issues[0]


def test_attempt_model_evidence_accepts_each_declared_restoration_role(tmp_path: Path) -> None:
    for group in ("A-only", "B-only", "ScaleGuard"):
        case_root = tmp_path / group
        case_root.mkdir()
        summary, preflight_entry, job, manifest_payload = _model_evidence_fixture(
            case_root,
            group=group,
        )
        assert (
            experiments._validate_attempt_model_evidence(
                json.dumps(summary).encode(),
                preflight_entry,
                manifest_payload,
                attempt_dir=case_root / "attempt",
                job=job,
            )
            == []
        )


def _pointer_fixture(tmp_path: Path, *, status: str = "failed") -> tuple[Path, dict[str, Any]]:
    attempt = tmp_path / "attempt-1"
    attempt.mkdir(parents=True)
    pointer = tmp_path / "pointer.json"
    document = {
        "schema_version": 1,
        "status": status,
        "stage": "experiment",
        "attempt_id": attempt.name,
        "attempt_dir": str(attempt.resolve()),
        "started_at_utc": "2026-07-27T00:00:00Z",
        "completed_at_utc": None if status == "running" else "2026-07-27T00:01:00Z",
        "experiment_group": "ScaleGuard",
        "experiment_sample_id": "sample",
        "files": {},
        "hardware": None,
    }
    pointer.write_text(json.dumps(document), encoding="utf-8")
    return pointer, document


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda pointer: pointer.update(schema_version=2), "wrong schema or stage"),
        (lambda pointer: pointer.update(status="unknown"), "invalid status"),
        (
            lambda pointer: pointer.update(status="running"),
            "running wrapper attempt pointer cannot be completed",
        ),
        (
            lambda pointer: pointer.update(completed_at_utc="2026-07-26T00:00:00Z"),
            "completed before it started",
        ),
        (lambda pointer: pointer.update(attempt_dir="relative"), "unsafe attempt directory"),
    ],
)
def test_attempt_pointer_rejects_invalid_lifecycle(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], Any],
    message: str,
) -> None:
    path, pointer = _pointer_fixture(tmp_path)
    mutate(pointer)
    path.write_text(json.dumps(pointer), encoding="utf-8")

    evidence, issues = experiments._inspect_wrapper_attempt(
        path,
        job={"group": "ScaleGuard", "sample_id": "sample"},
        project_commit=COMMIT,
        manifest_sha256=None,
    )

    assert evidence is None
    assert len(issues) == 1
    assert message in issues[0]


def test_attempt_pointer_reports_failed_status_and_succeeded_binding_errors(
    tmp_path: Path,
) -> None:
    failed, _ = _pointer_fixture(tmp_path / "failed")
    evidence, issues = experiments._inspect_wrapper_attempt(
        failed,
        job={"group": "ScaleGuard", "sample_id": "sample"},
        project_commit=COMMIT,
        manifest_sha256=None,
    )
    assert evidence is not None
    assert issues == ["wrapper_attempt_status:failed"]

    succeeded, pointer = _pointer_fixture(tmp_path / "succeeded", status="succeeded")
    pointer["experiment_group"] = "AB-fixed"
    succeeded.write_text(json.dumps(pointer), encoding="utf-8")
    evidence, issues = experiments._inspect_wrapper_attempt(
        succeeded,
        job={"group": "ScaleGuard", "sample_id": "sample"},
        project_commit=COMMIT,
        manifest_sha256=None,
    )
    assert evidence is not None
    assert issues[0] == "wrapper_attempt_experiment_binding_mismatch"
    assert "wrapper_attempt_files_invalid:" in issues[1]


def _experiment_manifest(group: str) -> dict[str, Any]:
    identity = group == "B-only"
    no_scale = group == "A-only"
    return {
        "mock": False,
        "status": "succeeded",
        "completion_level": "AB_INTEGRATED" if group == "ScaleGuard" else "STATIC_READY",
        "target_reached": True,
        "provenance": {
            "restoration_backend": (
                "scaleguard_identity_observation" if identity else "4kagent_upstream"
            ),
            "scale_backend": "chain_of_zoom",
        },
        "restoration_metadata": (
            {
                "backend": "scaleguard_identity_observation",
                "algorithmic_restoration": False,
            }
            if identity
            else {"backend": "4kagent_upstream"}
        ),
        "restoration_process": None if identity else {"returncode": 0},
        "scale_session_process": None if no_scale else {"returncode": 0},
        "steps": (
            []
            if no_scale
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
        "final_metrics": {"selected_scale": 1 if no_scale else 4},
    }


def test_group_manifest_contract_reports_each_unverified_runtime_layer() -> None:
    assert experiments.manifest_experiment_issues({}, "undeclared") == [
        "manifest_experiment_group_undeclared"
    ]
    assert experiments.manifest_experiment_issues(
        {"mock": True, "steps": {}},
        "ScaleGuard",
    ) == ["manifest_experiment_mock", "manifest_experiment_steps_missing"]

    a_only = _experiment_manifest("A-only")
    a_only["steps"] = [{"candidate": {}}]
    a_only["scale_session_process"] = {"returncode": 0}
    a_only["status"] = "failed"
    assert {
        "manifest_unexpected_coz_step",
        "manifest_unexpected_coz_session",
        "manifest_experiment_outcome_invalid",
    }.issubset(experiments.manifest_experiment_issues(a_only, "A-only"))

    b_only = _experiment_manifest("B-only")
    b_only["restoration_metadata"]["algorithmic_restoration"] = True
    b_only["steps"][0]["candidate"]["mock"] = True
    b_only["scale_session_process"]["returncode"] = 1
    b_only["status"] = "failed"
    assert {
        "manifest_identity_observation_unverified",
        "manifest_coz_candidate_unverified",
        "manifest_coz_session_process_unverified",
        "manifest_experiment_outcome_invalid",
    }.issubset(experiments.manifest_experiment_issues(b_only, "B-only"))

    ab_fixed = _experiment_manifest("AB-fixed")
    ab_fixed["steps"] = []
    assert experiments.manifest_experiment_issues(ab_fixed, "AB-fixed") == [
        "manifest_coz_step_count:0"
    ]

    scaleguard = _experiment_manifest("ScaleGuard")
    scaleguard["completion_level"] = "STATIC_READY"
    assert experiments.manifest_experiment_issues(scaleguard, "ScaleGuard") == [
        "manifest_scaleguard_outcome_invalid"
    ]


def test_manifest_binding_reports_all_config_and_commit_mismatches() -> None:
    group = experiments.load_ablation_protocol(PROTOCOL).groups[-1]
    assert experiments._manifest_binding_issues(
        {},
        group=group,
        sample_id="sample",
        seed=7,
        project_commit=COMMIT,
    ) == ["manifest_config_missing"]

    manifest = _experiment_manifest("ScaleGuard")
    manifest["config"] = {
        "runtime": {
            "experiment_group": "AB-fixed",
            "experiment_sample_id": "other",
        },
        "fourkagent": {"mode": "identity"},
        "coz": {"mode": "command", "seed": 8},
        "controller": {
            "target_factor": 8,
            "max_coz_steps": 2,
            "acceptance_policy": "fixed",
        },
    }
    manifest["provenance"]["project_commit"] = "b" * 40
    issues = experiments._manifest_binding_issues(
        manifest,
        group=group,
        sample_id="sample",
        seed=7,
        project_commit=COMMIT,
    )
    assert {
        "manifest_experiment_group_mismatch",
        "manifest_experiment_sample_id_mismatch",
        "manifest_fourkagent_mode_mismatch",
        "manifest_coz_seed_mismatch",
        "manifest_controller_target_factor_mismatch",
        "manifest_controller_max_coz_steps_mismatch",
        "manifest_controller_acceptance_policy_mismatch",
        "manifest_project_commit_mismatch",
    }.issubset(issues)


def test_duplicate_attempts_are_failed_without_duplicating_existing_issues() -> None:
    jobs = [
        {
            "status": "passed",
            "issues": ["duplicate_wrapper_attempt_id"],
            "wrapper_attempt": {"attempt_id": "same", "attempt_dir": "/same"},
        },
        {
            "status": "passed",
            "issues": [],
            "wrapper_attempt": {"attempt_id": "same", "attempt_dir": "/same"},
        },
        {"status": "passed", "issues": [], "wrapper_attempt": None},
    ]

    experiments._apply_attempt_uniqueness(jobs)

    assert jobs[0]["issues"].count("duplicate_wrapper_attempt_id") == 1
    assert jobs[1]["issues"] == [
        "duplicate_wrapper_attempt_id",
        "duplicate_wrapper_attempt_dir",
    ]
    assert jobs[0]["status"] == jobs[1]["status"] == "failed"
    assert jobs[2]["status"] == "passed"


def _signed_receipt(path: Path, receipt: dict[str, Any]) -> None:
    unsigned = copy.deepcopy(receipt)
    unsigned.pop("receipt_sha256", None)
    receipt["receipt_sha256"] = experiments._canonical_sha256(unsigned)
    path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")


def _suite_envelope(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    project = tmp_path / "project"
    runner = project / experiments.INTEGRATION_RUNNER
    runner.parent.mkdir(parents=True)
    runner.write_text("#!/bin/sh\n", encoding="utf-8")
    output = tmp_path / "suite"
    inputs = output / "inputs"
    inputs.mkdir(parents=True)
    snapshot = inputs / f"{hashlib.sha256(b'input').hexdigest()}.png"
    snapshot.write_bytes(b"input")
    base = tmp_path / "base.yaml"
    base.write_bytes(BASE_CONFIG.read_bytes())
    receipt_path = output / "suite-receipt.json"
    receipt: dict[str, Any] = {
        "schema_version": experiments.RECEIPT_SCHEMA,
        "status": "passed",
        "plan_only": False,
        "started_at_utc": "2026-07-27T00:00:00Z",
        "completed_at_utc": "2026-07-27T01:00:00Z",
        "project_root": str(project.resolve()),
        "project_commit": COMMIT,
        "output_directory": str(output.resolve()),
        "protocol": {
            **_file_entry(PROTOCOL),
            "name": "core-ablation",
            "status": "executable",
            "integration_runner": experiments.INTEGRATION_RUNNER,
        },
        "base_config": _file_entry(base),
        "integration_runner": _file_entry(runner),
        "groups": [dict(record) for record in experiments._GROUP_CONTRACT],
        "seeds": [7],
        "inputs": [
            {
                "requested_path": "/authorized/input.png",
                "source_path": "/authorized/input.png",
                "snapshot_path": str(snapshot.resolve()),
                "size_bytes": snapshot.stat().st_size,
                "sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
            }
        ],
        "jobs": [],
        "issues": [],
        "counts": {
            "total": 0,
            "planned": 0,
            "running": 0,
            "passed": 0,
            "failed": 0,
        },
        "receipt_sha256": "",
    }
    _signed_receipt(receipt_path, receipt)
    return receipt_path, receipt


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda receipt: receipt.update(status="planned"),
            "not a passed real suite",
        ),
        (
            lambda receipt: receipt.update(completed_at_utc="2026-07-26T00:00:00Z"),
            "completion precedes",
        ),
        (
            lambda receipt: receipt.update(project_root="relative"),
            "unsafe project root",
        ),
        (
            lambda receipt: receipt.update(project_commit="invalid"),
            "invalid project commit",
        ),
        (
            lambda receipt: receipt.update(output_directory="/"),
            "unsafe output directory",
        ),
        (
            lambda receipt: receipt["protocol"].update(name="other"),
            "invalid protocol binding",
        ),
        (
            lambda receipt: receipt.update(groups=[*receipt["groups"][:-1], {"id": "other"}]),
            "group semantics differ",
        ),
        (
            lambda receipt: receipt.update(seeds=[7, 7]),
            "invalid seed set",
        ),
        (
            lambda receipt: receipt.update(inputs=[]),
            "no input snapshots",
        ),
    ],
)
def test_suite_receipt_rejects_outer_envelope_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate: Callable[[dict[str, Any]], Any],
    message: str,
) -> None:
    path, receipt = _suite_envelope(tmp_path)
    mutate(receipt)
    _signed_receipt(path, receipt)
    monkeypatch.setattr(experiments, "_clean_commit", lambda *_args: COMMIT)

    with pytest.raises(ExperimentProtocolError, match=message):
        experiments.validate_ablation_suite_receipt(path)


def test_suite_receipt_rejects_self_digest_and_clean_commit_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, receipt = _suite_envelope(tmp_path)
    receipt["receipt_sha256"] = "0" * 64
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ExperimentProtocolError, match="self digest is invalid"):
        experiments.validate_ablation_suite_receipt(path)

    path, receipt = _suite_envelope(tmp_path / "commit")
    _signed_receipt(path, receipt)
    monkeypatch.setattr(experiments, "_clean_commit", lambda *_args: "b" * 40)
    with pytest.raises(ExperimentProtocolError, match="another project commit"):
        experiments.validate_ablation_suite_receipt(path)


def test_suite_receipt_rejects_missing_jobs_and_inconsistent_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, receipt = _suite_envelope(tmp_path)
    monkeypatch.setattr(experiments, "_clean_commit", lambda *_args: COMMIT)
    with pytest.raises(ExperimentProtocolError, match="has no jobs"):
        experiments.validate_ablation_suite_receipt(path)

    receipt["jobs"] = [{}]
    receipt["counts"]["total"] = 1
    _signed_receipt(path, receipt)
    with pytest.raises(ExperimentProtocolError, match="incomplete job matrix"):
        experiments.validate_ablation_suite_receipt(path)


def test_suite_file_and_input_snapshot_reject_byte_identity_drift(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.write_bytes(b"artifact")
    entry = _file_entry(artifact)

    invalid = dict(entry)
    invalid["size_bytes"] = -1
    with pytest.raises(ExperimentProtocolError, match="invalid byte identity"):
        experiments._validated_suite_file(invalid, context="artifact")

    changed = dict(entry)
    changed["sha256"] = "0" * 64
    with pytest.raises(ExperimentProtocolError, match="byte identity changed"):
        experiments._validated_suite_file(changed, context="artifact")

    output = tmp_path / "suite"
    (output / "inputs").mkdir(parents=True)
    wrong = tmp_path / "wrong.png"
    wrong.write_bytes(b"input")
    record = {
        "requested_path": "/source",
        "source_path": "/source",
        "snapshot_path": str(wrong.resolve()),
        "size_bytes": wrong.stat().st_size,
        "sha256": hashlib.sha256(wrong.read_bytes()).hexdigest(),
    }
    with pytest.raises(ExperimentProtocolError, match="snapshot byte identity changed"):
        experiments._validated_suite_inputs([record], output_root=output)

    duplicate = copy.deepcopy(record)
    correct = output / "inputs" / f"{record['sha256']}.png"
    correct.write_bytes(wrong.read_bytes())
    record["snapshot_path"] = str(correct.resolve())
    duplicate["snapshot_path"] = str(correct.resolve())
    with pytest.raises(ExperimentProtocolError, match="invalid or duplicate identity"):
        experiments._validated_suite_inputs([record, duplicate], output_root=output)


def test_declared_group_sequence_is_exact() -> None:
    protocol = experiments.load_ablation_protocol(PROTOCOL)
    groups = experiments._suite_group_specs(
        [dict(record) for record in experiments._GROUP_CONTRACT],
        protocol=protocol,
    )
    assert tuple(groups) == EXPERIMENT_GROUPS
