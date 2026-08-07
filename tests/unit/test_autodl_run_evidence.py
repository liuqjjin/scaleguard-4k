from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from scaleguard.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_evidence_module() -> ModuleType:
    path = PROJECT_ROOT / "scripts" / "autodl" / "_extract_run_evidence.py"
    specification = importlib.util.spec_from_file_location(
        "scaleguard_autodl_run_evidence",
        path,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _write_config(path: Path, *, group: str | None) -> None:
    semantics = {
        None: ("upstream", 4, 1, "trusted"),
        "A-only": ("upstream", 1, 0, "fixed"),
        "B-only": ("identity", 4, 1, "fixed"),
        "AB-fixed": ("upstream", 4, 1, "fixed"),
        "ScaleGuard": ("upstream", 4, 1, "trusted"),
    }
    restoration_mode, target, steps, policy = semantics[group]
    experiment_fields = (
        ""
        if group is None
        else (f'  experiment_group: "{group}"\n  experiment_sample_id: "sample-{group.lower()}"\n')
    )
    path.write_text(
        f"""
runtime:
  run_root: relative-runs
{experiment_fields}fourkagent:
  mode: {restoration_mode}
  checkout: third_party/fourkagent
  depictqa_command: ["python", "serve.py"]
  depictqa_cwd: third_party/depictqa
  perception_model_path: weights/fourkagent-qwen
  toolbox_root: weights/fourkagent-toolbox
  hps_root: weights/fourkagent-hps
  quality_model_path: weights/musiq
coz:
  mode: persistent
  checkout: third_party/chain-of-zoom
  model_path: weights/coz-sd3
  qwen_model_path: weights/coz-qwen
  sr_lora_path: weights/coz-sr-lora
  vae_path: weights/coz-vae
  vlm_lora_path: weights/coz-vlm-lora
metrics:
  quality_backend: pyiqa
  quality_model_path: weights/musiq
controller:
  target_factor: {target}
  max_coz_steps: {steps}
  color_strategy: none
  acceptance_policy: {policy}
""",
        encoding="utf-8",
    )


def _bound_config(raw: dict[str, Any], root: Path) -> dict[str, Any]:
    result = copy.deepcopy(raw)
    path_fields = {
        "fourkagent": (
            "checkout",
            "depictqa_cwd",
            "perception_model_path",
            "toolbox_root",
            "hps_root",
            "quality_model_path",
        ),
        "coz": (
            "checkout",
            "model_path",
            "qwen_model_path",
            "sr_lora_path",
            "vae_path",
            "vlm_lora_path",
        ),
        "metrics": ("quality_model_path",),
    }
    for section, fields in path_fields.items():
        for field in fields:
            value = result[section][field]
            result[section][field] = str((root / value).resolve())
    return result


def _process() -> dict[str, Any]:
    return {"returncode": 0}


def _sync_manifest(arguments: dict[str, Any], manifest: dict[str, Any]) -> None:
    arguments["manifest_path"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fixture(
    tmp_path: Path,
    make_image: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
    *,
    group: str | None = None,
    rollback: bool = False,
) -> tuple[ModuleType, dict[str, Any], dict[str, Any], dict[str, Any]]:
    module = _load_evidence_module()
    run_root = tmp_path / "relative-runs"
    run_dir = run_root / "attempt-1"
    run_dir.mkdir(parents=True)
    config_path = tmp_path / "runtime.yaml"
    _write_config(config_path, group=group)
    raw_config = load_config(config_path)
    bound_config = _bound_config(raw_config.as_dict(), tmp_path)

    expected_input = make_image(tmp_path / "source.png", size=(8, 6))
    normalized_input = run_dir / "input.png"
    shutil.copy2(expected_input, normalized_input)
    output_size = (8, 6) if group == "A-only" else (32, 24)
    internal_final = make_image(run_dir / "final.png", size=output_size)
    external_output = tmp_path / "published.png"
    shutil.copy2(internal_final, external_output)

    is_a_only = group == "A-only"
    is_b_only = group == "B-only"
    candidate_path = run_dir / "candidate.png"
    steps: list[dict[str, Any]] = []
    if not is_a_only:
        make_image(candidate_path, size=(32, 24))
        steps.append(
            {
                "accepted": not rollback,
                "decision": "rollback" if rollback else "stop",
                "candidate": {
                    "path": str(candidate_path.resolve()),
                    "sha256": module.sha256(candidate_path),
                    "mock": False,
                },
                "worker_metadata": {
                    "backend": "chain_of_zoom_persistent",
                    "candidate_sha256": module.sha256(candidate_path),
                },
            }
        )

    runtime_preflight = tmp_path / "runtime-preflight.json"
    runtime_preflight.write_text('{"schema_version":2,"status":"passed"}\n', encoding="utf-8")
    validated_provenance = {
        "runtime_evidence_verified": True,
        "runtime_profile_bound": True,
        "runtime_preflight_receipt": str(runtime_preflight.resolve()),
        "runtime_preflight_sha256": module.sha256(runtime_preflight),
        "runtime_config_path": str(config_path.resolve()),
        "runtime_config_sha256": module.sha256(config_path),
        "project_root": str(tmp_path.resolve()),
        "runtime_execution_binding": {"schema_version": 1, "fixture": True},
        "runtime_execution_binding_sha256": "a" * 64,
    }
    restoration_backend = "scaleguard_identity_observation" if is_b_only else "4kagent_upstream"
    status = "succeeded_with_rollback" if rollback else "succeeded"
    completion = (
        "COMPONENT_REPRODUCED"
        if rollback
        else ("STATIC_READY" if group in {"A-only", "B-only", "AB-fixed"} else "AB_INTEGRATED")
    )
    manifest = {
        "mock": False,
        "status": status,
        "completion_level": completion,
        "target_reached": not rollback,
        "run_id": run_dir.name,
        "started_at": "2026-07-27T00:00:00Z",
        "finished_at": "2026-07-27T00:01:00Z",
        "config": bound_config,
        "provenance": {
            **validated_provenance,
            "restoration_backend": restoration_backend,
            "scale_backend": "chain_of_zoom",
        },
        "input_image": {
            "path": str(normalized_input.resolve()),
            "sha256": module.sha256(normalized_input),
            "mock": False,
        },
        "restoration_metadata": (
            {
                "backend": "scaleguard_identity_observation",
                "algorithmic_restoration": False,
            }
            if is_b_only
            else {"backend": "4kagent_upstream"}
        ),
        "restoration_process": None if is_b_only else _process(),
        "scale_session_process": None if is_a_only else _process(),
        "events": [{"event": "restoration_completed"}],
        "steps": steps,
        "final_image": {
            "path": str(internal_final.resolve()),
            "sha256": module.sha256(internal_final),
            "mock": False,
        },
    }
    manifest_path = tmp_path / "scaleguard-run-manifest.json"
    arguments = {
        "expected_output": external_output.resolve(),
        "expected_input": expected_input.resolve(),
        "expected_config": config_path.resolve(),
        "project_root": tmp_path.resolve(),
        "run_dir": run_dir.resolve(),
        "wrapper_started_at": datetime(2026, 7, 26, tzinfo=timezone.utc),
        "expected_output_sha256": module.sha256(external_output),
        "manifest_path": manifest_path.resolve(),
        "runtime_preflight": runtime_preflight.resolve(),
        "stage": "integration" if group is None else "experiment",
    }
    _sync_manifest(arguments, manifest)

    calls: dict[str, Any] = {}

    def validate_manifest(path: Path) -> dict[str, Any]:
        calls["manifest_path"] = path
        return json.loads(path.read_text(encoding="utf-8"))

    def validate_preflight(
        path: Path,
        *,
        config_path: Path,
        project_root: Path,
    ) -> dict[str, Any]:
        calls["preflight"] = (path, config_path, project_root)
        return copy.deepcopy(validated_provenance)

    def bind_config(
        parsed: Any,
        *,
        project_root: Path,
        binding: dict[str, Any],
    ) -> Any:
        calls["raw_checkout"] = parsed.fourkagent.checkout
        calls["binding"] = binding
        calls["binding_root"] = project_root
        return SimpleNamespace(
            runtime=raw_config.runtime,
            as_dict=lambda: copy.deepcopy(bound_config),
        )

    monkeypatch.setattr(module, "validate_run_manifest", validate_manifest)
    monkeypatch.setattr(module, "validate_runtime_preflight", validate_preflight)
    monkeypatch.setattr(module, "bind_runtime_config", bind_config)
    return module, manifest, arguments, calls


def test_relative_runtime_paths_are_compared_after_preflight_binding(
    tmp_path: Path,
    make_image: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, manifest, arguments, calls = _fixture(
        tmp_path,
        make_image,
        monkeypatch,
    )

    summary = module.validate_manifest(manifest, **arguments)

    assert summary["completion_level"] == "AB_INTEGRATED"
    assert calls["raw_checkout"] == Path("third_party/fourkagent")
    assert Path(manifest["config"]["fourkagent"]["checkout"]).is_absolute()
    assert calls["manifest_path"] == arguments["manifest_path"]
    assert calls["preflight"][0] == arguments["runtime_preflight"]


@pytest.mark.parametrize(
    ("group", "rollback", "completion"),
    [
        ("A-only", False, "STATIC_READY"),
        ("B-only", False, "STATIC_READY"),
        ("AB-fixed", False, "STATIC_READY"),
        ("ScaleGuard", False, "AB_INTEGRATED"),
        ("ScaleGuard", True, "COMPONENT_REPRODUCED"),
    ],
)
def test_experiment_group_contracts_accept_only_the_declared_success_shape(
    tmp_path: Path,
    make_image: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
    group: str,
    rollback: bool,
    completion: str,
) -> None:
    module, manifest, arguments, _calls = _fixture(
        tmp_path,
        make_image,
        monkeypatch,
        group=group,
        rollback=rollback,
    )

    summary = module.validate_manifest(manifest, **arguments)

    assert summary["experiment_group"] == group
    assert summary["completion_level"] == completion
    assert summary["successful_coz_candidates"] == (0 if group == "A-only" else 1)


def test_scaleguard_accepts_a_quality_gate_stop_as_component_reproduced(
    tmp_path: Path,
    make_image: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, manifest, arguments, _calls = _fixture(
        tmp_path,
        make_image,
        monkeypatch,
        group="ScaleGuard",
        rollback=True,
    )
    manifest["steps"][0]["decision"] = "stop"
    _sync_manifest(arguments, manifest)

    summary = module.validate_manifest(manifest, **arguments)

    assert summary["completion_level"] == "COMPONENT_REPRODUCED"
    assert summary["successful_coz_candidates"] == 1


def test_scaleguard_accepts_post_adain_final_gate_rollback(
    tmp_path: Path,
    make_image: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, manifest, arguments, _calls = _fixture(
        tmp_path,
        make_image,
        monkeypatch,
        group="ScaleGuard",
        rollback=True,
    )
    manifest["steps"][0].update(accepted=True, decision="stop")
    manifest["events"].append({"event": "final_gate_rollback", "from_scale": 4.0, "to_scale": 2.0})
    manifest["final_metrics"] = {"selected_scale": 2.0}
    _sync_manifest(arguments, manifest)

    summary = module.validate_manifest(manifest, **arguments)

    assert summary["completion_level"] == "COMPONENT_REPRODUCED"
    assert summary["successful_coz_candidates"] == 1


def test_scaleguard_rejects_an_accepted_step_without_final_gate_rollback_evidence(
    tmp_path: Path,
    make_image: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, manifest, arguments, _calls = _fixture(
        tmp_path,
        make_image,
        monkeypatch,
        group="ScaleGuard",
        rollback=True,
    )
    manifest["steps"][0].update(accepted=True, decision="stop")
    _sync_manifest(arguments, manifest)

    with pytest.raises(
        module.EvidenceError,
        match="AB_INTEGRATED success or COMPONENT_REPRODUCED",
    ):
        module.validate_manifest(manifest, **arguments)


@pytest.mark.parametrize(
    ("group", "mutation", "message"),
    [
        (
            "A-only",
            lambda manifest: manifest["steps"].append({"candidate": None}),
            "exactly 0 CoZ step",
        ),
        (
            "B-only",
            lambda manifest: manifest.update(steps=[]),
            "exactly 1 CoZ step",
        ),
        (
            "B-only",
            lambda manifest: manifest["scale_session_process"].update(returncode=1),
            "successful persistent Chain-of-Zoom session",
        ),
        (
            "AB-fixed",
            lambda manifest: manifest.update(restoration_process=None),
            "successful real 4KAgent process",
        ),
        (
            "ScaleGuard",
            lambda manifest: manifest.update(completion_level="COMPONENT_REPRODUCED"),
            "AB_INTEGRATED success or COMPONENT_REPRODUCED",
        ),
    ],
)
def test_experiment_group_contracts_reject_missing_or_wrong_runtime_evidence(
    tmp_path: Path,
    make_image: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
    group: str,
    mutation: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    module, manifest, arguments, _calls = _fixture(
        tmp_path,
        make_image,
        monkeypatch,
        group=group,
    )
    mutation(manifest)
    _sync_manifest(arguments, manifest)

    with pytest.raises(module.EvidenceError, match=message):
        module.validate_manifest(manifest, **arguments)


def test_smoke_and_integration_require_exact_ab_integrated_completion(
    tmp_path: Path,
    make_image: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, manifest, arguments, _calls = _fixture(
        tmp_path,
        make_image,
        monkeypatch,
    )
    manifest["completion_level"] = "COMPONENT_REPRODUCED"
    _sync_manifest(arguments, manifest)

    with pytest.raises(module.EvidenceError, match="requires successful AB_INTEGRATED"):
        module.validate_manifest(manifest, **arguments)


def test_autodl_evidence_rejects_published_bytes_that_drift_from_the_run_artifact(
    tmp_path: Path,
    make_image: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, manifest, arguments, _calls = _fixture(
        tmp_path,
        make_image,
        monkeypatch,
    )
    external_output = arguments["expected_output"]
    make_image(external_output, size=(32, 24), color=(200, 20, 10))
    arguments["expected_output_sha256"] = module.sha256(external_output)

    with pytest.raises(
        module.EvidenceError,
        match="published output bytes differ from the internal final artifact",
    ):
        module.validate_manifest(manifest, **arguments)


def test_full_manifest_validation_failure_is_fail_closed(
    tmp_path: Path,
    make_image: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, manifest, arguments, _calls = _fixture(
        tmp_path,
        make_image,
        monkeypatch,
    )

    def reject(_path: Path) -> dict[str, Any]:
        raise module.ManifestValidationError("candidate artifact SHA-256 changed")

    monkeypatch.setattr(module, "validate_run_manifest", reject)

    with pytest.raises(module.EvidenceError, match="full run-manifest validation failed"):
        module.validate_manifest(manifest, **arguments)


def test_preflight_provenance_drift_is_rejected(
    tmp_path: Path,
    make_image: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, manifest, arguments, _calls = _fixture(
        tmp_path,
        make_image,
        monkeypatch,
        group="ScaleGuard",
    )
    manifest["provenance"]["runtime_preflight_sha256"] = "0" * 64
    _sync_manifest(arguments, manifest)

    with pytest.raises(module.EvidenceError, match="runtime provenance differs"):
        module.validate_manifest(manifest, **arguments)


def test_autodl_evidence_rejects_ambiguous_cli_result(tmp_path: Path) -> None:
    module = _load_evidence_module()
    log = tmp_path / "run.log"
    log.write_text(
        '{"status":"ok","status":"failed","run_dir":"/trusted","output":"/trusted.png"}\n',
        encoding="utf-8",
    )

    with pytest.raises(module.EvidenceError, match="emitted no successful run JSON"):
        module.find_cli_result(log)


def test_experiment_wrapper_uses_stage_and_preflight_evidence_arguments() -> None:
    runner = PROJECT_ROOT / "scripts" / "autodl" / "_run_scaleguard.sh"
    wrapper = PROJECT_ROOT / "scripts" / "autodl" / "run_experiment.sh"
    runner_text = runner.read_text(encoding="utf-8")
    wrapper_text = wrapper.read_text(encoding="utf-8")

    assert '--stage "${sg_stage}"' in runner_text
    assert '--runtime-preflight "${SG_RUN_DIR}/runtime-preflight.json"' in runner_text
    assert "--evidence-output" in runner_text
    assert '"${sg_here}/_run_scaleguard.sh" experiment "$@"' in wrapper_text
    assert "--group" not in runner_text
    assert "--group" not in wrapper_text
    subprocess.run(
        ["bash", "-n", str(runner), str(wrapper)],
        check=True,
        cwd=PROJECT_ROOT,
    )


def test_wrapper_gpu_receipt_fails_closed_when_monitor_stops_early(tmp_path: Path) -> None:
    runner = PROJECT_ROOT / "scripts" / "autodl" / "_run_scaleguard.sh"
    runner_text = runner.read_text(encoding="utf-8")
    execution_section = runner_text.split(
        'python3 -I - \\\n    "${SG_RUN_DIR}/execution.json"',
        1,
    )[1]
    execution_program = execution_section.split("<<'PY'\n", 1)[1].split(
        '\nPY\n\nif [[ "${sg_command_rc}"',
        1,
    )[0]

    commit = "a" * 40
    selected_gpus = [
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
    gpu_check = tmp_path / "gpu-check.json"
    gpu_check.write_text(
        json.dumps(
            {
                "status": "passed",
                "git_commit": commit,
                "selected_gpus": selected_gpus,
            }
        ),
        encoding="utf-8",
    )
    runtime_preflight = tmp_path / "runtime-preflight.json"
    runtime_preflight.write_text(
        json.dumps(
            {
                "gpu_preflight": {
                    "path": str(gpu_check.resolve()),
                    "sha256": hashlib.sha256(gpu_check.read_bytes()).hexdigest(),
                    "selected_gpus": selected_gpus,
                }
            }
        ),
        encoding="utf-8",
    )
    gpu_samples = tmp_path / "gpu-samples.csv"
    gpu_samples.write_text(
        "timestamp_utc,sample_kind,index,uuid,name,memory_used_mib,"
        "memory_total_mib,utilization_gpu_percent\n"
        "2026-08-08T00:00:00Z,inventory,0,GPU-0,NVIDIA GeForce RTX 4090,"
        "1024,24564,0\n"
        "2026-08-08T00:00:00Z,inventory,1,GPU-1,NVIDIA GeForce RTX 4090,"
        "2048,24564,0\n"
        "2026-08-08T00:00:01Z,workload,0,GPU-0,NVIDIA GeForce RTX 4090,"
        "2048,24564,50\n"
        "2026-08-08T00:00:01Z,workload,1,GPU-1,NVIDIA GeForce RTX 4090,"
        "3072,24564,60\n",
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    source = tmp_path / "input.png"
    upstream_lock = tmp_path / "upstream-lock.yaml"
    dependency_lock = tmp_path / "runtime-dependencies.yaml"
    for path in (config, source, upstream_lock, dependency_lock):
        path.write_bytes(path.name.encode())
    execution = tmp_path / "execution.json"
    missing_output = tmp_path / "output.png"

    completed = subprocess.run(
        [
            sys.executable,
            "-",
            str(execution),
            "experiment",
            "1",
            "1",
            "0",
            "0",
            "2026-08-08T00:00:00Z",
            "2026-08-08T00:00:42Z",
            "42",
            str(gpu_samples),
            str(config),
            str(source),
            str(missing_output),
            str(upstream_lock),
            str(dependency_lock),
            commit,
            "2",
            str(gpu_check),
            str(runtime_preflight),
            "60",
            "2026-08-08T00:00:00Z",
            "2026-08-08T00:00:42Z",
            "1",
            str(PROJECT_ROOT / "src"),
        ],
        input=execution_program,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    document = json.loads(execution.read_text(encoding="utf-8"))
    sampling = document["gpu_sampling"]
    assert sampling["window_duration_seconds"] == 42.0
    assert sampling["maximum_observed_gap_seconds"] == 1.0
    assert sampling["temporal_coverage_complete"] is False
    assert sampling["workload_sampling_complete"] is False
    assert sampling["evidence_complete"] is False


def test_experiment_pointer_is_machine_readable_and_hash_bound(tmp_path: Path) -> None:
    runner = PROJECT_ROOT / "scripts" / "autodl" / "_run_scaleguard.sh"
    runner_text = runner.read_text(encoding="utf-8")
    function_text = runner_text.split("sg_write_attempt_pointer() {", 1)[1]
    pointer_program = function_text.split("<<'PY'\n", 1)[1].split(
        "\nPY\n}\n\nsg_finalize_failed_stage",
        1,
    )[0]

    attempt = tmp_path / "attempt-1"
    (attempt / "gpu-preflight").mkdir(parents=True)
    files = {
        "execution.json": '{"status":"passed"}\n',
        "scaleguard-run-manifest.json": '{"status":"succeeded"}\n',
        "model-evidence.json": (
            '{"experiment_group":"ScaleGuard","experiment_sample_id":"sample-scaleguard"}\n'
        ),
        "experiment.log": "raw log\n",
        "gpu-samples.csv": "index,uuid,name\n0,GPU-a,RTX 4090\n",
        "nvidia-smi-before.txt": "temperature=41\n",
        "nvidia-smi-after.txt": "temperature=57\n",
        "gpu-preflight/gpu_inventory.csv": (
            "0, GPU-a, NVIDIA GeForce RTX 4090, 24564, 570.00\n"
            "1, GPU-b, NVIDIA GeForce RTX 4090, 24564, 570.00\n"
        ),
        "gpu-preflight/gpu_check.json": json.dumps(
            {
                "cuda_visible_devices": "0,1",
                "selected_gpus": [
                    {
                        "logical_index": 0,
                        "physical_index": "0",
                        "uuid": "GPU-a",
                        "name": "NVIDIA GeForce RTX 4090",
                        "memory_total_mib": 24564,
                        "driver_version": "570.00",
                    },
                    {
                        "logical_index": 1,
                        "physical_index": "1",
                        "uuid": "GPU-b",
                        "name": "NVIDIA GeForce RTX 4090",
                        "memory_total_mib": 24564,
                        "driver_version": "570.00",
                    },
                ],
            }
        ),
        "files.json": '{"schema_version":1,"files":[]}\n',
        "runtime-preflight.json": '{"schema_version":2,"status":"passed"}\n',
    }
    for relative, content in files.items():
        path = attempt / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    pointer = tmp_path / "attempt-pointer.json"
    subprocess.run(
        [
            sys.executable,
            "-",
            str(pointer),
            str(attempt),
            "running",
            "2026-07-27T00:00:00Z",
            "",
        ],
        input=pointer_program,
        text=True,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-",
            str(pointer),
            str(attempt),
            "succeeded",
            "2026-07-27T00:00:00Z",
            "2026-07-27T00:10:00Z",
        ],
        input=pointer_program,
        text=True,
        check=True,
    )
    document = json.loads(pointer.read_text(encoding="utf-8"))

    assert document["status"] == "succeeded"
    assert document["attempt_dir"] == str(attempt.resolve())
    assert document["experiment_group"] == "ScaleGuard"
    assert document["experiment_sample_id"] == "sample-scaleguard"
    assert set(document["files"]) == {
        "execution",
        "run_manifest",
        "model_evidence",
        "raw_log",
        "gpu_samples",
        "nvidia_smi_before",
        "nvidia_smi_after",
        "gpu_inventory",
        "gpu_preflight",
        "files_inventory",
        "runtime_preflight",
    }
    for item in document["files"].values():
        path = Path(item["path"])
        assert item["size_bytes"] == path.stat().st_size
        assert item["sha256"] == _load_evidence_module().sha256(path)
    assert document["hardware"]["selected_gpu_count"] == 2
    assert len(document["hardware"]["identity_sha256"]) == 64
    assert len(document["hardware"]["class_sha256"]) == 64


def test_experiment_help_exposes_required_machine_handoff() -> None:
    wrapper = PROJECT_ROOT / "scripts" / "autodl" / "run_experiment.sh"

    completed = subprocess.run(
        [str(wrapper), "--help"],
        check=True,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert "--evidence-output FILE" in completed.stdout
    assert "preflighted config" in completed.stdout
