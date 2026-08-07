from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _receipt_writer() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "scaleguard_test_gpu_preflight_binding",
        ROOT / "scripts" / "autodl" / "_write_preflight_receipt.py",
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _gpu_document(commit: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "passed",
        "git_commit": commit,
        "requirements": {"minimum_gpu_count": 2},
        "cuda_visible_devices": "GPU-first,GPU-second",
        "selected_gpus": [
            {
                "logical_index": 0,
                "physical_index": "0",
                "uuid": "GPU-first",
                "name": "NVIDIA GeForce RTX 4090",
                "memory_total_mib": 24564,
                "driver_version": "595.71.05",
            },
            {
                "logical_index": 1,
                "physical_index": "1",
                "uuid": "GPU-second",
                "name": "NVIDIA GeForce RTX 4090",
                "memory_total_mib": 24564,
                "driver_version": "595.71.05",
            },
        ],
    }


def test_gpu_preflight_binding_preserves_the_canonical_uuid_map(tmp_path: Path) -> None:
    writer = _receipt_writer()
    commit = "a" * 40
    path = tmp_path / "gpu-preflight" / "gpu_check.json"
    path.parent.mkdir()
    path.write_text(json.dumps(_gpu_document(commit)) + "\n", encoding="utf-8")

    binding = writer._gpu_preflight_binding(path, expected_path=path, commit=commit)

    assert binding["path"] == str(path.resolve())
    assert binding["cuda_visible_devices"] == "GPU-first,GPU-second"
    assert [item["uuid"] for item in binding["selected_gpus"]] == [
        "GPU-first",
        "GPU-second",
    ]


@pytest.mark.parametrize("mutation", ["physical-index", "duplicate-uuid", "gpu-count"])
def test_gpu_preflight_binding_rejects_topology_and_identity_mismatch(
    tmp_path: Path,
    mutation: str,
) -> None:
    writer = _receipt_writer()
    commit = "b" * 40
    document = _gpu_document(commit)
    selected = document["selected_gpus"]
    assert isinstance(selected, list)
    assert isinstance(selected[1], dict)
    if mutation == "physical-index":
        selected[1]["physical_index"] = "2"
    elif mutation == "duplicate-uuid":
        selected[1]["uuid"] = "GPU-first"
    else:
        requirements = document["requirements"]
        assert isinstance(requirements, dict)
        requirements["minimum_gpu_count"] = 1
    path = tmp_path / "gpu-preflight" / "gpu_check.json"
    path.parent.mkdir()
    path.write_text(json.dumps(document) + "\n", encoding="utf-8")

    with pytest.raises(writer.RuntimePreflightError):
        writer._gpu_preflight_binding(path, expected_path=path, commit=commit)
