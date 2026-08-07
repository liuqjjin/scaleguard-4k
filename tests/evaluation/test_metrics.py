from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

from scaleguard.cli import main
from scaleguard.evaluation import metrics as metrics_module
from scaleguard.evaluation.evidence import EvaluationEvidenceError, canonical_sha256
from scaleguard.evaluation.metrics import (
    METRIC_RECEIPT_SCHEMA,
    evaluate_metric_receipt,
    psnr_rgb,
    ssim_rgb,
    verify_metric_receipt,
)
from scaleguard.manifest import ManifestValidationError

from ._fixtures import artifact, write_summary_manifest


@pytest.fixture(autouse=True)
def _accept_minimal_metric_fixtures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        metrics_module,
        "validate_run_manifest",
        lambda path, **_kwargs: json.loads(path.read_text(encoding="utf-8")),
    )


def _fail_project_root_resolution() -> Path:
    raise AssertionError("project root must not be resolved")


def _rgb(path: Path, value: int, *, size: tuple[int, int] = (16, 16)) -> Path:
    Image.new("RGB", size, (value, value, value)).save(path)
    return path


def _manifest(
    path: Path,
    *,
    run_id: str,
    source: Path,
    output: Path,
    mock: bool = False,
) -> Path:
    manifest = write_summary_manifest(
        path,
        run_id=run_id,
        source=source,
        final=output,
        mock=mock,
    )
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    for key, image_path in (("input_image", source), ("final_image", output)):
        with Image.open(image_path) as image:
            raw[key]["width"] = image.width
            raw[key]["height"] = image.height
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    return manifest


def test_rgb_psnr_and_ssim_definitions() -> None:
    black = np.zeros((11, 11, 3), dtype=np.uint8)
    white = np.full((11, 11, 3), 255, dtype=np.uint8)

    assert psnr_rgb(black, black) == "infinity"
    assert psnr_rgb(black, white) == pytest.approx(0.0)
    assert ssim_rgb(black, black) == pytest.approx(1.0)
    assert ssim_rgb(black, white) == pytest.approx(0.01**2 / (1.0 + 0.01**2))
    with pytest.raises(EvaluationEvidenceError, match="uint8 HxWx3"):
        psnr_rgb(black.astype(np.float32), black)
    with pytest.raises(EvaluationEvidenceError, match="at least 11x11"):
        ssim_rgb(black[:10], black[:10])


def test_pyiqa_environment_forces_safe_checkpoint_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TORCH_FORCE_WEIGHTS_ONLY_LOAD", raising=False)
    monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    with metrics_module._offline_environment(tmp_path):
        assert os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] == "1"
        assert "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD" not in os.environ

    assert "TORCH_FORCE_WEIGHTS_ONLY_LOAD" not in os.environ
    assert os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] == "1"


def test_canonical_pyiqa_hashes_match_the_checked_in_weight_lock() -> None:
    lock = json.loads(
        (Path(__file__).resolve().parents[2] / "weights-lock.json").read_text(encoding="utf-8")
    )
    by_id = {artifact["id"]: artifact for artifact in lock["artifacts"]}
    expected = {
        "lpips": ("scaleguard-pyiqa-lpips-v01-alex", "metrics/lpips"),
        "musiq": ("scaleguard-pyiqa-musiq-koniq", "metrics/pyiqa"),
        "clipiqa": ("scaleguard-pyiqa-clipiqa-openai-rn50", "metrics/clipiqa/RN50.pt"),
    }

    for metric_name, (artifact_id, destination) in expected.items():
        artifact_record = by_id[artifact_id]
        assert metrics_module._PYIQA_WEIGHT_SHA256[metric_name] == artifact_record["known_sha256"]
        assert artifact_record["destination"] == destination


def test_single_sample_receipt_binds_all_evidence_and_is_self_hashed(tmp_path: Path) -> None:
    source = _rgb(tmp_path / "source.png", 20)
    final = _rgb(tmp_path / "final.png", 80)
    reference = _rgb(tmp_path / "reference.png", 80)
    manifest = _manifest(
        tmp_path / "manifest.json",
        run_id="sample-1",
        source=source,
        output=final,
    )
    output = tmp_path / "nested" / "metric-receipt.json"

    receipt = evaluate_metric_receipt([manifest], [reference], output)

    assert receipt["schema_version"] == METRIC_RECEIPT_SCHEMA
    assert receipt["status"] == "completed"
    assert receipt["counts"] == {
        "samples": 1,
        "samples_with_issues": 0,
        "metrics_requested": 2,
        "metrics_measured": 2,
        "metrics_failed": 0,
        "metrics_not_run": 0,
    }
    sample = receipt["samples"][0]
    assert sample["run_id"] == "sample-1"
    assert sample["manifest"]["sha256"]
    assert sample["input_image"]["sha256"] == artifact(source)["sha256"]
    assert sample["output_image"]["sha256"] == artifact(final)["sha256"]
    assert sample["reference_image"]["sha256"] == artifact(reference)["sha256"]
    assert sample["output_image"]["image_contract"]["mode"] == "RGB"
    assert [result["value"] for result in sample["metrics"]] == [
        "infinity",
        pytest.approx(1.0),
    ]
    on_disk = json.loads(output.read_text(encoding="utf-8"))
    body = dict(on_disk)
    digest = body.pop("receipt_sha256")
    assert digest == canonical_sha256(body)


def test_metric_receipt_verifier_replays_scores_and_rejects_self_signed_forgery(
    tmp_path: Path,
) -> None:
    source = _rgb(tmp_path / "source.png", 20)
    final = _rgb(tmp_path / "final.png", 80)
    reference = _rgb(tmp_path / "reference.png", 70)
    manifest = _manifest(
        tmp_path / "manifest.json",
        run_id="replayed",
        source=source,
        output=final,
    )
    receipt_path = tmp_path / "metric-receipt.json"
    evaluate_metric_receipt([manifest], [reference], receipt_path)

    verified = verify_metric_receipt(receipt_path, artifact_root=tmp_path)

    assert verified["verified"] is True
    assert verified["research_eligible"] is True
    assert verified["samples"][0]["metrics"]["psnr"]["status"] == "measured"

    forged = json.loads(receipt_path.read_text(encoding="utf-8"))
    forged["samples"][0]["metrics"][0]["value"] = 99.0
    forged.pop("receipt_sha256")
    forged["receipt_sha256"] = canonical_sha256(forged)
    receipt_path.write_text(json.dumps(forged), encoding="utf-8")

    with pytest.raises(EvaluationEvidenceError, match="score differs on replay"):
        verify_metric_receipt(receipt_path, artifact_root=tmp_path)


def test_metric_receipt_verifier_rejects_manifest_identity_drift(tmp_path: Path) -> None:
    source = _rgb(tmp_path / "source.png", 20)
    final = _rgb(tmp_path / "final.png", 80)
    reference = _rgb(tmp_path / "reference.png", 80)
    manifest = _manifest(
        tmp_path / "manifest.json",
        run_id="identity",
        source=source,
        output=final,
    )
    receipt_path = tmp_path / "metric-receipt.json"
    evaluate_metric_receipt(
        [manifest],
        [reference],
        receipt_path,
        metric_names=("psnr",),
    )
    forged = json.loads(receipt_path.read_text(encoding="utf-8"))
    forged["samples"][0]["manifest"]["sha256"] = "f" * 64
    forged.pop("receipt_sha256")
    forged["receipt_sha256"] = canonical_sha256(forged)
    receipt_path.write_text(json.dumps(forged), encoding="utf-8")

    with pytest.raises(EvaluationEvidenceError, match="manifest SHA256 changed"):
        verify_metric_receipt(receipt_path, artifact_root=tmp_path)


def test_batch_evaluation_preserves_pair_order_and_crop_contract(tmp_path: Path) -> None:
    manifests: list[Path] = []
    references: list[Path] = []
    for index, value in enumerate((30, 90)):
        source = _rgb(tmp_path / f"source-{index}.png", 10)
        final = _rgb(tmp_path / f"final-{index}.png", value, size=(20, 18))
        reference = _rgb(tmp_path / f"reference-{index}.png", value, size=(20, 18))
        manifests.append(
            _manifest(
                tmp_path / f"manifest-{index}.json",
                run_id=f"sample-{index}",
                source=source,
                output=final,
            )
        )
        references.append(reference)

    receipt = evaluate_metric_receipt(
        manifests,
        references,
        tmp_path / "batch.json",
        metric_names=("psnr",),
        crop_border=2,
    )

    assert receipt["status"] == "completed"
    assert receipt["contract"]["crop_border"] == 2
    assert [sample["run_id"] for sample in receipt["samples"]] == [
        "sample-0",
        "sample-1",
    ]
    assert receipt["counts"]["metrics_measured"] == 2


@pytest.mark.parametrize(
    ("reference_mode", "reference_size", "expected"),
    [
        ("RGB", (17, 16), "dimensions differ"),
        ("L", (16, 16), "mode RGB"),
    ],
)
def test_invalid_image_contract_writes_issue_receipt(
    tmp_path: Path,
    reference_mode: str,
    reference_size: tuple[int, int],
    expected: str,
) -> None:
    source = _rgb(tmp_path / "source.png", 0)
    final = _rgb(tmp_path / "final.png", 10)
    reference = tmp_path / "reference.png"
    Image.new(reference_mode, reference_size, 10).save(reference)
    manifest = _manifest(
        tmp_path / "manifest.json",
        run_id="bad-pair",
        source=source,
        output=final,
    )

    receipt = evaluate_metric_receipt([manifest], [reference], tmp_path / "receipt.json")

    assert receipt["status"] == "completed_with_issues"
    assert receipt["counts"]["samples_with_issues"] == 1
    assert receipt["counts"]["metrics_not_run"] == 2
    assert expected in receipt["issues"][0]
    assert receipt["samples"][0]["metrics"] == []


def test_orientation_icc_and_ssim_window_contracts_fail_closed(tmp_path: Path) -> None:
    source = _rgb(tmp_path / "source.png", 0)
    final = tmp_path / "final.png"
    reference = tmp_path / "reference.png"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (16, 16), 10).save(final, exif=exif)
    _rgb(reference, 10)
    manifest = _manifest(
        tmp_path / "manifest.json",
        run_id="orientation",
        source=source,
        output=final,
    )

    orientation_receipt = evaluate_metric_receipt(
        [manifest], [reference], tmp_path / "orientation.json"
    )
    assert "non-identity EXIF orientation" in orientation_receipt["issues"][0]

    Image.new("RGB", (16, 16), 10).save(final, icc_profile=b"profile-a")
    Image.new("RGB", (16, 16), 10).save(reference, icc_profile=b"profile-b")
    manifest = _manifest(
        tmp_path / "manifest-icc.json",
        run_id="icc",
        source=source,
        output=final,
    )
    icc_receipt = evaluate_metric_receipt([manifest], [reference], tmp_path / "icc.json")
    assert "ICC profiles differ" in icc_receipt["issues"][0]

    _rgb(final, 10, size=(10, 10))
    _rgb(reference, 10, size=(10, 10))
    manifest = _manifest(
        tmp_path / "manifest-small.json",
        run_id="small",
        source=source,
        output=final,
    )
    small_receipt = evaluate_metric_receipt([manifest], [reference], tmp_path / "small.json")
    assert "at least 11 pixels" in small_receipt["issues"][0]


def test_tampered_manifest_artifact_fails_closed(tmp_path: Path) -> None:
    source = _rgb(tmp_path / "source.png", 0)
    final = _rgb(tmp_path / "final.png", 10)
    reference = _rgb(tmp_path / "reference.png", 10)
    manifest = _manifest(
        tmp_path / "manifest.json",
        run_id="tampered",
        source=source,
        output=final,
    )
    _rgb(final, 11)

    receipt = evaluate_metric_receipt([manifest], [reference], tmp_path / "receipt.json")

    assert receipt["status"] == "completed_with_issues"
    assert "SHA256 mismatch" in receipt["issues"][0]
    assert receipt["counts"]["metrics_measured"] == 0


def test_duplicate_manifest_keys_write_an_issue_receipt(tmp_path: Path) -> None:
    source = _rgb(tmp_path / "source.png", 0)
    final = _rgb(tmp_path / "final.png", 10)
    reference = _rgb(tmp_path / "reference.png", 10)
    manifest = _manifest(
        tmp_path / "manifest.json",
        run_id="ambiguous",
        source=source,
        output=final,
    )
    original = manifest.read_text(encoding="utf-8").lstrip()
    manifest.write_text('{"run_id":"forged",' + original[1:], encoding="utf-8")

    receipt = evaluate_metric_receipt([manifest], [reference], tmp_path / "receipt.json")

    assert receipt["status"] == "completed_with_issues"
    assert receipt["counts"]["metrics_measured"] == 0
    assert "duplicate JSON object key 'run_id'" in receipt["issues"][0]


def test_failed_run_status_is_retained_as_an_issue(tmp_path: Path) -> None:
    source = _rgb(tmp_path / "source.png", 0)
    final = _rgb(tmp_path / "final.png", 10)
    reference = _rgb(tmp_path / "reference.png", 10)
    manifest = _manifest(
        tmp_path / "manifest.json",
        run_id="failed-run",
        source=source,
        output=final,
    )
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    raw["status"] = "failed"
    manifest.write_text(json.dumps(raw), encoding="utf-8")

    receipt = evaluate_metric_receipt(
        [manifest],
        [reference],
        tmp_path / "receipt.json",
        metric_names=("psnr",),
    )

    assert receipt["status"] == "completed_with_issues"
    assert receipt["samples"][0]["issues"] == ["run_status:failed"]


def test_mock_run_is_measured_but_never_issue_free(tmp_path: Path) -> None:
    source = _rgb(tmp_path / "source.png", 0)
    final = _rgb(tmp_path / "final.png", 10)
    reference = _rgb(tmp_path / "reference.png", 10)
    manifest = _manifest(
        tmp_path / "manifest.json",
        run_id="mock",
        source=source,
        output=final,
        mock=True,
    )

    receipt = evaluate_metric_receipt(
        [manifest],
        [reference],
        tmp_path / "receipt.json",
        metric_names=("psnr",),
    )

    assert receipt["status"] == "completed_with_issues"
    assert receipt["samples"][0]["issues"] == ["mock_run"]
    assert receipt["samples"][0]["metrics"][0]["status"] == "measured"


def test_missing_pyiqa_weight_records_failure_without_importing_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _rgb(tmp_path / "source.png", 0)
    final = _rgb(tmp_path / "final.png", 10)
    reference = _rgb(tmp_path / "reference.png", 10)
    manifest = _manifest(
        tmp_path / "manifest.json",
        run_id="no-weight",
        source=source,
        output=final,
    )
    imported: list[str] = []
    monkeypatch.setattr(
        "scaleguard.evaluation.metrics.importlib.import_module",
        lambda name: imported.append(name),
    )

    receipt = evaluate_metric_receipt(
        [manifest],
        [reference],
        tmp_path / "receipt.json",
        metric_names=("musiq",),
    )

    assert imported == []
    assert receipt["status"] == "completed_with_issues"
    result = receipt["samples"][0]["metrics"][0]
    assert result["status"] == "failed"
    assert result["value"] is None
    assert "explicit local weight is required" in result["issue"]


def test_pyiqa_adapter_records_version_weight_device_and_blocks_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _rgb(tmp_path / "source.png", 0)
    final = _rgb(tmp_path / "final.png", 10)
    reference = _rgb(tmp_path / "reference.png", 10)
    manifest = _manifest(
        tmp_path / "manifest.json",
        run_id="pyiqa",
        source=source,
        output=final,
    )
    weight = tmp_path / "musiq.pth"
    weight.write_bytes(b"locked-musiq")
    monkeypatch.setitem(
        metrics_module._PYIQA_WEIGHT_SHA256,
        "musiq",
        artifact(weight)["sha256"],
    )
    calls: list[dict[str, Any]] = []

    class FakeTensor:
        def detach(self) -> FakeTensor:
            return self

        def cpu(self) -> FakeTensor:
            return self

        def item(self) -> float:
            return 73.25

    class FakeMetric:
        lower_better = False

        def __call__(self, path: str) -> FakeTensor:
            assert path == str(final.resolve())
            assert socket.create_connection(("example.invalid", 443)) is None
            return FakeTensor()

    def create_metric(name: str, **options: object) -> FakeMetric:
        calls.append({"name": name, **options})
        return FakeMetric()

    monkeypatch.setattr(
        "scaleguard.evaluation.metrics.importlib.metadata.version",
        lambda name: "0.1.16",
    )
    monkeypatch.setattr(
        "scaleguard.evaluation.metrics.importlib.import_module",
        lambda name: SimpleNamespace(create_metric=create_metric),
    )

    receipt = evaluate_metric_receipt(
        [manifest],
        [reference],
        tmp_path / "receipt.json",
        metric_names=("musiq",),
        pyiqa_weights={"musiq": weight},
        device="cpu",
    )

    assert receipt["status"] == "completed_with_issues"
    result = receipt["samples"][0]["metrics"][0]
    assert result["status"] == "failed"
    assert "network access is disabled" in result["issue"]
    assert calls[0]["name"] == "musiq"
    assert calls[0]["device"] == "cpu"
    assert calls[0]["pretrained_model_path"] == str(weight.resolve())
    assert result["backend_version"] == "0.1.16"
    assert result["parameters"]["weight"]["sha256"] == artifact(weight)["sha256"]


def test_pyiqa_adapter_success_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _rgb(tmp_path / "source.png", 0)
    final = _rgb(tmp_path / "final.png", 10)
    reference = _rgb(tmp_path / "reference.png", 10)
    manifest = _manifest(
        tmp_path / "manifest.json",
        run_id="pyiqa-success",
        source=source,
        output=final,
    )
    weight = tmp_path / "musiq.pth"
    weight.write_bytes(b"locked-musiq")
    monkeypatch.setitem(
        metrics_module._PYIQA_WEIGHT_SHA256,
        "musiq",
        artifact(weight)["sha256"],
    )

    class FakeTensor:
        def detach(self) -> FakeTensor:
            return self

        def cpu(self) -> FakeTensor:
            return self

        def item(self) -> float:
            return 73.25

    fake_metric = SimpleNamespace(
        lower_better=False,
        __call__=lambda _path: FakeTensor(),
    )

    class CallableMetric:
        lower_better = False

        def __call__(self, _path: str) -> FakeTensor:
            return FakeTensor()

    monkeypatch.setattr(
        "scaleguard.evaluation.metrics.importlib.metadata.version",
        lambda name: "0.1.16",
    )
    monkeypatch.setattr(
        "scaleguard.evaluation.metrics.importlib.import_module",
        lambda name: SimpleNamespace(create_metric=lambda *args, **kwargs: CallableMetric()),
    )

    receipt = evaluate_metric_receipt(
        [manifest],
        [reference],
        tmp_path / "receipt.json",
        metric_names=("musiq",),
        pyiqa_weights={"musiq": weight},
    )

    assert fake_metric.lower_better is False
    assert receipt["status"] == "completed"
    result = receipt["samples"][0]["metrics"][0]
    assert result["status"] == "measured"
    assert result["value"] == pytest.approx(73.25)


def test_learned_metric_is_not_research_eligible_when_replay_source_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _rgb(tmp_path / "source.png", 0)
    final = _rgb(tmp_path / "final.png", 10)
    manifest = _manifest(
        tmp_path / "manifest.json",
        run_id="learned-replay",
        source=source,
        output=final,
    )
    weight = tmp_path / "musiq.pth"
    weight.write_bytes(b"locked-musiq")
    monkeypatch.setitem(
        metrics_module._PYIQA_WEIGHT_SHA256,
        "musiq",
        artifact(weight)["sha256"],
    )

    class FakeTensor:
        def detach(self) -> FakeTensor:
            return self

        def cpu(self) -> FakeTensor:
            return self

        def item(self) -> float:
            return 61.5

    class CallableMetric:
        lower_better = False

        def __call__(self, _path: str) -> FakeTensor:
            return FakeTensor()

    monkeypatch.setattr(
        "scaleguard.evaluation.metrics.importlib.metadata.version",
        lambda name: "0.1.16",
    )
    monkeypatch.setattr(
        "scaleguard.evaluation.metrics.importlib.import_module",
        lambda name: SimpleNamespace(create_metric=lambda *args, **kwargs: CallableMetric()),
    )
    receipt_path = tmp_path / "receipt.json"
    evaluate_metric_receipt(
        [manifest],
        None,
        receipt_path,
        metric_names=("musiq",),
        pyiqa_weights={"musiq": weight},
    )
    weight.unlink()

    verified = verify_metric_receipt(receipt_path, artifact_root=tmp_path)

    assert verified["verified"] is False
    assert verified["research_eligible"] is False
    assert verified["issues"] == ["learned_metric_source_unavailable:musiq"]
    assert verified["samples"][0]["metrics"]["musiq"]["status"] == "unverified"


def test_pyiqa_initialization_failures_are_receipted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _rgb(tmp_path / "source.png", 0)
    final = _rgb(tmp_path / "final.png", 10)
    reference = _rgb(tmp_path / "reference.png", 10)
    manifest = _manifest(
        tmp_path / "manifest.json",
        run_id="pyiqa-failures",
        source=source,
        output=final,
    )
    weight = tmp_path / "weight.pth"
    weight.write_bytes(b"not-the-rn50-checkpoint")
    monkeypatch.setattr(
        "scaleguard.evaluation.metrics.importlib.metadata.version",
        lambda name: "0.1.15",
    )

    version_receipt = evaluate_metric_receipt(
        [manifest],
        [reference],
        tmp_path / "version.json",
        metric_names=("musiq",),
        pyiqa_weights={"musiq": weight},
    )
    assert "version mismatch" in version_receipt["samples"][0]["metrics"][0]["issue"]
    assert (
        version_receipt["metric_requests"][0]["parameters"]["weight"]["sha256"]
        == artifact(weight)["sha256"]
    )

    monkeypatch.setattr(
        "scaleguard.evaluation.metrics.importlib.metadata.version",
        lambda name: "0.1.16",
    )
    missing_backbone = evaluate_metric_receipt(
        [manifest],
        [reference],
        tmp_path / "lpips.json",
        metric_names=("lpips",),
        pyiqa_weights={"lpips": weight},
    )
    assert "requires --pyiqa-backbone" in missing_backbone["samples"][0]["metrics"][0]["issue"]

    bad_clipiqa = evaluate_metric_receipt(
        [manifest],
        [reference],
        tmp_path / "clipiqa.json",
        metric_names=("clipiqa",),
        pyiqa_weights={"clipiqa": weight},
    )
    assert "pinned canonical checkpoint" in bad_clipiqa["samples"][0]["metrics"][0]["issue"]

    monkeypatch.setattr(
        "scaleguard.evaluation.metrics.importlib.import_module",
        lambda name: SimpleNamespace(
            create_metric=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("broken"))
        ),
    )
    monkeypatch.setitem(
        metrics_module._PYIQA_WEIGHT_SHA256,
        "musiq",
        artifact(weight)["sha256"],
    )
    init_failure = evaluate_metric_receipt(
        [manifest],
        [reference],
        tmp_path / "init.json",
        metric_names=("musiq",),
        pyiqa_weights={"musiq": weight},
    )
    assert "cannot initialize offline" in init_failure["samples"][0]["metrics"][0]["issue"]


def test_metric_request_validation_rejects_ambiguous_batches(tmp_path: Path) -> None:
    source = _rgb(tmp_path / "source.png", 0)
    final = _rgb(tmp_path / "final.png", 10)
    reference = _rgb(tmp_path / "reference.png", 10)
    manifest = _manifest(
        tmp_path / "manifest.json",
        run_id="validation",
        source=source,
        output=final,
    )
    output = tmp_path / "receipt.json"

    with pytest.raises(EvaluationEvidenceError, match="at least one metric"):
        evaluate_metric_receipt([manifest], [reference], output, metric_names=())
    with pytest.raises(EvaluationEvidenceError, match="must be unique"):
        evaluate_metric_receipt([manifest], [reference], output, metric_names=("psnr", "psnr"))
    with pytest.raises(EvaluationEvidenceError, match="must be unique within"):
        evaluate_metric_receipt(
            [manifest, manifest],
            [reference, reference],
            output,
            metric_names=("psnr",),
        )
    with pytest.raises(EvaluationEvidenceError, match="unrequested"):
        evaluate_metric_receipt(
            [manifest],
            [reference],
            output,
            metric_names=("psnr",),
            pyiqa_weights={"musiq": tmp_path / "missing.pth"},
        )
    with pytest.raises(EvaluationEvidenceError, match="crop_border must be zero"):
        evaluate_metric_receipt(
            [manifest],
            [reference],
            output,
            metric_names=("musiq",),
            crop_border=1,
        )


def test_no_reference_metric_accepts_no_reference_image(tmp_path: Path) -> None:
    source = _rgb(tmp_path / "source.png", 0)
    final = _rgb(tmp_path / "final.png", 10)
    manifest = _manifest(
        tmp_path / "manifest.json",
        run_id="no-reference",
        source=source,
        output=final,
    )

    receipt = evaluate_metric_receipt(
        [manifest],
        None,
        tmp_path / "receipt.json",
        metric_names=("musiq",),
    )

    assert receipt["samples"][0]["reference_image"] is None
    assert "counts must match" not in " ".join(receipt["issues"])


def test_metric_receipt_rejects_output_aliases_and_requires_full_manifest_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _rgb(tmp_path / "source.png", 0)
    final = _rgb(tmp_path / "final.png", 10)
    reference = _rgb(tmp_path / "reference.png", 10)
    manifest = _manifest(
        tmp_path / "manifest.json",
        run_id="validation",
        source=source,
        output=final,
    )

    with pytest.raises(EvaluationEvidenceError, match=r"would overwrite .* final image"):
        evaluate_metric_receipt([manifest], [reference], final, metric_names=("psnr",))

    def reject(*_args: object, **_kwargs: object) -> None:
        raise ManifestValidationError("forged runtime evidence")

    monkeypatch.setattr(metrics_module, "validate_run_manifest", reject)
    receipt = evaluate_metric_receipt(
        [manifest],
        [reference],
        tmp_path / "receipt.json",
        metric_names=("psnr",),
    )
    assert "forged runtime evidence" in receipt["issues"][0]


def test_cli_runs_single_and_batch_metrics(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.delenv("SCALEGUARD_PROJECT_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("scaleguard.cli.find_project_root", _fail_project_root_resolution)
    manifests: list[Path] = []
    references: list[Path] = []
    for index in range(2):
        source = _rgb(tmp_path / f"source-{index}.png", 0)
        final = _rgb(tmp_path / f"final-{index}.png", 10 + index)
        reference = _rgb(tmp_path / f"reference-{index}.png", 10 + index)
        manifests.append(
            _manifest(
                tmp_path / f"manifest-{index}.json",
                run_id=f"cli-{index}",
                source=source,
                output=final,
            )
        )
        references.append(reference)
    argv = ["evaluation", "metrics"]
    for manifest, reference in zip(manifests, references, strict=True):
        argv.extend(["--manifest", str(manifest), "--reference", str(reference)])
    output = tmp_path / "receipt.json"
    argv.extend(
        [
            "--metric",
            "psnr",
            "--output",
            str(output),
            "--artifact-root",
            str(tmp_path),
        ]
    )

    assert main(argv) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "completed"
    assert payload["samples"] == 2
    assert output.is_file()


def test_cli_returns_one_for_metric_issues_and_two_for_bad_pairing(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setenv("SCALEGUARD_PROJECT_ROOT", str(tmp_path))
    source = _rgb(tmp_path / "source.png", 0)
    final = _rgb(tmp_path / "final.png", 10)
    reference = _rgb(tmp_path / "reference.png", 10)
    manifest = _manifest(
        tmp_path / "manifest.json",
        run_id="cli-issue",
        source=source,
        output=final,
    )
    output = tmp_path / "receipt.json"

    assert (
        main(
            [
                "evaluation",
                "metrics",
                "--manifest",
                str(manifest),
                "--reference",
                str(reference),
                "--metric",
                "musiq",
                "--output",
                str(output),
            ]
        )
        == 1
    )
    assert json.loads(capsys.readouterr().out)["metrics_failed"] == 1
    assert (
        main(
            [
                "evaluation",
                "metrics",
                "--manifest",
                str(manifest),
                "--manifest",
                str(manifest),
                "--reference",
                str(reference),
                "--output",
                str(tmp_path / "bad.json"),
            ]
        )
        == 2
    )
    assert "counts must match" in capsys.readouterr().err
