from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import TracebackType

import pytest
from PIL import Image

from scaleguard.backends.fake import (
    FakeRestorationBackend,
    FakeScaleBackend,
    FakeScaleSession,
)
from scaleguard.config import (
    ControllerConfig,
    CoZConfig,
    FourKAgentConfig,
    MetricConfig,
    PipelineConfig,
    RuntimeConfig,
)
from scaleguard.contracts import WorkerResult
from scaleguard.controller.trusted_scale import TrustedScaleController
from scaleguard.errors import ArtifactError
from scaleguard.images import inspect_image
from scaleguard.manifest import validate_run_manifest


def pipeline_config(tmp_path: Path, target_factor: int) -> PipelineConfig:
    return PipelineConfig(
        runtime=RuntimeConfig(
            run_root=tmp_path / "运行 logs",
            process_timeout_seconds=2.0,
            gpu_poll_interval_seconds=0.01,
        ),
        metrics=MetricConfig(
            min_quality_gain=-10.0,
            max_scale_nrmse=10.0,
            max_scale_edge_mae=10.0,
        ),
        controller=ControllerConfig(
            target_factor=target_factor,
            max_coz_steps=2,
            color_strategy="none",
        ),
    )


def evidence_config(tmp_path: Path, target_factor: int) -> PipelineConfig:
    base = pipeline_config(tmp_path, target_factor)
    return replace(
        base,
        fourkagent=FourKAgentConfig(
            mode="command",
            command=("unused-restoration-adapter",),
        ),
        coz=CoZConfig(
            mode="command",
            command=("unused-scale-adapter",),
        ),
        controller=replace(
            base.controller,
            accept_unvalidated_quality_proxy=True,
        ),
    )


@pytest.mark.parametrize("target_factor", [1, 2, 4, 8, 16])
def test_fake_pipeline_realizes_each_supported_factor_on_a_non_square_image(
    tmp_path: Path,
    make_image: Callable[..., Path],
    target_factor: int,
) -> None:
    source = make_image(tmp_path / "输入 图像.JPG", size=(7, 5), image_format="JPEG")
    output = tmp_path / "导出 结果" / f"factor {target_factor}.png"
    config = pipeline_config(tmp_path, target_factor)
    controller = TrustedScaleController(
        config,
        FakeRestorationBackend(),
        FakeScaleBackend(),
    )

    returned = controller.run(source, output, run_id=f"倍率-{target_factor}")

    assert returned == output
    with Image.open(output) as result:
        assert result.format == "PNG"
        assert result.size == (7 * target_factor, 5 * target_factor)

    manifest = json.loads(
        (config.runtime.run_root / f"倍率-{target_factor}" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["status"] == "succeeded"
    assert manifest["mock"] is True
    assert manifest["requested_factor"] == target_factor
    assert manifest["achieved_factor"] == target_factor
    assert manifest["target_reached"] is True
    assert manifest["final_image"]["width"] == 7 * target_factor
    assert manifest["final_image"]["height"] == 5 * target_factor


class NeverScaleBackend:
    name = "must_not_run"
    mock = True

    def session(self, run_dir: Path) -> None:
        del run_dir
        raise AssertionError("CoZ session must not start for an already-satisfied target")


class EvidenceRestorationBackend(FakeRestorationBackend):
    name = "evidence_4kagent"
    mock = False

    def restore(
        self,
        source: Path,
        destination: Path,
        *,
        bridge_factor: int,
        run_dir: Path,
    ) -> WorkerResult:
        result = super().restore(
            source,
            destination,
            bridge_factor=bridge_factor,
            run_dir=run_dir,
        )
        return WorkerResult(
            image=replace(result.image, mock=False),
            metadata={**result.metadata, "mock": False},
        )


class EvidenceScaleSession(FakeScaleSession):
    name = "evidence_coz_session"
    mock = False

    def upscale_once(
        self,
        source: Path,
        destination: Path,
        *,
        step_index: int,
        seed: int,
    ) -> WorkerResult:
        result = super().upscale_once(
            source,
            destination,
            step_index=step_index,
            seed=seed,
        )
        return WorkerResult(
            image=replace(result.image, mock=False),
            metadata={**result.metadata, "mock": False},
        )


class EvidenceScaleBackend:
    name = "evidence_coz"
    mock = False

    def session(self, run_dir: Path) -> EvidenceScaleSession:
        del run_dir
        return EvidenceScaleSession()


class CloseFailureEvidenceSession(EvidenceScaleSession):
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        raise RuntimeError("synthetic close failure")


class CloseFailureEvidenceBackend:
    name = "close_failure_evidence_coz"
    mock = False

    def session(self, run_dir: Path) -> CloseFailureEvidenceSession:
        del run_dir
        return CloseFailureEvidenceSession()


def test_already_satisfied_target_skips_the_terminal_sr_session(
    tmp_path: Path,
    make_image: Callable[..., Path],
) -> None:
    source = make_image(tmp_path / "already 4k.png", size=(11, 7))
    output = tmp_path / "output.png"
    config = pipeline_config(tmp_path, 1)
    controller = TrustedScaleController(
        config,
        FakeRestorationBackend(),
        NeverScaleBackend(),  # type: ignore[arg-type]
    )

    controller.run(source, output, run_id="already-target")

    with Image.open(output) as result:
        assert result.size == (11, 7)
    manifest = json.loads(
        (config.runtime.run_root / "already-target" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["steps"] == []
    assert not any(event["event"] == "scale_worker_failed" for event in manifest["events"])


def test_completion_level_requires_recorded_coz_evidence(
    tmp_path: Path,
    make_image: Callable[..., Path],
) -> None:
    source = make_image(tmp_path / "source.png", size=(6, 4))
    config = evidence_config(tmp_path, 1)
    controller = TrustedScaleController(
        config,
        EvidenceRestorationBackend(),
        EvidenceScaleBackend(),
    )

    controller.run(source, tmp_path / "output.png", run_id="restoration-only")

    manifest = json.loads(
        (config.runtime.run_root / "restoration-only" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["mock"] is False
    assert manifest["completion_level"] == "STATIC_READY"


def test_unverified_candidate_cannot_claim_component_reproduced(
    tmp_path: Path,
    make_image: Callable[..., Path],
) -> None:
    source = make_image(tmp_path / "source.png", size=(6, 4))
    config = replace(
        evidence_config(tmp_path, 4),
        metrics=MetricConfig(
            min_quality_gain=10.0,
            max_scale_nrmse=10.0,
            max_scale_edge_mae=10.0,
        ),
    )
    controller = TrustedScaleController(
        config,
        EvidenceRestorationBackend(),
        EvidenceScaleBackend(),
    )

    controller.run(source, tmp_path / "output.png", run_id="candidate-rejected")

    manifest = json.loads(
        (config.runtime.run_root / "candidate-rejected" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["steps"][0]["candidate"] is not None
    assert manifest["steps"][0]["accepted"] is False
    assert manifest["status"] == "succeeded_with_rollback"
    assert manifest["achieved_factor"] == 1
    assert manifest["target_reached"] is False
    assert manifest["completion_level"] == "STATIC_READY"


def test_unverified_accepted_candidate_cannot_claim_ab_integrated(
    tmp_path: Path,
    make_image: Callable[..., Path],
) -> None:
    source = make_image(tmp_path / "source.png", size=(6, 4))
    config = evidence_config(tmp_path, 4)
    controller = TrustedScaleController(
        config,
        EvidenceRestorationBackend(),
        EvidenceScaleBackend(),
    )

    controller.run(source, tmp_path / "output.png", run_id="candidate-accepted")

    manifest = json.loads(
        (config.runtime.run_root / "candidate-accepted" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["steps"][0]["accepted"] is True
    assert manifest["achieved_factor"] == 4
    assert manifest["target_reached"] is True
    assert manifest["completion_level"] == "STATIC_READY"


class SecondStepFailureSession(EvidenceScaleSession):
    def upscale_once(
        self,
        source: Path,
        destination: Path,
        *,
        step_index: int,
        seed: int,
    ) -> WorkerResult:
        if step_index == 2:
            raise RuntimeError("synthetic second-step failure")
        return super().upscale_once(
            source,
            destination,
            step_index=step_index,
            seed=seed,
        )


class SecondStepFailureBackend:
    name = "second_step_failure"
    mock = False

    def session(self, run_dir: Path) -> SecondStepFailureSession:
        del run_dir
        return SecondStepFailureSession()


def test_partial_real_scale_path_cannot_claim_ab_integration(
    tmp_path: Path,
    make_image: Callable[..., Path],
) -> None:
    source = make_image(tmp_path / "source.png", size=(6, 4))
    config = evidence_config(tmp_path, 16)
    controller = TrustedScaleController(
        config,
        EvidenceRestorationBackend(),
        SecondStepFailureBackend(),
    )

    controller.run(source, tmp_path / "output.png", run_id="partial-real-path")

    manifest = json.loads(
        (config.runtime.run_root / "partial-real-path" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["steps"][0]["accepted"] is True
    assert manifest["steps"][1]["accepted"] is False
    assert manifest["achieved_factor"] == 4
    assert manifest["target_reached"] is False
    assert manifest["status"] == "succeeded_with_rollback"
    assert manifest["completion_level"] == "STATIC_READY"


def test_failed_coz_session_boundary_cannot_promote_completion_level(
    tmp_path: Path,
    make_image: Callable[..., Path],
) -> None:
    source = make_image(tmp_path / "source.png", size=(6, 4))
    config = evidence_config(tmp_path, 4)
    controller = TrustedScaleController(
        config,
        EvidenceRestorationBackend(),
        CloseFailureEvidenceBackend(),
    )

    controller.run(source, tmp_path / "output.png", run_id="close-failure")

    manifest = json.loads(
        (config.runtime.run_root / "close-failure" / "manifest.json").read_text(encoding="utf-8")
    )
    assert len(manifest["steps"]) == 1
    assert manifest["steps"][0]["accepted"] is True
    assert manifest["status"] == "succeeded_with_rollback"
    assert manifest["completion_level"] == "STATIC_READY"
    assert any(event["event"] == "scale_session_failed" for event in manifest["events"])


@pytest.mark.parametrize(
    "run_id",
    [
        "",
        ".",
        "..",
        "../outside",
        "nested/run",
        r"nested\run",
        "line\nbreak",
        "*",
        "run?",
        "run[1]",
    ],
)
def test_run_id_must_be_a_single_safe_unicode_component(
    tmp_path: Path,
    make_image: Callable[..., Path],
    run_id: str,
) -> None:
    source = make_image(tmp_path / "source.png")
    config = pipeline_config(tmp_path, 1)
    controller = TrustedScaleController(
        config,
        FakeRestorationBackend(),
        FakeScaleBackend(),
    )

    with pytest.raises(ValueError, match="one non-glob printable path component"):
        controller.run(source, tmp_path / "output.png", run_id=run_id)

    assert not (tmp_path / "outside").exists()
    assert controller.last_run_dir is None


def test_relative_run_root_cannot_escape_the_project_root(
    tmp_path: Path,
    make_image: Callable[..., Path],
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    source = make_image(tmp_path / "source.png")
    config = replace(
        pipeline_config(tmp_path, 1),
        runtime=RuntimeConfig(run_root=Path("../outside")),
    )
    controller = TrustedScaleController(
        config,
        FakeRestorationBackend(),
        FakeScaleBackend(),
        project_root=project_root,
    )

    with pytest.raises(ValueError, match="must resolve inside the project root"):
        controller.run(source, tmp_path / "output.png", run_id="safe-name")

    assert not (tmp_path / "outside").exists()


def test_external_output_cannot_overwrite_run_evidence(
    tmp_path: Path,
    make_image: Callable[..., Path],
) -> None:
    source = make_image(tmp_path / "source.png")
    config = pipeline_config(tmp_path, 1)
    controller = TrustedScaleController(
        config,
        FakeRestorationBackend(),
        FakeScaleBackend(),
    )
    manifest_path = config.runtime.run_root / "collision" / "manifest.json"

    with pytest.raises(ValueError, match="outside the immutable run directory"):
        controller.run(source, manifest_path, run_id="collision")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["error"]["type"] == "ValueError"
    assert manifest["final_image"] is None


def test_external_output_publication_is_canonical_and_no_clobber(
    tmp_path: Path,
    make_image: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = make_image(tmp_path / "source.png")
    config = pipeline_config(tmp_path, 1)
    controller = TrustedScaleController(
        config,
        FakeRestorationBackend(),
        FakeScaleBackend(),
    )
    destination = tmp_path / "published.png"
    destination.write_bytes(b"pre-existing evidence")
    resolution_calls: list[Path] = []
    resolve_output = controller._resolve_external_output

    def record_resolution(
        output: Path,
        run_dir: Path,
        *,
        source: Path,
        overwrite: bool,
    ) -> Path:
        resolution_calls.append(output)
        return resolve_output(
            output,
            run_dir,
            source=source,
            overwrite=overwrite,
        )

    monkeypatch.setattr(controller, "_resolve_external_output", record_resolution)

    with pytest.raises(FileExistsError):
        controller.run(
            source,
            tmp_path / "nested" / ".." / "published.png",
            run_id="no-clobber",
        )

    assert resolution_calls == [tmp_path / "nested" / ".." / "published.png"]
    assert destination.read_bytes() == b"pre-existing evidence"
    with pytest.raises(FileExistsError):
        controller._publish_no_clobber(source, destination)
    assert destination.read_bytes() == b"pre-existing evidence"
    assert list(tmp_path.glob(".published.png.*.tmp")) == []
    manifest_path = config.runtime.run_root / "no-clobber" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["error"]["type"] == "FileExistsError"
    assert validate_run_manifest(manifest_path)["status"] == "failed"


def test_explicit_overwrite_atomically_replaces_a_non_input_output(
    tmp_path: Path,
    make_image: Callable[..., Path],
) -> None:
    source = make_image(tmp_path / "source.png", size=(8, 5), color=(10, 20, 30))
    destination = make_image(
        tmp_path / "published.png",
        size=(2, 2),
        color=(240, 10, 10),
    )
    previous_bytes = destination.read_bytes()
    config = pipeline_config(tmp_path, 1)
    controller = TrustedScaleController(
        config,
        FakeRestorationBackend(),
        FakeScaleBackend(),
    )

    returned = controller.run(
        source,
        destination,
        run_id="explicit-overwrite",
        overwrite=True,
    )

    assert returned == destination.resolve()
    assert destination.read_bytes() != previous_bytes
    with Image.open(destination) as image:
        assert image.size == (8, 5)
    assert list(tmp_path.glob(".published.png.*.tmp")) == []
    manifest_path = config.runtime.run_root / "explicit-overwrite" / "manifest.json"
    assert validate_run_manifest(manifest_path)["status"] == "succeeded"


def test_explicit_overwrite_cannot_alias_the_input(
    tmp_path: Path,
    make_image: Callable[..., Path],
) -> None:
    source = make_image(tmp_path / "source.png", size=(8, 5))
    source_bytes = source.read_bytes()
    config = pipeline_config(tmp_path, 1)
    controller = TrustedScaleController(
        config,
        FakeRestorationBackend(),
        FakeScaleBackend(),
    )

    with pytest.raises(ValueError, match="cannot overwrite the input image"):
        controller.run(
            source,
            source,
            run_id="input-alias",
            overwrite=True,
        )

    assert source.read_bytes() == source_bytes
    manifest_path = config.runtime.run_root / "input-alias" / "manifest.json"
    assert validate_run_manifest(manifest_path)["status"] == "failed"


class BadBridgeRestorationBackend(FakeRestorationBackend):
    def restore(
        self,
        source: Path,
        destination: Path,
        *,
        bridge_factor: int,
        run_dir: Path,
    ) -> WorkerResult:
        del bridge_factor
        return super().restore(
            source,
            destination,
            bridge_factor=1,
            run_dir=run_dir,
        )


def test_restoration_bridge_must_realize_its_exact_declared_scale(
    tmp_path: Path,
    make_image: Callable[..., Path],
) -> None:
    source = make_image(tmp_path / "source.png", size=(8, 5))
    config = pipeline_config(tmp_path, 2)
    controller = TrustedScaleController(
        config,
        BadBridgeRestorationBackend(),
        FakeScaleBackend(),
    )

    with pytest.raises(ArtifactError, match="expected approximately 16x10"):
        controller.run(source, tmp_path / "output.png", run_id="bad-bridge")

    manifest = json.loads(
        (config.runtime.run_root / "bad-bridge" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "failed"
    assert manifest["achieved_factor"] is None
    assert manifest["target_reached"] is False
    assert (
        validate_run_manifest(config.runtime.run_root / "bad-bridge" / "manifest.json")["status"]
        == "failed"
    )


def test_non_finite_metrics_fail_closed_without_writing_nonstandard_json(
    tmp_path: Path,
    make_image: Callable[..., Path],
) -> None:
    source = make_image(tmp_path / "single-pixel.png", size=(1, 1))
    config = pipeline_config(tmp_path, 4)
    controller = TrustedScaleController(
        config,
        FakeRestorationBackend(),
        FakeScaleBackend(),
    )

    with pytest.raises(ValueError, match="both image dimensions to be at least 2"):
        controller.run(source, tmp_path / "output.png", run_id="non-finite")

    manifest_path = config.runtime.run_root / "non-finite" / "manifest.json"
    payload = manifest_path.read_text(encoding="utf-8")
    assert "NaN" not in payload
    manifest = json.loads(payload, parse_constant=lambda value: pytest.fail(value))
    assert manifest["status"] == "failed"
    assert validate_run_manifest(manifest_path)["status"] == "failed"


def test_mock_manifest_marks_every_generated_artifact_and_worker_result(
    tmp_path: Path,
    make_image: Callable[..., Path],
) -> None:
    source = make_image(tmp_path / "source.png", size=(4, 3))
    output = tmp_path / "output.png"
    config = pipeline_config(tmp_path, 16)
    controller = TrustedScaleController(
        config,
        FakeRestorationBackend(),
        FakeScaleBackend(),
    )

    controller.run(source, output, run_id="mock-evidence")

    manifest = json.loads(
        (config.runtime.run_root / "mock-evidence" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["mock"] is True
    assert manifest["completion_level"] == "STATIC_READY"
    assert manifest["provenance"]["quality_backend_is_proxy"] is True
    assert manifest["provenance"]["quality_thresholds_calibrated"] is False
    assert manifest["restored_image"]["mock"] is True
    assert manifest["final_image"]["mock"] is True
    assert len(manifest["steps"]) == 2
    for step in manifest["steps"]:
        assert step["candidate"]["mock"] is True
        assert step["worker_metadata"]["mock"] is True


class FailingScaleSession:
    name = "failing_scale_session"
    mock = True

    def __enter__(self) -> FailingScaleSession:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback

    def upscale_once(
        self,
        source: Path,
        destination: Path,
        *,
        step_index: int,
        seed: int,
    ) -> WorkerResult:
        del source, destination, step_index, seed
        raise RuntimeError("synthetic worker crash")

    def accept(self, candidate: WorkerResult, *, step_index: int) -> None:
        del candidate, step_index

    def rollback(self, *, step_index: int) -> None:
        del step_index


class FailingScaleBackend:
    name = "failing_scale"
    mock = True

    def session(self, run_dir: Path) -> FailingScaleSession:
        del run_dir
        return FailingScaleSession()


def test_scale_worker_exception_rolls_back_to_the_last_trusted_image(
    tmp_path: Path,
    make_image: Callable[..., Path],
) -> None:
    source = make_image(tmp_path / "source.png", size=(8, 5), color=(31, 83, 149))
    output = tmp_path / "output.png"
    config = pipeline_config(tmp_path, 4)
    controller = TrustedScaleController(
        config,
        FakeRestorationBackend(),
        FailingScaleBackend(),
    )

    controller.run(source, output, run_id="worker-crash")

    with Image.open(source) as expected, Image.open(output) as actual:
        assert actual.convert("RGB").tobytes() == expected.convert("RGB").tobytes()
    manifest = json.loads(
        (config.runtime.run_root / "worker-crash" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "succeeded_with_rollback"
    assert manifest["steps"][0]["decision"] == "rollback"
    assert manifest["steps"][0]["accepted"] is False
    assert manifest["steps"][0]["candidate"] is None
    assert "synthetic worker crash" in manifest["steps"][0]["reason"]
    assert any(event["event"] == "scale_worker_failed" for event in manifest["events"])


class StartupFailureSession(FailingScaleSession):
    def __enter__(self) -> StartupFailureSession:
        raise RuntimeError("synthetic startup failure")


class StartupFailureBackend:
    name = "startup_failure"
    mock = True

    def session(self, run_dir: Path) -> StartupFailureSession:
        del run_dir
        return StartupFailureSession()


def test_scale_worker_startup_exception_also_falls_back_to_the_trusted_image(
    tmp_path: Path,
    make_image: Callable[..., Path],
) -> None:
    source = make_image(tmp_path / "source.png", size=(8, 5))
    output = tmp_path / "output.png"
    config = pipeline_config(tmp_path, 4)
    controller = TrustedScaleController(
        config,
        FakeRestorationBackend(),
        StartupFailureBackend(),
    )

    controller.run(source, output, run_id="startup-crash")

    manifest = json.loads(
        (config.runtime.run_root / "startup-crash" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "succeeded_with_rollback"
    assert output.is_file()
    assert "synthetic startup failure" in manifest["steps"][0]["reason"]


class DriftScaleSession(FailingScaleSession):
    def upscale_once(
        self,
        source: Path,
        destination: Path,
        *,
        step_index: int,
        seed: int,
    ) -> WorkerResult:
        del seed
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as image:
            Image.new("RGB", (image.width * 4, image.height * 4), (255, 255, 255)).save(
                destination,
                "PNG",
            )
        return WorkerResult(
            image=inspect_image(destination, mock=True, stage=f"drift_{step_index}"),
            metadata={"mock": True, "backend": self.name},
        )


class DriftScaleBackend:
    name = "drift_scale"
    mock = True

    def session(self, run_dir: Path) -> DriftScaleSession:
        del run_dir
        return DriftScaleSession()


def test_consistency_gate_rolls_back_a_structurally_drifting_candidate(
    tmp_path: Path,
    make_image: Callable[..., Path],
) -> None:
    source = make_image(tmp_path / "source.png", size=(8, 5), color=(20, 20, 20))
    output = tmp_path / "output.png"
    config = PipelineConfig(
        runtime=RuntimeConfig(run_root=tmp_path / "runs"),
        metrics=MetricConfig(
            min_quality_gain=-10.0,
            max_scale_nrmse=0.01,
            max_scale_edge_mae=10.0,
        ),
        controller=ControllerConfig(target_factor=4, color_strategy="none"),
    )
    controller = TrustedScaleController(
        config,
        FakeRestorationBackend(),
        DriftScaleBackend(),
    )

    controller.run(source, output, run_id="drift")

    manifest = json.loads(
        (config.runtime.run_root / "drift" / "manifest.json").read_text(encoding="utf-8")
    )
    step = manifest["steps"][0]
    assert step["decision"] == "rollback"
    assert step["accepted"] is False
    assert "scale_nrmse" in step["reason"]
    assert manifest["status"] == "succeeded_with_rollback"


def test_post_color_gates_reject_a_damaging_adain_result_and_keep_trusted_bytes(
    tmp_path: Path,
    make_image: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = make_image(tmp_path / "source.png", size=(8, 5), color=(30, 80, 140))
    output = tmp_path / "output.png"
    config = replace(
        pipeline_config(tmp_path, 4),
        metrics=MetricConfig(
            min_quality_gain=-10.0,
            max_scale_nrmse=0.01,
            max_scale_edge_mae=10.0,
        ),
        controller=ControllerConfig(
            target_factor=4,
            max_coz_steps=2,
            color_strategy="adain",
        ),
    )

    def destructive_adain(_candidate: Path, _reference: Path, destination: Path) -> None:
        make_image(destination, size=(32, 20), color=(255, 255, 255))

    monkeypatch.setattr(
        "scaleguard.controller.trusted_scale.apply_adain",
        destructive_adain,
    )
    controller = TrustedScaleController(
        config,
        FakeRestorationBackend(),
        FakeScaleBackend(),
    )

    controller.run(source, output, run_id="color-fallback")

    with Image.open(output) as image:
        assert image.getpixel((0, 0)) != (255, 255, 255)
    manifest = json.loads(
        (config.runtime.run_root / "color-fallback" / "manifest.json").read_text(encoding="utf-8")
    )
    attempts = [
        event for event in manifest["events"] if event["event"] == "final_candidate_evaluated"
    ]
    assert attempts[0]["label"] == "adain"
    assert attempts[0]["accepted"] is False
    assert attempts[1]["label"] == "trusted"
    assert attempts[1]["accepted"] is True
    assert manifest["final_metrics"]["after_color_alignment"] is False
    assert manifest["target_reached"] is True
    assert manifest["status"] == "succeeded"


class RecordingQuality:
    name = "gradient_proxy_v1"
    is_proxy = True

    def __init__(self) -> None:
        self.sizes: list[tuple[int, int]] = []

    def score(self, image: Path) -> float:
        with Image.open(image) as opened:
            self.sizes.append(opened.size)
        return 0.0


def test_quality_gain_always_compares_images_at_the_same_dimensions(
    tmp_path: Path,
    make_image: Callable[..., Path],
) -> None:
    source = make_image(tmp_path / "source.png", size=(9, 5))
    output = tmp_path / "output.png"
    config = pipeline_config(tmp_path, 4)
    quality = RecordingQuality()
    controller = TrustedScaleController(
        config,
        FakeRestorationBackend(),
        FakeScaleBackend(),
        quality=quality,
    )

    controller.run(source, output, run_id="same-size-quality")

    assert len(quality.sizes) == 4
    assert quality.sizes[0] == quality.sizes[1] == (36, 20)
    assert quality.sizes[2] == quality.sizes[3] == (36, 20)
