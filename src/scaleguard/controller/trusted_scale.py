"""Explicit continue/stop/rollback control over one-step CoZ states."""

from __future__ import annotations

import dataclasses
import math
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from PIL import Image

from scaleguard.backends.base import RestorationBackend, ScaleBackend
from scaleguard.color import apply_adain
from scaleguard.config import PipelineConfig, validate_config
from scaleguard.contracts import (
    CompletionLevel,
    Decision,
    ManifestRecorder,
    MetricRecord,
    RunManifest,
    RunStatus,
    ScaleStepRecord,
    utc_now,
)
from scaleguard.controller.policy import build_scale_plan
from scaleguard.evaluation.calibration import verify_calibration_receipt
from scaleguard.images import assert_scale, inspect_image, normalize_to_png
from scaleguard.imaging.forward_models import (
    ForwardModel,
    build_forward_model,
    evaluate_measurement_consistency,
)
from scaleguard.metrics.quality import (
    QualityEvaluator,
    bicubic_baseline,
    build_quality_evaluator,
)
from scaleguard.metrics.scale import evaluate_scale_consistency
from scaleguard.runtime.gpu_lifecycle import GpuLifecycle, GpuPhase, PhaseEvent


class TrustedScaleController:
    """Orchestrate 4KAgent once, then make each CoZ scale state explicit."""

    def __init__(
        self,
        config: PipelineConfig,
        restoration: RestorationBackend,
        scale_backend: ScaleBackend,
        *,
        quality: QualityEvaluator | None = None,
        provenance: dict[str, Any] | None = None,
        project_root: Path | None = None,
    ) -> None:
        validate_config(config)
        self.config = config
        self.restoration = restoration
        self.scale_backend = scale_backend
        self.project_root = (project_root or Path.cwd()).resolve()
        quality_model_path = config.metrics.quality_model_path
        if quality_model_path is not None and not quality_model_path.is_absolute():
            quality_model_path = self.project_root / quality_model_path
        self.quality = quality or build_quality_evaluator(
            config.metrics.quality_backend,
            config.metrics.quality_metric,
            config.metrics.quality_device,
            quality_model_path,
        )
        self.provenance = provenance or {}
        self.last_run_dir: Path | None = None
        self.measurement: ForwardModel | None = None
        if config.metrics.measurement_enabled:
            self.measurement = build_forward_model(
                config.metrics.measurement_model,
                config.metrics.measurement_parameters,
            )
        self.calibration_valid = False
        self.calibration_reasons: list[str] = ["calibration_receipt_not_configured"]
        receipt = config.metrics.calibration_receipt
        if receipt is not None:
            resolved_receipt = (
                receipt if receipt.is_absolute() else self.project_root / receipt
            ).resolve()
            try:
                self.calibration_valid, self.calibration_reasons = verify_calibration_receipt(
                    resolved_receipt, config
                )
            except (OSError, ValueError) as error:
                self.calibration_reasons = [
                    f"calibration_receipt_unreadable:{type(error).__name__}"
                ]

    def run(self, source: Path, output: Path, *, run_id: str | None = None) -> Path:
        plan = build_scale_plan(
            self.config.controller.target_factor,
            self.config.controller.max_coz_steps,
        )
        resolved_run_id = self._validate_run_id(self._new_run_id() if run_id is None else run_id)
        run_dir = self._create_run_dir(resolved_run_id)
        self.last_run_dir = run_dir
        normalized_input = run_dir / "input.png"
        normalize_to_png(source, normalized_input)
        input_artifact = inspect_image(normalized_input, mock=False, stage="input")
        mock = self.restoration.mock or self.scale_backend.mock
        manifest = RunManifest(
            schema_version="1.0",
            run_id=resolved_run_id,
            status=RunStatus.RUNNING,
            completion_level=CompletionLevel.STATIC_READY,
            started_at=utc_now(),
            finished_at=None,
            mock=mock,
            config=self.config.as_dict(),
            provenance={
                **self.provenance,
                "restoration_backend": self.restoration.name,
                "scale_backend": self.scale_backend.name,
                "quality_backend": self.quality.name,
                "quality_backend_is_proxy": self.quality.is_proxy,
                "quality_thresholds_calibrated": self.calibration_valid,
                "quality_calibration_reasons": self.calibration_reasons,
                "quality_calibration_receipt": (
                    str(self.config.metrics.calibration_receipt)
                    if self.config.metrics.calibration_receipt is not None
                    else None
                ),
            },
            input_image=input_artifact,
            requested_factor=self.config.controller.target_factor,
        )
        recorder = ManifestRecorder(run_dir / "manifest.json", manifest)
        lifecycle = GpuLifecycle(lambda event: self._record_phase(recorder, event))
        rolled_back = False
        session_boundary_valid = False
        try:
            self._assert_output_outside_run(output, run_dir)
            restored_path = run_dir / "states" / "scale_00_restored.png"
            with lifecycle.enter(GpuPhase.RESTORATION, self.config.fourkagent.tool_gpu):
                restored_result = self.restoration.restore(
                    normalized_input,
                    restored_path,
                    bridge_factor=plan.bridge_factor,
                    run_dir=run_dir,
                )
            if mock and not restored_result.image.mock:
                restored_result = dataclasses.replace(
                    restored_result,
                    image=dataclasses.replace(restored_result.image, mock=True),
                )
            assert_scale(
                input_artifact,
                restored_result.image,
                factor=plan.bridge_factor,
                tolerance_pixels=0,
            )
            manifest.restored_image = restored_result.image
            manifest.restoration_metadata = restored_result.metadata
            manifest.restoration_process = restored_result.process
            recorder.event(
                "restoration_completed",
                bridge_factor=plan.bridge_factor,
                metadata=restored_result.metadata,
            )
            trusted = restored_result.image
            trusted_scale = float(plan.bridge_factor)

            if plan.coz_steps:
                session_trusted = trusted
                session_scale = trusted_scale
                session_started_at = utc_now()
                try:
                    with (
                        lifecycle.enter(GpuPhase.COZ, self.config.coz.visible_devices),
                        self.scale_backend.session(run_dir) as session,
                    ):
                        for index in range(1, plan.coz_steps + 1):
                            started_at = utc_now()
                            destination = run_dir / "states" / f"scale_{index:02d}_candidate.png"
                            try:
                                result = session.upscale_once(
                                    trusted.path,
                                    destination,
                                    step_index=index,
                                    seed=self.config.coz.seed + index - 1,
                                )
                                if mock and not result.image.mock:
                                    result = dataclasses.replace(
                                        result,
                                        image=dataclasses.replace(result.image, mock=True),
                                    )
                                assert_scale(trusted, result.image, factor=4, tolerance_pixels=0)
                                metrics = self._metrics(
                                    baseline_source=trusted.path,
                                    candidate=result.image.path,
                                    observation=normalized_input,
                                    work_dir=run_dir / "metrics" / f"scale_{index:02d}",
                                )
                                decision, accepted, reason = self._decide(
                                    metrics, index, plan.coz_steps
                                )
                                step = ScaleStepRecord(
                                    index=index,
                                    input_scale=trusted_scale,
                                    candidate_scale=trusted_scale * 4.0,
                                    trusted_before=trusted,
                                    candidate=result.image,
                                    metrics=metrics,
                                    decision=decision,
                                    accepted=accepted,
                                    reason=reason,
                                    started_at=started_at,
                                    finished_at=utc_now(),
                                    worker_metadata=result.metadata,
                                    process=result.process,
                                )
                                if accepted:
                                    self._session_action(
                                        session,
                                        "accept",
                                        candidate=result,
                                        step_index=index,
                                    )
                                else:
                                    self._session_action(session, "rollback", step_index=index)
                                recorder.append_step(step)
                                if accepted:
                                    trusted = result.image
                                    trusted_scale *= 4.0
                                if decision is not Decision.CONTINUE:
                                    rolled_back = rolled_back or not accepted
                                    break
                            except Exception as error:
                                rolled_back = True
                                recorder.append_step(
                                    ScaleStepRecord(
                                        index=index,
                                        input_scale=trusted_scale,
                                        candidate_scale=trusted_scale * 4.0,
                                        trusted_before=trusted,
                                        candidate=None,
                                        metrics=None,
                                        decision=Decision.ROLLBACK,
                                        accepted=False,
                                        reason=(
                                            f"scale worker failure: {type(error).__name__}: {error}"
                                        ),
                                        started_at=started_at,
                                        finished_at=utc_now(),
                                    )
                                )
                                recorder.event(
                                    "scale_worker_failed",
                                    step_index=index,
                                    error_type=type(error).__name__,
                                    message=str(error),
                                )
                                self._session_action(session, "rollback", step_index=index)
                                break
                    evidence_method = getattr(session, "evidence", None)
                    if callable(evidence_method):
                        manifest.scale_session_process = evidence_method()
                        recorder.event("scale_session_process_recorded")
                    session_boundary_valid = True
                except Exception as error:
                    # A worker that cannot start or close cannot establish a trusted
                    # session boundary. Return to the pre-session scale.
                    rolled_back = True
                    trusted = session_trusted
                    trusted_scale = session_scale
                    if not manifest.steps:
                        recorder.append_step(
                            ScaleStepRecord(
                                index=1,
                                input_scale=trusted_scale,
                                candidate_scale=trusted_scale * 4.0,
                                trusted_before=trusted,
                                candidate=None,
                                metrics=None,
                                decision=Decision.ROLLBACK,
                                accepted=False,
                                reason=(f"scale session failure: {type(error).__name__}: {error}"),
                                started_at=session_started_at,
                                finished_at=utc_now(),
                            )
                        )
                    recorder.event(
                        "scale_session_failed",
                        error_type=type(error).__name__,
                        message=str(error),
                    )

            intended_scale = trusted_scale
            final_state = run_dir / "final.png"
            attempts: list[tuple[str, Path, float]] = []
            if self.config.controller.color_strategy == "adain" and trusted.path != restored_path:
                color_candidate = run_dir / "states" / "final_color_candidate.png"
                try:
                    apply_adain(trusted.path, restored_path, color_candidate)
                    attempts.append(("adain", color_candidate, trusted_scale))
                except Exception as error:
                    recorder.event(
                        "final_color_alignment_failed",
                        error_type=type(error).__name__,
                        message=str(error),
                    )
            attempts.append(("trusted", trusted.path, trusted_scale))
            if session_boundary_valid:
                for step in reversed(manifest.steps):
                    if step.accepted:
                        attempts.append(
                            ("previous_trusted", step.trusted_before.path, step.input_scale)
                        )
            attempts.append(("restored", restored_path, float(plan.bridge_factor)))

            seen_paths: set[Path] = set()
            final_metrics: MetricRecord | None = None
            final_reason = ""
            final_label = ""
            selected_scale = 0.0
            for attempt_index, (label, candidate_path, candidate_scale) in enumerate(
                attempts,
                start=1,
            ):
                resolved_candidate = candidate_path.resolve()
                if resolved_candidate in seen_paths:
                    continue
                seen_paths.add(resolved_candidate)
                metrics = self._metrics(
                    baseline_source=restored_path,
                    candidate=resolved_candidate,
                    observation=normalized_input,
                    work_dir=run_dir / "metrics" / "final" / f"attempt_{attempt_index:02d}",
                )
                accepted, reason = self._final_gate(
                    metrics,
                    require_quality=candidate_scale > plan.bridge_factor,
                )
                recorder.event(
                    "final_candidate_evaluated",
                    label=label,
                    scale=candidate_scale,
                    accepted=accepted,
                    reason=reason,
                    metrics=self._metric_payload(metrics),
                )
                if accepted:
                    shutil.copy2(resolved_candidate, final_state)
                    final_metrics = metrics
                    final_reason = reason
                    final_label = label
                    selected_scale = candidate_scale
                    break
            if final_metrics is None:
                raise ValueError("no retained scale state passed the post-color final gates")
            if selected_scale < intended_scale:
                rolled_back = True
                recorder.event(
                    "final_gate_rollback",
                    from_scale=intended_scale,
                    to_scale=selected_scale,
                )
            recorder.event(
                "final_color_alignment",
                strategy="adain" if final_label == "adain" else "none",
                reference=str(restored_path),
            )
            manifest.final_image = inspect_image(
                final_state,
                mock=mock,
                stage="final_output",
            )
            manifest.final_metrics = {
                "after_color_alignment": final_label == "adain",
                "selected_state": final_label,
                "selected_scale": selected_scale,
                "gate_passed": True,
                "gate_reason": final_reason,
                "metrics": self._metric_payload(final_metrics),
            }
            manifest.achieved_factor = int(selected_scale)
            manifest.target_reached = (
                manifest.achieved_factor == self.config.controller.target_factor
            )
            manifest.status = (
                RunStatus.SUCCEEDED_WITH_ROLLBACK
                if rolled_back or not manifest.target_reached
                else RunStatus.SUCCEEDED
            )
            official_runtime = (
                self.restoration.name == "4kagent_upstream"
                and self.scale_backend.name == "chain_of_zoom"
                and restored_result.process is not None
                and restored_result.metadata.get("backend") == "4kagent_upstream"
                and self.provenance.get("runtime_evidence_verified") is True
            )
            if not mock and session_boundary_valid and official_runtime:
                recorded_candidates = [
                    step for step in manifest.steps if step.candidate is not None
                ]
                if recorded_candidates:
                    manifest.completion_level = CompletionLevel.COMPONENT_REPRODUCED
                if manifest.target_reached and any(step.accepted for step in recorded_candidates):
                    manifest.completion_level = CompletionLevel.AB_INTEGRATED
            manifest.finished_at = utc_now()
            recorder.event("run_completed", output=str(output.resolve()))
            recorder.write()
            from scaleguard.manifest import validate_run_manifest

            validate_run_manifest(recorder.path)
            self._assert_output_outside_run(output, run_dir)
            output.parent.mkdir(parents=True, exist_ok=True)
            self._atomic_copy(final_state, output)
            recorder.event("external_output_written", output=str(output.resolve()))
            return output
        except Exception as error:
            manifest.status = RunStatus.FAILED
            manifest.finished_at = utc_now()
            manifest.error = {"type": type(error).__name__, "message": str(error)}
            recorder.write()
            raise

    def _metrics(
        self,
        *,
        baseline_source: Path,
        candidate: Path,
        observation: Path,
        work_dir: Path,
    ) -> MetricRecord:
        with Image.open(candidate) as candidate_image:
            size = candidate_image.size
            baseline_path = work_dir / "bicubic_baseline.png"
            bicubic_baseline(baseline_source, size, baseline_path)
        quality_baseline = self.quality.score(baseline_path)
        quality_candidate = self.quality.score(candidate)
        scale = evaluate_scale_consistency(candidate, baseline_source)
        measurement_nrmse: float | None = None
        measurement_model: str | None = None
        if self.measurement is not None:
            measurement = evaluate_measurement_consistency(
                candidate,
                observation,
                self.measurement,
            )
            measurement_nrmse = measurement.nrmse
            measurement_model = measurement.model
        record = MetricRecord(
            quality_baseline=quality_baseline,
            quality_candidate=quality_candidate,
            quality_gain=quality_candidate - quality_baseline,
            quality_backend=self.quality.name,
            scale_nrmse=scale.nrmse,
            scale_edge_mae=scale.edge_mae,
            measurement_nrmse=measurement_nrmse,
            measurement_model=measurement_model,
        )
        finite_metrics = {
            "quality_baseline": record.quality_baseline,
            "quality_candidate": record.quality_candidate,
            "quality_gain": record.quality_gain,
            "scale_nrmse": record.scale_nrmse,
            "scale_edge_mae": record.scale_edge_mae,
        }
        if record.measurement_nrmse is not None:
            finite_metrics["measurement_nrmse"] = record.measurement_nrmse
        non_finite = [name for name, value in finite_metrics.items() if not math.isfinite(value)]
        if non_finite:
            raise ValueError("non-finite metrics are forbidden: " + ", ".join(non_finite))
        return record

    def _final_gate(
        self,
        metrics: MetricRecord,
        *,
        require_quality: bool,
    ) -> tuple[bool, str]:
        thresholds = self.config.metrics
        if metrics.scale_nrmse > thresholds.max_scale_nrmse:
            return (
                False,
                (f"scale_nrmse={metrics.scale_nrmse:.6f} exceeds {thresholds.max_scale_nrmse:.6f}"),
            )
        if metrics.scale_edge_mae > thresholds.max_scale_edge_mae:
            return (
                False,
                (
                    f"scale_edge_mae={metrics.scale_edge_mae:.6f} exceeds "
                    f"{thresholds.max_scale_edge_mae:.6f}"
                ),
            )
        if (
            metrics.measurement_nrmse is not None
            and metrics.measurement_nrmse > thresholds.max_measurement_nrmse
        ):
            return (
                False,
                (
                    f"measurement_nrmse={metrics.measurement_nrmse:.6f} exceeds "
                    f"{thresholds.max_measurement_nrmse:.6f}"
                ),
            )
        if require_quality and metrics.quality_gain < thresholds.min_quality_gain:
            return (
                False,
                (
                    f"quality_gain={metrics.quality_gain:.6f} is below "
                    f"{thresholds.min_quality_gain:.6f}"
                ),
            )
        return True, "post-color final gates passed"

    @staticmethod
    def _metric_payload(metrics: MetricRecord) -> dict[str, Any]:
        return {
            "quality_baseline": metrics.quality_baseline,
            "quality_candidate": metrics.quality_candidate,
            "quality_gain": metrics.quality_gain,
            "quality_backend": metrics.quality_backend,
            "scale_nrmse": metrics.scale_nrmse,
            "scale_edge_mae": metrics.scale_edge_mae,
            "measurement_nrmse": metrics.measurement_nrmse,
            "measurement_model": metrics.measurement_model,
        }

    def _decide(
        self,
        metrics: MetricRecord,
        step_index: int,
        total_steps: int,
    ) -> tuple[Decision, bool, str]:
        thresholds = self.config.metrics
        if metrics.scale_nrmse > thresholds.max_scale_nrmse:
            return (
                Decision.ROLLBACK,
                False,
                (f"scale_nrmse={metrics.scale_nrmse:.6f} exceeds {thresholds.max_scale_nrmse:.6f}"),
            )
        if metrics.scale_edge_mae > thresholds.max_scale_edge_mae:
            return (
                Decision.ROLLBACK,
                False,
                (
                    f"scale_edge_mae={metrics.scale_edge_mae:.6f} exceeds "
                    f"{thresholds.max_scale_edge_mae:.6f}"
                ),
            )
        if (
            metrics.measurement_nrmse is not None
            and metrics.measurement_nrmse > thresholds.max_measurement_nrmse
        ):
            return (
                Decision.ROLLBACK,
                False,
                (
                    f"measurement_nrmse={metrics.measurement_nrmse:.6f} exceeds "
                    f"{thresholds.max_measurement_nrmse:.6f}"
                ),
            )
        if metrics.quality_gain < thresholds.min_quality_gain:
            return (
                Decision.STOP,
                False,
                (
                    f"quality_gain={metrics.quality_gain:.6f} is below "
                    f"{thresholds.min_quality_gain:.6f}"
                ),
            )
        if step_index >= total_steps:
            return Decision.STOP, True, "target scale accepted"
        return Decision.CONTINUE, True, "all gates passed"

    @staticmethod
    def _record_phase(recorder: ManifestRecorder, event: PhaseEvent) -> None:
        recorder.event(
            "gpu_phase",
            phase=event.phase.value,
            action=event.action,
            devices=event.devices,
            phase_at=event.at,
        )

    @staticmethod
    def _session_action(session: Any, action: str, **fields: Any) -> None:
        method = getattr(session, action, None)
        if not callable(method):
            raise TypeError(f"scale session does not implement required {action}()")
        method(**fields)

    @staticmethod
    def _validate_run_id(run_id: str) -> str:
        if (
            not run_id
            or len(run_id) > 128
            or run_id in {".", ".."}
            or "/" in run_id
            or "\\" in run_id
            or any(character in "*?[]" for character in run_id)
            or any(ord(character) < 32 or ord(character) == 127 for character in run_id)
        ):
            raise ValueError(
                "run_id must be one non-glob printable path component of at most 128 characters"
            )
        return run_id

    def _create_run_dir(self, run_id: str) -> Path:
        configured_root = self.config.runtime.run_root
        if configured_root.is_absolute():
            run_root = configured_root.resolve()
        else:
            run_root = (self.project_root / configured_root).resolve()
            if run_root != self.project_root and not run_root.is_relative_to(self.project_root):
                raise ValueError("relative runtime.run_root must resolve inside the project root")
        run_root.mkdir(parents=True, exist_ok=True)
        run_dir = run_root / run_id
        run_dir.mkdir(exist_ok=False)
        return run_dir.resolve()

    @staticmethod
    def _assert_output_outside_run(output: Path, run_dir: Path) -> None:
        try:
            resolved_output = output.expanduser().resolve()
        except (OSError, RuntimeError) as error:
            raise ValueError(f"output path cannot be resolved safely: {output}: {error}") from error
        if resolved_output.is_relative_to(run_dir):
            raise ValueError(
                f"external output must be outside the immutable run directory: {resolved_output}"
            )

    @staticmethod
    def _atomic_copy(source: Path, destination: Path) -> None:
        temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
        try:
            shutil.copy2(source, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _new_run_id() -> str:
        return f"{utc_now().replace(':', '').replace('-', '')[:15]}-{uuid.uuid4().hex[:8]}"
