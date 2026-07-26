"""Evidence-preserving calibration and experiment summaries."""

from scaleguard.evaluation.calibration import (
    CalibrationParameters,
    calibrate_from_manifests,
    verify_calibration_receipt,
)
from scaleguard.evaluation.metrics import (
    METRIC_RECEIPT_SCHEMA,
    PYIQA_METRICS,
    SUPPORTED_METRICS,
    evaluate_metric_receipt,
    psnr_rgb,
    ssim_rgb,
)
from scaleguard.evaluation.summary import EXPERIMENT_GROUPS, summarize_paired_manifests

__all__ = [
    "EXPERIMENT_GROUPS",
    "METRIC_RECEIPT_SCHEMA",
    "PYIQA_METRICS",
    "SUPPORTED_METRICS",
    "CalibrationParameters",
    "calibrate_from_manifests",
    "evaluate_metric_receipt",
    "psnr_rgb",
    "ssim_rgb",
    "summarize_paired_manifests",
    "verify_calibration_receipt",
]
