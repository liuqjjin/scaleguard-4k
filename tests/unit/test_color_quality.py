from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from scaleguard.color import apply_adain
from scaleguard.metrics.quality import (
    GradientQualityEvaluator,
    PyiqaQualityEvaluator,
    build_quality_evaluator,
)


def test_adain_matches_reference_channel_statistics_at_candidate_resolution(
    tmp_path: Path,
) -> None:
    candidate_array = np.zeros((10, 14, 3), dtype=np.uint8)
    candidate_array[:, :7] = (20, 80, 140)
    candidate_array[:, 7:] = (180, 140, 60)
    reference_array = np.zeros((5, 7, 3), dtype=np.uint8)
    reference_array[:2] = (40, 60, 80)
    reference_array[2:] = (100, 130, 160)
    candidate = tmp_path / "candidate.png"
    reference = tmp_path / "reference.png"
    destination = tmp_path / "aligned.png"
    Image.fromarray(candidate_array, "RGB").save(candidate)
    Image.fromarray(reference_array, "RGB").save(reference)

    apply_adain(candidate, reference, destination)

    with Image.open(destination) as aligned_image, Image.open(reference) as reference_image:
        aligned = np.asarray(aligned_image, dtype=np.float32)
        resized_reference = np.asarray(
            reference_image.resize(aligned_image.size, Image.Resampling.BICUBIC),
            dtype=np.float32,
        )
    assert aligned_image.size == (14, 10)
    assert aligned.mean(axis=(0, 1)) == pytest.approx(
        resized_reference.mean(axis=(0, 1)),
        abs=1.0,
    )


def test_quality_factory_selects_the_deterministic_proxy() -> None:
    evaluator = build_quality_evaluator("gradient_proxy", "unused", "cpu")

    assert isinstance(evaluator, GradientQualityEvaluator)
    assert evaluator.is_proxy is True


def test_gradient_proxy_rejects_an_image_without_two_dimensional_gradients(
    tmp_path: Path,
    make_image: Callable[..., Path],
) -> None:
    image = make_image(tmp_path / "line.png", size=(1, 4))

    with pytest.raises(ValueError, match="both image dimensions to be at least 2"):
        GradientQualityEvaluator().score(image)


def test_pyiqa_backend_normalizes_lower_is_better_scores(
    tmp_path: Path,
    make_image: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeValue:
        def detach(self) -> FakeValue:
            return self

        def cpu(self) -> FakeValue:
            return self

        def item(self) -> float:
            return 2.5

    class FakeMetric:
        lower_better = True

        def __call__(self, _path: str) -> FakeValue:
            return FakeValue()

    fake_pyiqa = SimpleNamespace(create_metric=lambda _name, device: FakeMetric())
    monkeypatch.setitem(sys.modules, "pyiqa", fake_pyiqa)
    image = make_image(tmp_path / "image.png")

    evaluator = PyiqaQualityEvaluator("niqe", "cpu")

    assert evaluator.name == "pyiqa:niqe"
    assert evaluator.score(image) == -2.5


def test_pyiqa_backend_has_an_actionable_optional_dependency_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "pyiqa", raising=False)
    original_import = __import__

    def fake_import(name: str, *args, **kwargs):
        if name == "pyiqa":
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(RuntimeError, match=r"install scaleguard-4k\[metrics\]"):
        build_quality_evaluator("pyiqa", "musiq", "cpu")


def test_quality_factory_rejects_an_unknown_backend() -> None:
    with pytest.raises(ValueError, match="unknown quality backend"):
        build_quality_evaluator("unknown", "metric", "cpu")
