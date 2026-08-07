from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scaleguard.errors import ConfigurationError
from scaleguard.imaging.forward_models import (
    ResizeModel,
    build_forward_model,
    evaluate_measurement_consistency,
)
from scaleguard.metrics.scale import evaluate_scale_consistency


def test_cross_scale_consistency_is_zero_for_an_exact_constant_reconstruction(
    tmp_path: Path,
    make_image: Callable[..., Path],
) -> None:
    trusted = make_image(tmp_path / "trusted.png", size=(7, 5), color=(100, 100, 100))
    candidate = make_image(tmp_path / "candidate.png", size=(28, 20), color=(100, 100, 100))

    result = evaluate_scale_consistency(candidate, trusted)

    assert result.nrmse == pytest.approx(0.0, abs=1e-7)
    assert result.edge_mae == pytest.approx(0.0, abs=1e-7)


def test_cross_scale_consistency_rejects_dimensions_without_an_edge_domain(
    tmp_path: Path,
    make_image: Callable[..., Path],
) -> None:
    trusted = make_image(tmp_path / "trusted.png", size=(1, 4))
    candidate = make_image(tmp_path / "candidate.png", size=(4, 16))

    with pytest.raises(ValueError, match="both trusted dimensions to be at least 2"):
        evaluate_scale_consistency(candidate, trusted)


def test_cross_scale_nrmse_has_a_known_numeric_value_for_a_constant_shift(
    tmp_path: Path,
    make_image: Callable[..., Path],
) -> None:
    trusted = make_image(tmp_path / "trusted.png", size=(6, 4), color=(100, 100, 100))
    candidate = make_image(tmp_path / "candidate.png", size=(24, 16), color=(120, 120, 120))

    result = evaluate_scale_consistency(candidate, trusted)

    expected = (20.0 / 255.0) / ((100.0 / 255.0) + 1e-6)
    assert result.nrmse == pytest.approx(expected, rel=1e-5)
    assert result.edge_mae == pytest.approx(0.0, abs=1e-7)


def test_cross_scale_edge_error_detects_structure_drift(tmp_path: Path) -> None:
    trusted_array = np.zeros((6, 8, 3), dtype=np.uint8)
    trusted_array[:, 4:, :] = 180
    candidate_array = np.zeros((24, 32, 3), dtype=np.uint8)
    candidate_array[12:, :, :] = 180
    trusted = tmp_path / "trusted.png"
    candidate = tmp_path / "candidate.png"
    Image.fromarray(trusted_array, "RGB").save(trusted)
    Image.fromarray(candidate_array, "RGB").save(candidate)

    result = evaluate_scale_consistency(candidate, trusted)

    assert result.nrmse > 0.5
    assert result.edge_mae > 0.05


def test_measurement_consistency_is_zero_when_the_forward_observation_matches(
    tmp_path: Path,
    make_image: Callable[..., Path],
) -> None:
    candidate = make_image(tmp_path / "candidate.png", size=(24, 16), color=(100, 100, 100))
    observation = make_image(tmp_path / "observation.png", size=(6, 4), color=(100, 100, 100))

    result = evaluate_measurement_consistency(candidate, observation, ResizeModel())

    assert result.model == "resize_lanczos"
    assert result.nrmse == pytest.approx(0.0, abs=1e-7)


def test_measurement_nrmse_has_a_known_numeric_value_for_a_constant_shift(
    tmp_path: Path,
    make_image: Callable[..., Path],
) -> None:
    candidate = make_image(tmp_path / "candidate.png", size=(24, 16), color=(120, 120, 120))
    observation = make_image(tmp_path / "observation.png", size=(6, 4), color=(100, 100, 100))

    result = evaluate_measurement_consistency(candidate, observation, ResizeModel())

    expected = (20.0 / 255.0) / ((100.0 / 255.0) + 1e-6)
    assert result.nrmse == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize(
    ("name", "expected_name"),
    [
        ("resize", "resize_lanczos"),
        ("gaussian_psf", "gaussian_psf_resize"),
        ("jpeg", "jpeg_resize"),
        ("poisson_gaussian", "poisson_gaussian_resize"),
        ("haze", "uniform_haze_resize"),
    ],
)
def test_forward_model_registry_is_explicit(name: str, expected_name: str) -> None:
    assert build_forward_model(name, {}).name == expected_name


def test_forward_model_identity_binds_parameters_and_seed_domain() -> None:
    narrow = build_forward_model("gaussian_psf", {"sigma": 0.8})
    wide = build_forward_model("gaussian_psf", {"sigma": 1.6})
    assert narrow.identity != wide.identity
    assert narrow.identity["parameters"]["sigma"] == 0.8

    for invalid_seed in (-1, True, 2**63, 1.0):
        with pytest.raises(ConfigurationError, match="seed must be an integer between"):
            build_forward_model("poisson_gaussian", {"seed": invalid_seed})


@pytest.mark.parametrize(
    ("name", "parameters"),
    [
        ("gaussian_psf", {"sigma": 0.8}),
        ("jpeg", {"quality": 80}),
        ("poisson_gaussian", {"peak_photons": 50, "read_noise_std": 0.0, "seed": 7}),
        ("haze", {"transmission": 0.7, "atmospheric_light": 0.9}),
    ],
)
def test_forward_models_map_reconstructions_to_the_requested_observation_size(
    name: str,
    parameters: dict[str, float | int],
) -> None:
    image = Image.new("RGB", (12, 8), (70, 110, 150))
    model = build_forward_model(name, parameters)

    first = model.apply(image, (5, 3))
    second = model.apply(image, (5, 3))

    assert first.mode == "RGB"
    assert first.size == (5, 3)
    assert first.tobytes() == second.tobytes()


@pytest.mark.parametrize(
    ("name", "parameters", "message"),
    [
        ("jpeg", {"quality": 0}, "JPEG quality"),
        ("jpeg", {"quality": 101}, "JPEG quality"),
        ("haze", {"transmission": "opaque"}, "must be numeric"),
        ("unknown", {}, "unknown measurement model"),
    ],
)
def test_forward_model_configuration_errors_are_actionable(
    name: str,
    parameters: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        build_forward_model(name, parameters)
