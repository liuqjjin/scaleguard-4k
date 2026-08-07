from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import pytest
from PIL import Image, ImageOps

import scaleguard.images as images
from scaleguard.errors import ArtifactError
from scaleguard.images import discover_single_output, inspect_image, normalize_to_png


def test_nested_worker_output_is_discovered_without_a_fixed_filename(
    tmp_path: Path,
    make_image: Callable[..., Path],
) -> None:
    expected = make_image(tmp_path / "scale_1" / "sample" / "结果.jpg", size=(16, 12))

    assert discover_single_output(tmp_path) == expected


def test_ambiguous_worker_outputs_are_rejected_with_candidate_names(
    tmp_path: Path,
    make_image: Callable[..., Path],
) -> None:
    make_image(tmp_path / "first.png", size=(16, 12))
    make_image(tmp_path / "nested" / "second.jpg", size=(16, 12))

    with pytest.raises(ArtifactError) as captured:
        discover_single_output(tmp_path)

    message = str(captured.value)
    assert "ambiguous" in message
    assert "2 candidates" in message
    assert "first.png" in message
    assert "nested/second.jpg" in message


def test_expected_dimensions_disambiguate_nested_outputs(
    tmp_path: Path,
    make_image: Callable[..., Path],
) -> None:
    make_image(tmp_path / "preview.png", size=(4, 3))
    expected = make_image(tmp_path / "nested" / "final.png", size=(16, 12))

    assert discover_single_output(tmp_path, expected_size=(16, 12)) == expected


def test_missing_expected_dimensions_report_the_contract(
    tmp_path: Path,
    make_image: Callable[..., Path],
) -> None:
    make_image(tmp_path / "wrong.png", size=(15, 12))

    with pytest.raises(ArtifactError, match=r"no readable image with dimensions 16x12"):
        discover_single_output(tmp_path, expected_size=(16, 12))


def test_worker_output_symlink_cannot_escape_its_private_directory(
    tmp_path: Path,
    make_image: Callable[..., Path],
) -> None:
    private_output = tmp_path / "private"
    private_output.mkdir()
    escaped = make_image(tmp_path / "escaped.png")
    (private_output / "linked.png").symlink_to(escaped)

    with pytest.raises(ArtifactError, match="escaped its private output directory"):
        discover_single_output(private_output)


def test_image_inspection_and_normalization_reject_missing_or_invalid_inputs(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.png"
    with pytest.raises(ArtifactError, match="does not exist"):
        inspect_image(missing, mock=False, stage="input")

    invalid = tmp_path / "invalid.png"
    invalid.write_bytes(b"not an image")
    with pytest.raises(ArtifactError, match="not a readable image"):
        inspect_image(invalid, mock=False, stage="input")
    with pytest.raises(ArtifactError, match="cannot normalize image"):
        normalize_to_png(invalid, tmp_path / "normalized.png")


def test_image_inspection_uses_one_byte_snapshot_for_metadata_and_hash(
    tmp_path: Path,
    make_image: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = make_image(tmp_path / "first.png", size=(11, 7))
    current = make_image(tmp_path / "current.png", size=(19, 13))
    snapshot = first.read_bytes()
    digest = hashlib.sha256(snapshot).hexdigest()

    monkeypatch.setattr(
        images,
        "load_regular_file_snapshot",
        lambda _path, _label: (snapshot, digest),
    )

    artifact = inspect_image(current, mock=False, stage="race")

    assert (artifact.width, artifact.height) == (11, 7)
    assert artifact.sha256 == digest


def test_normalize_to_png_applies_exif_orientation_from_one_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "oriented.jpg"
    destination = tmp_path / "normalized.png"
    original = Image.new("RGB", (3, 2))
    original.putdata(
        [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (0, 255, 255),
            (255, 0, 255),
        ]
    )
    exif = Image.Exif()
    exif[274] = 6
    original.save(source, format="JPEG", quality=100, subsampling=0, exif=exif)
    with Image.open(source) as encoded:
        expected = ImageOps.exif_transpose(encoded).convert("RGB").copy()

    normalize_to_png(source, destination)

    with Image.open(destination) as normalized:
        assert normalized.format == "PNG"
        assert normalized.mode == "RGB"
        assert normalized.size == (2, 3)
        assert normalized.tobytes() == expected.tobytes()
