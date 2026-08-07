"""Image validation, normalization, and artifact discovery."""

from __future__ import annotations

import hashlib
import io
import os
import tempfile
from pathlib import Path

from PIL import Image, ImageOps

from scaleguard.contracts import ImageArtifact
from scaleguard.errors import ArtifactError, ScaleGuardError
from scaleguard.provenance import load_regular_file_snapshot

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def inspect_image(path: Path, *, mock: bool, stage: str) -> ImageArtifact:
    """Validate an image eagerly and return immutable provenance."""

    try:
        payload, digest = load_regular_file_snapshot(path, f"{stage} image")
        with Image.open(io.BytesIO(payload)) as image:
            image.verify()
        with Image.open(io.BytesIO(payload)) as image:
            width, height = image.size
            media_type = Image.MIME.get(image.format or "", "application/octet-stream")
    except ScaleGuardError as error:
        if not path.exists():
            raise ArtifactError(f"{stage} image does not exist: {path}") from error
        raise ArtifactError(f"{stage} artifact is unsafe: {path}: {error}") from error
    except (OSError, ValueError) as error:
        raise ArtifactError(f"{stage} artifact is not a readable image: {path}: {error}") from error
    if width <= 0 or height <= 0:
        raise ArtifactError(f"{stage} image has invalid dimensions {width}x{height}: {path}")
    return ImageArtifact(
        path=path.resolve(),
        sha256=digest,
        width=width,
        height=height,
        media_type=media_type,
        mock=mock,
        stage=stage,
    )


def normalize_to_png(source: Path, destination: Path) -> None:
    """Decode one immutable snapshot and write a canonical oriented RGB PNG."""

    resolved_destination = _distinct_destination(source, destination, role="normalized image")
    try:
        payload, _digest = load_regular_file_snapshot(source, "source image")
        with Image.open(io.BytesIO(payload)) as image:
            normalized = ImageOps.exif_transpose(image).convert("RGB")
            encoded = io.BytesIO()
            normalized.save(encoded, format="PNG", optimize=False)
    except (OSError, ValueError, ScaleGuardError) as error:
        raise ArtifactError(f"cannot normalize image {source}: {error}") from error
    _write_bytes_atomic(resolved_destination, encoded.getvalue())


def copy_artifact(source: Path, destination: Path) -> None:
    resolved_destination = _distinct_destination(source, destination, role="artifact copy")
    try:
        payload, _digest = load_regular_file_snapshot(source, "source artifact")
    except ScaleGuardError as error:
        raise ArtifactError(f"cannot copy artifact {source}: {error}") from error
    _write_bytes_atomic(resolved_destination, payload)


def _distinct_destination(source: Path, destination: Path, *, role: str) -> Path:
    try:
        resolved_source = source.expanduser().resolve()
        resolved_destination = destination.expanduser().resolve()
    except (OSError, RuntimeError) as error:
        raise ArtifactError(f"cannot resolve {role} paths safely: {error}") from error
    if resolved_source == resolved_destination:
        raise ArtifactError(f"{role} would overwrite its source: {resolved_source}")
    return resolved_destination


def _write_bytes_atomic(destination: Path, payload: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        descriptor: int | None = None
        try:
            descriptor = os.open(
                destination.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            os.fsync(descriptor)
        except OSError:
            pass
        finally:
            if descriptor is not None:
                os.close(descriptor)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def image_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def discover_single_output(
    root: Path,
    *,
    expected_size: tuple[int, int] | None = None,
    ignored: set[Path] | None = None,
) -> Path:
    """Resolve one worker output without assuming its input filename.

    Nested upstream layouts are permitted. If an expected size is supplied, only
    exact-size images are considered; ambiguity is always an error.
    """

    root_resolved = root.resolve()
    ignored_resolved = {path.resolve() for path in (ignored or set())}
    candidates = []
    for path in image_files(root):
        resolved = path.resolve()
        if not resolved.is_relative_to(root_resolved):
            raise ArtifactError(f"worker image escaped its private output directory: {path}")
        if resolved not in ignored_resolved:
            candidates.append(path)
    if expected_size is not None:
        matching: list[Path] = []
        for candidate in candidates:
            try:
                with Image.open(candidate) as image:
                    if image.size == expected_size:
                        matching.append(candidate)
            except OSError:
                continue
        candidates = matching
    if not candidates:
        expected = (
            f" with dimensions {expected_size[0]}x{expected_size[1]}" if expected_size else ""
        )
        raise ArtifactError(f"worker produced no readable image{expected} under {root}")
    if len(candidates) != 1:
        rendered = ", ".join(str(path.relative_to(root)) for path in candidates[:8])
        suffix = " ..." if len(candidates) > 8 else ""
        raise ArtifactError(
            f"worker output is ambiguous under {root}: {len(candidates)} candidates: "
            f"{rendered}{suffix}"
        )
    return candidates[0]


def assert_scale(
    previous: ImageArtifact,
    candidate: ImageArtifact,
    *,
    factor: int,
    tolerance_pixels: int = 2,
) -> None:
    expected_width = previous.width * factor
    expected_height = previous.height * factor
    if (
        abs(candidate.width - expected_width) > tolerance_pixels
        or abs(candidate.height - expected_height) > tolerance_pixels
    ):
        raise ArtifactError(
            f"scale worker returned {candidate.width}x{candidate.height}; expected approximately "
            f"{expected_width}x{expected_height} for a {factor}x step"
        )
