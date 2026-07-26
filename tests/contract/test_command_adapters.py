from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

from scaleguard.backends.command import CommandRestorationBackend, CommandScaleBackend
from scaleguard.config import CoZConfig, FourKAgentConfig, RuntimeConfig


def write_worker(path: Path) -> Path:
    path.write_text(
        """
import argparse
from pathlib import Path
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--factor", type=int, required=True)
args = parser.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)
with Image.open(args.input) as image:
    image.convert("RGB").resize(
        (image.width * args.factor, image.height * args.factor),
        Image.Resampling.NEAREST,
    ).save(args.output, "PNG")
print(f"wrote {args.output}")
""",
        encoding="utf-8",
    )
    return path


def test_command_restoration_backend_normalizes_private_io_and_preserves_evidence(
    tmp_path: Path,
    make_image: Callable[..., Path],
) -> None:
    worker = write_worker(tmp_path / "fake worker.py")
    config = FourKAgentConfig(
        mode="command",
        command=(
            sys.executable,
            str(worker),
            "--input",
            "{input}",
            "--output",
            "{output}",
            "--factor",
            "{bridge_factor}",
        ),
    )
    backend = CommandRestorationBackend(
        config,
        RuntimeConfig(process_timeout_seconds=2.0, gpu_poll_interval_seconds=0.01),
    )
    source = make_image(tmp_path / "输入 image.jpg", size=(5, 3), image_format="JPEG")
    destination = tmp_path / "states" / "restored.png"
    run_dir = tmp_path / "run with spaces"

    result = backend.restore(
        source,
        destination,
        bridge_factor=2,
        run_dir=run_dir,
    )

    assert result.image.path == destination.resolve()
    assert (result.image.width, result.image.height) == (10, 6)
    assert result.image.mock is False
    assert result.metadata == {"backend": "4kagent_command", "bridge_factor": 2}
    assert result.process is not None
    assert result.process.returncode == 0
    assert result.process.argv[0] == sys.executable
    assert (run_dir / "workers" / "4kagent-command" / "input" / "source.png").is_file()


def test_command_scale_backend_produces_exactly_one_4x_state(
    tmp_path: Path,
    make_image: Callable[..., Path],
) -> None:
    worker = write_worker(tmp_path / "fake-coz.py")
    config = CoZConfig(
        mode="command",
        command=(
            sys.executable,
            str(worker),
            "--input",
            "{input}",
            "--output",
            "{output}",
            "--factor",
            "4",
        ),
        seed=19,
    )
    backend = CommandScaleBackend(
        config,
        RuntimeConfig(process_timeout_seconds=2.0, gpu_poll_interval_seconds=0.01),
    )
    source = make_image(tmp_path / "source.png", size=(5, 3))
    destination = tmp_path / "state.png"
    run_dir = tmp_path / "run"

    with backend.session(run_dir) as session:
        result = session.upscale_once(source, destination, step_index=2, seed=20)
        session.accept(result, step_index=2)
        session.rollback(step_index=2)

    assert (result.image.width, result.image.height) == (20, 12)
    assert result.metadata == {"backend": "coz_command", "seed": 20}
    assert result.process is not None
    assert result.process.returncode == 0
    assert result.image.mock is False
