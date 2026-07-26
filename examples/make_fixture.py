#!/usr/bin/env python3
"""Generate a small deterministic image for CPU smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    width, height = 192, 128
    x = np.linspace(0, 1, width, dtype=np.float32)
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    rgb = np.stack(
        (
            np.broadcast_to(x, (height, width)),
            np.broadcast_to(y, (height, width)),
            0.25 + 0.5 * np.broadcast_to(x * y, (height, width)),
        ),
        axis=-1,
    )
    image = Image.fromarray(np.uint8(np.clip(rgb, 0, 1) * 255), mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((18, 18, 92, 72), outline=(245, 245, 245), width=3)
    draw.text((26, 40), "ScaleGuard", fill=(255, 255, 255))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)


if __name__ == "__main__":
    main()
