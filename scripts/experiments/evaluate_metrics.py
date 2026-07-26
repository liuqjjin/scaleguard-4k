#!/usr/bin/env python3
"""Run the ScaleGuard metric receipt command."""

from __future__ import annotations

import sys

from scaleguard.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["evaluation", "metrics", *sys.argv[1:]]))
