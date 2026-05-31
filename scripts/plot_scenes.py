#!/usr/bin/env python3
"""Backward-compatible CLI; implementation lives in ``uavlab.viz.plot_scenes``."""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from uavlab.viz.plot_scenes import main

if __name__ == "__main__":
    main()
