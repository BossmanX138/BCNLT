"""Isolated entry point for the fixed BossCrafts regression suite."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.regression_acceptance import worker_main

if __name__ == "__main__":
    raise SystemExit(worker_main())
