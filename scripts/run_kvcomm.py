#!/usr/bin/env python3
"""KVCOMM-style prefix-anchor calibration runner.

This is an adapter over ``run_kv_boundary_calibration.py``.  It uses the
source/target prefix-end KV difference as an online offset anchor, while
keeping the existing baseline files read-only.
"""
from __future__ import annotations
import sys
from pathlib import Path

from run_kv_boundary_calibration import main

if __name__ == "__main__":
    if "--method" not in sys.argv:
        sys.argv[1:1] = ["--method", "prefix_offset"]
    raise SystemExit(main())
