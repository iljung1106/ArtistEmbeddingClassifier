#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DDP training entrypoint.

This wraps `train_style_ddp.py` so you can run:
  python scripts/train_ddp.py
"""

from __future__ import annotations

import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    import train_style_ddp as ts

    ts.run_ddp_training()


if __name__ == "__main__":
    main()


