#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Face -> (optional) eye extraction entrypoint.

This wraps `anime_face_eye_extract.py` so you can run:
  python scripts/extract_faces_eyes.py --help
"""

from __future__ import annotations

import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    from anime_face_eye_extract import main as _main

    _main()


if __name__ == "__main__":
    main()


