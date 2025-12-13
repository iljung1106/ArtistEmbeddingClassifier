#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset crawling entrypoint (Danbooru artist list via Selenium + Gelbooru downloads).

This wraps `crawler_api.py` so you can run:
  python scripts/crawl_dataset.py --help
"""

from __future__ import annotations

import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    import crawler_api

    crawler_api.main_cli()


if __name__ == "__main__":
    main()


