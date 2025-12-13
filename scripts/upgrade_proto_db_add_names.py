#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Upgrade an existing prototype DB .pt file to include artist names (label_names).

This is useful for older prototype files that only store:
  - centers: [N, D]
  - labels:  [N]

We infer label_names from `dataset/` folder (sorted artist directories), matching
`train_style_ddp.TriViewDataset` label assignment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def infer_label_names(dataset_dir: Path) -> list[str]:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset dir not found: {dataset_dir}")
    names = sorted([p.name for p in dataset_dir.iterdir() if p.is_dir()])
    if not names:
        raise RuntimeError(f"No artist folders found under: {dataset_dir}")
    return names


def main() -> None:
    p = argparse.ArgumentParser(description="Add label_names to an existing prototype DB .pt")
    p.add_argument("--in", dest="in_path", required=True, help="Input .pt prototype file")
    p.add_argument("--out", dest="out_path", default=None, help="Output .pt (default: overwrite input)")
    p.add_argument("--dataset-dir", type=str, default="dataset", help="Dataset root to infer artist names from")
    args = p.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path) if args.out_path else in_path
    dataset_dir = Path(args.dataset_dir)

    obj = torch.load(str(in_path), map_location="cpu")
    if not isinstance(obj, dict) or "centers" not in obj or "labels" not in obj:
        raise ValueError("Unsupported prototype file format (expected dict with centers+labels).")

    if "label_names" in obj and isinstance(obj["label_names"], list) and obj["label_names"]:
        print("label_names already present; nothing to do.")
        if out_path != in_path:
            torch.save(obj, str(out_path))
            print("saved:", out_path)
        return

    label_names = infer_label_names(dataset_dir)
    obj["label_names"] = label_names

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, str(out_path))
    print("saved:", out_path)


if __name__ == "__main__":
    main()


