#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a Hugging Face Spaces-ready bundle directory from this repo.

The output folder can be uploaded (or git-pushed) to a new Gradio Space.
We intentionally do NOT rename files in this repo. Instead, the Space README
will specify `app_file: webui_gradio.py` to avoid conflicts with the `app/` package.

Usage:
  python scripts/make_hf_space_bundle.py --out hf_space
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path, *, ignore_globs: list[str] | None = None) -> None:
    ignore = None
    if ignore_globs:
        ignore = shutil.ignore_patterns(*ignore_globs)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=ignore)


def write_space_readme(dst: Path) -> None:
    text = """---
title: ArtistEmbeddingClassifier
sdk: gradio
app_file: webui_gradio.py
license: gpl-3.0
---

### ArtistEmbeddingClassifier (Gradio Space)

This Space bundles the model checkpoint + prototype DB and runs the Gradio UI.

Notes:
- This project is GPL-3.0.
- `yolov5_anime/` is from [zymk9/yolov5_anime](https://github.com/zymk9/yolov5_anime) (GPL-3.0).
- `anime-eyes-cascade.xml` is from [recette-lemon/Haar-Cascade-Anime-Eye-Detector](https://github.com/recette-lemon/Haar-Cascade-Anime-Eye-Detector) (GPL-3.0).
"""
    (dst / "README.md").write_text(text, encoding="utf-8")


def write_space_requirements(dst: Path) -> None:
    # IMPORTANT for Spaces:
    # - HF GPU base images already install torch + gradio + spaces.
    # - If we pin/downgrade these here, pip will try to replace huge packages and may fail.
    # Keep this list minimal and only add what is NOT guaranteed by the base image.
    text = """pillow
pyyaml
tqdm

# OpenCV for face/eye extraction (headless build for Spaces)
opencv-python-headless

# torchvision is NOT in the base image; model code imports it
torchvision

# matplotlib for Grad-CAM colormaps
matplotlib
"""
    (dst / "requirements.txt").write_text(text, encoding="utf-8")


def write_space_packages(dst: Path) -> None:
    # Helps OpenCV on Spaces.
    text = """libgl1
libglib2.0-0
"""
    (dst / "packages.txt").write_text(text, encoding="utf-8")


def write_lfs_gitattributes(dst: Path) -> None:
    # If you push via git, this ensures large weights are handled via LFS.
    text = """*.pt filter=lfs diff=lfs merge=lfs -text
"""
    (dst / ".gitattributes").write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Create Hugging Face Space bundle directory")
    ap.add_argument("--out", type=str, default="hf_space", help="Output folder name")
    args = ap.parse_args()

    out_dir = (ROOT / args.out).resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Core app code
    copy_file(ROOT / "webui_gradio.py", out_dir / "webui_gradio.py")
    copy_file(ROOT / "train_style_ddp.py", out_dir / "train_style_ddp.py")  # needed by app/model_io.py
    copy_tree(ROOT / "app", out_dir / "app", ignore_globs=["__pycache__"])

    # Assets required by the UI
    copy_file(ROOT / "anime-eyes-cascade.xml", out_dir / "anime-eyes-cascade.xml")
    copy_file(ROOT / "yolov5x_anime.pt", out_dir / "yolov5x_anime.pt")

    # Bundle checkpoints/prototypes
    (out_dir / "checkpoints_style").mkdir(exist_ok=True)
    copy_file(ROOT / "checkpoints_style" / "stage3_epoch24.pt", out_dir / "checkpoints_style" / "stage3_epoch24.pt")
    copy_file(
        ROOT / "checkpoints_style" / "per_artist_prototypes_90_10_full.pt",
        out_dir / "checkpoints_style" / "per_artist_prototypes_90_10_full.pt",
    )

    # Vendor yolov5_anime (strip heavy demo assets)
    copy_tree(
        ROOT / "yolov5_anime",
        out_dir / "yolov5_anime",
        ignore_globs=[
            "__pycache__",
            ".git",
            "inference",
            "tutorial.ipynb",
            "Dockerfile",
            # We bundle yolov5x_anime.pt at repo root; don't include extra weights.
            "*.pt",
            "weights/*.pt",
        ],
    )

    # Licensing/attribution
    for fn in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
        if (ROOT / fn).exists():
            copy_file(ROOT / fn, out_dir / fn)

    # Space metadata
    write_space_readme(out_dir)
    write_space_requirements(out_dir)
    write_space_packages(out_dir)
    write_lfs_gitattributes(out_dir)

    print("✅ Created Space bundle at:", out_dir)
    print("Next: upload/push the contents of that folder to your Hugging Face Space repo.")


if __name__ == "__main__":
    main()


