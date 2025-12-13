### ArtistEmbeddingClassifier

Train an **artist embedding model** from anime images (whole / face / eyes), and evaluate using **per-artist prototypes**.

This repo includes:

- **Face detection**: `yolov5_anime/` (GPL-3.0) from [zymk9/yolov5_anime](https://github.com/zymk9/yolov5_anime)
- **Anime eye Haar cascade**: `anime-eyes-cascade.xml` (GPL-3.0) from [recette-lemon/Haar-Cascade-Anime-Eye-Detector](https://github.com/recette-lemon/Haar-Cascade-Anime-Eye-Detector)

See `THIRD_PARTY_NOTICES.md` for attribution details.

### Project layout

- **Raw dataset (whole images)**: `dataset/<artist_name>/*`
- **Extracted faces**: `dataset_face/<artist_name>/*`
- **Extracted eyes**: `dataset_eyes/<artist_name>/*`
- **Training**: `train_style_ddp.py` (DDP)
- **Convenience entrypoints**: `scripts/*.py`

### Setup

Create a venv and install dependencies:

```bash
python -m venv .venv
. .venv/bin/activate  # Linux/macOS
# .\.venv\Scripts\Activate.ps1  # Windows PowerShell

pip install -r requirements.txt
```

Notes:
- **PyTorch** install varies by CUDA version; if you already have a working torch install, keep it.
- `yolov5_anime/requirements.txt` is included via `requirements.txt`.

### Weights

- **Recommended/default face detector weights**: `yolov5x_anime.pt` (already in repo root)
- The lightweight alternative is `yolov5_anime/weights/yolov5s_anime.pt`

### 1) Crawl dataset (optional)

This crawler:
- collects top artists from Danbooru via Selenium
- downloads images for each artist from Gelbooru via API

Run:

```bash
python scripts/crawl_dataset.py --help
python scripts/crawl_dataset.py --output-dir dataset --use-cache auto
```

Environment variables (recommended):
- `GELBOORU_API_KEY`
- `GELBOORU_USER_ID`

### 2) Extract faces + eyes

Given `dataset/` (whole images), produce `dataset_face/` and `dataset_eyes/`:

```bash
python scripts/extract_faces_eyes.py --help
python scripts/extract_faces_eyes.py --input ./dataset --out-face ./dataset_face --out-eye ./dataset_eyes
```

Defaults:
- face detector: `--yolo-dir ./yolov5_anime --weights ./yolov5x_anime.pt`
- eye cascade: `--cascade ./anime-eyes-cascade.xml`

### 3) Train (DDP)

```bash
python scripts/train_ddp.py
```

Checkpoints/logs are written to `checkpoints_style/`.

### 4) Evaluate (prototype-based)

Prototype evaluation script:

- **Strict 90/10 per view per artist (train+val merged pools)**:

```bash
python scripts/eval_prototypes_strict_90_10.py --ckpt ./checkpoints_style/stage3_epoch24.pt
```

### Web UI (Gradio)

Run:

```bash
python webui_gradio.py
```

The UI loads checkpoints and prototype databases from `./checkpoints_style/` and lets you:
- classify an uploaded image
- add new prototypes and save back to the prototype DB `.pt`

If your environment blocks localhost access (common with some proxy/security setups), run:

```bash
python webui_gradio.py --share
```


