### ArtistEmbeddingClassifier

Train an **artist embedding model** from anime images (whole / face / eye), and evaluate using **per-artist prototypes**.

This repo includes:

- **Face detection**: `yolov5_anime/` (GPL-3.0) from [zymk9/yolov5_anime](https://github.com/zymk9/yolov5_anime)
- **Anime eye Haar cascade**: `anime-eyes-cascade.xml` (GPL-3.0) from [recette-lemon/Haar-Cascade-Anime-Eye-Detector](https://github.com/recette-lemon/Haar-Cascade-Anime-Eye-Detector)

See `THIRD_PARTY_NOTICES.md` for attribution details.

### Hugging Face Space demo

- **Space**: [ij/ArtistEmbeddingClassifier](https://huggingface.co/spaces/ij/ArtistEmbeddingClassifier)

### Video

- **Test video**: 

https://github.com/user-attachments/assets/74c822e6-8e09-41c4-b0b3-08c767caebf9

### Background

최근 웹소설 표지 및 삽화 등 창작 분야에서 AI 생성 이미지 활용이 증가함에 따라, 특정 작가의 화풍 도용 논란이 발생하고 있다. 이는 창작자가 의도하지 않았더라도 다음과 같은 원인으로 발생한다. 첫 째, 타인에게 제공받은 AI 이미지가 특정 작가의 화풍을 모방한 경우이다. 둘 째, 확산 모델이 학습 데이터를 그대로 암기하여 특정 작가를 프롬프트에 입력하지 않아도 유사한 이미지를 생성하는 경우이다.

본 프로젝트는 AI 생성 이미지가 기존 작가의 화풍과 얼마나 유사한지 사전에 검증하는 기능을 구현한다. 이를 통해 창작자가 결과물을 자기 검열하여 의도치 않은 화풍 도용 분쟁을 미연에 방지할 수 있도록 한다.

### Result

입력 이미지에 대해 유사 작가와 그 유사도를 제시하고, 필요한 작가를 추가하는 기능을 성공적으로 구현하였다. 이를 통해 사용자가 AI 생성 이미지의 화풍 유사도를 사전에 진단하여 의도치 않은 분쟁을 예방하고 창작의 투명성을 높일 수 있다. 또, 기존 전문가의 직관에 의존하던 화풍 판단을 객관적 데이터로 전환하여 논란의 여지를 최소화하며, 나아가 추상적인 예술적 개념을 넘어 데이터 관점에서 화풍을 새롭게 정의하는 기준을 제시하고자 한다.

### Future Work

- **벡터 검색 엔진 최적화**: 코사인 유사도 방식의 한계를 극복하기 위해, 대규모 데이터에서도 고속 검색이 가능한 최신 유사도 측정 알고리즘을 도입한다. 동시에, 정밀도 저하를 최소화하며 메모리 사용량을 최소화하여 시스템 효율성을 극대화한다.
- **추가 벡터 생성 성능 강화**: 작가를 추가하기 위해 Few-shot의 이미지만을 투입하는 과정에서 작가의 다양하고 고유한 화풍을 정확하게 임베딩할 수 있도록 개선한다. 이를 통해 기존 학습 데이터와의 품질 격차를 해소한다.
- **모델 일반화 성능 강화**: 학습되지 않은 새로운 이미지나 미세한 스타일 차이 까지 명확하게 구분할 수 있도록 Head의 성능을 지속적으로 개선하여 전반적인 정확도를 향상시킨다.

### Project layout

- **Raw dataset (whole images)**: `dataset/<artist_name>/*`
- **Extracted faces**: `dataset_face/<artist_name>/*`
- **Extracted eye crops**: `dataset_eyes/<artist_name>/*`
- **Training**: `train_style_ddp.py` (DDP)
- **Convenience entrypoints**: `scripts/*.py`
- **Gradio app**: `webui_gradio.py`

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

### 2) Extract faces + eye

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
- **Classify** an uploaded image (artist shown once; best similarity if multiple prototypes exist)
- **XGrad-CAM** visualization for each view (whole / face / eye)
- **View contribution** bars (how much each view contributed)
- **Artists (in DB)** tab showing which labels exist and how many prototypes each has
- **Add prototype (temporary)**: creates *session-only* prototypes via random (whole, face, eye) triplets + K-means
  - Temporary prototypes are **not persisted** to disk on Spaces and will be lost on restart/idle.

Prototype DB selection:
- The **default** prototype DB is `per_artist_prototypes_90_10_full.pt` (if present)
- The dropdown shows **all** `*.pt` files under `./checkpoints_style/` (excluding obvious training checkpoints like `stage*_epoch*.pt`)

If your environment blocks localhost access (common with some proxy/security setups), run:

```bash
python webui_gradio.py --share
```

### Deploy to Hugging Face Spaces (Gradio)

This repo includes a bundler that prepares a Space-ready folder:

```bash
python scripts/make_hf_space_bundle.py --out hf_space
```

Then push the **contents** of `hf_space/` to your Space repo (not the folder itself). Large `.pt` files must be pushed with **Git LFS**:

```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes
git commit -m "Track .pt with Git LFS"
git add -A
git commit -m "Update Space bundle"
git push
```

### Pretrained model (Hugging Face)

- **Model page**: [ij/ArtistEmbedder](https://huggingface.co/ij/ArtistEmbedder) (GPL-3.0)


