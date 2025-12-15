from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

try:
    import spaces  # Hugging Face Spaces helper package
except Exception:  # noqa: BLE001
    spaces = None

# Detect if running on HF Spaces (ZeroGPU requires special handling)
_ON_SPACES = bool(os.getenv("SPACE_ID") or os.getenv("HF_SPACE"))

def _patch_fastapi_starlette_middleware_unpack() -> None:
    """
    Work around FastAPI/Starlette version mismatches where Starlette's Middleware
    iterates as (cls, args, kwargs) but FastAPI expects (cls, options).

    The user reported: ValueError: too many values to unpack (expected 2)
    in fastapi.applications.FastAPI.build_middleware_stack.
    """
    try:
        import fastapi.applications as fa
        from starlette.middleware import Middleware as StarletteMiddleware
    except Exception:
        return

    # Idempotent: don't patch multiple times.
    if getattr(fa.FastAPI.build_middleware_stack, "_aec_patched", False):
        return

    orig = fa.FastAPI.build_middleware_stack

    def patched_build_middleware_stack(self):  # noqa: ANN001
        # Mostly copied from FastAPI, but with robust handling of Middleware objects.
        debug = self.debug
        error_handler = None
        exception_handlers = {}
        if self.exception_handlers:
            exception_handlers = self.exception_handlers
            error_handler = exception_handlers.get(500) or exception_handlers.get(Exception)

        from starlette.middleware.errors import ServerErrorMiddleware
        from starlette.middleware.exceptions import ExceptionMiddleware
        from fastapi.middleware.asyncexitstack import AsyncExitStackMiddleware

        middleware = (
            [StarletteMiddleware(ServerErrorMiddleware, handler=error_handler, debug=debug)]
            + self.user_middleware
            + [
                StarletteMiddleware(ExceptionMiddleware, handlers=exception_handlers, debug=debug),
                StarletteMiddleware(AsyncExitStackMiddleware),
            ]
        )

        app = self.router
        for m in reversed(middleware):
            # Starlette Middleware object
            if hasattr(m, "cls") and hasattr(m, "args") and hasattr(m, "kwargs"):
                app = m.cls(app=app, *list(m.args), **dict(m.kwargs))
                continue

            # Old-style tuple/list
            if isinstance(m, (tuple, list)):
                if len(m) == 2:
                    cls, options = m
                    app = cls(app=app, **options)
                    continue
                if len(m) == 3:
                    cls, args, kwargs = m
                    app = cls(app=app, *list(args), **dict(kwargs))
                    continue

            # Fallback to original behavior for unexpected types
            return orig(self)

        return app

    patched_build_middleware_stack._aec_patched = True  # type: ignore[attr-defined]
    fa.FastAPI.build_middleware_stack = patched_build_middleware_stack


_patch_fastapi_starlette_middleware_unpack()

import gradio as gr

if spaces is not None:
    # Hugging Face GPU Spaces require at least one @spaces.GPU-decorated function.
    # We decorate a tiny no-op marker and also (optionally) wrap inference-heavy calls.
    @spaces.GPU
    def _spaces_gpu_marker():  # noqa: D401
        """Marker function for Hugging Face GPU Spaces."""
        return True

def _launch_compat(demo: gr.Blocks, **kwargs):
    """
    Launch Gradio across versions by only passing supported kwargs.
    Some versions don't support e.g. `show_api=...`.
    """
    import inspect

    sig = inspect.signature(demo.launch)
    allowed = set(sig.parameters.keys())
    safe_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
    return demo.launch(**safe_kwargs)

def _patch_gradio_client_bool_jsonschema() -> None:
    """
    Work around gradio_client JSON-schema parsing bug where it assumes schema is a dict,
    but JSON Schema allows booleans for additionalProperties (true/false).

    Error seen:
      TypeError: argument of type 'bool' is not iterable
      in gradio_client/utils.py:get_type -> if "const" in schema:
    """
    try:
        import gradio_client.utils as gcu
    except Exception:
        return

    # Idempotent: patch once.
    if getattr(getattr(gcu, "get_type", None), "_aec_patched", False):
        return

    orig_get_type = gcu.get_type

    def patched_get_type(schema):  # noqa: ANN001
        if isinstance(schema, bool):
            # additionalProperties: false/true
            return "object"
        if schema is None:
            return "object"
        if not isinstance(schema, dict):
            return "object"
        return orig_get_type(schema)

    patched_get_type._aec_patched = True  # type: ignore[attr-defined]
    gcu.get_type = patched_get_type

    # Also patch the deeper helper that assumes schema is always a dict.
    orig_inner = getattr(gcu, "_json_schema_to_python_type", None)
    if callable(orig_inner) and not getattr(orig_inner, "_aec_patched", False):
        def patched_inner(schema, defs=None):  # noqa: ANN001
            # JSON Schema allows boolean schemas: https://json-schema.org/
            if isinstance(schema, bool):
                return "typing.Any"
            if schema is None:
                return "typing.Any"
            if not isinstance(schema, dict):
                return "typing.Any"
            return orig_inner(schema, defs)

        patched_inner._aec_patched = True  # type: ignore[attr-defined]
        gcu._json_schema_to_python_type = patched_inner


_patch_gradio_client_bool_jsonschema()

from app.model_io import LoadedModel, embed_triview, load_style_model
from app.proto_db import PrototypeDB, load_prototype_db, topk_predictions_unique_labels
from app.view_extractor import AnimeFaceEyeExtractor, ExtractorCfg
from app.visualization import ViewAnalysis, analyze_views, format_view_weights_html


ROOT = Path(__file__).resolve().parent
CKPT_DIR = ROOT / "checkpoints_style"


def _list_pt_files(folder: Path) -> List[str]:
    if not folder.exists():
        return []
    return [str(p) for p in sorted(folder.glob("*.pt"))]

def _list_ckpt_files(folder: Path) -> List[str]:
    files = _list_pt_files(folder)
    # heuristics: training checkpoints usually look like "stageX_epochY.pt"
    ckpts = [f for f in files if "stage" in Path(f).name.lower() and "epoch" in Path(f).name.lower()]
    return ckpts if ckpts else files


def _list_proto_files(folder: Path) -> List[str]:
    """
    List prototype DB candidates.

    On Spaces, users may upload prototype DBs with arbitrary names. We therefore:
    - include all *.pt in checkpoints_style
    - but try to exclude obvious training checkpoints like stageX_epochY.pt
    """
    files = _list_pt_files(folder)
    out: List[str] = []
    for f in files:
        name = Path(f).name.lower()
        # exclude training checkpoints
        if ("stage" in name) and ("epoch" in name):
            continue
        out.append(f)
    return out if out else files


def _guess_default_ckpt(files: List[str]) -> Optional[str]:
    # prefer stage3_epoch24.pt if present
    for f in files:
        if Path(f).name.lower() == "stage3_epoch24.pt":
            return f
    return files[-1] if files else None


def _guess_default_proto(files: List[str]) -> Optional[str]:
    # Prefer the strict 90/10 prototype DB if present.
    for f in files:
        if Path(f).name.lower() == "per_artist_prototypes_90_10_full.pt":
            return f
    # Otherwise, try to prefer a file with "proto" in name
    for f in files:
        if "proto" in Path(f).name.lower():
            return f
    return files[0] if files else None


def _pil_to_tensor(im: Image.Image, T) -> torch.Tensor:
    # `T` is torchvision transform pipeline from train_style_ddp.make_val_transforms
    return T(im.convert("RGB"))


@dataclass
class State:
    lm: Optional[LoadedModel] = None
    ckpt_path: Optional[str] = None
    db: Optional[PrototypeDB] = None
    proto_path: Optional[str] = None
    extractor: Optional[AnimeFaceEyeExtractor] = None


APP_STATE = State()


def load_all(ckpt_path: str, proto_path: str, device: str) -> str:
    if not ckpt_path:
        return "❌ No checkpoint selected."
    if not proto_path:
        return "❌ No prototype DB selected."

    # Force CPU on HF Spaces (ZeroGPU doesn't allow CUDA init in main process)
    if _ON_SPACES:
        device = "cpu"

    try:
        lm = load_style_model(ckpt_path, device=device)
        db = load_prototype_db(proto_path, try_dataset_dir=str(ROOT / "dataset"))
    except Exception as e:
        return f"❌ Load failed: {e}"

    if db.dim != lm.embed_dim:
        return f"❌ Dim mismatch: model embed_dim={lm.embed_dim} but prototypes dim={db.dim}"

    APP_STATE.lm = lm
    APP_STATE.ckpt_path = ckpt_path
    APP_STATE.db = db
    APP_STATE.proto_path = proto_path

    # initialize view extractor (whole -> face/eye) with defaults
    try:
        cfg = ExtractorCfg(
            yolo_dir=ROOT / "yolov5_anime",
            weights=ROOT / "yolov5x_anime.pt",
            cascade=ROOT / "anime-eyes-cascade.xml",
            yolo_device="cpu" if _ON_SPACES else ("0" if torch.cuda.is_available() else "cpu"),
        )
        APP_STATE.extractor = AnimeFaceEyeExtractor(cfg)
    except Exception:
        APP_STATE.extractor = None

    return f"✅ Loaded checkpoint `{Path(ckpt_path).name}` (stage={lm.stage_i}) and proto DB `{Path(proto_path).name}` (N={db.centers.shape[0]})"


def classify_and_analyze(
    whole_img,
    topk: int,
):
    """
    Classify and analyze an image in one pass.
    Returns: status, table_rows, view_weights_html,
             gcam_whole, gcam_face, gcam_eye, face_preview, eye_preview
    """
    empty_result = ("", [], "", None, None, None, None, None)

    if APP_STATE.lm is None or APP_STATE.db is None:
        return ("❌ Click **Load** first.",) + empty_result[1:]

    lm = APP_STATE.lm
    db = APP_STATE.db
    ex = APP_STATE.extractor

    def _to_pil(x):
        if x is None:
            return None
        if isinstance(x, Image.Image):
            return x
        return Image.fromarray(x)

    w = _to_pil(whole_img)
    if w is None:
        return ("❌ Provide a whole image.",) + empty_result[1:]

    try:
        # Extract face and eye
        face_pil = None
        eye_pil = None
        if ex is not None:
            rgb = np.array(w.convert("RGB"))
            face_rgb, eye_rgb = ex.extract(rgb)
            if face_rgb is not None:
                face_pil = Image.fromarray(face_rgb)
            if eye_rgb is not None:
                eye_pil = Image.fromarray(eye_rgb)

        # Prepare tensors
        wt = _pil_to_tensor(w, lm.T_w)
        ft = _pil_to_tensor(face_pil, lm.T_f) if face_pil is not None else None
        et = _pil_to_tensor(eye_pil, lm.T_e) if eye_pil is not None else None

        # Classification
        z = embed_triview(lm, whole=wt, face=ft, eyes=et)
        preds = topk_predictions_unique_labels(db, z, topk=int(topk))
        rows = [[name, float(score)] for (name, score) in preds]

        # Analysis (XGrad-CAM + view weights)
        views = {"whole": wt, "face": ft, "eyes": et}
        original_images = {"whole": w, "face": face_pil, "eyes": eye_pil}
        analysis = analyze_views(lm.model, views, original_images, lm.device)
        view_weights_html = format_view_weights_html(analysis)

        return (
            "✅ Done",
            rows,
            view_weights_html,
            analysis.gradcam_heatmaps.get("whole"),
            analysis.gradcam_heatmaps.get("face"),
            analysis.gradcam_heatmaps.get("eyes"),
            face_pil,
            eye_pil,
        )
    except Exception as e:
        return (f"❌ Failed: {e}",) + empty_result[1:]


def list_artists_in_db():
    """
    List all artists present in the currently loaded prototype DB.
    Returns: status, rows [artist, prototype_count]
    """
    if APP_STATE.db is None:
        return "❌ Click **Load** first.", []

    db = APP_STATE.db
    # Count prototypes per label id
    counts: dict[int, int] = {}
    for lid in db.labels.detach().cpu().tolist():
        counts[int(lid)] = counts.get(int(lid), 0) + 1

    rows: list[list] = []
    for lid, name in enumerate(db.label_names):
        c = int(counts.get(int(lid), 0))
        if c > 0:
            rows.append([name, c])

    rows.sort(key=lambda r: (-int(r[1]), str(r[0]).lower()))
    return f"✅ {len(rows)} artists in DB (total prototypes: {int(db.centers.shape[0])}).", rows


def _gallery_item_to_pil(item) -> Optional[Image.Image]:
    """Convert a Gradio gallery item to PIL Image (handles various formats)."""
    if item is None:
        return None
    # Already a PIL Image
    if isinstance(item, Image.Image):
        return item
    # Tuple format: (image, caption)
    if isinstance(item, (tuple, list)) and len(item) >= 1:
        return _gallery_item_to_pil(item[0])
    # Dict format: {"image": ..., "caption": ...} or {"name": filepath, ...}
    if isinstance(item, dict):
        if "image" in item:
            return _gallery_item_to_pil(item["image"])
        if "name" in item:
            return Image.open(item["name"]).convert("RGB")
        if "path" in item:
            return Image.open(item["path"]).convert("RGB")
    # String path
    if isinstance(item, str):
        return Image.open(item).convert("RGB")
    # Numpy array
    if isinstance(item, np.ndarray):
        return Image.fromarray(item).convert("RGB")
    return None


def _kmeans_cosine(Z: torch.Tensor, K: int, iters: int = 20, seed: int = 42) -> torch.Tensor:
    """
    K-means clustering in cosine space (CPU only).
    Returns K cluster centers (normalized).
    """
    Z = torch.nn.functional.normalize(Z, dim=1)
    N, D = Z.shape
    if N <= K:
        return Z.clone()

    # Initialize centers randomly
    import random
    random.seed(seed)
    init_idx = random.sample(range(N), K)
    C = Z[init_idx].clone()

    for _ in range(iters):
        # Assign each point to nearest center
        sim = Z @ C.t()
        assign = sim.argmax(dim=1)

        # Recompute centers
        new_C = torch.zeros(K, D, dtype=Z.dtype)
        counts = torch.zeros(K, dtype=torch.long)
        for i, c in enumerate(assign.tolist()):
            new_C[c] += Z[i]
            counts[c] += 1

        # Handle empty clusters
        for k in range(K):
            if counts[k] == 0:
                # Reinitialize from a random point
                new_C[k] = Z[random.randint(0, N - 1)]
                counts[k] = 1

        C = new_C / counts.unsqueeze(1).clamp_min(1).float()
        C = torch.nn.functional.normalize(C, dim=1)

    return C


def add_prototype(
    label_name: str,
    images: List,
    k_prototypes: int,
    n_triplets: int,
) -> str:
    """
    Add temporary prototypes using random triplet combinations and K-means clustering.
    Similar to the eval process: extract views, create random triplets, embed, cluster.
    """
    import random

    if APP_STATE.lm is None or APP_STATE.db is None:
        return "❌ Click **Load** first."
    lm = APP_STATE.lm
    db = APP_STATE.db
    ex = APP_STATE.extractor

    label_name = (label_name or "").strip()
    if not label_name:
        return "❌ Label name is required."
    if not images:
        return "❌ Upload at least 1 image."

    k_prototypes = max(1, int(k_prototypes))
    n_triplets = max(1, int(n_triplets))

    # Step 1: Extract whole/face/eye from all uploaded images
    wholes: List[Image.Image] = []
    faces: List[Image.Image] = []
    eyes_list: List[Image.Image] = []
    errors: List[str] = []

    for i, x in enumerate(images):
        try:
            im = _gallery_item_to_pil(x)
            if im is None:
                errors.append(f"Image {i}: could not parse format {type(x)}")
                continue

            wholes.append(im)

            # Extract face and eye
            if ex is not None:
                rgb = np.array(im.convert("RGB"))
                face_rgb, eyes_rgb = ex.extract(rgb)
                if face_rgb is not None:
                    faces.append(Image.fromarray(face_rgb))
                if eyes_rgb is not None:
                    eyes_list.append(Image.fromarray(eyes_rgb))
        except Exception as e:
            errors.append(f"Image {i}: {e}")
            continue

    if not wholes:
        err_detail = "; ".join(errors[:3]) if errors else "unknown error"
        return f"❌ Could not process any images. Details: {err_detail}"

    # Step 2: Create random triplet combinations
    # If we have fewer faces/eyes than wholes, we still try to make triplets
    triplets: List[Tuple[Image.Image, Optional[Image.Image], Optional[Image.Image]]] = []
    for _ in range(n_triplets):
        w = random.choice(wholes)
        f = random.choice(faces) if faces else None
        e = random.choice(eyes_list) if eyes_list else None
        triplets.append((w, f, e))

    # Step 3: Embed all triplets
    zs: List[torch.Tensor] = []
    for w, f, e in triplets:
        try:
            wt = _pil_to_tensor(w, lm.T_w)
            ft = _pil_to_tensor(f, lm.T_f) if f is not None else None
            et = _pil_to_tensor(e, lm.T_e) if e is not None else None
            z = embed_triview(lm, whole=wt, face=ft, eyes=et)
            zs.append(z)
        except Exception:
            continue

    if not zs:
        return "❌ Could not embed any triplets."

    Z = torch.stack(zs, dim=0)
    Z = torch.nn.functional.normalize(Z, dim=1)

    # Step 4: Run K-means to get K prototype centers
    actual_k = min(k_prototypes, len(zs))
    if actual_k < k_prototypes:
        # Not enough embeddings for requested K
        pass

    centers = _kmeans_cosine(Z, actual_k, iters=20, seed=42)

    # Step 5: Add all K prototypes to the DB
    added_ids = []
    for center in centers:
        lid = db.add_center(label_name, center)
        added_ids.append(lid)

    return (
        f"✅ Added {len(added_ids)} temporary prototype(s) for `{label_name}` "
        f"(from {len(wholes)} images, {len(triplets)} triplets, K-means K={actual_k}). "
        f"DB now N={db.centers.shape[0]}. "
        f"⚠️ Session-only — lost on Space restart."
    )


def build_ui() -> gr.Blocks:
    ckpts = _list_ckpt_files(CKPT_DIR)
    protos = _list_proto_files(CKPT_DIR)

    with gr.Blocks(title="ArtistEmbeddingClassifier") as demo:
        gr.Markdown("### ArtistEmbeddingClassifier — Gradio UI\nLoads checkpoint + prototype DB from `./checkpoints_style/`.")

        with gr.Row():
            ckpt_dd = gr.Dropdown(choices=ckpts, value=_guess_default_ckpt(ckpts), label="Checkpoint (.pt)")
            proto_dd = gr.Dropdown(choices=protos, value=_guess_default_proto(protos), label="Prototype DB (.pt)")
            device_dd = gr.Dropdown(choices=["auto", "cpu"], value="auto", label="Device")
            load_btn = gr.Button("Load", variant="primary")

        status = gr.Markdown("")
        load_btn.click(load_all, inputs=[ckpt_dd, proto_dd, device_dd], outputs=[status])

        with gr.Tab("Classify"):
            with gr.Row():
                with gr.Column(scale=1):
                    whole = gr.Image(label="Upload image", type="pil")
                    with gr.Row():
                        topk = gr.Slider(1, 20, value=5, step=1, label="Top-K")
                        run_btn = gr.Button("Run", variant="primary")
                    out_status = gr.Markdown("")

                with gr.Column(scale=1):
                    view_weights_display = gr.HTML(label="View Contribution")

            # Classification results
            gr.Markdown("### 🎯 Classification Results")
            table = gr.Dataframe(headers=["Artist", "Similarity"], datatype=["str", "number"], interactive=False)

            # XGrad-CAM heatmaps
            gr.Markdown("### 🔥 XGrad-CAM Attention Maps")
            gr.Markdown("*Where the model focused in each view:*")
            with gr.Row():
                gcam_whole = gr.Image(label="Whole Image", type="pil")
                gcam_face = gr.Image(label="Face", type="pil")
                gcam_eye = gr.Image(label="Eye", type="pil")

            # Extracted views
            gr.Markdown("### 👁️ Auto-Extracted Views")
            with gr.Row():
                face_prev = gr.Image(label="Detected Face", type="pil")
                eye_prev = gr.Image(label="Detected Eye", type="pil")

            run_btn.click(
                classify_and_analyze,
                inputs=[whole, topk],
                outputs=[out_status, table, view_weights_display, gcam_whole, gcam_face, gcam_eye, face_prev, eye_prev],
            )

        with gr.Tab("Add prototype (temporary)"):
            gr.Markdown(
                "### ⚠️ Temporary Prototypes Only\n"
                "Add prototypes using random triplet combinations and K-means clustering (same as eval process).\n"
                "1. Upload multiple whole images\n"
                "2. Face and eye are auto-extracted from each\n"
                "3. Random triplets (whole + face + eye) are created\n"
                "4. K-means clustering creates K prototype centers\n\n"
                "**These prototypes are session-only** — lost when the Space restarts."
            )
            label = gr.Textbox(label="Label name (artist)", placeholder="e.g. new_artist")
            imgs = gr.Gallery(label="Whole images (1+)", columns=4, rows=2, height=240, allow_preview=True)
            uploader = gr.Files(label="Upload image files (whole)", file_types=["image"], file_count="multiple")
            with gr.Row():
                k_proto = gr.Slider(1, 8, value=4, step=1, label="K (prototypes to create)")
                n_trips = gr.Slider(4, 64, value=16, step=4, label="N (random triplets to sample)")
            add_btn = gr.Button("Add temporary prototypes", variant="primary")
            add_status = gr.Markdown("")

            def _files_to_gallery(files):
                if not files:
                    return []
                out = []
                for f in files:
                    try:
                        im = Image.open(f.name).convert("RGB")
                        out.append(im)
                    except Exception:
                        continue
                return out

            uploader.change(_files_to_gallery, inputs=[uploader], outputs=[imgs])
            add_btn.click(add_prototype, inputs=[label, imgs, k_proto, n_trips], outputs=[add_status])

        with gr.Tab("Artists (in DB)"):
            gr.Markdown(
                "### Artists in Prototype DB\n"
                "Shows which artist labels exist in the currently loaded prototype database "
                "(including any temporary prototypes added in this session)."
            )
            refresh_artists = gr.Button("Refresh", variant="secondary")
            artists_status = gr.Markdown("")
            artists_table = gr.Dataframe(headers=["Artist", "#Prototypes"], datatype=["str", "number"], interactive=False)
            refresh_artists.click(list_artists_in_db, inputs=[], outputs=[artists_status, artists_table])

    return demo


if __name__ == "__main__":
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    demo = build_ui()

    ap = argparse.ArgumentParser(description="ArtistEmbeddingClassifier Gradio UI")
    # Hugging Face Spaces runs behind a proxy and expects binding to 0.0.0.0:$PORT.
    default_host = os.getenv("GRADIO_SERVER_NAME")
    if not default_host:
        default_host = "0.0.0.0" if os.getenv("SPACE_ID") or os.getenv("HF_SPACE") else "127.0.0.1"
    default_port = int(os.getenv("PORT") or os.getenv("GRADIO_SERVER_PORT") or "7860")

    ap.add_argument("--host", type=str, default=default_host)
    ap.add_argument("--port", type=int, default=default_port)
    ap.add_argument("--share", action="store_true", help="Create a public share link")
    args = ap.parse_args()

    # Re-apply patch right before launching (in case import order changed).
    _patch_fastapi_starlette_middleware_unpack()

    try:
        _launch_compat(demo, server_name=args.host, server_port=args.port, show_api=False, share=args.share, ssr_mode=False)
    except ValueError as e:
        # Some environments block localhost checks; fall back to share link.
        msg = str(e)
        if "localhost is not accessible" in msg and not args.share:
            _launch_compat(demo, server_name=args.host, server_port=args.port, show_api=False, share=True, ssr_mode=False)
        else:
            raise


