from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

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
    files = _list_pt_files(folder)
    # heuristics: prototype db files usually contain "proto" in filename
    protos = [f for f in files if "proto" in Path(f).name.lower()]
    return protos if protos else files


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

    # initialize view extractor (whole -> face/eyes) with defaults
    try:
        cfg = ExtractorCfg(
            yolo_dir=ROOT / "yolov5_anime",
            weights=ROOT / "yolov5x_anime.pt",
            cascade=ROOT / "anime-eyes-cascade.xml",
            yolo_device=("0" if torch.cuda.is_available() else "cpu"),
        )
        APP_STATE.extractor = AnimeFaceEyeExtractor(cfg)
    except Exception:
        APP_STATE.extractor = None

    return f"✅ Loaded checkpoint `{Path(ckpt_path).name}` (stage={lm.stage_i}) and proto DB `{Path(proto_path).name}` (N={db.centers.shape[0]})"


def classify(
    whole_img,
    topk: int,
):
    """
    Classify using auto-extracted face/eyes from whole image.
    Returns: status, table_rows, face_preview, eyes_preview
    """
    if APP_STATE.lm is None or APP_STATE.db is None:
        return "❌ Click **Load** first.", [], None, None

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
        return "❌ Provide a whole image.", [], None, None

    try:
        face_pil = None
        eyes_pil = None
        if ex is not None:
            rgb = np.array(w.convert("RGB"))
            face_rgb, eyes_rgb = ex.extract(rgb)
            if face_rgb is not None:
                face_pil = Image.fromarray(face_rgb)
            if eyes_rgb is not None:
                eyes_pil = Image.fromarray(eyes_rgb)

        wt = _pil_to_tensor(w, lm.T_w)
        ft = _pil_to_tensor(face_pil, lm.T_f) if face_pil is not None else None
        et = _pil_to_tensor(eyes_pil, lm.T_e) if eyes_pil is not None else None
        z = embed_triview(lm, whole=wt, face=ft, eyes=et)
        preds = topk_predictions_unique_labels(db, z, topk=int(topk))
    except Exception as ex:
        return f"❌ Inference failed: {ex}", [], None, None

    rows = [[name, float(score)] for (name, score) in preds]
    return "✅ OK", rows, (face_pil if "face_pil" in locals() else None), (eyes_pil if "eyes_pil" in locals() else None)


def add_prototype(
    label_name: str,
    images: List,
    save_back: bool,
) -> str:
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

    zs: List[torch.Tensor] = []
    for x in images:
        try:
            im = x if isinstance(x, Image.Image) else Image.fromarray(x)
            face_pil = None
            eyes_pil = None
            if ex is not None:
                rgb = np.array(im.convert("RGB"))
                face_rgb, eyes_rgb = ex.extract(rgb)
                if face_rgb is not None:
                    face_pil = Image.fromarray(face_rgb)
                if eyes_rgb is not None:
                    eyes_pil = Image.fromarray(eyes_rgb)

            wt = _pil_to_tensor(im, lm.T_w)
            ft = _pil_to_tensor(face_pil, lm.T_f) if face_pil is not None else None
            et = _pil_to_tensor(eyes_pil, lm.T_e) if eyes_pil is not None else None
            z = embed_triview(lm, whole=wt, face=ft, eyes=et)
            zs.append(z)
        except Exception:
            continue

    if not zs:
        return "❌ Could not embed any uploaded images."

    center = torch.stack(zs, dim=0).mean(dim=0)
    lid = db.add_center(label_name, center)

    msg = f"✅ Added prototype for `{label_name}` (label_id={lid}). DB now N={db.centers.shape[0]}."

    if save_back:
        out_path = db.save(APP_STATE.proto_path)
        msg += f" Saved to `{out_path}`."
    return msg


def save_db_as(path_text: str) -> str:
    if APP_STATE.db is None:
        return "❌ Nothing loaded."
    out = (path_text or "").strip()
    if not out:
        return "❌ Provide an output path."
    out_path = Path(out)
    if not out_path.is_absolute():
        out_path = (CKPT_DIR / out_path).resolve()
    APP_STATE.db.save(out_path)
    APP_STATE.proto_path = str(out_path)
    return f"✅ Saved prototype DB to `{out_path}`"


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
                whole = gr.Image(label="Whole image (required)", type="pil")
                face_prev = gr.Image(label="Extracted face (auto)", type="pil")
                eyes_prev = gr.Image(label="Extracted eyes (auto)", type="pil")
            with gr.Row():
                topk = gr.Slider(1, 20, value=5, step=1, label="Top-K")
                run_btn = gr.Button("Run", variant="primary")

            out_status = gr.Markdown("")
            table = gr.Dataframe(headers=["label", "cosine_sim"], datatype=["str", "number"], interactive=False)
            run_btn.click(classify, inputs=[whole, topk], outputs=[out_status, table, face_prev, eyes_prev])

        with gr.Tab("Add prototype"):
            gr.Markdown(
                "Add a new prototype to the loaded prototype DB by averaging embeddings of uploaded whole images.\n"
                "Multiple prototypes per label are allowed."
            )
            label = gr.Textbox(label="Label name (artist)", placeholder="e.g. new_artist")
            imgs = gr.Gallery(label="Whole images (1+)", columns=4, rows=2, height=240, allow_preview=True)
            uploader = gr.Files(label="Upload image files (whole)", file_types=["image"], file_count="multiple")
            save_back = gr.Checkbox(value=True, label="Save back to selected prototype DB file after adding")
            add_btn = gr.Button("Add prototype", variant="primary")
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
            add_btn.click(add_prototype, inputs=[label, imgs, save_back], outputs=[add_status])

            gr.Markdown("Save DB as (optional):")
            save_path = gr.Textbox(label="Output path (relative paths go under ./checkpoints_style/)", placeholder="prototypes_custom.pt")
            save_btn = gr.Button("Save As")
            save_btn.click(save_db_as, inputs=[save_path], outputs=[add_status])

    return demo


if __name__ == "__main__":
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    demo = build_ui()

    ap = argparse.ArgumentParser(description="ArtistEmbeddingClassifier Gradio UI")
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="Create a public share link")
    args = ap.parse_args()

    # Re-apply patch right before launching (in case import order changed).
    _patch_fastapi_starlette_middleware_unpack()

    try:
        demo.launch(server_name=args.host, server_port=args.port, show_api=False, share=args.share)
    except ValueError as e:
        # Some environments block localhost checks; fall back to share link.
        msg = str(e)
        if "localhost is not accessible" in msg and not args.share:
            demo.launch(server_name=args.host, server_port=args.port, show_api=False, share=True)
        else:
            raise


