from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr
import numpy as np
import torch
from PIL import Image

from app.model_io import LoadedModel, embed_triview, load_style_model
from app.proto_db import PrototypeDB, load_prototype_db, topk_predictions


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
    # try to prefer a file with "prototype" in name
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
    return f"✅ Loaded checkpoint `{Path(ckpt_path).name}` (stage={lm.stage_i}) and proto DB `{Path(proto_path).name}` (N={db.centers.shape[0]})"


def classify(
    whole_img,
    face_img,
    eyes_img,
    topk: int,
) -> Tuple[str, List[List[object]]]:
    if APP_STATE.lm is None or APP_STATE.db is None:
        return "❌ Click **Load** first.", []

    lm = APP_STATE.lm
    db = APP_STATE.db

    def _to_pil(x):
        if x is None:
            return None
        if isinstance(x, Image.Image):
            return x
        return Image.fromarray(x)

    w = _to_pil(whole_img)
    f = _to_pil(face_img)
    e = _to_pil(eyes_img)

    try:
        wt = _pil_to_tensor(w, lm.T_w) if w is not None else None
        ft = _pil_to_tensor(f, lm.T_f) if f is not None else None
        et = _pil_to_tensor(e, lm.T_e) if e is not None else None
        z = embed_triview(lm, whole=wt, face=ft, eyes=et)
        preds = topk_predictions(db, z, topk=int(topk))
    except Exception as ex:
        return f"❌ Inference failed: {ex}", []

    rows = [[name, float(score)] for (name, score) in preds]
    return "✅ OK", rows


def add_prototype(
    label_name: str,
    images: List,
    save_back: bool,
) -> str:
    if APP_STATE.lm is None or APP_STATE.db is None:
        return "❌ Click **Load** first."
    lm = APP_STATE.lm
    db = APP_STATE.db

    label_name = (label_name or "").strip()
    if not label_name:
        return "❌ Label name is required."
    if not images:
        return "❌ Upload at least 1 image."

    zs: List[torch.Tensor] = []
    for x in images:
        try:
            im = x if isinstance(x, Image.Image) else Image.fromarray(x)
            wt = _pil_to_tensor(im, lm.T_w)
            z = embed_triview(lm, whole=wt, face=None, eyes=None)
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
                face = gr.Image(label="Face (optional)", type="pil")
                eyes = gr.Image(label="Eyes (optional)", type="pil")
            with gr.Row():
                topk = gr.Slider(1, 20, value=5, step=1, label="Top-K")
                run_btn = gr.Button("Run", variant="primary")

            out_status = gr.Markdown("")
            table = gr.Dataframe(headers=["label", "cosine_sim"], datatype=["str", "number"], interactive=False)
            run_btn.click(classify, inputs=[whole, face, eyes, topk], outputs=[out_status, table])

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
    demo.launch(server_name="127.0.0.1", server_port=7860, show_api=False)


