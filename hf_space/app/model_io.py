from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch

# ZeroGPU on HF Spaces: CUDA must not be initialized in the main process
_ON_SPACES = bool(os.getenv("SPACE_ID") or os.getenv("HF_SPACE"))


@dataclass(frozen=True)
class LoadedModel:
    model: torch.nn.Module
    device: torch.device
    stage_i: int
    embed_dim: int
    T_w: object
    T_f: object
    T_e: object


def _pick_device(device: str) -> torch.device:
    if device.strip().lower() == "cpu":
        return torch.device("cpu")
    if _ON_SPACES:
        # ZeroGPU: can't init CUDA in main process
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_style_model(
    ckpt_path: str | Path,
    *,
    device: str = "auto",
) -> LoadedModel:
    """
    Loads `train_style_ddp.TriViewStyleNet` from a checkpoint saved by `train_style_ddp.py`.
    Returns the model and deterministic val transforms based on the checkpoint stage.
    """
    import train_style_ddp as ts

    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(str(ckpt_path))

    # On Spaces, always use CPU (ZeroGPU forbids CUDA in main process)
    if _ON_SPACES:
        dev = torch.device("cpu")
    elif device == "auto":
        dev = _pick_device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = _pick_device(device)

    ck = torch.load(str(ckpt_path), map_location="cpu")
    meta = ck.get("meta", {}) if isinstance(ck, dict) else {}
    stage_i = int(meta.get("stage", 1))
    stage_i = max(1, min(stage_i, len(ts.cfg.stages)))
    stage = ts.cfg.stages[stage_i - 1]

    T_w, T_f, T_e = ts.make_val_transforms(stage["sz_whole"], stage["sz_face"], stage["sz_eyes"])

    model = ts.TriViewStyleNet(out_dim=ts.cfg.embed_dim, mix_p=ts.cfg.mixstyle_p, share_backbone=True)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model.load_state_dict(state, strict=False)
    model.eval()
    model = model.to(dev)
    try:
        model = model.to(memory_format=torch.channels_last)
    except Exception:
        pass

    return LoadedModel(
        model=model,
        device=dev,
        stage_i=stage_i,
        embed_dim=int(ts.cfg.embed_dim),
        T_w=T_w,
        T_f=T_f,
        T_e=T_e,
    )


def embed_triview(
    lm: LoadedModel,
    *,
    whole: Optional[torch.Tensor],
    face: Optional[torch.Tensor],
    eyes: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Computes a single fused embedding for a triview sample.
    Each view tensor must be CHW (already normalized) and will be batched.
    Missing views can be None.
    """
    if whole is None and face is None and eyes is None:
        raise ValueError("At least one of whole/face/eyes must be provided.")

    views = {}
    masks = {}
    for k, v in (("whole", whole), ("face", face), ("eyes", eyes)):
        if v is None:
            views[k] = None
            masks[k] = torch.zeros(1, dtype=torch.bool, device=lm.device)
        else:
            vb = v.unsqueeze(0).to(lm.device)
            views[k] = vb
            masks[k] = torch.ones(1, dtype=torch.bool, device=lm.device)

    # Use lazy dtype detection to avoid CUDA init at import time (ZeroGPU compatibility)
    import train_style_ddp as _ts
    _dtype = _ts._get_amp_dtype() if hasattr(_ts, "_get_amp_dtype") else torch.float16
    # On CPU or Spaces, skip autocast entirely to avoid touching CUDA
    use_amp = (lm.device.type == "cuda") and not _ON_SPACES
    if use_amp:
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=_dtype, enabled=True):
            z, _, _ = lm.model(views, masks)
    else:
        with torch.no_grad():
            z, _, _ = lm.model(views, masks)
    z = torch.nn.functional.normalize(z.float(), dim=1)
    return z.squeeze(0).detach().cpu()


