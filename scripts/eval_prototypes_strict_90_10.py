#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prototype evaluation (strict 90/10 split per view per artist, using merged train+val pools).

This script mirrors the "strict 90/10 full coverage" prototype-eval logic.
"""

from __future__ import annotations

import argparse
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import train_style_ddp as ts


TripletWithID = Tuple[str, str, str, int]


@dataclass
class Args:
    ckpt: str
    out: str
    k_per_artist: int
    build_ratio: float
    batch_size: int
    num_workers: int
    seed: int
    chunk_size: int


def parse_args() -> Args:
    p = argparse.ArgumentParser(description="Eval prototypes (strict 90/10 split per view)")
    p.add_argument("--ckpt", type=str, default="./checkpoints_style/stage3_epoch24.pt")
    p.add_argument("--out", type=str, default="./checkpoints_style/per_artist_prototypes_90_10_full.pt")
    p.add_argument("--k-per-artist", type=int, default=4)
    p.add_argument("--build-ratio", type=float, default=0.9)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0, help="0 is safest on Windows/spawn.")
    p.add_argument("--seed", type=int, default=ts.cfg.seed)
    p.add_argument("--chunk-size", type=int, default=5000)
    a = p.parse_args()
    return Args(
        ckpt=a.ckpt,
        out=a.out,
        k_per_artist=a.k_per_artist,
        build_ratio=a.build_ratio,
        batch_size=a.batch_size,
        num_workers=a.num_workers,
        seed=a.seed,
        chunk_size=a.chunk_size,
    )


def kmeans_cosine(Z_cpu: torch.Tensor, K: int, *, iters: int = 20, seed: int = 1337, device: torch.device) -> torch.Tensor:
    Z = torch.nn.functional.normalize(Z_cpu.to(device), dim=1)
    N, D = Z.shape
    if N <= K:
        return Z.detach().cpu()
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    init_idx = torch.randperm(N, generator=g, device=device)[:K]
    C = Z[init_idx].clone()
    assign = torch.full((N,), -1, device=device, dtype=torch.long)
    for _ in range(iters):
        sim = Z @ C.t()
        new_assign = sim.argmax(dim=1)
        if (new_assign == assign).all():
            assign = new_assign
            break
        assign = new_assign
        C = torch.zeros(K, D, device=device, dtype=Z.dtype)
        C.index_add_(0, assign, Z)
        counts_raw = torch.bincount(assign, minlength=K)
        empty = (counts_raw == 0)
        counts = counts_raw.clamp_min(1).unsqueeze(1).to(Z.dtype)
        C = C / counts
        if empty.any():
            ridx = torch.randperm(N, generator=g, device=device)[: int(empty.sum().item())]
            C[empty] = Z[ridx]
        C = torch.nn.functional.normalize(C, dim=1)
    return C.detach().cpu()


class TripletDatasetWithID(Dataset):
    def __init__(self, triplets: Sequence[TripletWithID], T_w, T_f, T_e):
        self.triplets = list(triplets)
        self.T_w = T_w
        self.T_f = T_f
        self.T_e = T_e

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int):
        pw, pf, pe, aid = self.triplets[idx]
        try:
            im_w = Image.open(pw).convert("RGB")
            im_f = Image.open(pf).convert("RGB")
            im_e = Image.open(pe).convert("RGB")
        except (UnidentifiedImageError, OSError):
            return None
        return dict(whole=self.T_w(im_w), face=self.T_f(im_f), eyes=self.T_e(im_e), aid=int(aid))


def collate_triplets_with_id(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None, None, None
    Ws = torch.stack([b["whole"] for b in batch], dim=0)
    Fs = torch.stack([b["face"] for b in batch], dim=0)
    Es = torch.stack([b["eyes"] for b in batch], dim=0)
    A = torch.tensor([b["aid"] for b in batch], dtype=torch.long)
    return Ws, Fs, Es, A


def extract_embeddings_with_id(
    *,
    model: ts.TriViewStyleNet,
    triplets: Sequence[TripletWithID],
    T_w,
    T_f,
    T_e,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if not triplets:
        return None, None
    ds = TripletDatasetWithID(triplets, T_w, T_f, T_e)

    def _run_loader(nw: int, pin: bool):
        dl = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=nw,
            pin_memory=pin,
            collate_fn=collate_triplets_with_id,
        )
        feats: List[torch.Tensor] = []
        aids: List[torch.Tensor] = []
        model.eval()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=ts.amp_dtype, enabled=(device.type == "cuda")):
            for Wb, Fb, Eb, Ab in dl:
                if Wb is None:
                    continue
                Wb = Wb.to(device, non_blocking=True)
                Fb = Fb.to(device, non_blocking=True)
                Eb = Eb.to(device, non_blocking=True)
                views = {"whole": Wb, "face": Fb, "eyes": Eb}
                masks = {k: torch.ones(Wb.size(0), dtype=torch.bool, device=device) for k in views}
                z_fused, _, _ = model(views, masks)
                feats.append(z_fused.detach().cpu())
                aids.append(Ab.detach().cpu())
        return feats, aids

    try:
        feats, aids = _run_loader(num_workers, pin=True)
    except Exception:
        feats, aids = _run_loader(0, pin=False)

    if not feats:
        return None, None
    Z = torch.nn.functional.normalize(torch.cat(feats, dim=0), dim=1)
    A = torch.cat(aids, dim=0).long()
    return Z, A


def merge_dicts(d1: Dict[int, List], d2: Dict[int, List]) -> Dict[int, List]:
    out = defaultdict(list)
    for k, v in d1.items():
        out[k].extend(list(v))
    for k, v in d2.items():
        out[k].extend(list(v))
    return dict(out)


def main() -> None:
    a = parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    ck = torch.load(a.ckpt, map_location="cpu")
    meta = ck.get("meta", {})
    stage_i = int(meta.get("stage", 1))
    stage = ts.cfg.stages[stage_i - 1]
    print(f"loaded ckpt={a.ckpt} (stage={stage_i})")

    # use deterministic transforms for prototype building/eval
    T_w_val, T_f_val, T_e_val = ts.make_val_transforms(stage["sz_whole"], stage["sz_face"], stage["sz_eyes"])

    train_ds = ts.TriViewDataset(ts.cfg.data_root, ts.cfg.folders, split="train", T_whole=T_w_val, T_face=T_f_val, T_eyes=T_e_val)
    val_ds = ts.TriViewDataset(ts.cfg.data_root, ts.cfg.folders, split="val", T_whole=T_w_val, T_face=T_f_val, T_eyes=T_e_val)

    # merge pools (train+val)
    wholes_all = merge_dicts(train_ds.whole_paths_by_artist, val_ds.whole_paths_by_artist)
    faces_all = merge_dicts(train_ds.face_paths_by_artist, val_ds.face_paths_by_artist)
    eyes_all = merge_dicts(train_ds.eyes_paths_by_artist, val_ds.eyes_paths_by_artist)

    build_data = {}
    eval_data = {}
    for aid in wholes_all.keys():
        W_list = list({str(p) for p in wholes_all.get(aid, [])})
        F_list = list({str(p) for p in faces_all.get(aid, [])})
        E_list = list({str(p) for p in eyes_all.get(aid, [])})
        random.shuffle(W_list)
        random.shuffle(F_list)
        random.shuffle(E_list)
        if len(W_list) < 2 or len(F_list) < 2 or len(E_list) < 2:
            continue
        mw = max(1, int(len(W_list) * a.build_ratio))
        mf = max(1, int(len(F_list) * a.build_ratio))
        me = max(1, int(len(E_list) * a.build_ratio))
        if mw == len(W_list):
            mw -= 1
        if mf == len(F_list):
            mf -= 1
        if me == len(E_list):
            me -= 1
        W_b, W_e = W_list[:mw], W_list[mw:]
        F_b, F_e = F_list[:mf], F_list[mf:]
        E_b, E_e = E_list[:me], E_list[me:]
        if not (W_b and W_e and F_b and F_e and E_b and E_e):
            continue
        build_data[aid] = {"W": W_b, "F": F_b, "E": E_b}
        eval_data[aid] = {"W": W_e, "F": F_e, "E": E_e}

    print("valid artists:", len(build_data))

    model = ts.TriViewStyleNet(out_dim=ts.cfg.embed_dim, mix_p=ts.cfg.mixstyle_p, share_backbone=True).to(device)
    model = model.to(memory_format=torch.channels_last)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()

    # build triplets: use all build wholes once, random face/eyes from build pools
    build_triplets: List[TripletWithID] = []
    for aid, d in build_data.items():
        for pw in d["W"]:
            pf = random.choice(d["F"])
            pe = random.choice(d["E"])
            build_triplets.append((pw, pf, pe, int(aid)))
    print("build triplets:", len(build_triplets))

    Z_build, A_build = extract_embeddings_with_id(
        model=model,
        triplets=build_triplets,
        T_w=T_w_val,
        T_f=T_f_val,
        T_e=T_e_val,
        batch_size=a.batch_size,
        num_workers=a.num_workers,
        device=device,
    )
    if Z_build is None or A_build is None:
        raise RuntimeError("No build embeddings extracted.")

    # prototypes per artist
    aid_to_idx = defaultdict(list)
    for i, aid in enumerate(A_build.tolist()):
        aid_to_idx[int(aid)].append(i)

    proto_centers_list: List[torch.Tensor] = []
    proto_labels_list: List[torch.Tensor] = []
    for aid, idxs in aid_to_idx.items():
        Zi = Z_build[torch.tensor(idxs, dtype=torch.long)]
        if Zi.shape[0] <= a.k_per_artist:
            proto_centers_list.append(Zi)
            proto_labels_list.append(torch.full((Zi.shape[0],), aid, dtype=torch.long))
        else:
            centers = kmeans_cosine(Zi, K=a.k_per_artist, iters=20, seed=a.seed, device=device)
            proto_centers_list.append(centers)
            proto_labels_list.append(torch.full((a.k_per_artist,), aid, dtype=torch.long))

    proto_centers = torch.cat(proto_centers_list, dim=0)
    proto_labels = torch.cat(proto_labels_list, dim=0)
    print("total prototypes:", proto_centers.shape[0])

    # eval triplets: use all eval wholes once, random face/eyes from eval pools
    eval_triplets: List[TripletWithID] = []
    valid_proto_artists = set(proto_labels.unique().tolist())
    for aid, d in eval_data.items():
        if int(aid) not in valid_proto_artists:
            continue
        for pw in d["W"]:
            pf = random.choice(d["F"])
            pe = random.choice(d["E"])
            eval_triplets.append((pw, pf, pe, int(aid)))
    print("eval triplets:", len(eval_triplets))

    Z_eval, Y_eval = extract_embeddings_with_id(
        model=model,
        triplets=eval_triplets,
        T_w=T_w_val,
        T_f=T_f_val,
        T_e=T_e_val,
        batch_size=a.batch_size,
        num_workers=a.num_workers,
        device=device,
    )
    if Z_eval is None or Y_eval is None:
        raise RuntimeError("No eval embeddings extracted.")

    # nearest-prototype classification (cosine)
    with torch.no_grad():
        C = torch.nn.functional.normalize(proto_centers.to(device), dim=1)
        Z = torch.nn.functional.normalize(Z_eval.to(device), dim=1)
        correct = 0
        total = Z.shape[0]
        for i in range(0, total, a.chunk_size):
            zc = Z[i : i + a.chunk_size]
            yc = Y_eval[i : i + a.chunk_size].to(device)
            sim = zc @ C.t()
            pred_idx = sim.argmax(dim=1)
            pred_labels = proto_labels.to(device)[pred_idx]
            correct += (pred_labels == yc).sum().item()
        acc = correct / max(1, total)
    print(f"prototype accuracy (strict 90/10): {acc:.4f}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save(
        dict(
            centers=proto_centers,
            labels=proto_labels,
            k_per_artist=a.k_per_artist,
            ckpt=a.ckpt,
            split_method="90_10_strict_per_view_per_artist",
            build_ratio=a.build_ratio,
            acc=acc,
        ),
        a.out,
    )
    print("saved:", a.out)


if __name__ == "__main__":
    main()


