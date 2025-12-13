#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, math, random, glob, time, subprocess, sys, zlib, gc, warnings, atexit
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

import numpy as np
from PIL import Image, ImageFile
from PIL.Image import DecompressionBombWarning

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist

from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms

# tqdm (auto-install if missing)
try:
    from tqdm.auto import tqdm
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tqdm"])
    from tqdm.auto import tqdm

# ------------------------- Config -------------------------
@dataclass
class Cfg:
    data_root: str = "./"
    folders: dict = None
    stages: list = None
    P: int = 16
    K: int = 2
    embed_dim: int = 256
    workers: int = 8
    weight_decay: float = 0.01
    alpha_proxy: float = 32.0
    margin_proxy: float = 0.2
    supcon_tau: float = 0.07
    mv_tau: float = 0.10
    mixstyle_p: float = 0.10
    out_dir: str = "./checkpoints_style"
    seed: int = 1337
    max_steps_per_epoch: Optional[int] = None  # None이면 데이터 길이에 따라 자동
    print_every: int = 50
    use_compile: bool = False

cfg = Cfg(
    folders=dict(whole="dataset", face="dataset_face", eyes="dataset_eyes"),
    stages=[
        dict(sz_whole=224, sz_face=192, sz_eyes=128, epochs=12, lr=3e-4,   P=64, K=2),
        dict(sz_whole=384, sz_face=320, sz_eyes=192, epochs=12, lr=1.5e-4, P=24, K=2),
        dict(sz_whole=512, sz_face=384, sz_eyes=224, epochs=24, lr=8e-5,   P=12, K=2),
    ],
)

# ------------------------- Device & determinism -------------------------
def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_all(cfg.seed)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    amp_dtype = torch.bfloat16
else:
    amp_dtype = torch.float16

# --- PIL safety/verbosity tweaks ---
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 300_000_000
warnings.filterwarnings("ignore", category=DecompressionBombWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.TiffImagePlugin")

# ------------------------- Robust multiprocessing for DataLoader -------------------------
def _init_mp_ctx():
    method = mp.get_start_method(allow_none=True)
    if method is None:
        preferred = 'fork' if sys.platform.startswith('linux') else 'spawn'
        try:
            mp.set_start_method(preferred, force=True)
        except Exception:
            pass
        method = mp.get_start_method(allow_none=True) or preferred
    print(f"[mp] using '{method}'.")
    return mp.get_context(method)

MP_CTX = _init_mp_ctx()

_DL_TRACK = []
def _track_dl(dl):
    _DL_TRACK.append(dl); return dl

def _close_dl(dl):
    try:
        it = getattr(dl, "_iterator", None)
        if it is not None:
            it._shutdown_workers()
            dl._iterator = None
    except Exception:
        pass

@atexit.register
def _cleanup_all_dls():
    for dl in list(_DL_TRACK):
        _close_dl(dl)
    _DL_TRACK.clear()

def _should_fallback_workers(err: Exception) -> bool:
    s = str(err)
    return ("Can't get attribute" in s or
            "PicklingError" in s or
            ("AttributeError" in s and "__main__" in s))

# ------------------------- Helpers -------------------------
def stable_int(s: str) -> int:
    return zlib.adler32(s.encode("utf-8")) & 0xffffffff

def l2n(x, eps=1e-8):
    return F.normalize(x, dim=-1, eps=eps)

# ------------------------- Dataset -------------------------
class TriViewDataset(Dataset):
    """
    - whole / face / eyes 각각에 대해 9:1로 train/val split (경로 해시 기반).
    - __getitem__에서는 해당 작가의 view pool에서 랜덤으로 뽑아서 tri-view 구성.
    - 파일명 매칭 전혀 사용 X, 작가(label)만 동일하면 아무 이미지나 조합.
    - index는 whole 기반으로 만들고, label/gid/path 는 whole 기준.
    """

    def __init__(self, root, folders, split="train",
                 T_whole=None, T_face=None, T_eyes=None):
        assert split in ("train", "val")
        self.split = split
        self.root = Path(root)
        self.dirs = {k: self.root / v for k, v in folders.items()}
        self.T = dict(whole=T_whole, face=T_face, eyes=T_eyes)

        # artist 목록
        whole_root = self.dirs["whole"]
        artists = sorted([d.name for d in whole_root.iterdir() if d.is_dir()])
        self.artist2id = {a: i for i, a in enumerate(artists)}
        self.id2artist = {v: k for k, v in self.artist2id.items()}
        self.num_classes = len(self.artist2id)

        # artist별 view pool (split 별)
        self.whole_paths_by_artist: Dict[int, List[Path]] = {aid: [] for aid in self.id2artist.keys()}
        self.face_paths_by_artist:  Dict[int, List[Path]] = {aid: [] for aid in self.id2artist.keys()}
        self.eyes_paths_by_artist:  Dict[int, List[Path]] = {aid: [] for aid in self.id2artist.keys()}

        def view_split(paths: List[Path], split: str) -> List[Path]:
            train_list, val_list = [], []
            for p in paths:
                h = stable_int(str(p)) % 10
                if split == "train":
                    if h < 9:  # 0~8 => train
                        train_list.append(p)
                else:
                    if h >= 9:  # 9 => val
                        val_list.append(p)
            return train_list if split == "train" else val_list

        # whole / face / eyes 각각에 대해 artist별 split
        for artist_name, aid in self.artist2id.items():
            # whole
            w_dir = self.dirs["whole"] / artist_name
            if w_dir.is_dir():
                w_all = sorted([p for p in w_dir.iterdir() if p.is_file()])
            else:
                w_all = []
            self.whole_paths_by_artist[aid] = view_split(w_all, split)

            # face
            f_dir = self.dirs["face"] / artist_name
            if f_dir.is_dir():
                f_all = sorted([p for p in f_dir.iterdir() if p.is_file()])
            else:
                f_all = []
            self.face_paths_by_artist[aid] = view_split(f_all, split)

            # eyes
            e_dir = self.dirs["eyes"] / artist_name
            if e_dir.is_dir():
                e_all = sorted([p for p in e_dir.iterdir() if p.is_file()])
            else:
                e_all = []
            self.eyes_paths_by_artist[aid] = view_split(e_all, split)

        # index: whole 기반 anchor
        self.index = []
        for aid, w_list in self.whole_paths_by_artist.items():
            for wp in w_list:
                rec = {
                    "label": aid,
                    "whole": str(wp),
                    "gid": stable_int(str(wp)),
                    "path": str(wp),
                }
                self.index.append(rec)

    def __len__(self):
        return len(self.index)

    def _load_one(self, path: Optional[Path], T):
        if path is None:
            return None
        try:
            im = Image.open(path).convert("RGB")
        except Exception:
            return None
        if T is not None:
            return T(im)
        else:
            return transforms.ToTensor()(im)

    def __getitem__(self, i):
        rec = self.index[i]
        aid = rec["label"]

        W_pool = self.whole_paths_by_artist.get(aid, [])
        F_pool = self.face_paths_by_artist.get(aid, [])
        E_pool = self.eyes_paths_by_artist.get(aid, [])

        pw = random.choice(W_pool) if W_pool else None
        pf = random.choice(F_pool) if F_pool else None
        pe = random.choice(E_pool) if E_pool else None

        xw = self._load_one(pw, self.T["whole"]) if pw is not None else None
        xf = self._load_one(pf, self.T["face"])  if pf is not None else None
        xe = self._load_one(pe, self.T["eyes"])  if pe is not None else None

        gid = torch.tensor([rec["gid"]], dtype=torch.long)
        return dict(
            whole=xw,
            face=xf,
            eyes=xe,
            label=torch.tensor(aid, dtype=torch.long),
            gid=gid,
            path=rec["path"],
        )

# ------------------------- PK batch sampler -------------------------
class PKBatchSampler(Sampler):
    """P개 클래스 × K개 이미지를 한 배치로 뽑는 샘플러."""
    def __init__(self, dataset: TriViewDataset, P: int, K: int):
        self.P, self.K = int(P), int(K)
        from collections import defaultdict
        self.by_cls = defaultdict(list)
        for idx, rec in enumerate(dataset.index):
            self.by_cls[rec["label"]].append(idx)
        self.labels = list(self.by_cls.keys())
        for lst in self.by_cls.values():
            random.shuffle(lst)

    def __iter__(self):
        while True:
            P, K = self.P, self.K
            if len(self.labels) >= P:
                classes = random.sample(self.labels, P)
            else:
                classes = random.choices(self.labels, k=P)
            batch = []
            for c in classes:
                pool = self.by_cls[c]
                if len(pool) >= K:
                    picks = random.sample(pool, K)
                else:
                    picks = [random.choice(pool) for _ in range(K)]
                batch.extend(picks)
            yield batch

    def __len__(self):  # not used
        return 10**9

# ------------------------- Collate & transforms -------------------------
def collate_triview(batch):
    labels = torch.stack([b["label"] for b in batch])
    gids   = torch.stack([b["gid"] for b in batch]).squeeze(1)
    paths  = [b["path"] for b in batch]
    views, masks = {}, {}
    for k in ("whole", "face", "eyes"):
        xs = [b[k] for b in batch]
        mask = torch.tensor([x is not None for x in xs], dtype=torch.bool)
        if any(mask):
            ex = next(x for x in xs if x is not None)
            zeros = torch.zeros_like(ex)
            xs = [x if x is not None else zeros for x in xs]
            views[k] = torch.stack(xs, dim=0)
        else:
            views[k] = None
        masks[k] = mask
    return dict(views=views, masks=masks, labels=labels, gids=gids, paths=paths)

def make_transforms(sz_w, sz_f, sz_e):
    def aug(s):
        return transforms.Compose([
            transforms.RandomResizedCrop(s, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1,
                                   saturation=0.05, hue=0.02),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
    return aug(sz_w), aug(sz_f), aug(sz_e)

def make_val_transforms(sz_w, sz_f, sz_e):
    def val(s):
        return transforms.Compose([
            transforms.Resize(int(s*1.15)),
            transforms.CenterCrop(s),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
    return val(sz_w), val(sz_f), val(sz_e)

# ------------------------- Model & heads -------------------------
class MixStyle(nn.Module):
    def __init__(self, p=0.3, alpha=0.1):
        super().__init__()
        self.p = p; self.alpha = alpha
    def forward(self, x):
        if not self.training or self.p <= 0.0:
            return x
        B,C,H,W = x.shape
        mu = x.mean([2,3], keepdim=True)
        var = x.var([2,3], unbiased=False, keepdim=True)
        sigma = (var+1e-5).sqrt()
        perm = torch.randperm(B, device=x.device)
        mu2, sigma2 = mu[perm], sigma[perm]
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((B,1,1,1)).to(x.device)
        mu_mix = mu*lam + mu2*(1-lam)
        sigma_mix = sigma*lam + sigma2*(1-lam)
        x_norm = (x - mu)/sigma
        apply = (torch.rand(B,1,1,1, device=x.device) < self.p).float()
        mixed = x_norm * sigma_mix + mu_mix
        return mixed*apply + x*(1-apply)

class SqueezeExcite(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        m = max(8, c//r)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, m, 1), nn.GELU(),
            nn.Conv2d(m, c, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.net(x)

class ConvBlock(nn.Module):
    def __init__(self, ci, co, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(ci, co, k, s, p, bias=False)
        self.gn   = nn.GroupNorm(16, co)
        self.act  = nn.GELU()
    def forward(self, x):
        return self.act(self.gn(self.conv(x)))

class ResBlock(nn.Module):
    def __init__(self, ci, co, down=False, mix=None):
        super().__init__()
        self.c1 = ConvBlock(ci, co, 3, 1, 1)
        self.c2 = ConvBlock(co, co, 3, 1, 1)
        self.se = SqueezeExcite(co)
        self.down = down
        self.pool = nn.AvgPool2d(2) if down else nn.Identity()
        self.proj = nn.Conv2d(ci, co, 1, 1, 0, bias=False) if ci != co else nn.Identity()
        self.mix  = mix
    def forward(self, x):
        h = self.c1(x)
        if self.mix is not None:
            h = self.mix(h)
        h = self.c2(h)
        h = self.se(h)
        if self.down:
            h = self.pool(h); x = self.pool(x)
        return F.gelu(h + self.proj(x))

def matrix_sqrt_newton_schulz(A, iters=5):
    B,C,_ = A.shape
    normA = A.reshape(B, -1).norm(dim=1).view(B,1,1).clamp(min=1e-8)
    Y = A / normA
    I = torch.eye(C, device=A.device).expand(B, C, C)
    Z = I.clone()
    for _ in range(iters):
        T = 0.5 * (3.0*I - Z.bmm(Y))
        Y = Y.bmm(T)
        Z = T.bmm(Z)
    return Y * (normA.sqrt())

class GramHead(nn.Module):
    def __init__(self, c_in, c_red=64, proj=128):
        super().__init__()
        self.red  = nn.Conv2d(c_in, c_red, 1, bias=False)
        self.proj = nn.Linear(c_red*c_red, proj)
    def forward(self, x):
        f = self.red(x)
        B,C,H,W = f.shape
        Fm = f.flatten(2)
        G  = torch.bmm(Fm, Fm.transpose(1,2)) / (H*W)
        return self.proj(G.reshape(B, C*C))

class CovISqrtHead(nn.Module):
    def __init__(self, c_in, c_red=64, proj=128):
        super().__init__()
        self.red  = nn.Conv2d(c_in, c_red, 1, bias=False)
        self.proj = nn.Linear(c_red*c_red, proj)
    def forward(self, x):
        with torch.amp.autocast('cuda', enabled=False):
            f = self.red(x.float())
            B,C,H,W = f.shape
            Fm = f.flatten(2)
            mu = Fm.mean(-1, keepdim=True)
            Xc = Fm - mu
            cov = torch.bmm(Xc, Xc.transpose(1,2)) / (H*W - 1 + 1e-5)
            cov = matrix_sqrt_newton_schulz(cov.float(), iters=5)
            return self.proj(cov.reshape(B, C*C))

def spectrum_hist(x, K=16, O=8):
    B,C,H,W = x.shape
    spec = torch.fft.rfft2(x, norm='ortho').abs().mean(1)
    H2, W2 = spec.shape[-2], spec.shape[-1]
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H2, device=x.device),
        torch.linspace(0, 1,  W2, device=x.device),
        indexing="ij"
    )
    rr = (yy**2 + xx**2).sqrt().clamp(0, 1 - 1e-8)
    th = (torch.atan2(yy, xx + 1e-9) + math.pi/2)
    rb = (rr * K).long().clamp(0, K-1)
    ob = (th / math.pi * O).long().clamp(0, O-1)
    mag = torch.log1p(spec)
    rad = torch.zeros(B, K, device=x.device)
    ang = torch.zeros(B, O, device=x.device)
    rbf = rb.reshape(-1); obf = ob.reshape(-1)
    for b in range(B):
        m = mag[b].reshape(-1)
        rad[b].scatter_add_(0, rbf, m)
        ang[b].scatter_add_(0, obf, m)
    rad = rad / (rad.sum(-1, keepdim=True)+1e-6)
    ang = ang / (ang.sum(-1, keepdim=True)+1e-6)
    return torch.cat([rad, ang], dim=1)

class SpectrumHead(nn.Module):
    def __init__(self, c_in, proj=64, K=16, O=8):
        super().__init__()
        self.proj = nn.Linear(K+O, proj)
    def forward(self, x):
        with torch.amp.autocast('cuda', enabled=False):
            h = spectrum_hist(x.float())
            return self.proj(h)

class StatsHead(nn.Module):
    def __init__(self, c_in, proj=64):
        super().__init__()
        c = min(64, c_in)
        self.red = nn.Conv2d(c_in, c, 1, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(c*2, 128),
            nn.GELU(),
            nn.Linear(128, proj),
        )
    def forward(self, x):
        f = self.red(x)
        mu = f.mean([2,3])
        lv = torch.log(f.var([2,3], unbiased=False)+1e-5)
        return self.mlp(torch.cat([mu, lv], dim=1))

class ViewEncoder(nn.Module):
    """
    - Normalize([0.5],[0.5])된 RGB 입력
    - RGB -> Lab 변환
    - backbone + 스타일 헤드 4개 (Gram/Cov/Spectrum/Stats)
    - 브랜치 attention
    """
    def __init__(self, mix_p=0.3, out_dim=256):
        super().__init__()
        self.mix = MixStyle(p=mix_p, alpha=0.1)
        ch = [32, 64, 128, 192, 256]

        self.stem = nn.Sequential(
            ConvBlock(3, ch[0], 3, 1, 1),
            ConvBlock(ch[0], ch[0], 3, 1, 1),
        )
        self.b1 = ResBlock(ch[0], ch[1], down=True,  mix=self.mix)
        self.b2 = ResBlock(ch[1], ch[2], down=True,  mix=self.mix)
        self.b3 = ResBlock(ch[2], ch[3], down=True,  mix=None)
        self.b4 = ResBlock(ch[3], ch[4], down=True,  mix=None)

        self.h_gram3 = GramHead(ch[3])
        self.h_cov3  = CovISqrtHead(ch[3])
        self.h_sp3   = SpectrumHead(ch[3])
        self.h_st3   = StatsHead(ch[3])

        self.h_gram4 = GramHead(ch[4])
        self.h_cov4  = CovISqrtHead(ch[4])
        self.h_sp4   = SpectrumHead(ch[4])
        self.h_st4   = StatsHead(ch[4])

        fdim = (128+128+64+64)*2  # 768
        self.fdim = fdim

        self.branch_gate = nn.Sequential(
            nn.LayerNorm(fdim),
            nn.Linear(fdim, 4, bias=True),
        )

        self.fuse = nn.Sequential(
            nn.Linear(fdim, 512),
            nn.GELU(),
            nn.Linear(512, out_dim),
        )

    def _rgb_to_lab(self, x: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast('cuda', enabled=False):
            x_f = x.float()
            rgb = (x_f * 0.5 + 0.5).clamp(0.0, 1.0)

            thresh = 0.04045
            low = rgb / 12.92
            high = ((rgb + 0.055) / 1.055).pow(2.4)
            rgb_lin = torch.where(rgb <= thresh, low, high)

            rgb_lin = rgb_lin.permute(0, 2, 3, 1)
            M = rgb_lin.new_tensor([
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041],
            ])
            xyz = torch.matmul(rgb_lin, M.T)

            Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
            xyz = xyz / rgb_lin.new_tensor([Xn, Yn, Zn])

            eps = 0.008856
            kappa = 903.3

            def f(t):
                t = t.clamp(min=1e-6)
                return torch.where(
                    t > eps,
                    t.pow(1.0 / 3.0),
                    (kappa * t + 16.0) / 116.0,
                )

            f_xyz = f(xyz)
            fx, fy, fz = f_xyz[..., 0], f_xyz[..., 1], f_xyz[..., 2]

            L = 116.0 * fy - 16.0
            a = 500.0 * (fx - fy)
            b = 200.0 * (fy - fz)

            L_scaled = L / 100.0
            a_scaled = (a + 128.0) / 255.0
            b_scaled = (b + 128.0) / 255.0

            lab = torch.stack([L_scaled, a_scaled, b_scaled], dim=-1)
            lab = lab.permute(0, 3, 1, 2)

        return lab.to(dtype=x.dtype)

    def forward(self, x):
        x_lab = self._rgb_to_lab(x)

        f0 = self.stem(x_lab)
        f1 = self.b1(f0)
        f2 = self.b2(f1)
        f3 = self.b3(f2)
        f4 = self.b4(f3)

        g3 = self.h_gram3(f3)
        c3 = self.h_cov3(f3)
        sp3 = self.h_sp3(f3)
        st3 = self.h_st3(f3)

        g4 = self.h_gram4(f4)
        c4 = self.h_cov4(f4)
        sp4 = self.h_sp4(f4)
        st4 = self.h_st4(f4)

        b_gram = torch.cat([g3, g4], dim=1)
        b_cov  = torch.cat([c3, c4], dim=1)
        b_sp   = torch.cat([sp3, sp4], dim=1)
        b_st   = torch.cat([st3, st4], dim=1)

        flat = torch.cat([b_gram, b_cov, b_sp, b_st], dim=1)  # [B,768]

        gate_logits = self.branch_gate(flat)
        w = torch.softmax(gate_logits, dim=-1)
        w0, w1, w2, w3 = w[:,0:1], w[:,1:2], w[:,2:3], w[:,3:4]

        flat_weighted = torch.cat([
            b_gram * w0,
            b_cov  * w1,
            b_sp   * w2,
            b_st   * w3,
        ], dim=1)

        view_vec = self.fuse(flat_weighted)
        return view_vec

class TriViewStyleNet(nn.Module):
    def __init__(self, out_dim=256, mix_p=0.3, share_backbone: bool = True):
        super().__init__()
        if share_backbone:
            shared = ViewEncoder(mix_p=mix_p, out_dim=out_dim)
            self.enc_whole = shared
            self.enc_face  = shared
            self.enc_eyes  = shared
        else:
            self.enc_whole = ViewEncoder(mix_p=mix_p, out_dim=out_dim)
            self.enc_face  = ViewEncoder(mix_p=mix_p, out_dim=out_dim)
            self.enc_eyes  = ViewEncoder(mix_p=mix_p, out_dim=out_dim)
        self.view_gate = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, 1, bias=True),
        )
    def forward(self, views, masks):
        outs, alphas = {}, []
        for k, enc in (("whole", self.enc_whole),
                       ("face",  self.enc_face),
                       ("eyes",  self.enc_eyes)):
            if views[k] is None:
                outs[k] = None
                alphas.append(None)
                continue
            vk = enc(views[k].to(memory_format=torch.channels_last))
            outs[k] = l2n(vk)
            score = self.view_gate(outs[k]).squeeze(1)
            score = torch.where(
                masks[k].to(score.device),
                score,
                torch.full_like(score, -1e4),
            )
            alphas.append(score)
        scores = [a for a in alphas if a is not None]
        if len(scores) == 0:
            raise RuntimeError("All views are missing in this batch.")
        A = torch.stack(scores, dim=1)      # [B, num_views]
        W = F.softmax(A, dim=1)
        present = [outs[k] for k in ("whole","face","eyes") if outs[k] is not None]
        Z = torch.stack(present, dim=1)     # [B, num_views, dim]
        fused = l2n((W.unsqueeze(-1) * Z).sum(dim=1))  # [B, dim]
        return fused, outs, W

# ------------------------- Losses -------------------------
class ProxyAnchorLoss(nn.Module):
    def __init__(self, num_classes, dim, alpha=16.0, margin=0.1, neg_weight=0.25):
        super().__init__()
        self.proxies = nn.Parameter(torch.randn(num_classes, dim))
        nn.init.normal_(self.proxies, std=0.01)
        self.alpha = float(alpha)
        self.margin = float(margin)
        self.neg_weight = float(neg_weight)
    def forward(self, z, y):
        with torch.amp.autocast('cuda', enabled=False):
            z = F.normalize(z.float(), dim=-1)
            P = F.normalize(self.proxies.float(), dim=-1)
            sim = z @ P.t()
            C = sim.size(1)
            yOH = F.one_hot(y, num_classes=C).float()
            pos_e = torch.clamp(-self.alpha * (sim - self.margin),
                                min=-60.0, max=60.0)
            neg_e = torch.clamp( self.alpha * (sim + self.margin),
                                min=-60.0, max=60.0)
            pos_term = torch.exp(pos_e) * yOH
            neg_term = torch.exp(neg_e) * (1.0 - yOH)
            pos_sum = pos_term.sum(0)
            neg_sum = neg_term.sum(0)
            num_pos = (yOH.sum(0) > 0)
            L_pos = torch.log1p(pos_sum[num_pos]).sum() / (num_pos.sum().clamp_min(1.0))
            L_neg = torch.log1p(neg_sum).sum() / C
            return L_pos + self.neg_weight * L_neg

class SupConLoss(nn.Module):
    def __init__(self, tau=0.07):
        super().__init__()
        self.tau = tau
    def forward(self, feats, labels):
        feats = l2n(feats)
        sim = feats @ feats.t() / self.tau
        logits = sim - torch.eye(sim.size(0), device=sim.device) * 1e9
        pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)) & \
                   (~torch.eye(len(labels), device=labels.device, dtype=torch.bool))
        numer = (torch.exp(logits) * pos_mask).sum(1)
        denom = torch.exp(logits).sum(1).clamp_min(1e-8)
        valid = (pos_mask.sum(1) > 0)
        loss  = -torch.log((numer+1e-12) / denom)
        return (loss[valid].mean() if valid.any() else torch.tensor(0.0, device=feats.device))

class MultiViewInfoNCE(nn.Module):
    def __init__(self, tau=0.1):
        super().__init__()
        self.tau = tau
    def forward(self, feats, gids):
        feats = l2n(feats)
        sim = feats @ feats.t() / self.tau
        logits = sim - torch.eye(sim.size(0), device=sim.device) * 1e9
        pos_mask = (gids.unsqueeze(1) == gids.unsqueeze(0)) & \
                   (~torch.eye(len(gids), device=gids.device, dtype=torch.bool))
        numer = (torch.exp(logits) * pos_mask).sum(1)
        denom = torch.exp(logits).sum(1).clamp_min(1e-8)
        valid = (pos_mask.sum(1) > 0)
        loss  = -torch.log((numer+1e-12) / denom)
        return (loss[valid].mean() if valid.any() else torch.tensor(0.0, device=feats.device))

# --------------------- Logging / checkpoints / schedulers -----------------
os.makedirs(cfg.out_dir, exist_ok=True)
LOG_TXT = os.path.join(cfg.out_dir, "train.log")
METRICS_CSV = os.path.join(cfg.out_dir, "metrics_epoch.csv")
if not os.path.exists(METRICS_CSV):
    with open(METRICS_CSV, "w", encoding="utf-8") as f:
        f.write("timestamp,stage,epoch,steps,P,K,train_loss,train_proxy,train_sup,train_mv,"
                "val_proxy,proxy_top1,knn_r1,knn_r5,kmeans_acc,nmi,ari\n")

def wlog_global(msg, also_print=False):
    ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts_str}] {msg}"
    with open(LOG_TXT, "a", encoding="utf-8", buffering=1) as _logf:
        _logf.write(line + "\n")
    if also_print:
        tqdm.write(line)

def write_epoch_metrics(stage_i, epoch_i, steps, P, K,
                        tr_mean, tr_p, tr_s, tr_m,
                        val_proxy, proxy_top1,
                        knn_r1, knn_r5,
                        kmeans_acc, nmi, ari):
    ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    def fmt(x):
        if x is None:
            return "nan"
        try:
            if hasattr(x, "item"):
                x = float(x.item())
            else:
                x = float(x)
        except Exception:
            return "nan"
        if np.isnan(x) or np.isinf(x):
            return "nan"
        return f"{x:.6f}"
    with open(METRICS_CSV, "a", encoding="utf-8") as fh:
        fh.write(
            f"{ts_str},{stage_i},{epoch_i},{steps},{P},{K},"
            f"{fmt(tr_mean)},{fmt(tr_p)},{fmt(tr_s)},{fmt(tr_m)},"
            f"{fmt(val_proxy)},{fmt(proxy_top1)},"
            f"{fmt(knn_r1)},{fmt(knn_r5)},"
            f"{fmt(kmeans_acc)},{fmt(nmi)},{fmt(ari)}\n"
        )

def save_ckpt(path, model, proxy_loss, optim, sched, meta, is_main: bool):
    if not is_main:
        return
    base_model = model.module if isinstance(model, nn.parallel.DistributedDataParallel) else model
    torch.save({
        "model": base_model.state_dict(),
        "proxies": proxy_loss.state_dict(),
        "optim": optim.state_dict() if optim else None,
        "sched": sched.state_dict() if sched else None,
        "meta": meta,
    }, path)

def find_latest_checkpoint(out_dir):
    paths = glob.glob(os.path.join(out_dir, "stage*_epoch*.pt"))
    best, best_stage, best_epoch = None, -1, -1
    for p in paths:
        m = re.search(r"stage(\d+)_epoch(\d+)\.pt$", os.path.basename(p))
        if not m:
            continue
        si, ep = int(m.group(1)), int(m.group(2))
        if (si > best_stage) or (si == best_stage and ep > best_epoch):
            best, best_stage, best_epoch = p, si, ep
    return best, best_stage, best_epoch

def _pick_from_schedule(sched, default_val, ep):
    if not sched:
        return int(default_val)
    if isinstance(sched, dict):
        items = sorted([(int(k), int(v)) for k,v in sched.items()], key=lambda x: x[0])
    else:
        items = sorted([(int(k), int(v)) for k,v in sched], key=lambda x: x[0])
    val = int(default_val)
    for k,v in items:
        if ep >= k:
            val = int(v)
    return int(val)

def resolve_epoch_PK(stage: dict, ep: int):
    P = int(stage.get("P", cfg.P))
    K = int(stage.get("K", cfg.K))
    P = _pick_from_schedule(stage.get("P_schedule"), P, ep)
    K = _pick_from_schedule(stage.get("K_schedule"), K, ep)
    bs_sched = stage.get("bs_schedule")
    if bs_sched:
        bs = _pick_from_schedule(bs_sched, P*K, ep)
        if bs % K != 0:
            wlog_global(f"[batch] bs_schedule value {bs} not divisible by K={K}; rounding down to {bs//K*K}", also_print=True)
            bs = (bs // K) * K
        P = max(1, bs // K)
    return int(P), int(K)

def estimate_steps_per_epoch(train_len: int, global_batch: int, max_steps: Optional[int]):
    if max_steps is not None:
        return int(max_steps)
    return max(1, math.ceil(train_len / max(1, global_batch)))

def build_train_loader(ds: TriViewDataset, P: int, K: int):
    bs = PKBatchSampler(ds, P, K)
    dl = DataLoader(
        ds,
        batch_sampler=bs,
        num_workers=cfg.workers,
        pin_memory=True,
        collate_fn=collate_triview,
        persistent_workers=False,
        prefetch_factor=2 if cfg.workers > 0 else None,
        multiprocessing_context=MP_CTX,
    )
    return _track_dl(dl)

def make_cosine_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        rem = max(1, total_steps - warmup_steps)
        progress = (step - warmup_steps) / rem
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

# ------------------------------ DDP worker --------------------------------
def ddp_train_worker(rank: int, world_size: int):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    seed_all(cfg.seed + rank)
    is_main = (rank == 0)

    # class count
    artists_dir = os.path.join(cfg.data_root, cfg.folders['whole'])
    num_classes_total = len([
        d for d in os.listdir(artists_dir)
        if os.path.isdir(os.path.join(artists_dir, d))
    ])
    if is_main:
        wlog_global(f"[DDP] world_size={world_size}, num_classes_total={num_classes_total}", also_print=True)

    # model & losses
    base_model = TriViewStyleNet(
        out_dim=cfg.embed_dim,
        mix_p=cfg.mixstyle_p,
        share_backbone=True,
    ).to(device)
    base_model = base_model.to(memory_format=torch.channels_last)

    if cfg.use_compile and hasattr(torch, "compile"):
        try:
            base_model = torch.compile(base_model, mode="reduce-overhead", fullgraph=False)
        except Exception:
            pass

    model = nn.parallel.DistributedDataParallel(
        base_model,
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=False,
    )

    proxy_loss = ProxyAnchorLoss(
        num_classes=num_classes_total,
        dim=cfg.embed_dim,
        alpha=cfg.alpha_proxy,
        margin=cfg.margin_proxy,
        neg_weight=0.25,
    ).to(device)

    supcon     = SupConLoss(tau=cfg.supcon_tau).to(device)
    mv_infonce = MultiViewInfoNCE(tau=cfg.mv_tau).to(device)

    # resume
    resume_info = None
    ckpt_path, ck_stage, ck_epoch = find_latest_checkpoint(cfg.out_dir)
    if ckpt_path is not None:
        ck = torch.load(ckpt_path, map_location="cpu")
        try:
            model.module.load_state_dict(ck["model"], strict=False)
        except Exception as e:
            if is_main:
                wlog_global(f"[resume] WARNING: model state load failed: {e}", also_print=True)
        try:
            proxy_loss.load_state_dict(ck["proxies"])
        except Exception as e:
            if is_main:
                wlog_global(f"[resume] WARNING: proxy state load failed: {e}", also_print=True)

        meta = ck.get("meta", {})
        last_stage = int(meta.get("stage", ck_stage or 1))
        last_epoch = int(meta.get("epoch", ck_epoch or 0))
        start_stage = last_stage
        start_epoch = last_epoch + 1
        if start_stage <= len(cfg.stages) and start_epoch > cfg.stages[start_stage-1]["epochs"]:
            start_stage += 1
            start_epoch = 1

        resume_info = dict(
            ckpt=ck,
            path=ckpt_path,
            last_stage=last_stage,
            last_epoch=last_epoch,
            start_stage=start_stage,
            start_epoch=start_epoch,
        )
        if is_main:
            wlog_global(
                f"[resume] Found {ckpt_path} (stage {last_stage}, epoch {last_epoch}). "
                f"Resuming at stage {start_stage}, epoch {start_epoch}.",
                also_print=True,
            )
    else:
        if is_main:
            wlog_global("[resume] No checkpoint found; training from scratch.", also_print=True)

    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
    global_step = 0

    proxy_lr_mult = 5.0
    RAMP_EPOCHS   = 3
    WARMUP_EPOCHS = 1
    VALIDATE_EVERY = 4  # N epoch마다 검증

    from tqdm.auto import tqdm as tqdm_local

    # Stage loop
    for si, stage in enumerate(cfg.stages, 1):
        if resume_info and si < resume_info["start_stage"]:
            if is_main:
                wlog_global(f"[resume] Skipping stage {si}; already completed.", also_print=True)
            continue

        # datasets per stage
        T_w_tr, T_f_tr, T_e_tr   = make_transforms(stage["sz_whole"], stage["sz_face"], stage["sz_eyes"])
        T_w_val, T_f_val, T_e_val = make_val_transforms(stage["sz_whole"], stage["sz_face"], stage["sz_eyes"])

        train_ds = TriViewDataset(cfg.data_root, cfg.folders, split="train",
                                  T_whole=T_w_tr,  T_face=T_f_tr,  T_eyes=T_e_tr)
        val_ds   = TriViewDataset(cfg.data_root, cfg.folders, split="val",
                                  T_whole=T_w_val, T_face=T_f_val, T_eyes=T_e_val)

        # steps_per_epoch schedule (global batch 기준)
        steps_list = []
        for ep_tmp in range(1, stage["epochs"]+1):
            P_tmp, K_tmp = resolve_epoch_PK(stage, ep_tmp)
            global_batch = P_tmp * K_tmp * world_size
            steps = estimate_steps_per_epoch(
                len(train_ds),
                global_batch,
                cfg.max_steps_per_epoch,
            )
            steps_list.append(steps)
        total_steps_stage = int(sum(steps_list))
        warmup_steps = int(steps_list[0] * WARMUP_EPOCHS)

        params = [
            {"params": model.parameters(),      "lr": stage["lr"]},
            {"params": proxy_loss.parameters(), "lr": stage["lr"] * proxy_lr_mult},
        ]
        optim = torch.optim.AdamW(params, weight_decay=cfg.weight_decay)
        sched = make_cosine_with_warmup(optim, warmup_steps=warmup_steps, total_steps=total_steps_stage)

        start_ep = 1
        if resume_info and si == resume_info["start_stage"]:
            start_ep = resume_info["start_epoch"]
            if resume_info["last_stage"] == si and start_ep > 1:
                try:
                    if resume_info["ckpt"].get("optim") is not None:
                        optim.load_state_dict(resume_info["ckpt"]["optim"])
                    if resume_info["ckpt"].get("sched") is not None:
                        sched.load_state_dict(resume_info["ckpt"]["sched"])
                    if is_main:
                        wlog_global(f"[resume] Loaded optimizer/scheduler from {resume_info['path']}.", also_print=True)
                except Exception as e:
                    if is_main:
                        wlog_global(f"[resume] WARNING: could not load optimizer/scheduler state: {e}", also_print=True)

        stage_msg = (f"\n=== [DDP] Stage {si}/{len(cfg.stages)} :: "
                     f"wh/face/eyes={stage['sz_whole']}/{stage['sz_face']}/{stage['sz_eyes']} | "
                     f"epochs={stage['epochs']} | lr={stage['lr']} | classes={num_classes_total} ===")
        if is_main:
            print(stage_msg)
            wlog_global(stage_msg)

        # epoch loop
        for ep in range(start_ep, stage["epochs"]+1):
            P_e, K_e = resolve_epoch_PK(stage, ep)
            B_e = P_e * K_e     # local batch
            train_dl = build_train_loader(train_ds, P_e, K_e)

            steps_per_epoch = steps_list[ep-1]
            model.train()
            proxy_loss.train()

            running = {"proxy":0.0, "supcon":0.0, "mv":0.0, "tot":0.0}
            ep_sum_tot = ep_sum_p = ep_sum_s = ep_sum_m = 0.0
            ramp = min(1.0, ep / RAMP_EPOCHS)

            if is_main:
                tbar = tqdm_local(range(1, steps_per_epoch+1),
                                  desc=f"[train-DDP] stage{si} ep{ep} (P={P_e},K={K_e},B={B_e},rank={rank})",
                                  leave=True)
            else:
                tbar = range(1, steps_per_epoch+1)

            train_iter = iter(train_dl)

            for it in tbar:
                try:
                    batch = next(train_iter)
                except Exception as e:
                    if _should_fallback_workers(e) and cfg.workers > 0:
                        if is_main:
                            print("[mp] Worker pickling error detected. Rebuilding loaders with num_workers=0.")
                        cfg.workers = 0
                        train_dl = build_train_loader(train_ds, P_e, K_e)
                        train_iter = iter(train_dl)
                        batch = next(train_iter)
                    else:
                        raise

                labels = batch["labels"].to(device, non_blocking=True)
                gids   = batch["gids"].to(device, non_blocking=True)
                views = {
                    k: (v.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                        if v is not None else None)
                    for k,v in batch["views"].items()
                }
                masks = {k: v.to(device, non_blocking=True) for k,v in batch["masks"].items()}

                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    z_fused, z_views_dict, W = model(views, masks)

                    Z_all, Y_all, G_all = [], [], []
                    for vk in ("whole","face","eyes"):
                        zk = z_views_dict.get(vk)
                        if zk is None:
                            continue
                        mk = masks[vk]
                        if mk.any():
                            Z_all.append(zk[mk])
                            Y_all.append(labels[mk])
                            G_all.append(gids[mk])
                    if len(Z_all) == 0:
                        Z_all, Y_all, G_all = [z_fused], [labels], [gids]
                    Z_all = torch.cat(Z_all, dim=0)
                    Y_all = torch.cat(Y_all, dim=0)
                    G_all = torch.cat(G_all, dim=0)

                    L_proxy = proxy_loss(z_fused, labels)
                    L_sup   = supcon(Z_all, Y_all)
                    L_mv    = mv_infonce(Z_all, G_all)
                    L_total = L_proxy + (0.5 * ramp) * L_sup + (0.5 * ramp) * L_mv

                optim.zero_grad(set_to_none=True)
                scaler.scale(L_total).backward()
                scaler.step(optim)
                scaler.update()
                sched.step()
                global_step += 1

                running["proxy"]  += L_proxy.item()
                running["supcon"] += L_sup.item()
                running["mv"]     += L_mv.item()
                running["tot"]    += L_total.item()

                ep_sum_tot += L_total.item()
                ep_sum_p   += L_proxy.item()
                ep_sum_s   += L_sup.item()
                ep_sum_m   += L_mv.item()

                if is_main and (it % cfg.print_every == 0 or it == steps_per_epoch):
                    denom = min(cfg.print_every, it % cfg.print_every or cfg.print_every)
                    tbar.set_postfix({
                        "L": f"{running['tot']/denom:.3f}",
                        "proxy": f"{running['proxy']/denom:.3f}",
                        "sup": f"{running['supcon']/denom:.3f}",
                        "mv": f"{running['mv']/denom:.3f}",
                        "lr": f"{optim.param_groups[0]['lr']:.2e}",
                    })
                    msg = (f"stage{si} ep{ep:02d} it{it:05d}/{steps_per_epoch} | "
                           f"P={P_e} K={K_e} B={B_e} | "
                           f"L={running['tot']/denom:.3f} "
                           f"(proxy={running['proxy']/denom:.3f}, "
                           f"sup={running['supcon']/denom:.3f}, "
                           f"mv={running['mv']/denom:.3f}) | "
                           f"lr={optim.param_groups[0]['lr']:.2e}")
                    wlog_global(msg)
                    running = {k:0.0 for k in running}

            # ===== 검증 (proxy loss + proxy Top1만) =====
            proxy_top1   = float("nan")
            kmeans_acc   = float("nan")  # 사용 안 하지만 CSV 포맷 때문에 남겨둠
            nmi          = float("nan")
            ari          = float("nan")
            knn_r1       = float("nan")
            knn_r5       = float("nan")
            val_proxy_mean = float("nan")

            do_val = (VALIDATE_EVERY <= 0) or (ep % VALIDATE_EVERY == 0) or (ep == stage["epochs"])

            if do_val:
                from torch.utils.data.distributed import DistributedSampler

                val_sampler = DistributedSampler(
                    val_ds,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=False,
                    drop_last=False,
                )
                val_sampler.set_epoch(ep)

                val_dl_ddp = DataLoader(
                    val_ds,
                    batch_size=B_e,
                    sampler=val_sampler,
                    num_workers=min(8, cfg.workers),
                    pin_memory=True,
                    collate_fn=collate_triview,
                    persistent_workers=False,
                    multiprocessing_context=MP_CTX,
                )

                model.eval()
                proxy_loss.eval()

                local_loss_sum = 0.0
                local_loss_cnt = 0.0
                local_correct  = 0.0
                local_total    = 0.0

                with torch.no_grad():
                    Pn = F.normalize(proxy_loss.proxies.detach(), dim=1).to(device)

                with torch.no_grad(), torch.amp.autocast('cuda', dtype=amp_dtype):
                    for batch in val_dl_ddp:
                        labels = batch["labels"].to(device, non_blocking=True)
                        views  = {
                            k: (v.to(device).to(memory_format=torch.channels_last) if v is not None else None)
                            for k, v in batch["views"].items()
                        }
                        masks  = {k: v.to(device, non_blocking=True) for k, v in batch["masks"].items()}

                        z_fused, _, _ = model(views, masks)
                        L = proxy_loss(z_fused, labels)

                        z_norm = F.normalize(z_fused, dim=1)
                        logits = z_norm @ Pn.t()
                        pred = logits.argmax(dim=1)
                        correct = (pred == labels).float().sum().item()

                        bs = float(labels.size(0))
                        local_loss_sum += L.item()
                        local_loss_cnt += 1.0
                        local_correct  += correct
                        local_total    += bs

                t = torch.tensor(
                    [local_loss_sum, local_loss_cnt, local_correct, local_total],
                    device=device,
                )
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                total_loss_sum = float(t[0].item())
                total_loss_cnt = max(1.0, float(t[1].item()))
                total_correct  = float(t[2].item())
                total_total    = max(1.0, float(t[3].item()))

                val_proxy_mean = total_loss_sum / total_loss_cnt
                proxy_top1     = total_correct / total_total

                if is_main:
                    print(f"[val] ep{ep:02d} proxy-loss ~ {val_proxy_mean:.3f}, Top1={proxy_top1:.4f}")
                    wlog_global(f"[val] ep{ep:02d} proxy-loss ~ {val_proxy_mean:.3f}, Top1={proxy_top1:.4f}")

                dist.barrier()

                _close_dl(val_dl_ddp)
                del val_dl_ddp
                gc.collect()
                time.sleep(0.05)

            # ----- Epoch metrics & checkpoint (rank0) -----
            train_mean = ep_sum_tot / steps_per_epoch
            train_p    = ep_sum_p   / steps_per_epoch
            train_s    = ep_sum_s   / steps_per_epoch
            train_m    = ep_sum_m   / steps_per_epoch

            write_epoch_metrics(
                si, ep, steps_per_epoch, P_e, K_e,
                train_mean, train_p, train_s, train_m,
                val_proxy_mean, proxy_top1,
                knn_r1, knn_r5,
                kmeans_acc, nmi, ari,
            )

            ck = os.path.join(cfg.out_dir, f"stage{si}_epoch{ep}.pt")
            save_ckpt(
                ck, model, proxy_loss, optim, sched,
                meta=dict(
                    stage=si, epoch=ep,
                    P=P_e, K=K_e, steps=steps_per_epoch,
                    val_every=VALIDATE_EVERY,
                    proxy_top1=proxy_top1,
                    knn_r1=knn_r1, knn_r5=knn_r5,
                ),
                is_main=is_main,
            )
            if is_main:
                print(f"Saved: {ck}")
                wlog_global(f"Saved: {ck}")

            _close_dl(train_dl)
            del train_dl
            gc.collect()
            time.sleep(0.1)

    dist.destroy_process_group()
    if is_main:
        print("\n[DDP] Training finished or paused. Checkpoints in:", cfg.out_dir)
        print("Logs:", LOG_TXT, " | CSV:", METRICS_CSV)
        print("Tip: Re-run this script to RESUME (DDP).")

# ------------------------------ entry point --------------------------------
def run_ddp_training():
    if not torch.cuda.is_available():
        print("CUDA not available; DDP training requires GPU.")
        return

    world_size = torch.cuda.device_count()
    print(f"[DDP] Launching training on {world_size} GPUs...")
    mp.spawn(
        ddp_train_worker,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )

if __name__ == "__main__":
    run_ddp_training()
