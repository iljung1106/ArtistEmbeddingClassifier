from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class PrototypeDB:
    centers: torch.Tensor  # [N,D] float32
    labels: torch.Tensor  # [N] int64
    label_names: List[str]  # id -> name
    source_path: Optional[Path] = None

    @property
    def dim(self) -> int:
        return int(self.centers.shape[1])

    def id_to_name(self, idx: int) -> str:
        if 0 <= idx < len(self.label_names):
            return self.label_names[idx]
        return str(idx)

    def ensure_label_id(self, name: str) -> int:
        name = str(name).strip()
        if not name:
            raise ValueError("Empty label name.")
        try:
            i = self.label_names.index(name)
            return int(i)
        except ValueError:
            self.label_names.append(name)
            return len(self.label_names) - 1

    def add_center(self, label_name: str, center: torch.Tensor) -> int:
        if center.ndim != 1:
            raise ValueError("center must be 1D embedding vector.")
        center = torch.nn.functional.normalize(center.float(), dim=0).view(1, -1)
        lid = self.ensure_label_id(label_name)
        self.centers = torch.cat([self.centers, center], dim=0)
        self.labels = torch.cat([self.labels, torch.tensor([lid], dtype=torch.long)], dim=0)
        return lid

    def save(self, path: Optional[str | Path] = None) -> Path:
        out = Path(path) if path is not None else self.source_path
        if out is None:
            raise ValueError("No output path specified for saving prototype DB.")
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            dict(
                centers=self.centers.detach().cpu(),
                labels=self.labels.detach().cpu(),
                label_names=list(self.label_names),
            ),
            str(out),
        )
        self.source_path = out
        return out


def _infer_label_names_from_dataset(dataset_root: Path) -> Optional[List[str]]:
    # `train_style_ddp.TriViewDataset` assigns IDs based on sorted directory names under dataset/<artist>.
    if not dataset_root.exists():
        return None
    artists = sorted([p.name for p in dataset_root.iterdir() if p.is_dir()])
    return artists if artists else None


def load_prototype_db(path: str | Path, *, try_dataset_dir: str | Path = "dataset") -> PrototypeDB:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    obj = torch.load(str(p), map_location="cpu")
    if not isinstance(obj, dict) or "centers" not in obj or "labels" not in obj:
        raise ValueError(f"Unsupported prototype file format: {p}")

    centers = obj["centers"].float()
    labels = obj["labels"].long()

    label_names = obj.get("label_names")
    if not isinstance(label_names, list) or not all(isinstance(x, str) for x in label_names):
        inferred = _infer_label_names_from_dataset(Path(try_dataset_dir))
        if inferred is None:
            max_id = int(labels.max().item()) if labels.numel() else -1
            label_names = [str(i) for i in range(max_id + 1)]
        else:
            label_names = inferred

    return PrototypeDB(centers=centers, labels=labels, label_names=label_names, source_path=p)


def topk_predictions(
    db: PrototypeDB,
    z: torch.Tensor,
    *,
    topk: int = 5,
) -> List[Tuple[str, float]]:
    """
    Returns [(label_name, score)] sorted by score desc (cosine similarity).
    `z` is 1D embedding (D).
    """
    if z.ndim != 1:
        raise ValueError("z must be 1D.")
    Z = torch.nn.functional.normalize(z.float(), dim=0).view(1, -1)
    C = torch.nn.functional.normalize(db.centers.float(), dim=1)
    sim = (Z @ C.t()).squeeze(0)  # [N]
    k = int(max(1, min(topk, sim.numel()))) if sim.numel() else 0
    if k == 0:
        return []
    vals, idxs = torch.topk(sim, k=k)
    out: List[Tuple[str, float]] = []
    for v, i in zip(vals.tolist(), idxs.tolist()):
        lid = int(db.labels[i].item())
        out.append((db.id_to_name(lid), float(v)))
    return out


def topk_predictions_unique_labels(
    db: PrototypeDB,
    z: torch.Tensor,
    *,
    topk: int = 5,
) -> List[Tuple[str, float]]:
    """
    Like topk_predictions(), but dedupes by label:
    if a label has multiple prototypes, only the highest score is kept.
    """
    if z.ndim != 1:
        raise ValueError("z must be 1D.")
    Z = torch.nn.functional.normalize(z.float(), dim=0).view(1, -1)
    C = torch.nn.functional.normalize(db.centers.float(), dim=1)
    sim = (Z @ C.t()).squeeze(0)  # [N]
    if sim.numel() == 0:
        return []

    best_by_label: dict[int, float] = {}
    # iterate all prototypes once; keep max per label id
    for i in range(sim.numel()):
        lid = int(db.labels[i].item())
        s = float(sim[i].item())
        prev = best_by_label.get(lid)
        if prev is None or s > prev:
            best_by_label[lid] = s

    items = sorted(best_by_label.items(), key=lambda kv: kv[1], reverse=True)
    items = items[: max(1, int(topk))]
    return [(db.id_to_name(lid), float(score)) for (lid, score) in items]


