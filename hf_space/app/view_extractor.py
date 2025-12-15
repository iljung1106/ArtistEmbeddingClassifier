from __future__ import annotations

import itertools
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch


def _patch_torch_load_for_old_ckpt() -> None:
    """
    Matches `anime_face_eye_extract._patch_torch_load_for_old_ckpt()` to load older YOLOv5 checkpoints
    on newer torch versions.
    """
    import numpy as _np

    try:
        torch.serialization.add_safe_globals([_np.core.multiarray._reconstruct, _np.ndarray])
    except Exception:
        pass

    _orig_load = torch.load

    def _patched_load(*args, **kwargs):  # noqa: ANN001
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)

    torch.load = _patched_load


def _pre(gray: np.ndarray) -> np.ndarray:
    import cv2

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _expand(box, margin: float, W: int, H: int):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = (x2 - x1) * (1 + margin)
    h = (y2 - y1) * (1 + margin)
    nx1 = int(round(cx - w / 2))
    ny1 = int(round(cy - h / 2))
    nx2 = int(round(cx + w / 2))
    ny2 = int(round(cy + h / 2))
    nx1 = max(0, min(W, nx1))
    ny1 = max(0, min(H, ny1))
    nx2 = max(0, min(W, nx2))
    ny2 = max(0, min(H, ny2))
    return nx1, ny1, nx2, ny2


def _shrink(img: np.ndarray, limit: int):
    import cv2

    h, w = img.shape[:2]
    m = max(h, w)
    if m <= limit:
        return img, 1.0
    s = limit / float(m)
    nh, nw = int(h * s), int(w * s)
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return small, s


def _pad_to_square_rgb(img: np.ndarray) -> np.ndarray:
    """
    Pad an RGB crop to a square (1:1) using edge-padding.
    This guarantees 1:1 aspect ratio without stretching content.
    """
    if img is None or img.size == 0:
        return img
    h, w = img.shape[:2]
    if h == w:
        return img
    s = max(h, w)
    pad_y = s - h
    pad_x = s - w
    top = pad_y // 2
    bottom = pad_y - top
    left = pad_x // 2
    right = pad_x - left
    return np.pad(img, ((top, bottom), (left, right), (0, 0)), mode="edge")


def _square_box_from_rect(rect, *, scale: float, W: int, H: int):
    """
    Convert a rectangle (x1,y1,x2,y2) into a square box centered on the rect,
    scaled by `scale`, clamped to image bounds.
    """
    x1, y1, x2, y2 = [int(v) for v in rect]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = max(1.0, float(x2 - x1))
    bh = max(1.0, float(y2 - y1))
    side = max(bw, bh) * float(scale)
    nx1 = int(round(cx - side / 2.0))
    ny1 = int(round(cy - side / 2.0))
    nx2 = int(round(cx + side / 2.0))
    ny2 = int(round(cy + side / 2.0))
    nx1 = max(0, min(W, nx1))
    ny1 = max(0, min(H, ny1))
    nx2 = max(0, min(W, nx2))
    ny2 = max(0, min(H, ny2))
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    return nx1, ny1, nx2, ny2


def _split_box_by_midline(box, mid_x: int):
    """
    If a box crosses the vertical midline, split into left/right boxes.
    Returns list of (tag, box).
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    if x1 < mid_x < x2:
        left = (x1, y1, mid_x, y2)
        right = (mid_x, y1, x2, y2)
        out = []
        if left[2] > left[0]:
            out.append(("left", left))
        if right[2] > right[0]:
            out.append(("right", right))
        return out
    tag = "left" if (x1 + x2) / 2.0 <= mid_x else "right"
    return [(tag, (x1, y1, x2, y2))]


def _best_pair(boxes, W: int, H: int):
    clean = [(int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in boxes]
    if len(clean) < 2:
        return []

    def cxcy(b):
        x1, y1, x2, y2 = b
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def area(b):
        x1, y1, x2, y2 = b
        return max(1, (x2 - x1) * (y2 - y1))

    best = None
    best_s = 1e9
    for b1, b2 in itertools.combinations(clean, 2):
        c1x, c1y = cxcy(b1)
        c2x, c2y = cxcy(b2)
        a1, a2 = area(b1), area(b2)
        horiz = 0.0 if c1x < c2x else 0.5
        y_aln = abs(c1y - c2y) / max(1.0, H)
        szsim = abs(a1 - a2) / float(max(a1, a2))
        gap = abs(c2x - c1x) / max(1.0, W)
        if 0.05 <= gap <= 0.5:
            gap_pen = 0.0
        else:
            gap_pen = 0.5 * ((0.5 + abs(gap - 0.05) * 10) if gap < 0.05 else (gap - 0.5) * 2.0)
        mean_y = (c1y + c2y) / 2.0 / max(1.0, H)
        upper = 0.3 * max(0.0, (mean_y - 0.67) * 2.0)
        s = y_aln + szsim + gap_pen + upper + horiz
        if s < best_s:
            best_s = s
            best = (b1, b2)

    if best is None:
        return []
    b1, b2 = best
    left, right = (b1, b2) if (b1[0] + b1[2]) <= (b2[0] + b2[2]) else (b2, b1)
    return [("left", left), ("right", right)]


@dataclass
class ExtractorCfg:
    yolo_dir: Path
    weights: Path
    cascade: Path
    imgsz: int = 640
    conf: float = 0.5
    iou: float = 0.5
    yolo_device: str = "cpu"  # "cpu" or "0"
    eye_roi_frac: float = 0.70
    eye_min_size: int = 12
    eye_margin: float = 0.60
    neighbors: int = 9
    eye_downscale_limit_roi: int = 512
    eye_downscale_limit_face: int = 768
    eye_fallback_to_face: bool = True


class AnimeFaceEyeExtractor:
    """
    Single-image view extractor (whole -> face crop, eyes crop) based on `anime_face_eye_extract.py`.
    Designed for use in the Gradio UI: caches YOLO model + Haar cascade.
    """

    def __init__(self, cfg: ExtractorCfg):
        self.cfg = cfg
        self._model = None
        self._device = None
        self._stride = 32
        self._tl = threading.local()

    def _init_detector(self) -> None:
        if self._model is not None:
            return

        ydir = self.cfg.yolo_dir.resolve()
        if not ydir.exists():
            raise RuntimeError(f"yolov5_anime dir not found: {ydir}")
        if str(ydir) not in sys.path:
            sys.path.insert(0, str(ydir))

        _patch_torch_load_for_old_ckpt()

        from models.experimental import attempt_load
        from utils.torch_utils import select_device

        self._device = select_device(self.cfg.yolo_device)
        self._model = attempt_load(str(self.cfg.weights), map_location=self._device)
        self._model.eval()

        self._stride = int(self._model.stride.max())
        s = int(self.cfg.imgsz)
        s = int(np.ceil(s / self._stride) * self._stride)
        self.cfg.imgsz = s

    def _letterbox_compat(self, img0, new_shape, stride):
        from utils.datasets import letterbox
        try:
            lb = letterbox(img0, new_shape, stride=stride, auto=False)
        except TypeError:
            try:
                lb = letterbox(img0, new_shape, auto=False)
            except TypeError:
                lb = letterbox(img0, new_shape)
        return lb[0]

    def _detect_faces(self, rgb: np.ndarray):
        import cv2
        self._init_detector()
        from utils.general import non_max_suppression, scale_coords

        img0 = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        h0, w0, _ = img0.shape
        img = self._letterbox_compat(img0, self.cfg.imgsz, self._stride)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        im = torch.from_numpy(img).to(self._device)
        im = im.float() / 255.0
        if im.ndim == 3:
            im = im[None]

        with torch.no_grad():
            pred = self._model(im)[0]
        pred = non_max_suppression(pred, conf_thres=self.cfg.conf, iou_thres=self.cfg.iou, classes=None, agnostic=False)

        boxes = []
        det = pred[0]
        if det is not None and len(det):
            det[:, :4] = scale_coords((self.cfg.imgsz, self.cfg.imgsz), det[:, :4], (h0, w0)).round()
            for *xyxy, conf, cls in det.tolist():
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                boxes.append((x1, y1, x2, y2))
        return boxes

    def _get_cascade(self):
        import cv2

        c = getattr(self._tl, "cascade", None)
        if c is None:
            c = cv2.CascadeClassifier(str(self.cfg.cascade))
            if c.empty():
                raise RuntimeError(f"cascade load fail: {self.cfg.cascade}")
            self._tl.cascade = c
        return c

    def _detect_eyes_in_roi(self, rgb_roi: np.ndarray):
        import cv2

        gray = cv2.cvtColor(rgb_roi, cv2.COLOR_RGB2GRAY)
        proc = _pre(gray)
        H, W = proc.shape[:2]
        min_side = max(1, min(W, H))
        dyn_min = int(0.07 * min_side)
        min_sz = max(8, int(self.cfg.eye_min_size), dyn_min)

        cascade = self._get_cascade()
        raw = cascade.detectMultiScale(
            proc,
            scaleFactor=1.15,
            minNeighbors=int(self.cfg.neighbors),
            minSize=(min_sz, min_sz),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        try:
            arr = np.asarray(raw if not isinstance(raw, tuple) else raw[0])
        except Exception:
            arr = np.empty((0, 4), dtype=int)
        if arr.size == 0:
            return []
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        boxes = []
        for r in arr:
            x, y, w, h = [int(v) for v in r[:4]]
            if w <= 0 or h <= 0:
                continue
            boxes.append((x, y, x + w, y + h))
        return boxes

    @staticmethod
    def _pick_best_face(boxes):
        if not boxes:
            return None
        # choose largest-area face
        def area(b):
            x1, y1, x2, y2 = b
            return max(1, (x2 - x1) * (y2 - y1))

        return max(boxes, key=area)

    def extract(self, whole_rgb: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Args:
            whole_rgb: HWC RGB uint8
        Returns:
            (face_rgb, eye_rgb) as RGB uint8 crops (or None if not found)
        """
        import cv2

        boxes = self._detect_faces(whole_rgb)
        face_box = self._pick_best_face(boxes)
        if face_box is None:
            return None, None

        x1, y1, x2, y2 = face_box
        H0, W0 = whole_rgb.shape[:2]
        x1 = max(0, min(W0, x1))
        x2 = max(0, min(W0, x2))
        y1 = max(0, min(H0, y1))
        y2 = max(0, min(H0, y2))
        if x2 <= x1 or y2 <= y1:
            return None, None

        face = whole_rgb[y1:y2, x1:x2].copy()

        # eye detection on upper ROI
        H, W = face.shape[:2]
        roi_h = int(H * float(self.cfg.eye_roi_frac))
        roi = face[0: max(1, roi_h), :]

        roi_small, s_roi = _shrink(roi, int(self.cfg.eye_downscale_limit_roi))
        face_small, s_face = _shrink(face, int(self.cfg.eye_downscale_limit_face))

        eyes_roi = self._detect_eyes_in_roi(roi_small)
        eyes_roi = [(int(a / s_roi), int(b / s_roi), int(c / s_roi), int(d / s_roi)) for (a, b, c, d) in eyes_roi]
        labs = _best_pair(eyes_roi, W, roi.shape[0])
        origin = "roi" if labs else None

        eyes_full = []
        if self.cfg.eye_fallback_to_face and (not labs):
            eyes_full = self._detect_eyes_in_roi(face_small)
            eyes_full = [(int(a / s_face), int(b / s_face), int(c / s_face), int(d / s_face)) for (a, b, c, d) in eyes_full]
            if len(eyes_full) >= 2:
                labs = _best_pair(eyes_full, W, H)
                origin = "face" if labs else origin

        if not labs:
            cand = eyes_roi
            cand_origin = "roi"
            if self.cfg.eye_fallback_to_face and len(eyes_full) >= 1:
                cand = eyes_full
                cand_origin = "face"
            if len(cand) >= 2:
                top2 = sorted(cand, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)[:2]
                top2 = sorted(top2, key=lambda b: (b[0] + b[2]))
                labs = [("left", top2[0]), ("right", top2[1])]
                origin = cand_origin
            elif len(cand) == 1:
                labs = [("left", cand[0])]
                origin = cand_origin

        eye_crop = None
        if labs:
            src_img = roi if origin == "roi" else face
            bound_h = roi.shape[0] if origin == "roi" else H
            mid_x = int(round(W / 2.0))

            # Build candidate eye boxes; split any box that crosses the midline
            candidates = []
            for tag, b in labs:
                candidates.extend(_split_box_by_midline(b, mid_x))

            # Deterministically choose the LEFT eye if present; otherwise fall back to largest
            left_boxes = [b for (t, b) in candidates if t == "left"]
            pick_from = left_boxes if left_boxes else [b for (_, b) in candidates]
            chosen = max(pick_from, key=lambda bb: max(1, (bb[2] - bb[0]) * (bb[3] - bb[1])))

            # Square crop around the chosen eye (no stretching); pad to square to guarantee 1:1.
            scale = 1.0 + float(self.cfg.eye_margin)
            sq = _square_box_from_rect(chosen, scale=scale, W=W, H=bound_h)
            if sq is not None:
                ex1, ey1, ex2, ey2 = sq
                crop = src_img[ey1:ey2, ex1:ex2]
                if crop.size > 0 and min(crop.shape[0], crop.shape[1]) >= int(self.cfg.eye_min_size):
                    eye_crop = _pad_to_square_rgb(crop.copy())

        return face, eye_crop


