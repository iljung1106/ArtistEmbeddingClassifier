#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
로컬용 얼굴→눈 추출 파이프라인
- detector: yolov5_anime (프로젝트 루트 바로 아래)
- weights:  ./yolov5s_anime.pt  (프로젝트 루트 바로 아래라고 가정)
- letterbox auto=False 호환
- GPU: 얼굴 / CPU: 눈/Haar/저장
- Colab 전용 경로, 드라이브 복사 제거
- 느리던 부분들 기본값 조금 완화

사용 예:
    python anime_face_eye_extract.py \
        --input ./dataset_raw \
        --out-face ./dataset_face \
        --out-eye ./dataset_eyes \
        --cascade ./anime-eyes-cascade.xml
"""

import os, sys, time, itertools, threading, shutil
from pathlib import Path
import argparse
import numpy as np
from queue import Queue

# OpenCV 내부 스레드 중첩 방지
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cv2
try:
    cv2.setNumThreads(1)
except Exception:
    pass

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# =========================================================
# 유틸
# =========================================================
def _iter_images(root: Path):
    for p in root.rglob("*"):
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"):
            yield p

def _read_rgb_pil(path: Path):
    with Image.open(path) as im:
        return np.array(im.convert("RGB"))

def _save_jpg(rgb, out_path: Path, quality=90):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ValueError("jpg encode failed")
    with open(out_path, "wb") as f:
        buf.tofile(f)

def _save_png(rgb, out_path: Path, level=2):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_PNG_COMPRESSION, int(level)])
    if not ok:
        raise ValueError("png encode failed")
    with open(out_path, "wb") as f:
        buf.tofile(f)

def _pre(gray):
    # 로컬에선 빠른 쪽으로
    # return cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def _shrink_for_eye(img, limit=900):
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= limit:
        return img, 1.0
    s = limit / float(m)
    nh, nw = int(h * s), int(w * s)
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return small, s

def _expand(box, margin, W, H):
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

# =========================================================
# torch 2.6+ pickle 우회
# =========================================================
def _patch_torch_load_for_old_ckpt():
    import torch, numpy as _np
    torch.serialization.add_safe_globals([
        _np.core.multiarray._reconstruct,
        _np.ndarray,
    ])
    _orig_load = torch.load

    def _patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)

    torch.load = _patched_load

# =========================================================
# 메인 파이프라인
# =========================================================
class Pipeline:
    def __init__(self, args):
        self.args = args
        self._model = None
        self._device = None
        self._stride = 32
        self._tl = threading.local()

    # ---------------------------
    # detector init
    # ---------------------------
    def _init_detector(self):
        ydir = Path(self.args.yolo_dir).resolve()
        if not ydir.exists():
            raise RuntimeError(f"yolov5_anime dir not found: {ydir}")
        sys.path.insert(0, str(ydir))

        _patch_torch_load_for_old_ckpt()

        import torch
        from models.experimental import attempt_load
        from utils.torch_utils import select_device

        self._device = select_device(self.args.device)
        self._model = attempt_load(str(self.args.weights), map_location=self._device)
        self._model.eval()

        self._stride = int(self._model.stride.max())
        imgz = self.args.imgsz
        if isinstance(imgz, int):
            imgz = (imgz, imgz)
        imgz = tuple(int(np.ceil(s / self._stride) * self._stride) for s in imgz)
        self.args.imgsz = imgz

    # ---------------------------
    # letterbox 호환
    # ---------------------------
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

    # ---------------------------
    # face batch detect
    # ---------------------------
    def _detect_faces_v5_batch(self, rgb_list):
        import torch
        from utils.general import non_max_suppression, scale_coords

        imgs = []
        shapes = []
        for rgb in rgb_list:
            img0 = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            shapes.append(img0.shape)
            img = self._letterbox_compat(img0, self.args.imgsz, self._stride)
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            imgs.append(img)

        im = np.stack(imgs, 0)
        im = torch.from_numpy(im).to(self._device)
        im = im.float() / 255.0
        if im.ndim == 3:
            im = im[None]

        with torch.no_grad():
            pred = self._model(im)[0]

        pred = non_max_suppression(
            pred,
            conf_thres=self.args.conf,
            iou_thres=self.args.iou,
            classes=None,
            agnostic=False,
        )

        outs = []
        for det, (h0, w0, _) in zip(pred, shapes):
            boxes = []
            if det is not None and len(det):
                det[:, :4] = scale_coords(self.args.imgsz, det[:, :4], (h0, w0)).round()
                for *xyxy, conf, cls in det.tolist():
                    x1, y1, x2, y2 = [int(v) for v in xyxy]
                    boxes.append((x1, y1, x2, y2))
            outs.append(boxes)
        return outs, rgb_list

    # ---------------------------
    # Haar
    # ---------------------------
    def _get_cascade(self):
        c = getattr(self._tl, "cascade", None)
        if c is None:
            c = cv2.CascadeClassifier(str(self.args.cascade))
            if c.empty():
                raise RuntimeError(f"cascade load fail: {self.args.cascade}")
            self._tl.cascade = c
        return c

    def _detect_eyes_in_roi(self, rgb_roi):
        gray = cv2.cvtColor(rgb_roi, cv2.COLOR_RGB2GRAY)
        proc = _pre(gray)

        H, W = proc.shape[:2]
        min_side = max(1, min(W, H))
        dyn_min = int(0.07 * min_side)
        min_sz = max(8, int(self.args.eye_min_size), dyn_min)

        cascade = self._get_cascade()
        raw = cascade.detectMultiScale(
            proc,
            scaleFactor=1.15,
            minNeighbors=self.args.neighbors,
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

    def _best_pair(self, boxes, W, H):
        clean = [(int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in boxes]
        if len(clean) < 2:
            return []
        def cxcy(b): x1,y1,x2,y2=b; return (x1+x2)/2.0, (y1+y2)/2.0
        def area(b): x1,y1,x2,y2=b; return max(1,(x2-x1)*(y2-y1))
        best=None; best_s=1e9
        for b1, b2 in itertools.combinations(clean, 2):
            c1x, c1y = cxcy(b1); c2x, c2y = cxcy(b2)
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
                best_s = s; best = (b1, b2)
        if best is None:
            return []
        b1, b2 = best
        left, right = (b1, b2) if (b1[0] + b1[2]) <= (b2[0] + b2[2]) else (b2, b1)
        return [("left", left), ("right", right)]

    # ---------------------------
    # run
    # ---------------------------
    def run(self):
        in_root = Path(self.args.input).resolve()
        out_face = Path(self.args.out_face).resolve()
        out_eye = Path(self.args.out_eye).resolve()

        if not in_root.exists():
            print("❌ input folder not found:", in_root)
            return
        if not Path(self.args.cascade).exists():
            print("❌ cascade not found:", self.args.cascade)
            return
        if not Path(self.args.yolo_dir).exists():
            print("❌ yolov5_anime dir not found:", self.args.yolo_dir)
            return
        if not Path(self.args.weights).exists():
            print("❌ yolov5_anime weights not found:", self.args.weights)
            return

        out_face.mkdir(parents=True, exist_ok=True)
        out_eye.mkdir(parents=True, exist_ok=True)

        files = list(_iter_images(in_root))
        if not files:
            print("⚠️ no images under", in_root)
            return

        self._init_detector()
        print(f"✅ yolov5_anime loaded: {self.args.weights}")

        total = len(files)
        print(
            f"🖼 {total} imgs | CPU threads={self.args.cpu_threads} | "
            f"READ_CHUNK={self.args.read_chunk} | batch={self.args.batch_size} | "
            f"imgsz={self.args.imgsz} | conf={self.args.conf} | iou={self.args.iou}"
        )

        saved = 0
        skip = 0
        fail = 0
        done = 0
        start_ts = time.time()
        counter_lock = threading.Lock()

        def _progress():
            if self.args.no_progress:
                return
            with counter_lock:
                _d, _s, _sk, _f = done, saved, skip, fail
            elapsed = max(1e-6, time.time() - start_ts)
            rate = _d / elapsed
            eta_min, eta_sec = 0, 0
            if rate > 0:
                eta = (total - _d) / rate
                eta_min = int(eta) // 60
                eta_sec = int(eta) % 60
            pct = (_d / total) * 100.0
            bar_len = 28
            filled = int(bar_len * _d / total)
            bar = "█" * filled + "·" * (bar_len - filled)
            sys.stdout.write(
                f"\r[{bar}] {pct:5.1f}%  done={_d}/{total}  saved={_s}  skipped={_sk}  "
                f"failed={_f}  {rate:5.1f} it/s  ETA {eta_min:02d}:{eta_sec:02d}"
            )
            sys.stdout.flush()

        cpu_q = Queue(maxsize=self.args.cpu_threads * 4)

        def postprocess_single(src: Path, rgb, faces_s):
            nonlocal saved, skip, fail, done
            start_local = time.time()
            try:
                if not faces_s:
                    with counter_lock:
                        done += 1
                        skip += 1
                    _progress()
                    return

                face_saved = 0
                for i, (x1, y1, x2, y2) in enumerate(faces_s):
                    if self.args.postprocess_timeout and (time.time() - start_local) > self.args.postprocess_timeout:
                        break

                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(rgb.shape[1], x2); y2 = min(rgb.shape[0], y2)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    face = rgb[y1:y2, x1:x2]
                    face_rel = src.parent.relative_to(in_root) if in_root in src.parents else Path(".")
                    face_dir_local = out_face / face_rel
                    face_dir_local.mkdir(parents=True, exist_ok=True)
                    stem = src.stem if i == 0 else f"{src.stem}_{i}"

                    face_local_path = face_dir_local / (f"{stem}.jpg" if self.args.save_jpg else f"{stem}.png")
                    if self.args.save_jpg:
                        _save_jpg(face, face_local_path, self.args.jpg_quality)
                    else:
                        _save_png(face, face_local_path, self.args.png_level)
                    face_saved += 1

                    # ===== 눈 =====
                    H, W = face.shape[:2]
                    roi_h = int(H * float(self.args.eye_roi_frac))
                    roi = face[0:roi_h, :]

                    roi_small, s_roi = _shrink_for_eye(roi, limit=self.args.eye_downscale_limit_roi)
                    face_small, s_face = _shrink_for_eye(face, limit=self.args.eye_downscale_limit_face)

                    eyes_roi = self._detect_eyes_in_roi(roi_small)
                    eyes_roi = [
                        (int(x1 / s_roi), int(y1 / s_roi), int(x2 / s_roi), int(y2 / s_roi))
                        for (x1, y1, x2, y2) in eyes_roi
                    ]
                    labs = self._best_pair(eyes_roi, W, roi_h)
                    origin = "roi" if labs else None

                    if self.args.eye_fallback_to_face and ((not labs) or len(labs) < 2):
                        eyes_full = self._detect_eyes_in_roi(face_small)
                        eyes_full = [
                            (int(x1 / s_face), int(y1 / s_face), int(x2 / s_face), int(y2 / s_face))
                            for (x1, y1, x2, y2) in eyes_full
                        ]
                        if len(eyes_full) >= 2:
                            labs = self._best_pair(eyes_full, W, H)
                            origin = "face" if labs else origin

                    if not labs:
                        cand = eyes_roi
                        cand_origin = "roi"
                        if self.args.eye_fallback_to_face:
                            # 위에서 eyes_full 만든 경우
                            if "eyes_full" in locals() and len(eyes_full) >= 1:
                                cand = eyes_full
                                cand_origin = "face"
                        if len(cand) >= 2:
                            top2 = sorted(
                                cand,
                                key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
                                reverse=True,
                            )[:2]
                            top2 = sorted(top2, key=lambda b: (b[0] + b[2]))
                            labs = [("left", top2[0]), ("right", top2[1])]
                            origin = cand_origin
                        elif len(cand) == 1:
                            labs = [("left", cand[0])]
                            origin = cand_origin

                    if labs:
                        cand_boxes = []
                        for _, box in labs:
                            x1b, y1b, x2b, y2b = [int(v) for v in box]
                            cand_boxes.append((x1b, y1b, x2b, y2b))
                        labeled = []
                        if len(cand_boxes) >= 2:
                            cand_boxes = sorted(cand_boxes, key=lambda b: (b[0] + b[2]))[:2]
                            labeled = [("left", cand_boxes[0]), ("right", cand_boxes[1])]
                        elif len(cand_boxes) == 1:
                            labeled = [("left", cand_boxes[0])]

                        if labeled:
                            src_img = roi if origin == "roi" else face
                            bound_h = roi_h if origin == "roi" else H
                            eye_dir_local = out_eye / face_rel
                            eye_dir_local.mkdir(parents=True, exist_ok=True)
                            for label, (bx1, by1, bx2, by2) in labeled:
                                ex1, ey1, ex2, ey2 = _expand(
                                    (bx1, by1, bx2, by2),
                                    self.args.eye_margin,
                                    W,
                                    bound_h,
                                )
                                crop = src_img[ey1:ey2, ex1:ex2]
                                if crop.size == 0 or min(crop.shape[0], crop.shape[1]) < self.args.eye_min_size:
                                    continue
                                eye_local_path = eye_dir_local / (
                                    f"{stem}_{label}.jpg" if self.args.save_jpg else f"{stem}_{label}.png"
                                )
                                if self.args.save_jpg:
                                    _save_jpg(crop, eye_local_path, self.args.jpg_quality)
                                else:
                                    _save_png(crop, eye_local_path, self.args.png_level)

                with counter_lock:
                    done += 1
                    if face_saved == 0:
                        skip += 1
                    else:
                        saved += 1
                _progress()

            except Exception as e:
                with counter_lock:
                    done += 1
                    fail += 1
                print("\n❗", f"{src} :: {e}")
                _progress()

        def cpu_worker():
            while True:
                item = cpu_q.get()
                if item is None:
                    cpu_q.task_done()
                    break
                src, rgb, faces_s = item
                postprocess_single(src, rgb, faces_s)
                cpu_q.task_done()

        workers = []
        for _ in range(self.args.cpu_threads):
            t = threading.Thread(target=cpu_worker, daemon=True)
            t.start()
            workers.append(t)

        for i in range(0, total, self.args.read_chunk):
            chunk_paths = files[i : i + self.args.read_chunk]
            rgbs = []
            ok_paths = []
            for p in chunk_paths:
                try:
                    rgbs.append(_read_rgb_pil(p))
                    ok_paths.append(p)
                except Exception as e:
                    with counter_lock:
                        fail += 1
                        done += 1
                    print("\n❗", f"{p} :: read error {e}")
                    _progress()

            if not rgbs:
                continue

            for j in range(0, len(rgbs), self.args.batch_size):
                sub_rgbs = rgbs[j : j + self.args.batch_size]
                sub_paths = ok_paths[j : j + self.args.batch_size]
                faces_list, _ = self._detect_faces_v5_batch(sub_rgbs)
                for pth, rgb, faces_s in zip(sub_paths, sub_rgbs, faces_list):
                    cpu_q.put((pth, rgb, faces_s))

            _progress()

        cpu_q.join()
        for _ in workers:
            cpu_q.put(None)
        for t in workers:
            t.join()

        if not self.args.no_progress:
            sys.stdout.write("\n")
        print(f"✅ Done. saved={saved}, skipped={skip}, failed={fail}")

# =========================================================
# CLI
# =========================================================
def parse_args():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Anime face → eyes extractor (local)")
    parser.add_argument("--input", type=str, required=True,
                        help="입력 이미지 루트 폴더")
    parser.add_argument("--out-face", type=str, default=str(base / "dataset_face"),
                        help="얼굴 저장 폴더")
    parser.add_argument("--out-eye", type=str, default=str(base / "dataset_eyes"),
                        help="눈 저장 폴더")
    parser.add_argument("--yolo-dir", type=str, default=str(base / "yolov5_anime"),
                        help="yolov5_anime 디렉터리")
    parser.add_argument("--weights", type=str, default=str(base / "yolov5x_anime.pt"),
                        help="yolo anime weights(.pt)")
    parser.add_argument("--cascade", type=str, default=str(base / "anime-eyes-cascade.xml"),
                        help="Haar cascade xml 경로")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0",
                        help="GPU 번호 혹은 cpu")
    parser.add_argument("--eye-roi-frac", type=float, default=0.70)
    parser.add_argument("--eye-min-size", type=int, default=12)
    parser.add_argument("--eye-margin", type=float, default=0.60)
    parser.add_argument("--neighbors", type=int, default=9)
    parser.add_argument("--eye-fallback-to-face", action="store_true",
                        help="ROI 실패시 얼굴 전체에서 다시 탐지")
    parser.add_argument("--eye-downscale-limit-roi", type=int, default=512)
    parser.add_argument("--eye-downscale-limit-face", type=int, default=768)
    parser.add_argument("--read-chunk", type=int, default=32,
                        help="한 번에 로딩할 이미지 개수 (I/O 부담 줄이기)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="GPU 추론 배치 사이즈")
    parser.add_argument("--cpu-threads", type=int, default=(os.cpu_count()),
                        help="CPU 후처리 워커 수")
    parser.add_argument("--save-jpg", action="store_true", default=True)
    parser.add_argument("--jpg-quality", type=int, default=80)
    parser.add_argument("--png-level", type=int, default=2)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--postprocess-timeout", type=float, default=15.0,
                        help="이미지 1장당 후처리 시간 제한(초). 0이면 무제한.")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    pipe = Pipeline(args)
    pipe.run()

if __name__ == "__main__":
    main()


#python anime_face_eye_extract.py --input ./dataset --out-face ./dataset_face --out-eye ./dataset_eyes