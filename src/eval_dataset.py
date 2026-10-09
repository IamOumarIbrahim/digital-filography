#!/usr/bin/env python3
"""eval_dataset.py: Full-dataset benchmark of digital filography against detector baselines.

For every image in the dataset manifest and every detector model:
- The model is run on the uncompressed (resized) photo. Its detections are that
  model's own baseline.
- Filography reconstructions are rendered at many thread counts and in 6 packed
  storage formats and run through the same model. Every thread matrix is really
  packed into bytes and decoded again before it is drawn (see packing.py and
  packing_uint7.py):

      uniform_fp16   uniform_uint8   uniform_uint7    all 8 values at one precision
      adjusted_fp16  adjusted_uint8  adjusted_uint7   coordinates at that precision,
                                                      colour + alpha as RGBA4444
- Each reconstruction is scored against the same model's baseline:

      recovery_rate = matched baseline objects / baseline objects
      precision     = matched baseline objects / predicted objects
      f1            = 2 * matched / (baseline objects + predicted objects)

  A prediction matches a baseline object when the class name is equal and
  IoU >= 0.50. Matching is greedy (highest confidence first) and each baseline
  object can be matched once, so extra detections count as false positives.
  Summary rows pool object counts across images (micro average).

What is deliberately NOT reported:
- No mAP. Matching at IoU 0.50 first and then averaging IoUs is not mAP.
  Add COCO AP separately if it is needed.
- No inference time. Placement workers compete with the detector for CPU and
  GPU, so the timings are not meaningful.
- "Recovery" is relative to each model's own output on the original photo. It
  measures how much of what the model sees in the original survives, not
  accuracy against COCO labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from itertools import islice
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from ultralytics import RTDETR, YOLO, YOLOWorld

# Add src to sys.path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from packing import bytes_per_thread as _bytes_per_thread_old
from packing import quantize_threads as _quantize_threads_old
from packing_uint7 import UINT7_FORMATS
from packing_uint7 import bytes_per_thread as _bytes_per_thread_7
from packing_uint7 import quantize_threads as _quantize_threads_7
from threader import ThreadPainter, load_image, render_threads

# The uint7 formats replace the uint6 ones. fp16 and uint8 still go through the
# original packing module, so their results are identical to earlier runs.
FORMATS = (
    "uniform_fp16", "uniform_uint8", "uniform_uint7",
    "adjusted_fp16", "adjusted_uint8", "adjusted_uint7",
)


def bytes_per_thread(tier: str) -> float:
    """Storage per thread in bytes for any supported format."""
    if tier in UINT7_FORMATS:
        return _bytes_per_thread_7(tier)
    return _bytes_per_thread_old(tier)


def quantize_threads(threads: np.ndarray, tier: str) -> tuple[np.ndarray, int]:
    """Pack threads into bytes and decode them again. Returns (decoded, packed byte count)."""
    if tier in UINT7_FORMATS:
        return _quantize_threads_7(threads, tier)
    return _quantize_threads_old(threads, tier)


SCHEMA_VERSION = 4
# One "tier" is one packed storage format. The CSV column is still called
# `quantization` so generate_graphs.py keeps working.
TIERS = FORMATS
BYTES_PER_THREAD = {tier: bytes_per_thread(tier) for tier in TIERS}
FLOP_CANDIDATES_PER_THREAD = 304
IOU_MATCH = 0.50
RELIABLE_RECOVERY = 0.50
# The end-of-run summary reports this thread count. The default grid
# (250..10000, step 250) contains it. If a custom grid omits it, the largest
# thread count in the grid is used instead of crashing.
SUMMARY_THREAD_COUNT = 10000


# -----------------------------------------------------------------------------
# Geometry, matching and quantization
# -----------------------------------------------------------------------------
def compute_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Compute Intersection over Union (IoU) between two boxes [x1, y1, x2, y2]."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter_w = max(0.0, xb - xa)
    inter_h = max(0.0, yb - ya)
    inter_area = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    if union_area <= 1e-9:
        return 0.0
    return float(inter_area / union_area)


def match_detections(
    preds: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    iou_thr: float = IOU_MATCH,
) -> tuple[list[float], list[float]]:
    """Greedily match predictions to baseline objects (same class, IoU >= iou_thr).

    Predictions are visited in descending confidence order and each baseline
    object is matched at most once. Returns (matched_ious, matched_confs); the
    number of true positives is len(matched_ious). Unmatched predictions are
    false positives.
    """
    matched_ious: list[float] = []
    matched_confs: list[float] = []
    used: set[int] = set()

    for p in sorted(preds, key=lambda d: d["conf"], reverse=True):
        best_iou = 0.0
        best_idx = -1
        for i, b in enumerate(baseline):
            if i in used or b["cls_name"] != p["cls_name"]:
                continue
            iou = compute_iou(p["box"], b["box"])
            if iou >= iou_thr and iou > best_iou:
                best_iou = iou
                best_idx = i
        if best_idx >= 0:
            used.add(best_idx)
            matched_ious.append(best_iou)
            matched_confs.append(p["conf"])

    return matched_ious, matched_confs


# -----------------------------------------------------------------------------
# Model loading and inference
# -----------------------------------------------------------------------------
def load_model(path: str):
    """Load a YOLO, RT-DETR or YOLO-World checkpoint based on the file name."""
    name = os.path.basename(path).lower()
    if name.startswith("rtdetr"):
        return RTDETR(path)
    if "world" in name:
        m = YOLOWorld(path)
        m.set_classes(["person", "car", "bus", "truck", "motorcycle", "bicycle"])
        return m
    return YOLO(path)


def class_name(names: Any, cid: int) -> str:
    """Resolve a class id to a lowercase name for dict or list style name tables."""
    if isinstance(names, dict):
        return str(names.get(cid, cid)).lower()
    try:
        return str(names[cid]).lower()
    except (IndexError, KeyError, TypeError):
        return str(cid)


def fresh_predict(
    model: Any,
    image: Image.Image | np.ndarray,
    device: str,
    conf: float = 0.25,
) -> Any:
    """Run stateless single-image inference and return the Results object."""
    results = model.predict(
        source=image,
        device=device,
        conf=conf,
        stream=False,
        verbose=False,
    )
    return results[0]


def extract_detections(res: Any) -> list[dict[str, Any]]:
    """Convert an Ultralytics Results object into a confidence-sorted detection list."""
    boxes = res.boxes
    if boxes is None or len(boxes) == 0:
        return []
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy()
    dets = [
        {
            "cls_name": class_name(res.names, int(cid)),
            "conf": float(c),
            "box": [float(v) for v in box],
        }
        for cid, c, box in zip(cls_ids, confs, xyxy)
    ]
    dets.sort(key=lambda d: d["conf"], reverse=True)
    return dets


def fmt_dets(dets: list[dict[str, Any]]) -> str:
    if not dets:
        return "none"
    return ", ".join(f"{d['cls_name']} ({d['conf']:.2f})" for d in dets)


# -----------------------------------------------------------------------------
# Row building and aggregation
# -----------------------------------------------------------------------------
def build_row(
    *,
    image_id: int,
    filename: str,
    model: str,
    image_type: str,
    thread_count: int,
    quantization: str,
    n_base: int,
    n_pred: int,
    n_tp: int,
    mean_conf: float,
    mean_iou: float,
    psnr: float | None,
    all_dets: str,
) -> dict[str, Any]:
    """Build one result row. recovery_rate/precision are None when undefined."""
    bpt = BYTES_PER_THREAD.get(quantization, 0)
    return {
        "image_id": image_id,
        "filename": filename,
        "model": model,
        "image_type": image_type,
        "thread_count": thread_count,
        "quantization": quantization,
        "bytes_per_thread": bpt,
        "storage_kb": round(thread_count * bpt / 1000.0, 2),
        "baseline_objects": n_base,
        "predicted_objects": n_pred,
        "recovered_objects": n_tp,
        "recovery_rate": round(n_tp / n_base, 4) if n_base else None,
        "precision": round(n_tp / n_pred, 4) if n_pred else None,
        "confidence": round(mean_conf, 4),
        "iou": round(mean_iou, 4),
        "psnr_db": psnr,
        "all_detections": all_dets,
    }


def _r(x: float | None, nd: int = 4) -> float | None:
    return None if x is None else round(float(x), nd)


def aggregate_group(model: str, count: int, tier: str, sub: list[dict[str, Any]]) -> dict[str, Any]:
    """Micro-average one (model, thread_count, tier) group across images."""
    n_base = sum(r["baseline_objects"] for r in sub)
    n_pred = sum(r["predicted_objects"] for r in sub)
    n_tp = sum(r["recovered_objects"] for r in sub)

    recovery = n_tp / n_base if n_base else None
    precision = n_tp / n_pred if n_pred else None
    f1 = 2.0 * n_tp / (n_base + n_pred) if (n_base + n_pred) else None
    mean_conf = sum(r["confidence"] * r["recovered_objects"] for r in sub) / n_tp if n_tp else None
    mean_iou = sum(r["iou"] * r["recovered_objects"] for r in sub) / n_tp if n_tp else None

    return {
        "model": model,
        "thread_count": count,
        "quantization": tier,
        "bytes_per_thread": BYTES_PER_THREAD[tier],
        "storage_kb": sub[0]["storage_kb"],
        "recovery_rate": _r(recovery),
        "precision": _r(precision),
        "f1": _r(f1),
        "mean_confidence": _r(mean_conf),
        "mean_iou": _r(mean_iou),
        "mean_psnr_db": round(float(np.mean([r["psnr_db"] for r in sub])), 2),
        "baseline_objects": n_base,
        "predicted_objects": n_pred,
        "recovered_objects": n_tp,
        "images_evaluated": len(sub),
        "images_with_baseline": sum(1 for r in sub if r["baseline_objects"] > 0),
    }


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


# -----------------------------------------------------------------------------
# Checkpointing
# -----------------------------------------------------------------------------
def read_checkpoint(path: str) -> tuple[dict[str, Any] | None, dict[int, list[dict[str, Any]]]]:
    """Read a checkpoint file. Returns (meta, finished_rows_by_image).

    Rows only count once their image's `_done` marker is present. Duplicate rows
    (from a crash and re-run of the same image) are collapsed by key.
    """
    meta: dict[str, Any] | None = None
    pending: dict[int, dict[tuple[str, int, str], dict[str, Any]]] = {}
    finished: dict[int, list[dict[str, Any]]] = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # last line may be cut off after a crash
            if "_meta" in rec:
                meta = rec["_meta"]
            elif "_done" in rec:
                i = rec["_done"]
                finished[i] = list(pending.pop(i, {}).values())
            else:
                key = (rec["model"], rec["thread_count"], rec["quantization"])
                pending.setdefault(rec["image_id"], {})[key] = rec
    return meta, finished


# Top-level worker function for multiprocessing on Windows
def worker_thread_placement(
    args_tuple: tuple[int, str, int, Sequence[float], float, float, int, list[int]],
) -> tuple[int, str, int, int, dict[int, np.ndarray]]:
    idx, img_path, size, bg, thread_width, alpha, seed, counts = args_tuple
    im = load_image(img_path, size, bg)
    w, h = im.size
    target = np.asarray(im, dtype=np.float32) / 255.0
    painter = ThreadPainter(target, alpha, bg, thread_width, seed)
    snaps: dict[int, np.ndarray] = {}
    painter.run(counts, lambda n, t, c: snaps.update({n: t.copy()}))
    snaps[0] = np.zeros((0, 8), dtype=np.float32)  # 0 threads = blank canvas
    return idx, img_path, w, h, snaps


def bounded_results(executor: ProcessPoolExecutor, tasks: Sequence[Any], window: int):
    """Yield worker results as they finish, with at most `window` tasks queued or waiting.

    Submitting every task at once and keeping every future alive kept all thread
    snapshots (about 6.5 MB per image) in RAM until the end of the run. Here a new
    task is only submitted when an earlier one has finished, and a finished future
    is dropped as soon as its result has been handed to the caller.
    """
    task_iter = iter(tasks)
    in_flight = {executor.submit(worker_thread_placement, t) for t in islice(task_iter, window)}
    while in_flight:
        done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
        for fut in done:
            nxt = next(task_iter, None)
            if nxt is not None:
                in_flight.add(executor.submit(worker_thread_placement, nxt))
            yield fut.result()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark digital filography reconstructions against each detector's own baseline."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="dataset",
        help="Path to folder containing COCO validation images and manifest",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Path to output directory for CSV files",
    )
    parser.add_argument(
        "--graphs-dir",
        type=str,
        default="graphs",
        help="Path to graphs directory for visual outputs",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["yolov8n.pt", "yolov10n.pt", "yolo11n.pt", "yolo12n.pt", "yolo26n.pt"],
        help="Model checkpoint paths (YOLO, rtdetr-*.pt, or *world*.pt)",
    )
    parser.add_argument(
        "--min-threads",
        type=int,
        default=250,
        help="Minimum thread count (default: 250)",
    )
    parser.add_argument(
        "--max-threads",
        type=int,
        default=10000,
        help="Maximum thread count (default: 10000)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=250,
        help="Thread count step increment (default: 250)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=640,
        help="Longest dimension canvas size in pixels (default: 640)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device: cuda or cpu",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(10, os.cpu_count() or 4),
        help="CPU worker processes for parallel thread placement (default: 10)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.graphs_dir, exist_ok=True)

    manifest_json_path = os.path.join(args.dataset_dir, "manifest.json")
    if not os.path.exists(manifest_json_path):
        print(f"Error: manifest not found at {manifest_json_path}", file=sys.stderr)
        return 1

    with open(manifest_json_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    image_files = [
        os.path.join(args.dataset_dir, item["filename"])
        for item in manifest_data
        if os.path.exists(os.path.join(args.dataset_dir, item["filename"]))
    ]

    total_images = len(image_files)
    thread_counts = list(range(args.min_threads, args.max_threads + 1, args.step))
    if total_images == 0 or not thread_counts:
        print("Error: no images found or empty thread-count grid.", file=sys.stderr)
        return 1

    summary_count = SUMMARY_THREAD_COUNT if SUMMARY_THREAD_COUNT in thread_counts else max(thread_counts)
    model_names = [os.path.splitext(os.path.basename(mf))[0] for mf in args.models]

    print("================================================================================")
    print("Full-Dataset Multi-Model Digital Filography Benchmark")
    print(f"Dataset images:   {total_images} (from {args.dataset_dir})")
    print(f"Models:           {args.models}")
    print(f"Device:           {args.device}")
    print(f"Confidence:       {args.conf}")
    print(f"Thread counts:    {len(thread_counts)} steps ({args.min_threads}..{args.max_threads}, step {args.step})")
    print(f"Packed formats:   {list(TIERS)} ({len(TIERS)} formats)")
    print("Bytes / thread:   " + ", ".join(f"{t}={BYTES_PER_THREAD[t]}" for t in TIERS))
    print(f"CPU Workers:      {args.workers} processes")
    print(f"Total per image:  {1 + len(thread_counts) * len(TIERS)} configs x {len(args.models)} models")
    print(f"Total inferences: {total_images * (1 + len(thread_counts) * len(TIERS)) * len(args.models):,}")
    if SUMMARY_THREAD_COUNT not in thread_counts:
        print(f"Warning: {SUMMARY_THREAD_COUNT} is not in the thread grid; the end summary will use N={summary_count}.")
    print("================================================================================")

    bg = (1.0, 1.0, 1.0)
    thread_width = 2.0
    alpha = 1.0
    seed = 0

    # --- checkpoint / resume (validated BEFORE any heavy work) ---
    signature = {
        "schema": SCHEMA_VERSION,
        "models": model_names,
        "thread_counts": thread_counts,
        "tiers": list(TIERS),
        "conf": args.conf,
        "size": args.size,
        "images": total_images,
        # Same image count but a different download must not resume an old checkpoint.
        "dataset_sha1": hashlib.sha1(
            "\n".join(sorted(os.path.basename(p) for p in image_files)).encode("utf-8")
        ).hexdigest()[:12],
    }
    ckpt_path = os.path.join(args.output_dir, "checkpoint_rows.jsonl")
    all_filo_rows: list[dict[str, Any]] = []
    # Accumulators for generation.csv (PSNR is identical across models)
    gen_metrics_accum: dict[tuple[int, str], dict[str, float]] = {
        (cnt, tier): {"psnr_sum": 0.0, "count": 0}
        for cnt in thread_counts for tier in TIERS
    }
    done_images: set[int] = set()

    if os.path.exists(ckpt_path):
        stored_sig, finished = read_checkpoint(ckpt_path)
        if stored_sig != signature:
            print(
                f"\nError: {ckpt_path} was written by a different run configuration or an older schema.\n"
                "Delete the output directory (or just that file) and start again.",
                file=sys.stderr,
            )
            return 1
        expected_rows = len(thread_counts) * len(TIERS) * len(model_names)
        for i, rows in finished.items():
            if len(rows) != expected_rows:
                continue  # incomplete image, redo it
            done_images.add(i)
            seen: set[tuple[int, str]] = set()
            for r in rows:
                all_filo_rows.append(r)
                key = (r["thread_count"], r["quantization"])
                if key not in seen:
                    seen.add(key)
                    g = gen_metrics_accum[key]
                    g["psnr_sum"] += r["psnr_db"]
                    g["count"] += 1
        print(f"Resuming: {len(done_images)} images already completed in {ckpt_path}")
    else:
        with open(ckpt_path, "w", encoding="utf-8") as ck:
            ck.write(json.dumps({"_meta": signature}) + "\n")

    # 1. Load models and establish model-relative baselines
    print("\n[Phase 1/4] Initializing models and establishing model-relative baselines...")
    models_dict: dict[str, Any] = {}
    for mf, mname in zip(args.models, model_names):
        models_dict[mname] = load_model(mf)

    # baseline_dets[img_idx][model_name] = list of detection dicts
    baseline_dets: dict[int, dict[str, list[dict[str, Any]]]] = {}
    baseline_rows: list[dict[str, Any]] = []

    t_base_start = time.time()
    for idx, img_path in enumerate(image_files):
        base_pil = load_image(img_path, args.size, bg)
        fname = os.path.basename(img_path)
        baseline_dets[idx] = {}

        for mname, m in models_dict.items():
            res = fresh_predict(m, base_pil, device=args.device, conf=args.conf)
            dets = extract_detections(res)
            baseline_dets[idx][mname] = dets
            baseline_rows.append(build_row(
                image_id=idx,
                filename=fname,
                model=mname,
                image_type="baseline_original",
                thread_count=0,
                quantization="none",
                n_base=len(dets),
                n_pred=len(dets),
                n_tp=len(dets),
                mean_conf=float(np.mean([d["conf"] for d in dets])) if dets else 0.0,
                mean_iou=1.0 if dets else 0.0,
                psnr=None,
                all_dets=fmt_dets(dets),
            ))

        if (idx + 1) % 25 == 0 or idx == total_images - 1:
            print(f"  Established baselines for {idx + 1}/{total_images} images ({time.time() - t_base_start:.1f}s)")

    print(f"Baseline phase complete in {time.time() - t_base_start:.1f}s.")

    # 2. Multi-core thread placement and filography inference
    print(f"\n[Phase 2/4] Executing multi-core thread placement ({args.workers} workers) and evaluation...")
    worker_tasks = [
        (idx, img_path, args.size, bg, thread_width, alpha, seed, thread_counts)
        for idx, img_path in enumerate(image_files)
    ]

    session_total = total_images - len(done_images)
    t_eval_start = time.time()
    completed_images = 0

    todo = [t for t in worker_tasks if t[0] not in done_images]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for idx, img_path, w, h, snaps in bounded_results(executor, todo, window=args.workers * 2):
            completed_images += 1
            fname = os.path.basename(img_path)

            # Load target image for PSNR calculation
            base_pil = load_image(img_path, args.size, bg)
            target_norm = np.asarray(base_pil, dtype=np.float32) / 255.0

            img_start_len = len(all_filo_rows)
            # One running canvas per storage format. Thread counts go up, so each
            # step only draws the new threads on top of the previous canvas. The
            # pixels are identical to drawing all N threads from scratch.
            canvases = {
                tier: np.full((h, w, 3), np.asarray(bg, dtype=np.float32).reshape(1, 1, 3), dtype=np.float32)
                for tier in TIERS
            }
            drawn = 0
            for count in thread_counts:
                raw_threads = snaps[count]

                for tier in TIERS:
                    # Real round trip: floats -> packed bytes -> floats.
                    q_threads, packed_bytes = quantize_threads(raw_threads, tier)
                    # 5.5 bytes per thread is not a whole number, so compare against
                    # the size rounded up to a whole byte.
                    expected_bytes = math.ceil(count * BYTES_PER_THREAD[tier])
                    if packed_bytes != expected_bytes:
                        raise RuntimeError(
                            f"{tier}: packed {count} threads into {packed_bytes} bytes, "
                            f"expected {expected_bytes}"
                        )
                    canvas = render_threads(q_threads[drawn:], w, h, bg, thread_width, canvas=canvases[tier])
                    pixels = np.clip(canvas * 255.0 + 0.5, 0, 255).astype(np.uint8)
                    filo_pil = Image.fromarray(pixels)

                    # PSNR against the original photo (same for every model)
                    mse = float(np.mean((canvas - target_norm) ** 2))
                    psnr = round(float(-10.0 * math.log10(max(mse, 1e-10))), 2)

                    gen_metrics_accum[(count, tier)]["psnr_sum"] += psnr
                    gen_metrics_accum[(count, tier)]["count"] += 1

                    for mname, m in models_dict.items():
                        base_list = baseline_dets[idx][mname]
                        res = fresh_predict(m, filo_pil, device=args.device, conf=args.conf)
                        preds = extract_detections(res)

                        matched_ious, matched_confs = match_detections(preds, base_list)

                        all_filo_rows.append(build_row(
                            image_id=idx,
                            filename=fname,
                            model=mname,
                            image_type="filography",
                            thread_count=count,
                            quantization=tier,
                            n_base=len(base_list),
                            n_pred=len(preds),
                            n_tp=len(matched_ious),
                            mean_conf=float(np.mean(matched_confs)) if matched_confs else 0.0,
                            mean_iou=float(np.mean(matched_ious)) if matched_ious else 0.0,
                            psnr=psnr,
                            all_dets=fmt_dets(preds),
                        ))

                drawn = count  # every format has now drawn threads 0..count-1

            with open(ckpt_path, "a", encoding="utf-8") as ck:
                ck.write("".join(json.dumps(r) + "\n" for r in all_filo_rows[img_start_len:]))
                ck.write(json.dumps({"_done": idx}) + "\n")

            elapsed = time.time() - t_eval_start
            rate = completed_images / max(elapsed, 1e-6)
            rem = (session_total - completed_images) / rate
            if completed_images % 5 == 0 or completed_images == session_total:
                print(
                    f"  Progress: {completed_images + len(done_images):3d}/{total_images} images completed "
                    f"({elapsed / 60.0:4.1f}m elapsed, ~{rem / 60.0:4.1f}m rem)"
                )

    print(f"\nAll {total_images} images successfully evaluated in {(time.time() - t_eval_start) / 60.0:.2f} minutes.")

    # 3. Export datasets and summaries
    print("\n[Phase 3/4] Exporting benchmark CSV datasets...")
    all_rows = baseline_rows + all_filo_rows
    fieldnames = list(all_rows[0].keys())

    # Master detailed CSV
    csv_master = os.path.join(args.output_dir, "yolo_benchmark_results.csv")
    with open(csv_master, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"  Saved master detailed CSV: {csv_master} ({len(all_rows):,} rows)")

    # Aggregated summary CSV (one row per model x thread count x tier)
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for r in all_filo_rows:
        groups[(r["model"], r["thread_count"], r["quantization"])].append(r)

    summary_rows: list[dict[str, Any]] = []
    summary_index: dict[tuple[str, int, str], dict[str, Any]] = {}
    for mname in model_names:
        for count in thread_counts:
            for tier in TIERS:
                sub = groups.get((mname, count, tier))
                if sub:
                    row = aggregate_group(mname, count, tier, sub)
                    summary_rows.append(row)
                    summary_index[(mname, count, tier)] = row

    csv_summary = os.path.join(args.output_dir, "dataset_evaluation_summary.csv")
    with open(csv_summary, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"  Saved dataset evaluation summary: {csv_summary} ({len(summary_rows):,} rows)")

    # Per-model detailed CSVs
    for mname in model_names:
        m_rows = [r for r in all_rows if r["model"] == mname]
        p_csv = os.path.join(args.output_dir, f"{mname}_results.csv")
        with open(p_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(m_rows)
        print(f"  Saved {mname} CSV: {p_csv} ({len(m_rows):,} rows)")

    # generation.csv across the dataset
    gen_rows = []
    pixels_m = args.size * args.size * 0.75 / 1e6  # reference 4:3 canvas, in megapixels
    flops_by_count = {c: c * FLOP_CANDIDATES_PER_THREAD * pixels_m for c in thread_counts}  # MFLOPs
    # NOTE: generation time is a fixed linear formula (0.5 s + 0.00265 s per thread),
    # not a value measured in this run.
    time_by_count = {c: round(0.5 + c * 0.00265, 1) for c in thread_counts}

    all_storages = [c * BYTES_PER_THREAD[t] / 1000.0 for c in thread_counts for t in TIERS]
    all_psnrs = [
        gen_metrics_accum[(c, t)]["psnr_sum"] / max(gen_metrics_accum[(c, t)]["count"], 1)
        for c in thread_counts for t in TIERS
    ]
    min_s, max_s = min(all_storages), max(all_storages)
    min_p, max_p = min(all_psnrs), max(all_psnrs)
    min_f, max_f = min(flops_by_count.values()), max(flops_by_count.values())
    min_t, max_t = min(time_by_count.values()), max(time_by_count.values())

    for count in thread_counts:
        flops = flops_by_count[count]
        flops_str = f"{flops / 1000.0:.2f} GFLOPs" if flops >= 1000 else f"{flops:.1f} MFLOPs"
        gen_time_s = time_by_count[count]

        for tier in TIERS:
            st_kb = count * BYTES_PER_THREAD[tier] / 1000.0
            avg_psnr = round(
                gen_metrics_accum[(count, tier)]["psnr_sum"] / max(gen_metrics_accum[(count, tier)]["count"], 1), 2
            )

            # Min-max normalized scores in [0, 1], higher is better
            score_s = (max_s - st_kb) / max(max_s - min_s, 1e-6)
            score_p = (avg_psnr - min_p) / max(max_p - min_p, 1e-6)
            score_t = (max_t - gen_time_s) / max(max_t - min_t, 1e-6)
            score_f = (max_f - flops) / max(max_f - min_f, 1e-6)

            avg_1 = round(100.0 * (score_s + score_t + score_f + score_p) / 4.0, 2)
            avg_2 = round(100.0 * (score_s + score_t + score_f + 2.0 * score_p) / 5.0, 2)
            avg_3 = round(100.0 * (score_s + score_t + score_f + 3.0 * score_p) / 6.0, 2)

            gen_rows.append({
                "Thread Count (N)": count,
                "Precision Tier": tier,
                "Bytes / Thread": f"{BYTES_PER_THREAD[tier]} B",
                "Storage (KB) \u2193": f"{st_kb:.2f} KB",
                "Generation Time \u2193": f"{gen_time_s:.1f} s",
                "Total Operations (FLOPs) \u2193": flops_str,
                "PSNR (vs Original) \u2191": f"{avg_psnr:.2f} dB",
                "Average (1:1:1:1) \u2191": avg_1,
                "Average (1:1:1:2) \u2191": avg_2,
                "Average (1:1:1:3) \u2191": avg_3,
            })

    csv_gen = os.path.join(args.output_dir, "generation.csv")
    with open(csv_gen, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(gen_rows[0].keys()))
        writer.writeheader()
        writer.writerows(gen_rows)
    print(f"  Saved dataset generation metrics: {csv_gen} ({len(gen_rows)} rows)")

    # 4. Regenerate graphs
    print("\n[Phase 4/4] Regenerating publication-quality graphs for the full dataset...")
    graph_cmd = [
        sys.executable,
        os.path.join(SRC_DIR, "generate_graphs.py"),
        "--benchmark-csv", csv_summary,
        "--generation-csv", csv_gen,
        "--output-dir", args.graphs_dir,
        "--dpi", "300",
        "--formats", "png", "svg",
    ]
    rc = subprocess.run(graph_cmd).returncode
    if rc != 0:
        print(f"\nWarning: graph generation exited with code {rc}. CSVs are saved. Re-run with:")
        print("  " + " ".join(f'"{c}"' if " " in c else c for c in graph_cmd))

    # 5. Concise console summary
    print("\n" + "=" * 90)
    print(f"DATASET BENCHMARK RESULTS SUMMARY (Model-relative recovery across {total_images} COCO images):")
    print("=" * 90)
    for mname in model_names:
        print(f"\n--- Architecture: {mname} ---")
        for tier in TIERS:
            tier_rows = sorted(
                (r for r in summary_rows if r["model"] == mname and r["quantization"] == tier),
                key=lambda r: r["thread_count"],
            )
            if not tier_rows:
                continue
            final = summary_index.get((mname, summary_count, tier))
            final_txt = (
                f"N={summary_count}: recovery {pct(final['recovery_rate'])}, "
                f"precision {pct(final['precision'])}, F1 {pct(final['f1'])}"
                if final else f"N={summary_count}: no data"
            )
            hits = [r for r in tier_rows if (r["recovery_rate"] or 0.0) >= RELIABLE_RECOVERY]
            if hits:
                first = hits[0]
                print(
                    f"  {tier:14s}: >=50% recovery first at N={first['thread_count']:5d} "
                    f"(precision {pct(first['precision'])}) | {final_txt}"
                )
            else:
                best = max(tier_rows, key=lambda r: r["recovery_rate"] or 0.0)
                print(
                    f"  {tier:14s}: never reached 50% recovery "
                    f"(peak {pct(best['recovery_rate'])} at N={best['thread_count']}) | {final_txt}"
                )

    print("\n================================================================================")
    print("Full-Dataset Benchmark & Visualizations Completed Successfully!")
    print("================================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())