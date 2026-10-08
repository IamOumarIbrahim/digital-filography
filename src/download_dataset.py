#!/usr/bin/env python3
"""download_dataset.py: Download diverse, balanced COCO 2017 validation images using FiftyOne.

Selectively downloads a curated, balanced set of 150 COCO images spanning multiple
object classes, scales, occlusions, and backgrounds, meeting target distribution
quotas while preventing person dominance.

Appends width and height pixel counts to dataset/manifest.csv and dataset/manifest.json.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


TARGET_CLASSES = [
    "person",
    "car", "truck", "bus",
    "chair", "couch", "dining table",
    "bottle", "cup", "bowl", "wine glass",
    "bird", "dog", "cat", "horse",
    "umbrella", "handbag", "backpack",
    "bicycle", "motorcycle",
    "traffic light", "airplane", "boat",
]

CATEGORIES = {
    "person": ["person"],
    "vehicle": ["car", "truck", "bus"],
    "indoor_geometry": ["chair", "couch", "dining table"],
    "small_objects": ["bottle", "cup", "bowl", "wine glass"],
    "animals": ["dog", "cat", "horse", "bird"],
    "false_positives": ["umbrella", "handbag", "backpack"],
    "thin_structures": ["bicycle", "motorcycle"],
    "other": ["traffic light", "airplane", "boat"],
}

TARGET_RANGES = {
    "person": {"target_imgs": (40, 46), "target_inst": (180, 220), "label": "person"},
    "vehicle": {"target_imgs": (25, 30), "target_inst": (90, 110), "label": "car / truck / bus"},
    "indoor_geometry": {"target_imgs": (20, 25), "target_inst": (70, 90), "label": "chair / couch / dining table"},
    "small_objects": {"target_imgs": (20, 25), "target_inst": (60, 80), "label": "bottle / cup / bowl / wine glass"},
    "animals": {"target_imgs": (25, 32), "target_inst": (60, 80), "label": "dog / cat / horse / bird"},
    "false_positives": {"target_imgs": (10, 15), "target_inst": (30, 40), "label": "umbrella / handbag / backpack"},
    "thin_structures": {"target_imgs": (10, 12), "target_inst": (25, 35), "label": "bicycle / motorcycle"},
    "other": {"target_imgs": (10, 15), "target_inst": (20, 36), "label": "Other (traffic light, etc.)"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download balanced diverse COCO test images for Digital Filography."
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default="dataset",
        help="Export folder for images (default: dataset)",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=380,
        help="Candidate pool size in FiftyOne Zoo (default: 380)",
    )
    parser.add_argument(
        "--final-samples",
        type=int,
        default=150,
        help="Final number of balanced samples to select (default: 150)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="COCO split (default: validation)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for repeatable sampling (default: 42)",
    )
    return parser.parse_args()


def optimize_balanced_subset(
    pool: list[dict[str, Any]],
    target_count: int = 150,
    seed: int = 42,
    restarts: int = 25,
    steps_per_restart: int = 5000,
) -> list[dict[str, Any]]:
    """Select exactly target_count images minimizing deviation from class and instance targets."""
    all_idx = list(range(len(pool)))
    pool_set = set(all_idx)

    best_subset_indices: set[int] | None = None
    best_loss = float("inf")

    def calc_loss(subset: set[int]) -> tuple[float, dict[str, int], dict[str, int]]:
        loss = 0.0
        img_cnts = {c: sum(1 for idx in subset if pool[idx]["counts"][c] > 0) for c in CATEGORIES}
        inst_cnts = {c: sum(pool[idx]["counts"][c] for idx in subset) for c in CATEGORIES}

        for c, spec in TARGET_RANGES.items():
            min_img, max_img = spec["target_imgs"]
            min_inst, max_inst = spec["target_inst"]

            if img_cnts[c] < min_img:
                loss += 35.0 * ((min_img - img_cnts[c]) ** 2)
            elif img_cnts[c] > max_img:
                loss += 35.0 * ((img_cnts[c] - max_img) ** 2)

            if inst_cnts[c] < min_inst:
                loss += 2.5 * ((min_inst - inst_cnts[c]) ** 2)
            elif inst_cnts[c] > max_inst:
                loss += 2.5 * ((inst_cnts[c] - max_inst) ** 2)

        return loss, img_cnts, inst_cnts

    for r_idx in range(restarts):
        rng = random.Random(seed + r_idx * 1337)
        cur_subset = set(rng.sample(all_idx, target_count))
        cur_loss, _, _ = calc_loss(cur_subset)

        for _ in range(steps_per_restart):
            in_elem = rng.choice(list(cur_subset))
            out_elem = rng.choice(list(pool_set - cur_subset))
            cand_subset = (cur_subset - {in_elem}) | {out_elem}
            cand_loss, _, _ = calc_loss(cand_subset)

            if cand_loss < cur_loss:
                cur_subset = cand_subset
                cur_loss = cand_loss
                if cur_loss < 10:
                    break

        if cur_loss < best_loss:
            best_loss = cur_loss
            best_subset_indices = cur_subset

    assert best_subset_indices is not None
    return [pool[idx] for idx in sorted(list(best_subset_indices))]


def main() -> int:
    args = parse_args()

    print("Initializing FiftyOne...")
    import fiftyone as fo
    import fiftyone.zoo as foz

    os.makedirs(args.export_dir, exist_ok=True)

    print("================================================================================")
    print("COCO 2017 Balanced Validation Dataset Downloader & Exporter")
    print(f"Export directory: {args.export_dir}")
    print(f"Candidate pool:   {args.pool_size}")
    print(f"Final samples:    {args.final_samples}")
    print(f"Random seed:      {args.seed}")
    print("================================================================================")

    # 1. Load candidate pool from FiftyOne zoo
    print(f"\n[1/4] Ensuring candidate pool of {args.pool_size} samples in FiftyOne Zoo...")
    pool_dataset = foz.load_zoo_dataset(
        "coco-2017",
        split=args.split,
        classes=TARGET_CLASSES,
        max_samples=args.pool_size,
        shuffle=True,
        seed=args.seed,
    )
    print(f"Candidate pool ready with {len(pool_dataset)} samples.")

    # 2. Extract metadata and category counts
    print("\n[2/4] Parsing candidate annotations and calculating category statistics...")
    pool_records = []
    for sample in pool_dataset:
        filepath = sample.filepath
        filename = os.path.basename(filepath)
        detections = []
        classes_in_sample: set[str] = set()
        cat_counts = {cat: 0 for cat in CATEGORIES}

        if hasattr(sample, "ground_truth") and sample.ground_truth is not None:
            for det in sample.ground_truth.detections:
                lbl = det.label.lower().strip()
                classes_in_sample.add(lbl)
                bbox = [round(float(x), 4) for x in det.bounding_box]
                detections.append({
                    "label": lbl,
                    "bbox_norm": bbox,
                    "confidence": float(getattr(det, "confidence", 1.0) or 1.0),
                })
                for cat_name, labels in CATEGORIES.items():
                    if lbl in labels:
                        cat_counts[cat_name] += 1
                        break

        pool_records.append({
            "sample": sample,
            "filepath": filepath,
            "filename": filename,
            "classes": sorted(list(classes_in_sample)),
            "num_objects": len(detections),
            "detections": detections,
            "counts": cat_counts,
        })

    # 3. Optimize balanced selection
    print(f"\n[3/4] Optimizing balanced selection for {args.final_samples} images...")
    selected = optimize_balanced_subset(
        pool_records,
        target_count=args.final_samples,
        seed=args.seed,
    )
    print(f"Selected {len(selected)} balanced images.")

    # 4. Clean export directory & copy selected images
    print(f"\n[4/4] Exporting images and appending pixel resolutions to manifest...")
    # Clean old images in export_dir
    for fname in os.listdir(args.export_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            os.remove(os.path.join(args.export_dir, fname))

    manifest_records = []
    csv_rows = []

    for item in selected:
        src_path = item["filepath"]
        dst_path = os.path.join(args.export_dir, item["filename"])
        shutil.copy2(src_path, dst_path)

        # Get exact image dimensions in pixels
        with Image.open(dst_path) as img:
            w_px, h_px = img.size

        manifest_records.append({
            "filename": item["filename"],
            "width": w_px,
            "height": h_px,
            "classes": item["classes"],
            "num_objects": item["num_objects"],
            "detections": item["detections"],
        })

        csv_rows.append({
            "filename": item["filename"],
            "classes": ", ".join(item["classes"]),
            "num_objects": item["num_objects"],
            "width": w_px,
            "height": h_px,
        })

    # Save manifest.json
    manifest_json_path = os.path.join(args.export_dir, "manifest.json")
    with open(manifest_json_path, "w", encoding="utf-8") as f:
        json.dump(manifest_records, f, indent=2)
    print(f"Saved manifest JSON: {manifest_json_path}")

    # Save manifest.csv
    manifest_csv_path = os.path.join(args.export_dir, "manifest.csv")
    df_manifest = pd.DataFrame(csv_rows)
    df_manifest.to_csv(manifest_csv_path, index=False, encoding="utf-8")
    print(f"Saved manifest CSV:  {manifest_csv_path}")

    # Print distribution summary
    print("\n" + "=" * 96)
    print(f"{'Object Category':<35} | {'Target Images':<14} | {'Actual Images':<14} | {'Target Inst':<12} | {'Actual Inst':<12}")
    print("=" * 96)

    cat_img_counts = {cat: 0 for cat in CATEGORIES}
    cat_inst_counts = {cat: 0 for cat in CATEGORIES}

    for item in selected:
        for cat, cnt in item["counts"].items():
            if cnt > 0:
                cat_img_counts[cat] += 1
                cat_inst_counts[cat] += cnt

    for cat, spec in TARGET_RANGES.items():
        t_imgs_str = f"{spec['target_imgs'][0]}–{spec['target_imgs'][1]}"
        a_imgs = cat_img_counts[cat]
        t_inst_str = f"{spec['target_inst'][0]}–{spec['target_inst'][1]}"
        a_inst = cat_inst_counts[cat]
        print(f"{spec['label']:<35} | {t_imgs_str:<14} | {a_imgs:<14} | {t_inst_str:<12} | {a_inst:<12}")

    print("=" * 96)
    print(f"Total Images: {len(selected)} | Verified in: {args.export_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
