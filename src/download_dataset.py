#!/usr/bin/env python3
"""download_dataset.py: Download the Phase 1 recognition test set (500 COCO images).

Phase 1 is a general recognition test, not a deployment for one specific use
case, so the set has NO per-class quota and is not limited to a few classes:

- Images come from the whole COCO 2017 label set (80 classes), not only
  person / vehicle images.
- Exactly --final-samples images are kept (default 500).
- Selection is a seeded random draw from a larger candidate pool, so the class
  mix follows what COCO photos really contain (some classes, such as person,
  will be common and others rare).
- One light safeguard keeps the set varied: every class that exists in the
  candidate pool must appear in at least --min-images-per-class images
  (default 1). This is a floor, not a quota. Use 0 to turn it off.
- Images with no labelled object are skipped, because they have no baseline.

Writes the images plus manifest.csv, manifest.json (file name, classes, object
count, width and height in pixels, detections) and class_summary.csv (how many
images and objects each class got) into the export folder.

Requires: fiftyone, numpy, pandas, Pillow
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter
from typing import Any

import pandas as pd
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a varied 500-image COCO test set for Digital Filography (Phase 1)."
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
        default=3000,
        help="Candidate pool size loaded from the FiftyOne Zoo (default: 3000)",
    )
    parser.add_argument(
        "--final-samples",
        type=int,
        default=500,
        help="Total number of images in the test set (default: 500)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="COCO split: validation or train (default: validation)",
    )
    parser.add_argument(
        "--min-images-per-class",
        type=int,
        default=1,
        help="Every class found in the pool appears in at least this many images; 0 disables (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for repeatable sampling (default: 42)",
    )
    return parser.parse_args()


def select_images(
    pool: list[dict[str, Any]],
    target_count: int,
    seed: int = 42,
    min_per_class: int = 1,
) -> list[dict[str, Any]]:
    """Pick exactly target_count records at random, then make sure no class is missing.

    Each record needs a "classes" list. There is no per-class quota. The only rule
    is the floor: every class in the pool gets at least min_per_class images (or
    all of its images if the pool has fewer). Missing classes are fixed by
    swapping one image of an already well-covered set for one that has the class.
    """
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    if len(pool) < target_count:
        raise ValueError(f"Pool has {len(pool)} usable images but {target_count} were requested; raise --pool-size.")

    rng = random.Random(seed)
    order = list(range(len(pool)))
    rng.shuffle(order)

    chosen: set[int] = set(order[:target_count])
    in_pool: Counter[str] = Counter(c for rec in pool for c in rec["classes"])
    counts: Counter[str] = Counter(c for i in chosen for c in pool[i]["classes"])

    def floor_for(cls: str) -> int:
        return min(min_per_class, in_pool[cls])

    if min_per_class > 0:
        for cls in sorted(in_pool, key=lambda c: (in_pool[c], c)):      # rarest classes first
            while counts[cls] < floor_for(cls):
                new = next((i for i in order if i not in chosen and cls in pool[i]["classes"]), None)
                victims = [
                    i for i in chosen
                    if cls not in pool[i]["classes"]
                    and all(counts[c] - 1 >= floor_for(c) for c in pool[i]["classes"])
                ]
                if new is None or not victims:
                    print(f"  Warning: could not raise class '{cls}' to {floor_for(cls)} image(s).")
                    break
                # Drop the image whose classes are the most over-represented.
                victim = max(victims, key=lambda i: (sum(counts[c] for c in pool[i]["classes"]), -order.index(i)))
                chosen.remove(victim)
                counts.subtract(pool[victim]["classes"])
                chosen.add(new)
                counts.update(pool[new]["classes"])

    assert len(chosen) == target_count
    return sorted((pool[i] for i in chosen), key=lambda rec: rec["filename"])


def main() -> int:
    args = parse_args()

    print("Initializing FiftyOne...")
    import fiftyone.zoo as foz

    os.makedirs(args.export_dir, exist_ok=True)

    print("================================================================================")
    print("COCO 2017 Phase 1 Recognition Test Set Downloader")
    print(f"Export directory:     {args.export_dir}")
    print(f"Split:                {args.split}")
    print(f"Candidate pool:       {args.pool_size}")
    print(f"Final samples:        {args.final_samples}")
    print(f"Min images per class: {args.min_images_per_class}")
    print(f"Random seed:          {args.seed}")
    print("================================================================================")

    # 1. Load candidate pool from the FiftyOne Zoo (all 80 COCO classes)
    print(f"\n[1/4] Ensuring candidate pool of {args.pool_size} samples in FiftyOne Zoo...")
    pool_dataset = foz.load_zoo_dataset(
        "coco-2017",
        split=args.split,
        label_types=["detections"],
        max_samples=args.pool_size,
        shuffle=True,
        seed=args.seed,
    )
    print(f"Candidate pool ready with {len(pool_dataset)} samples.")

    # 2. Extract metadata and class lists
    print("\n[2/4] Parsing candidate annotations...")
    pool_records: list[dict[str, Any]] = []
    skipped_unreadable = 0
    skipped_empty = 0
    for sample in pool_dataset:
        filepath = sample.filepath
        filename = os.path.basename(filepath)
        # Skip images corrupted by interrupted downloads
        try:
            with Image.open(filepath) as _img:
                _img.load()
        except Exception:
            print(f"  Skipping unreadable image: {filename}")
            skipped_unreadable += 1
            continue

        detections = []
        counts: Counter[str] = Counter()
        gt = getattr(sample, "ground_truth", None)
        if gt is not None:
            for det in gt.detections:
                lbl = det.label.lower().strip()
                counts[lbl] += 1
                detections.append({
                    "label": lbl,
                    "bbox_norm": [round(float(x), 4) for x in det.bounding_box],
                    "confidence": float(getattr(det, "confidence", 1.0) or 1.0),
                })

        if not detections:
            skipped_empty += 1      # nothing labelled, so nothing to recognise
            continue

        pool_records.append({
            "filepath": filepath,
            "filename": filename,
            "classes": sorted(counts),
            "class_counts": dict(counts),
            "num_objects": len(detections),
            "detections": detections,
        })

    print(f"Usable images: {len(pool_records)} (skipped {skipped_unreadable} unreadable, {skipped_empty} with no labels)")

    # 3. Select the test set
    print(f"\n[3/4] Selecting {args.final_samples} images...")
    selected = select_images(
        pool_records,
        target_count=args.final_samples,
        seed=args.seed,
        min_per_class=args.min_images_per_class,
    )
    print(f"Selected {len(selected)} images.")

    # 4. Clean export directory & copy selected images
    print("\n[4/4] Exporting images and writing the manifest...")
    for fname in os.listdir(args.export_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            os.remove(os.path.join(args.export_dir, fname))

    manifest_records = []
    csv_rows = []
    img_per_class: Counter[str] = Counter()
    obj_per_class: Counter[str] = Counter()

    for item in selected:
        dst_path = os.path.join(args.export_dir, item["filename"])
        shutil.copy2(item["filepath"], dst_path)

        with Image.open(dst_path) as img:
            w_px, h_px = img.size

        img_per_class.update(item["classes"])
        obj_per_class.update(item["class_counts"])

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

    manifest_json_path = os.path.join(args.export_dir, "manifest.json")
    with open(manifest_json_path, "w", encoding="utf-8") as f:
        json.dump(manifest_records, f, indent=2)
    print(f"Saved manifest JSON:  {manifest_json_path}")

    manifest_csv_path = os.path.join(args.export_dir, "manifest.csv")
    pd.DataFrame(csv_rows).to_csv(manifest_csv_path, index=False, encoding="utf-8")
    print(f"Saved manifest CSV:   {manifest_csv_path}")

    summary_rows = [
        {"class": cls, "images": img_per_class[cls], "objects": obj_per_class[cls]}
        for cls in sorted(img_per_class, key=lambda c: (-img_per_class[c], c))
    ]
    summary_path = os.path.join(args.export_dir, "class_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8")
    print(f"Saved class summary:  {summary_path}")

    # Distribution report (what the random draw produced, not a quota check)
    widths = [r["width"] for r in manifest_records]
    heights = [r["height"] for r in manifest_records]
    objs = [r["num_objects"] for r in manifest_records]
    print("\n" + "=" * 80)
    print(f"Total images: {len(selected)} | Classes covered: {len(img_per_class)} | Folder: {args.export_dir}")
    print(f"Objects per image: min {min(objs)}, mean {sum(objs) / len(objs):.1f}, max {max(objs)}")
    print(f"Resolution: width {min(widths)}-{max(widths)} px, height {min(heights)}-{max(heights)} px")
    print("-" * 80)
    print(f"{'Class':<22}{'Images':>8}{'Objects':>9}    {'Class':<22}{'Images':>8}{'Objects':>9}")
    half = (len(summary_rows) + 1) // 2
    for i in range(half):
        left = summary_rows[i]
        line = f"{left['class']:<22}{left['images']:>8}{left['objects']:>9}"
        if i + half < len(summary_rows):
            right = summary_rows[i + half]
            line += f"    {right['class']:<22}{right['images']:>8}{right['objects']:>9}"
        print(line)
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
