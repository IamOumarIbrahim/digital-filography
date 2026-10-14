#!/usr/bin/env python3
"""Resize dataset images to 640x640 using a bilinear central crop with bounded annotations.

Crop & resize behavior:
- Aspect-ratio preserving central crop: min(w, h) square is extracted from the image center.
- No black bars / letterboxing, no stretching or distortion.
- Resampled to 640x640 using PIL Bilinear interpolation (Image.Resampling.BILINEAR).
- Bounding boxes and segmentation polygons in annotations.json are translated,
  scaled, and bounded to the [0, 640] square.
- Annotations completely outside the central crop are cleanly dropped.
- Image width/height metadata in annotations.json are updated to 640x640.
- Supports both per-class subfolders (dataset/<class>/) and flat dataset/ layouts.

Usage:
    python src/resize_dataset.py                       # 640x640 on dataset/
    python src/resize_dataset.py --size 640            # custom square size
    python src/resize_dataset.py --workers 16          # parallel workers
    python src/resize_dataset.py --dataset-dir dataset/
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageOps

# ---------------------------------------------------------------------------
# Paths and Defaults
# ---------------------------------------------------------------------------

PROJECT_ROOT        = Path(__file__).resolve().parent.parent
CONFIG_PATH         = PROJECT_ROOT / "config" / "config.json"
DEFAULT_DATASET_DIR = PROJECT_ROOT / "dataset"
ANNOTATIONS_CACHE   = PROJECT_ROOT / "config" / "instances_val2017.json"

DEFAULT_SIZE    = 640
DEFAULT_WORKERS = 16

# ---------------------------------------------------------------------------
# Terminal Colors
# ---------------------------------------------------------------------------

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

if os.name == "nt":
    os.system("")

USE_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")

RESET, BOLD = "\033[0m", "\033[1m"
GREEN, CYAN, YELLOW, RED, GRAY, MAGENTA = "\033[32m", "\033[36m", "\033[33m", "\033[31m", "\033[90m", "\033[35m"

MESSAGE_STYLES = {
    "success": ("✔", GREEN),
    "info":    ("ℹ", CYAN),
    "warning": ("⚠", YELLOW),
    "error":   ("✖", RED),
}


def paint(text, *styles):
    if not USE_COLOR or not styles:
        return str(text)
    return "".join(styles) + str(text) + RESET


def say(kind, text):
    symbol, color = MESSAGE_STYLES[kind]
    print(paint(f"{symbol} ", BOLD, color) + text)


def show_progress(done, total, width=30):
    filled  = int(width * done / total) if total else width
    bar     = paint("█" * filled, GREEN) + paint("░" * (width - filled), GRAY)
    percent = int(100 * done / total) if total else 100
    sys.stdout.write(f"\r{bar} {done:,}/{total:,} ({percent}%)")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Geometry & Annotation Transforms
# ---------------------------------------------------------------------------

def transform_bbox(bbox: list[float], orig_w: int, orig_h: int, target_size: int = 640) -> list[float] | None:
    """Transform and clip a COCO bounding box [x, y, w, h] to the central crop square.

    Parameters
    ----------
    bbox : [x, y, w, h] in original pixel coordinates.
    orig_w, orig_h : original image dimensions.
    target_size : size of the square target (default: 640).

    Returns
    -------
    [new_x, new_y, new_w, new_h] bounded within [0, target_size],
    or None if the bounding box is outside the crop or has width/height < 1px.
    """
    s = min(orig_w, orig_h)
    crop_x = (orig_w - s) / 2.0
    crop_y = (orig_h - s) / 2.0
    scale = target_size / s

    x, y, w, h = bbox
    x1, y1 = float(x), float(y)
    x2, y2 = x1 + float(w), y1 + float(h)

    # Translate by crop offset and scale to target size
    nx1 = (x1 - crop_x) * scale
    ny1 = (y1 - crop_y) * scale
    nx2 = (x2 - crop_x) * scale
    ny2 = (y2 - crop_y) * scale

    # Clip to target boundary [0, target_size]
    cx1 = max(0.0, min(float(target_size), nx1))
    cy1 = max(0.0, min(float(target_size), ny1))
    cx2 = max(0.0, min(float(target_size), nx2))
    cy2 = max(0.0, min(float(target_size), ny2))

    nw = cx2 - cx1
    nh = cy2 - cy1

    if nw < 1.0 or nh < 1.0:
        return None

    return [round(cx1, 2), round(cy1, 2), round(nw, 2), round(nh, 2)]


def transform_segmentation(segmentation, orig_w: int, orig_h: int, target_size: int = 640):
    """Transform COCO polygon segmentation coordinates to the central crop square."""
    if not isinstance(segmentation, list):
        return segmentation

    s = min(orig_w, orig_h)
    crop_x = (orig_w - s) / 2.0
    crop_y = (orig_h - s) / 2.0
    scale = target_size / s

    new_polygons = []
    for poly in segmentation:
        if not isinstance(poly, list) or len(poly) < 6:
            continue
        new_poly = []
        for i in range(0, len(poly), 2):
            px = poly[i]
            py = poly[i + 1]
            npx = max(0.0, min(float(target_size), (px - crop_x) * scale))
            npy = max(0.0, min(float(target_size), (py - crop_y) * scale))
            new_poly.extend([round(npx, 2), round(npy, 2)])
        unique_pts = {(new_poly[i], new_poly[i + 1]) for i in range(0, len(new_poly), 2)}
        if len(unique_pts) >= 3:
            new_polygons.append(new_poly)

    return new_polygons if new_polygons else None


# ---------------------------------------------------------------------------
# Image Resizing (Bilinear Central Crop)
# ---------------------------------------------------------------------------

def crop_and_resize_image(img_path: Path, target_size: int = 640, output_path: Path | None = None) -> tuple[int, int, bool]:
    """Crops the central square of the image and resizes using bilinear interpolation.

    Parameters
    ----------
    img_path : Path to image file.
    target_size : Target width and height in pixels (default: 640).
    output_path : Path to save output image (defaults to in-place replacement).

    Returns
    -------
    (original_width, original_height, was_modified)
    """
    if output_path is None:
        output_path = img_path

    with Image.open(img_path) as img:
        img = ImageOps.exif_transpose(img)
        w, h = img.size
        if w == target_size and h == target_size:
            return w, h, False

        s = min(w, h)
        crop_x = (w - s) / 2.0
        crop_y = (h - s) / 2.0

        cropped = img.crop((crop_x, crop_y, crop_x + s, crop_y + s))
        resized = cropped.resize((target_size, target_size), resample=Image.Resampling.BILINEAR)

        temp_out = output_path.with_suffix(".part")
        resized.save(temp_out, format="JPEG", quality=95)
        temp_out.replace(output_path)

    return w, h, True


def _resize_worker(task):
    path, target_size = task
    try:
        w, h, modified = crop_and_resize_image(path, target_size)
        return path, w, h, modified, None
    except Exception as exc:  # noqa: BLE001
        return path, 0, 0, False, str(exc)


# ---------------------------------------------------------------------------
# Dataset Discovery & Annotation Processing
# ---------------------------------------------------------------------------

def load_master_annotations() -> dict | None:
    """Load cached COCO val2017 annotations if available."""
    if ANNOTATIONS_CACHE.exists():
        try:
            return json.loads(ANNOTATIONS_CACHE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
    return None


def get_dataset_groups(dataset_dir: Path) -> list[tuple[Path, list[Path], Path]]:
    """Find all image groups: subfolders or top-level dataset folder with images.

    Returns list of (group_dir, [image_paths], annotations_path).
    """
    groups = []
    if not dataset_dir.exists():
        return groups

    subdirs = [d for d in sorted(dataset_dir.iterdir()) if d.is_dir() and any(d.glob("*.jpg"))]
    if subdirs:
        for d in subdirs:
            groups.append((d, sorted(d.glob("*.jpg")), d / "annotations.json"))

    top_images = sorted(dataset_dir.glob("*.jpg"))
    if top_images:
        groups.append((dataset_dir, top_images, dataset_dir / "annotations.json"))

    return groups


def process_annotations_for_group(
    group_dir: Path,
    image_paths: list[Path],
    ann_path: Path,
    image_dimensions: dict[str, tuple[int, int]],
    target_size: int = 640,
    master_data: dict | None = None,
) -> dict:
    """Transform annotations for a specific image group and write annotations.json."""
    data = None
    if ann_path.exists():
        try:
            data = json.loads(ann_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = None

    file_names = {p.name for p in image_paths}

    # If no annotations.json exists in this folder, construct from master
    if data is None and master_data:
        relevant_images = [img for img in master_data.get("images", []) if img["file_name"] in file_names]
        rel_ids = {img["id"] for img in relevant_images}
        data = {
            "info":        master_data.get("info", {}),
            "licenses":    master_data.get("licenses", []),
            "images":      relevant_images,
            "annotations": [ann for ann in master_data.get("annotations", []) if ann["image_id"] in rel_ids],
            "categories":  master_data.get("categories", []),
        }

    if not data or not isinstance(data, dict):
        return {"total_before": 0, "kept": 0, "dropped": 0}

    # Build image lookup: id -> original (w, h)
    img_orig_dims = {}
    for img in data.get("images", []):
        fn = img.get("file_name")
        if fn in image_dimensions:
            orig_w, orig_h = image_dimensions[fn]
        else:
            orig_w = img.get("width", target_size)
            orig_h = img.get("height", target_size)
        img_orig_dims[img["id"]] = (orig_w, orig_h)
        # Update metadata dimensions to target size
        img["width"] = target_size
        img["height"] = target_size

    # Transform bounding boxes
    total_before = len(data.get("annotations", []))
    retained_anns = []
    dropped_count = 0

    for ann in data.get("annotations", []):
        img_id = ann.get("image_id")
        orig_w, orig_h = img_orig_dims.get(img_id, (target_size, target_size))
        raw_bbox = ann.get("bbox")
        if not raw_bbox or len(raw_bbox) != 4:
            continue

        new_bbox = transform_bbox(raw_bbox, orig_w, orig_h, target_size=target_size)
        if new_bbox is None:
            dropped_count += 1
            continue

        ann["bbox"] = new_bbox
        ann["area"] = round(new_bbox[2] * new_bbox[3], 2)

        if ann.get("segmentation"):
            new_seg = transform_segmentation(ann["segmentation"], orig_w, orig_h, target_size=target_size)
            if new_seg:
                ann["segmentation"] = new_seg

        retained_anns.append(ann)

    data["annotations"] = retained_anns

    try:
        ann_path.parent.mkdir(parents=True, exist_ok=True)
        ann_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        say("warning", f"Could not save annotations to {ann_path}: {exc}")

    return {
        "total_before": total_before,
        "kept": len(retained_anns),
        "dropped": dropped_count,
    }


# ---------------------------------------------------------------------------
# Main Resizing Pipeline
# ---------------------------------------------------------------------------

def resize_dataset(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    target_size: int = DEFAULT_SIZE,
    workers: int = DEFAULT_WORKERS,
) -> dict:
    """Crop and resize dataset to target_size x target_size using Bilinear interpolation.

    Performs:
    1. Bilinear central crop on all .jpg images.
    2. Coordinates transformation on bounding boxes and segmentations.
    3. Writes updated annotations.json alongside each image folder.
    """
    dataset_dir = Path(dataset_dir).resolve()
    groups = get_dataset_groups(dataset_dir)
    all_images = [img for _, img_list, _ in groups for img in img_list]

    if not all_images:
        say("warning", f"No images found in {dataset_dir}.")
        return {"images_resized": 0, "images_skipped": 0, "failed": [], "annotations": {}}

    print()
    say("info", f"Dataset resizing target: {paint(f'{target_size}x{target_size}', BOLD, GREEN)} (Bilinear Central Crop)")
    say("info", f"Layout: {len(groups)} folder(s), {len(all_images):,} total image(s), {workers} worker thread(s)")
    say("info", "No black bars, no distortion. Annotations bounded and clipped to square.")
    print()

    master_data = load_master_annotations()
    image_dimensions: dict[str, tuple[int, int]] = {}
    resized_count = 0
    skipped_count = 0
    failed_list = []

    tasks = [(p, target_size) for p in all_images]
    show_progress(0, len(tasks))

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_resize_worker, task): task[0] for task in tasks}
        for done, future in enumerate(as_completed(futures), start=1):
            path, orig_w, orig_h, modified, err = future.result()
            if err:
                failed_list.append((str(path), err))
            else:
                image_dimensions[path.name] = (orig_w, orig_h)
                if modified:
                    resized_count += 1
                else:
                    skipped_count += 1
            show_progress(done, len(tasks))
    print()
    t_elapsed = time.perf_counter() - t0

    # Process annotations for each folder group
    say("info", "Transforming and clipping annotations to 640x640 boundary...")
    total_kept = 0
    total_dropped = 0
    total_before = 0

    for group_dir, group_images, ann_path in groups:
        stats = process_annotations_for_group(
            group_dir=group_dir,
            image_paths=group_images,
            ann_path=ann_path,
            image_dimensions=image_dimensions,
            target_size=target_size,
            master_data=master_data,
        )
        total_before  += stats["total_before"]
        total_kept    += stats["kept"]
        total_dropped += stats["dropped"]
        if ann_path.exists():
            say("info", f"  • {ann_path.relative_to(PROJECT_ROOT)}: {stats['kept']:,} annotations kept ({stats['dropped']:,} clipped/dropped)")

    print()
    say("success", f"Resizing complete in {t_elapsed:.2f}s!")
    say("info", f"  • Images: {paint(f'{resized_count:,} resized', BOLD, GREEN)}, {skipped_count:,} already {target_size}x{target_size}")
    if total_before:
        say("info", f"  • Annotations: {paint(f'{total_kept:,} active', BOLD, GREEN)} (clipped to square, {total_dropped:,} outside crop removed)")
    if failed_list:
        say("error", f"  • Failed: {len(failed_list)} file(s)")

    return {
        "images_resized": resized_count,
        "images_skipped": skipped_count,
        "failed":         failed_list,
        "annotations": {
            "total_before": total_before,
            "kept":         total_kept,
            "dropped":      total_dropped,
        },
    }


# ---------------------------------------------------------------------------
# CLI Argument Parsing
# ---------------------------------------------------------------------------

def default_workers() -> int:
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            w = data.get("shared", {}).get("workers") or data.get("workers")
            if isinstance(w, int) and w > 0:
                return w
    except (OSError, ValueError):
        pass
    return DEFAULT_WORKERS


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Crop and resize dataset to 640x640 with bilinear interpolation and bounded annotations."
    )
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE,
                        help=f"target square dimensions in pixels (default: {DEFAULT_SIZE})")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR,
                        help=f"path to dataset root folder (default: {DEFAULT_DATASET_DIR})")
    parser.add_argument("--workers", type=int, default=None,
                        help="parallel worker threads (default from config or 16)")

    args = parser.parse_args(argv)
    workers = args.workers if args.workers is not None else default_workers()

    try:
        summary = resize_dataset(dataset_dir=args.dataset_dir, target_size=args.size, workers=workers)
        return 1 if summary["failed"] else 0
    except KeyboardInterrupt:
        print()
        say("warning", "Resizing interrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
