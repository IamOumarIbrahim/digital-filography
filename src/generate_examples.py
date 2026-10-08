#!/usr/bin/env python3
"""generate_examples.py: Generate publication visual examples across diverse COCO dataset images.

Generates and saves into example/:
1. Diverse object categories (Cat, Dog, Car, Chair).
2. Original uncompressed baseline photos with reference YOLO detections.
3. Sparse thread reconstructions (250 threads) showing geometric hallucination or missing objects.
4. Detection breakthrough reconstructions (2,500 threads) demonstrating threshold emergence.
5. High-density converged reconstructions (10,000 threads) in float16 and uint8.
6. Quantization noise collapse (10,000 threads in uint4) illustrating the sub-8-bit barrier.
7. Master Breakthrough Comparison Grid (PNG & SVG).
8. Master Convergence & Quantization Frontier Grid (PNG & SVG).
"""

from __future__ import annotations

import base64
import io
import math
import os
import sys
import time
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from threader import ThreadPainter, load_image, render_threads


def get_font(size: int = 15, bold: bool = True) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = ("segoeuib.ttf", "arialbd.ttf") if bold else ("segoeui.ttf", "arial.ttf")
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def quantize_threads(threads: np.ndarray, tier: str) -> np.ndarray:
    if tier == "float32":
        return threads.astype(np.float32)
    elif tier == "float16":
        return threads.astype(np.float16).astype(np.float32)
    elif tier == "uint8":
        return np.clip(np.round(threads * 255.0), 0, 255).astype(np.float32) / 255.0
    elif tier == "uint6":
        return np.clip(np.round(threads * 63.0), 0, 63).astype(np.float32) / 63.0
    elif tier == "uint4":
        return np.clip(np.round(threads * 15.0), 0, 15).astype(np.float32) / 15.0
    else:
        raise ValueError(f"Unknown tier: {tier}")


def add_banner(
    img: Image.Image,
    title: str,
    subtitle: str,
    banner_height: int = 46,
    bg_color: tuple[int, int, int] = (20, 22, 28),
    accent_color: tuple[int, int, int] = (41, 128, 185),
) -> Image.Image:
    """Add a clean, styled header banner to an annotated image."""
    w, h = img.size
    new_img = Image.new("RGB", (w, h + banner_height), bg_color)
    new_img.paste(img, (0, banner_height))

    draw = ImageDraw.Draw(new_img)
    f_title = get_font(15, bold=True)
    f_sub = get_font(12, bold=False)

    # Accent color bar on left edge
    draw.rectangle([(0, 0), (6, banner_height)], fill=accent_color)

    # Title & Subtitle text
    draw.text((16, 5), title, fill=(245, 245, 250), font=f_title)
    draw.text((16, 26), subtitle, fill=(180, 185, 200), font=f_sub)

    return new_img


def draw_bounding_boxes(
    img: Image.Image,
    detections: list[dict[str, Any]],
    box_color: tuple[int, int, int] = (46, 204, 113),
    text_color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Draw styled bounding boxes and tags on PIL image."""
    res = img.copy()
    draw = ImageDraw.Draw(res)
    f_box = get_font(13, bold=True)

    for det in detections:
        box = det["box"]  # [x1, y1, x2, y2]
        cname = det["cls_name"]
        conf = det["conf"]
        label = f"{cname} {conf:.2f}"

        # Draw thick box
        x1, y1, x2, y2 = box
        for offset in range(3):
            draw.rectangle([(x1 - offset, y1 - offset), (x2 + offset, y2 + offset)], outline=box_color)

        # Label background
        bbox = draw.textbbox((x1, y1), label, font=f_box)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tag_y1 = max(0, y1 - th - 6)
        draw.rectangle([(x1, tag_y1), (x1 + tw + 8, tag_y1 + th + 6)], fill=box_color)
        draw.text((x1 + 4, tag_y1 + 2), label, fill=text_color, font=f_box)

    return res


def build_composite_grid(
    cells: list[list[dict[str, Any]]],
    output_png: str,
    output_svg: str,
    bg_color: tuple[int, int, int] = (18, 18, 22),
) -> None:
    """Compose a 2D matrix of images into both PNG and SVG grids."""
    rows = len(cells)
    cols = len(cells[0])
    cell_w, cell_h = cells[0][0]["img"].size
    grid_w = cell_w * cols
    grid_h = cell_h * rows

    grid_img = Image.new("RGB", (grid_w, grid_h), bg_color)
    svg_elements: list[str] = []

    for r_idx in range(rows):
        for c_idx in range(cols):
            item = cells[r_idx][c_idx]
            x = c_idx * cell_w
            y = r_idx * cell_h
            grid_img.paste(item["img"], (x, y))

            buf = io.BytesIO()
            item["img"].save(buf, format="JPEG", quality=90)
            b64_tile = base64.b64encode(buf.getvalue()).decode("ascii")
            svg_elements.append(
                f'  <!-- Cell ({c_idx}, {r_idx}): {item.get("label", "")} -->\n'
                f'  <image href="data:image/jpeg;base64,{b64_tile}" xlink:href="data:image/jpeg;base64,{b64_tile}" '
                f'x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" preserveAspectRatio="none"/>'
            )

    grid_img.save(output_png, format="PNG")
    print(f"Saved grid PNG: {output_png} ({grid_w}x{grid_h} px)")

    svg_content = (
        f'<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{grid_w}" height="{grid_h}" viewBox="0 0 {grid_w} {grid_h}">\n'
        f'<rect width="{grid_w}" height="{grid_h}" fill="#121214"/>\n'
        + "\n".join(svg_elements)
        + "\n</svg>\n"
    )
    with open(output_svg, "w", encoding="utf-8") as f:
        f.write(svg_content)
    print(f"Saved grid SVG: {output_svg}")


def run_yolo_predict(model: YOLO, img: Image.Image, conf: float = 0.25) -> list[dict[str, Any]]:
    """Predict detections and return formatted box dictionaries."""
    res = model.predict(img, imgsz=640, conf=conf, verbose=False, device="cuda" if torch.cuda.is_available() else "cpu")[0]
    dets = []
    for b in res.boxes:
        cid = int(b.cls[0].item())
        cname = model.names.get(cid, str(cid)).lower()
        c_conf = float(b.conf[0].item())
        xyxy = [float(x) for x in b.xyxy[0].tolist()]
        dets.append({"cls_id": cid, "cls_name": cname, "conf": c_conf, "box": xyxy})
    return dets


def main() -> int:
    dataset_dir = "dataset"
    out_dir = "example"
    os.makedirs(out_dir, exist_ok=True)

    print("================================================================================")
    print("Generating High-Quality Diverse COCO Filography Examples")
    print("================================================================================")

    # 4 distinct categories
    sample_specs = [
        {"file": "000000416170.jpg", "category": "Cat", "accent": (41, 128, 185)},
        {"file": "000000546829.jpg", "category": "Dog", "accent": (39, 174, 96)},
        {"file": "000000144706.jpg", "category": "Car", "accent": (230, 126, 34)},
        {"file": "000000290771.jpg", "category": "Chair / Indoor", "accent": (142, 68, 173)},
    ]

    print("Loading YOLOv8n and YOLO11n models...")
    m_v8 = YOLO("yolov8n.pt")
    m_11 = YOLO("yolo11n.pt")

    size = 640
    bg = (1.0, 1.0, 1.0)
    thread_width = 2.0
    alpha = 1.0
    seed = 0

    grid_breakthrough_cells: list[list[dict[str, Any]]] = []
    grid_quantization_cells: list[list[dict[str, Any]]] = []

    item_idx = 1

    for spec in sample_specs:
        fname = spec["file"]
        cat_name = spec["category"]
        accent = spec["accent"]
        img_path = os.path.join(dataset_dir, fname)

        if not os.path.exists(img_path):
            print(f"Warning: {img_path} not found, skipping.")
            continue

        print(f"\nProcessing category: {cat_name} ({fname})...")
        base_pil = load_image(img_path, size, bg)
        w, h = base_pil.size

        # 1. Baseline detection
        base_dets = run_yolo_predict(m_v8, base_pil)
        dets_str = ", ".join([f"{d['cls_name']} ({d['conf']:.2f})" for d in base_dets]) if base_dets else "none"
        base_annotated = draw_bounding_boxes(base_pil, base_dets, box_color=(46, 204, 113))
        banner_base = add_banner(
            base_annotated,
            f"Original Reference Photo — {cat_name} ({w}x{h})",
            f"Baseline Detections: {dets_str}",
            accent_color=accent,
        )
        base_out = os.path.join(out_dir, f"{item_idx:02d}_{cat_name.lower().split()[0]}_baseline_original.png")
        banner_base.save(base_out)
        print(f"  Saved: {base_out}")
        item_idx += 1

        # Generate threads up to 10,000
        print(f"  Generating thread placement up to 10,000 threads for {fname}...")
        t_start = time.time()
        target = np.asarray(base_pil, dtype=np.float32) / 255.0
        painter = ThreadPainter(target, alpha, bg, thread_width, seed)
        snaps: dict[int, np.ndarray] = {}
        painter.run([250, 2500, 5000, 10000], lambda n, t, c: snaps.update({n: t.copy()}))
        print(f"  Completed thread placement in {time.time() - t_start:.1f}s.")

        # 2. Sparse (250 threads)
        t250 = snaps[250]
        c250 = render_threads(quantize_threads(t250, "float16"), w, h, bg, thread_width)
        pil250 = Image.fromarray(np.clip(c250 * 255.0 + 0.5, 0, 255).astype(np.uint8), "RGB")
        d250 = run_yolo_predict(m_v8, pil250)
        d250_str = ", ".join([f"{d['cls_name']} ({d['conf']:.2f})" for d in d250]) if d250 else "None (Below detection threshold)"
        ann250 = draw_bounding_boxes(pil250, d250, box_color=(231, 76, 60))
        banner250 = add_banner(
            ann250,
            f"{cat_name} Filography — 250 Threads (Sparse Reconstruction)",
            f"YOLOv8n Detections: {d250_str} (Storage: 2.0 KB)",
            accent_color=(231, 76, 60),
        )
        out250 = os.path.join(out_dir, f"{item_idx:02d}_{cat_name.lower().split()[0]}_250threads_sparse.png")
        banner250.save(out250)
        print(f"  Saved: {out250}")
        item_idx += 1

        # 3. Breakthrough (2,500 threads)
        t2500 = snaps[2500]
        c2500 = render_threads(quantize_threads(t2500, "float16"), w, h, bg, thread_width)
        pil2500 = Image.fromarray(np.clip(c2500 * 255.0 + 0.5, 0, 255).astype(np.uint8), "RGB")
        d2500 = run_yolo_predict(m_v8, pil2500)
        d2500_str = ", ".join([f"{d['cls_name']} ({d['conf']:.2f})" for d in d2500]) if d2500 else "Recovering structure"
        ann2500 = draw_bounding_boxes(pil2500, d2500, box_color=(52, 152, 219))
        banner2500 = add_banner(
            ann2500,
            f"{cat_name} Filography — 2,500 Threads (Breakthrough Threshold)",
            f"YOLOv8n Detections: {d2500_str} (Storage: 20.0 KB)",
            accent_color=(52, 152, 219),
        )
        out2500 = os.path.join(out_dir, f"{item_idx:02d}_{cat_name.lower().split()[0]}_2500threads_breakthrough.png")
        banner2500.save(out2500)
        print(f"  Saved: {out2500}")
        item_idx += 1

        # 4. Dense 10,000 threads float16
        t10k = snaps[10000]
        c10k_f16 = render_threads(quantize_threads(t10k, "float16"), w, h, bg, thread_width)
        pil10k_f16 = Image.fromarray(np.clip(c10k_f16 * 255.0 + 0.5, 0, 255).astype(np.uint8), "RGB")
        d10k_f16 = run_yolo_predict(m_v8, pil10k_f16)
        d10k_f16_str = ", ".join([f"{d['cls_name']} ({d['conf']:.2f})" for d in d10k_f16]) if d10k_f16 else "none"
        ann10k_f16 = draw_bounding_boxes(pil10k_f16, d10k_f16, box_color=(46, 204, 113))
        banner10k_f16 = add_banner(
            ann10k_f16,
            f"{cat_name} Filography — 10,000 Threads (float16 / 16-bit)",
            f"YOLOv8n Detections: {d10k_f16_str} (Storage: 80.0 KB)",
            accent_color=(46, 204, 113),
        )
        out10k_f16 = os.path.join(out_dir, f"{item_idx:02d}_{cat_name.lower().split()[0]}_10000threads_float16.png")
        banner10k_f16.save(out10k_f16)
        print(f"  Saved: {out10k_f16}")
        item_idx += 1

        # 5. Dense 10,000 threads uint8 (50% storage sweet spot)
        c10k_u8 = render_threads(quantize_threads(t10k, "uint8"), w, h, bg, thread_width)
        pil10k_u8 = Image.fromarray(np.clip(c10k_u8 * 255.0 + 0.5, 0, 255).astype(np.uint8), "RGB")
        d10k_u8 = run_yolo_predict(m_v8, pil10k_u8)
        d10k_u8_str = ", ".join([f"{d['cls_name']} ({d['conf']:.2f})" for d in d10k_u8]) if d10k_u8 else "none"
        ann10k_u8 = draw_bounding_boxes(pil10k_u8, d10k_u8, box_color=(39, 174, 96))
        banner10k_u8 = add_banner(
            ann10k_u8,
            f"{cat_name} Filography — 10,000 Threads (uint8 / 8-bit Sweet Spot)",
            f"YOLOv8n Detections: {d10k_u8_str} (Storage: 40.0 KB, 50% Reduction)",
            accent_color=(39, 174, 96),
        )
        out10k_u8 = os.path.join(out_dir, f"{item_idx:02d}_{cat_name.lower().split()[0]}_10000threads_uint8.png")
        banner10k_u8.save(out10k_u8)
        print(f"  Saved: {out10k_u8}")
        item_idx += 1

        # 6. Dense 10,000 threads uint4 (Quantization Step Collapse)
        c10k_u4 = render_threads(quantize_threads(t10k, "uint4"), w, h, bg, thread_width)
        pil10k_u4 = Image.fromarray(np.clip(c10k_u4 * 255.0 + 0.5, 0, 255).astype(np.uint8), "RGB")
        d10k_u4 = run_yolo_predict(m_v8, pil10k_u4)
        d10k_u4_str = ", ".join([f"{d['cls_name']} ({d['conf']:.2f})" for d in d10k_u4]) if d10k_u4 else "Zero Detections (Precision Noise Breakdown)"
        ann10k_u4 = draw_bounding_boxes(pil10k_u4, d10k_u4, box_color=(192, 57, 43))
        banner10k_u4 = add_banner(
            ann10k_u4,
            f"{cat_name} Filography — 10,000 Threads (uint4 / 4-bit Step Collapse)",
            f"YOLOv8n Detections: {d10k_u4_str} (Storage: 20.0 KB)",
            accent_color=(192, 57, 43),
        )
        out10k_u4 = os.path.join(out_dir, f"{item_idx:02d}_{cat_name.lower().split()[0]}_10000threads_uint4_collapse.png")
        banner10k_u4.save(out10k_u4)
        print(f"  Saved: {out10k_u4}")
        item_idx += 1

        # Add to composite grid rows
        grid_breakthrough_cells.append([
            {"img": banner_base, "label": f"{cat_name} Baseline"},
            {"img": banner250, "label": f"{cat_name} 250 threads"},
            {"img": banner2500, "label": f"{cat_name} 2500 threads"},
            {"img": banner10k_f16, "label": f"{cat_name} 10000 threads"},
        ])

        grid_quantization_cells.append([
            {"img": banner_base, "label": f"{cat_name} Baseline"},
            {"img": banner10k_f16, "label": f"{cat_name} float16"},
            {"img": banner10k_u8, "label": f"{cat_name} uint8"},
            {"img": banner10k_u4, "label": f"{cat_name} uint4"},
        ])

    # Build composite grids
    if grid_breakthrough_cells:
        print("\nComposing master multi-category breakthrough comparison grid...")
        png_grid1 = os.path.join(out_dir, "breakthrough_comparison_grid_coco.png")
        svg_grid1 = os.path.join(out_dir, "breakthrough_comparison_grid_coco.svg")
        build_composite_grid(grid_breakthrough_cells, png_grid1, svg_grid1)

        print("\nComposing master multi-category quantization frontier grid...")
        png_grid2 = os.path.join(out_dir, "convergence_and_quantization_grid_coco.png")
        svg_grid2 = os.path.join(out_dir, "convergence_and_quantization_grid_coco.svg")
        build_composite_grid(grid_quantization_cells, png_grid2, svg_grid2)

    print("\n================================================================================")
    print("New visual examples generated successfully in example/!")
    print("================================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
