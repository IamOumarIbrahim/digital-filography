#!/usr/bin/env python3
"""make_quantization_grid.py: Build comprehensive Quantization x Thread Count grid.

Grid structure:
- Columns: Baseline (Original) + Thread Counts [100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000] (11 columns total)
- Rows: Quantization tiers [float32, float16, uint8, uint6, uint5, uint4] (6 rows total)
- Each tile header: Storage size, PSNR vs baseline, MSE vs baseline
- Original resolution preserved: 640x427 per tile (no downsizing)
- Generates both PNG and standalone SVG
"""

from __future__ import annotations

import base64
import io
import math
import os
import sys
import time
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from threader import ThreadPainter, load_image, render_threads, save_png


def get_font(size: int = 14, bold: bool = True) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = (
        ("segoeuib.ttf", "arialbd.ttf") if bold else ("segoeui.ttf", "arial.ttf")
    )
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
    elif tier == "uint5":
        return np.clip(np.round(threads * 31.0), 0, 31).astype(np.float32) / 31.0
    elif tier == "uint4":
        return np.clip(np.round(threads * 15.0), 0, 15).astype(np.float32) / 15.0
    else:
        raise ValueError(f"Unknown tier: {tier}")


def main() -> int:
    image_path = r"D:\Downloads\house-cat_MIZQ6V1ZJU.jpg"
    out_dir = r"D:\Downloads\house-cat_MIZQ6V1ZJU_threads"
    stem = "house-cat_MIZQ6V1ZJU"

    counts = [100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000]
    tiers = [
        {"name": "float32", "bytes_per_thread": 32, "label": "float32 (32-bit)"},
        {"name": "float16", "bytes_per_thread": 16, "label": "float16 (16-bit)"},
        {"name": "uint8",   "bytes_per_thread": 8,  "label": "uint8 (8-bit)"},
        {"name": "uint6",   "bytes_per_thread": 6,  "label": "uint6 (6-bit)"},
        {"name": "uint5",   "bytes_per_thread": 5,  "label": "uint5 (5-bit)"},
        {"name": "uint4",   "bytes_per_thread": 4,  "label": "uint4 (4-bit)"},
    ]

    size = 640
    thread_width = 2.0
    alpha = 1.0
    bg = (1.0, 1.0, 1.0)
    seed = 0

    print("Loading baseline image...")
    baseline_img = load_image(image_path, size, bg)
    cell_w, cell_h = baseline_img.size
    target = np.asarray(baseline_img, dtype=np.float32) / 255.0

    print("Running thread placement optimization up to 5000 threads...")
    t0 = time.time()
    painter = ThreadPainter(target, alpha, bg, thread_width, seed)
    snapshots: dict[int, np.ndarray] = {}

    def on_snapshot(n: int, threads: np.ndarray, canvas: np.ndarray) -> None:
        snapshots[n] = threads.copy()
        print(f"  Reached {n:5d} threads ({time.time() - t0:5.1f}s)")

    painter.run(counts, on_snapshot)

    # Grid configuration
    # Column 0: Baseline, Columns 1..10: counts
    num_cols = 1 + len(counts)
    num_rows = len(tiers)
    col_header_h = 36
    cell_header_h = 28

    grid_w = cell_w * num_cols
    grid_h = col_header_h + cell_h * num_rows

    print(f"\nBuilding {grid_w}x{grid_h} px grid ({num_cols} cols x {num_rows} rows)...")
    grid_img = Image.new("RGB", (grid_w, grid_h), (18, 18, 20))
    draw = ImageDraw.Draw(grid_img)

    f_col_header = get_font(16, bold=True)
    f_cell_header = get_font(13, bold=True)

    svg_elements: list[str] = []

    # 1. Draw top column headers
    col_titles = ["BASELINE (ORIGINAL)"] + [f"{c:,} THREADS" for c in counts]
    for col_idx, title in enumerate(col_titles):
        cx = col_idx * cell_w
        draw.rectangle([(cx, 0), (cx + cell_w, col_header_h)], fill=(28, 28, 32))
        draw.line([(cx, 0), (cx, col_header_h)], fill=(50, 50, 56))

        bbox = f_col_header.getbbox(title)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = cx + (cell_w - tw) / 2
        ty = (col_header_h - th) / 2 - bbox[1]
        draw.text((tx, ty), title, fill=(240, 240, 245), font=f_col_header)

        svg_elements.append(
            f'  <rect x="{cx}" y="0" width="{cell_w}" height="{col_header_h}" fill="#1c1c20"/>\n'
            f'  <text x="{cx + cell_w / 2:.1f}" y="{col_header_h / 2:.1f}" fill="#f0f0f5" '
            f'font-family="Segoe UI, sans-serif" font-size="15" font-weight="700" '
            f'text-anchor="middle" dominant-baseline="central">{title}</text>'
        )

    # 2. Populate cells
    for row_idx, tier_info in enumerate(tiers):
        tier_name = tier_info["name"]
        tier_label = tier_info["label"]
        b_pt = tier_info["bytes_per_thread"]

        for col_idx in range(num_cols):
            x = col_idx * cell_w
            y = col_header_h + row_idx * cell_h

            if col_idx == 0:
                # Baseline column
                cell_pil = baseline_img.copy()
                header_text = f"{tier_name.upper()} | Baseline Reference | PSNR: ∞ | MSE: 0.0000"
            else:
                n = counts[col_idx - 1]
                raw_t = snapshots[n]
                recon_t = quantize_threads(raw_t, tier_name)
                canvas = render_threads(recon_t, cell_w, cell_h, bg, thread_width)

                mse = float(((target - canvas) ** 2).mean())
                psnr = 10.0 * math.log10(1.0 / max(mse, 1e-12))
                raw_kb = (n * b_pt) / 1000.0

                pixels = np.clip(canvas * 255.0 + 0.5, 0, 255).astype(np.uint8)
                cell_pil = Image.fromarray(pixels)
                header_text = f"{tier_name} ({raw_kb:.1f} KB) | PSNR: {psnr:5.2f} dB | MSE: {mse:.4f}"

            # Paste into grid image
            grid_img.paste(cell_pil, (x, y))

            # Draw header bar inside top of tile
            draw.rectangle([(x, y), (x + cell_w, y + cell_header_h)], fill=(0, 0, 0))
            bbox = f_cell_header.getbbox(header_text)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            tx = x + (cell_w - tw) / 2
            ty = y + (cell_header_h - th) / 2 - bbox[1]
            draw.text((tx, ty), header_text, fill=(255, 255, 255), font=f_cell_header)

            # Base64 encode for standalone SVG
            buffer = io.BytesIO()
            cell_pil.save(buffer, format="JPEG", quality=92)
            b64_str = base64.b64encode(buffer.getvalue()).decode("ascii")

            svg_elements.append(
                f'  <!-- Cell (row={row_idx}, col={col_idx}): {header_text} -->\n'
                f'  <image href="data:image/jpeg;base64,{b64_str}" xlink:href="data:image/jpeg;base64,{b64_str}" '
                f'x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" preserveAspectRatio="none"/>\n'
                f'  <rect x="{x}" y="{y}" width="{cell_w}" height="{cell_header_h}" fill="#000000"/>\n'
                f'  <text x="{x + cell_w / 2:.1f}" y="{y + cell_header_h / 2:.1f}" fill="#ffffff" '
                f'font-family="Segoe UI, sans-serif" font-size="12.5" font-weight="600" '
                f'text-anchor="middle" dominant-baseline="central">{header_text}</text>'
            )

        print(f"  Finished row {row_idx + 1}/{num_rows} ({tier_name})")

    # 3. Save files
    png_path = os.path.join(out_dir, f"{stem}_quantization_vs_counts_grid.png")
    svg_path = os.path.join(out_dir, f"{stem}_quantization_vs_counts_grid.svg")

    print(f"Saving PNG grid to {png_path}...")
    grid_img.save(png_path, format="PNG")
    print(f"Saved PNG ({os.path.getsize(png_path) / 1e6:.1f} MB)")

    print(f"Saving SVG grid to {svg_path}...")
    svg_content = (
        f'<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{grid_w}" height="{grid_h}" viewBox="0 0 {grid_w} {grid_h}">\n'
        f'<rect width="{grid_w}" height="{grid_h}" fill="#121214"/>\n'
        + "\n".join(svg_elements)
        + "\n</svg>\n"
    )
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg_content)
    print(f"Saved SVG ({os.path.getsize(svg_path) / 1e6:.1f} MB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
