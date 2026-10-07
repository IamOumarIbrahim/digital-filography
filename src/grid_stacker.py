#!/usr/bin/env python3
"""grid_stacker.py: Stack threader output images into a 2xN grid."""

from __future__ import annotations

import base64
import os
import sys
from typing import Sequence
from PIL import Image, ImageDraw, ImageFont


def get_font(size: int = 15) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_candidates = [
        "segoeuib.ttf",   # Segoe UI Bold
        "segoeui.ttf",    # Segoe UI Regular
        "arialbd.ttf",    # Arial Bold
        "arial.ttf",      # Arial
    ]
    for name in font_candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def create_grid(
    dir_path: str,
    stem: str,
    alpha_tag: str,
    counts: list[int],
    modes: list[str] = ("color", "bw"),
    header_height: int = 28,
) -> tuple[str, str]:
    font = get_font(15)

    # Inspect first image to determine cell size
    first_path = os.path.join(dir_path, f"{stem}_{modes[0]}_{counts[0]}threads_{alpha_tag}.png")
    if not os.path.isfile(first_path):
        raise FileNotFoundError(f"Missing first image: {first_path}")

    with Image.open(first_path) as im:
        cell_w, cell_h = im.size

    num_cols = len(modes)
    num_rows = len(counts)
    grid_w = cell_w * num_cols
    grid_h = cell_h * num_rows

    print(f"Generating {alpha_tag} grid: {grid_w}x{grid_h} px ({num_cols}x{num_rows} cells)...")

    # 1. Generate PNG grid
    grid_img = Image.new("RGB", (grid_w, grid_h), (0, 0, 0))
    draw = ImageDraw.Draw(grid_img)

    svg_elements: list[str] = []

    for row_idx, count in enumerate(counts):
        for col_idx, mode in enumerate(modes):
            filename = f"{stem}_{mode}_{count}threads_{alpha_tag}.png"
            img_path = os.path.join(dir_path, filename)
            if not os.path.isfile(img_path):
                raise FileNotFoundError(f"Missing file: {img_path}")

            x = col_idx * cell_w
            y = row_idx * cell_h

            with Image.open(img_path) as tile:
                tile_rgb = tile.convert("RGB")
                grid_img.paste(tile_rgb, (x, y))

            # Header inside the image at the top
            draw.rectangle([(x, y), (x + cell_w, y + header_height)], fill=(0, 0, 0))
            text = f"{count} threads"
            bbox = font.getbbox(text)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            tx = x + (cell_w - text_w) / 2
            ty = y + (header_height - text_h) / 2 - bbox[1]
            draw.text((tx, ty), text, fill=(255, 255, 255), font=font)

            # Read raw PNG for self-contained SVG embedding
            with open(img_path, "rb") as f:
                b64_data = base64.b64encode(f.read()).decode("ascii")

            svg_elements.append(
                f'  <!-- Cell ({col_idx}, {row_idx}): {mode} {count} threads -->\n'
                f'  <image href="data:image/png;base64,{b64_data}" xlink:href="data:image/png;base64,{b64_data}" '
                f'x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" preserveAspectRatio="none"/>\n'
                f'  <rect x="{x}" y="{y}" width="{cell_w}" height="{header_height}" fill="#000000"/>\n'
                f'  <text x="{x + cell_w / 2:.1f}" y="{y + header_height / 2:.1f}" fill="#ffffff" '
                f'font-family="Segoe UI, -apple-system, BlinkMacSystemFont, Arial, sans-serif" font-size="14" '
                f'font-weight="600" text-anchor="middle" dominant-baseline="central">{count} threads</text>'
            )

    png_out = os.path.join(dir_path, f"{stem}_grid_{alpha_tag}.png")
    grid_img.save(png_out, format="PNG")
    print(f"Saved PNG: {png_out}")

    # 2. Generate SVG grid
    svg_out = os.path.join(dir_path, f"{stem}_grid_{alpha_tag}.svg")
    svg_content = (
        f'<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{grid_w}" height="{grid_h}" viewBox="0 0 {grid_w} {grid_h}">\n'
        f'<rect width="{grid_w}" height="{grid_h}" fill="#000000"/>\n'
        + "\n".join(svg_elements)
        + "\n</svg>\n"
    )

    with open(svg_out, "w", encoding="utf-8") as f:
        f.write(svg_content)
    print(f"Saved SVG: {svg_out}")

    return png_out, svg_out


def find_tile_path(
    dir_path: str,
    stem: str,
    mode: str,
    count: int,
    alpha_tag: str,
    suffix: str = "",
) -> str:
    for c_str in (f"{count:04d}", f"{count:05d}", f"{count}"):
        candidate = os.path.join(dir_path, f"{stem}_{mode}_{c_str}threads_{alpha_tag}{suffix}.png")
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"Missing tile for {stem}_{mode}_[{count} padded/unpadded]_{alpha_tag}{suffix}.png"
    )


def create_dtype_comparison_grid(
    dir_path: str,
    stem: str,
    counts: list[int],
    mode: str = "color",
    alpha_tag: str = "alpha1.0",
    header_height: int = 28,
) -> tuple[str, str]:
    font = get_font(15)

    first_path = find_tile_path(dir_path, stem, mode, counts[0], alpha_tag, suffix="")
    with Image.open(first_path) as im:
        cell_w, cell_h = im.size

    columns = [
        {"dtype": "uint8", "suffix": "_uint8", "label": "uint8"},
        {"dtype": "float32", "suffix": "", "label": "float32"},
    ]
    num_cols = len(columns)
    num_rows = len(counts)
    grid_w = cell_w * num_cols
    grid_h = cell_h * num_rows

    print(
        f"Generating dtype comparison grid ({mode}, {alpha_tag}): "
        f"{grid_w}x{grid_h} px ({num_cols}x{num_rows} cells)..."
    )

    grid_img = Image.new("RGB", (grid_w, grid_h), (0, 0, 0))
    draw = ImageDraw.Draw(grid_img)
    svg_elements: list[str] = []

    for row_idx, count in enumerate(counts):
        for col_idx, col_info in enumerate(columns):
            suffix = col_info["suffix"]
            img_path = find_tile_path(dir_path, stem, mode, count, alpha_tag, suffix=suffix)

            x = col_idx * cell_w
            y = row_idx * cell_h

            with Image.open(img_path) as tile:
                tile_rgb = tile.convert("RGB")
                grid_img.paste(tile_rgb, (x, y))

            # Header inside the image at the top
            draw.rectangle([(x, y), (x + cell_w, y + header_height)], fill=(0, 0, 0))
            text = f"{col_info['label']} | {count} threads"
            bbox = font.getbbox(text)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            tx = x + (cell_w - text_w) / 2
            ty = y + (header_height - text_h) / 2 - bbox[1]
            draw.text((tx, ty), text, fill=(255, 255, 255), font=font)

            with open(img_path, "rb") as f:
                b64_data = base64.b64encode(f.read()).decode("ascii")

            svg_elements.append(
                f'  <!-- Cell ({col_idx}, {row_idx}): {col_info["label"]} {count} threads -->\n'
                f'  <image href="data:image/png;base64,{b64_data}" xlink:href="data:image/png;base64,{b64_data}" '
                f'x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" preserveAspectRatio="none"/>\n'
                f'  <rect x="{x}" y="{y}" width="{cell_w}" height="{header_height}" fill="#000000"/>\n'
                f'  <text x="{x + cell_w / 2:.1f}" y="{y + header_height / 2:.1f}" fill="#ffffff" '
                f'font-family="Segoe UI, -apple-system, BlinkMacSystemFont, Arial, sans-serif" font-size="14" '
                f'font-weight="600" text-anchor="middle" dominant-baseline="central">{text}</text>'
            )

    png_out = os.path.join(dir_path, f"{stem}_grid_uint8_vs_float32.png")
    grid_img.save(png_out, format="PNG")
    print(f"Saved PNG: {png_out}")

    svg_out = os.path.join(dir_path, f"{stem}_grid_uint8_vs_float32.svg")
    svg_content = (
        f'<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{grid_w}" height="{grid_h}" viewBox="0 0 {grid_w} {grid_h}">\n'
        f'<rect width="{grid_w}" height="{grid_h}" fill="#000000"/>\n'
        + "\n".join(svg_elements)
        + "\n</svg>\n"
    )

    with open(svg_out, "w", encoding="utf-8") as f:
        f.write(svg_content)
    print(f"Saved SVG: {svg_out}")

    return png_out, svg_out


def create_u8_vs_u4_grid(
    dir_path: str,
    stem: str,
    counts: list[int],
    mode: str = "color",
    alpha_tag: str = "alpha1.0",
    header_height: int = 28,
) -> tuple[str, str]:
    font = get_font(15)

    first_path = find_tile_path(dir_path, stem, mode, counts[0], alpha_tag, suffix="_uint8")
    with Image.open(first_path) as im:
        cell_w, cell_h = im.size

    columns = [
        {"dtype": "uint8", "suffix": "_uint8", "label": "uint8"},
        {"dtype": "uint4", "suffix": "_uint4", "label": "uint4"},
    ]
    num_cols = len(columns)
    num_rows = len(counts)
    grid_w = cell_w * num_cols
    grid_h = cell_h * num_rows

    print(
        f"Generating uint8 vs uint4 grid ({mode}, {alpha_tag}): "
        f"{grid_w}x{grid_h} px ({num_cols}x{num_rows} cells)..."
    )

    grid_img = Image.new("RGB", (grid_w, grid_h), (0, 0, 0))
    draw = ImageDraw.Draw(grid_img)
    svg_elements: list[str] = []

    for row_idx, count in enumerate(counts):
        for col_idx, col_info in enumerate(columns):
            suffix = col_info["suffix"]
            img_path = find_tile_path(dir_path, stem, mode, count, alpha_tag, suffix=suffix)

            x = col_idx * cell_w
            y = row_idx * cell_h

            with Image.open(img_path) as tile:
                tile_rgb = tile.convert("RGB")
                grid_img.paste(tile_rgb, (x, y))

            # Header inside the image at the top
            draw.rectangle([(x, y), (x + cell_w, y + header_height)], fill=(0, 0, 0))
            text = f"{col_info['label']} | {count} threads"
            bbox = font.getbbox(text)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            tx = x + (cell_w - text_w) / 2
            ty = y + (header_height - text_h) / 2 - bbox[1]
            draw.text((tx, ty), text, fill=(255, 255, 255), font=font)

            with open(img_path, "rb") as f:
                b64_data = base64.b64encode(f.read()).decode("ascii")

            svg_elements.append(
                f'  <!-- Cell ({col_idx}, {row_idx}): {col_info["label"]} {count} threads -->\n'
                f'  <image href="data:image/png;base64,{b64_data}" xlink:href="data:image/png;base64,{b64_data}" '
                f'x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" preserveAspectRatio="none"/>\n'
                f'  <rect x="{x}" y="{y}" width="{cell_w}" height="{header_height}" fill="#000000"/>\n'
                f'  <text x="{x + cell_w / 2:.1f}" y="{y + header_height / 2:.1f}" fill="#ffffff" '
                f'font-family="Segoe UI, -apple-system, BlinkMacSystemFont, Arial, sans-serif" font-size="14" '
                f'font-weight="600" text-anchor="middle" dominant-baseline="central">{text}</text>'
            )

    png_out = os.path.join(dir_path, f"{stem}_grid_uint8_vs_uint4.png")
    grid_img.save(png_out, format="PNG")
    print(f"Saved PNG: {png_out}")

    svg_out = os.path.join(dir_path, f"{stem}_grid_uint8_vs_uint4.svg")
    svg_content = (
        f'<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{grid_w}" height="{grid_h}" viewBox="0 0 {grid_w} {grid_h}">\n'
        f'<rect width="{grid_w}" height="{grid_h}" fill="#000000"/>\n'
        + "\n".join(svg_elements)
        + "\n</svg>\n"
    )

    with open(svg_out, "w", encoding="utf-8") as f:
        f.write(svg_content)
    print(f"Saved SVG: {svg_out}")

    return png_out, svg_out


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Stack filography output images into a 2xN grid."
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=r"D:\Downloads\house-cat_MIZQ6V1ZJU_threads",
        help="path to threads output directory",
    )
    parser.add_argument(
        "--stem",
        default="house-cat_MIZQ6V1ZJU",
        help="image name stem (default: house-cat_MIZQ6V1ZJU)",
    )
    parser.add_argument(
        "--counts",
        type=int,
        nargs="+",
        default=[100, 200, 300, 400, 500, 1000, 2000, 3000, 4000],
        help="thread counts to stack",
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        default=["alpha1.0", "alpha0.3"],
        help="alpha suffixes to generate grids for (default: alpha1.0 alpha0.3)",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["color", "bw"],
        help="modes for the 2 columns (default: color bw)",
    )
    parser.add_argument(
        "--header-height",
        type=int,
        default=28,
        help="header height in px inside the image (default: 28)",
    )
    parser.add_argument(
        "--compare-dtype",
        action="store_true",
        help="create a 2xN grid comparing uint8 vs float32",
    )
    parser.add_argument(
        "--compare-u8-u4",
        action="store_true",
        help="create a 2xN grid comparing uint8 vs uint4",
    )

    args = parser.parse_args(argv)

    if not os.path.isdir(args.folder):
        print(f"error: directory not found: {args.folder}", file=sys.stderr)
        return 1

    if args.compare_u8_u4:
        create_u8_vs_u4_grid(
            args.folder,
            args.stem,
            args.counts,
            mode="color",
            alpha_tag="alpha1.0",
            header_height=args.header_height,
        )
        return 0

    if args.compare_dtype:
        create_dtype_comparison_grid(
            args.folder,
            args.stem,
            args.counts,
            mode="color",
            alpha_tag="alpha1.0",
            header_height=args.header_height,
        )
        return 0

    alphas = [a if a.startswith("alpha") else f"alpha{a}" for a in args.alphas]

    for alpha in alphas:
        create_grid(
            args.folder,
            args.stem,
            alpha,
            args.counts,
            modes=args.modes,
            header_height=args.header_height,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
