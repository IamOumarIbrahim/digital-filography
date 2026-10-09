#!/usr/bin/env python3
"""threader.py: turn an image into digital filography (straight threads).

A filographic image is an N x 8 matrix with one thread per row:

    [x1, y1, x2, y2, r, g, b, a]          all values normalized to [0, 1]

Threads are drawn in order (row 1 first) on a plain background.

Usage:
    python threader.py path/to/image.jpg

For every combination of
    colour mode  : color, bw (grayscale threads)
    thread count : 5000, 10000, 15000, 20000
    thread alpha : 1.0, 0.3
the script writes a PNG and an SVG into a folder next to the input image and
prints the path of every file it created, one per line, on stdout.
Progress messages go to stderr.

The output keeps the aspect ratio of the input (longest side = --size px).

Requires: numpy, Pillow
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image, ImageOps

DEFAULT_COUNTS = (5000, 10000, 15000, 20000)
ALPHAS = (1.0, 0.3)
MODES = ("color", "bw")
RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def parse_color(text: str) -> tuple[float, float, float]:
    """Parse 'white', 'black' or '#rrggbb' into floats in [0, 1]."""
    named = {"white": (1.0, 1.0, 1.0), "black": (0.0, 0.0, 0.0)}
    key = text.strip().lower()
    if key in named:
        return named[key]
    digits = key.lstrip("#")
    if len(digits) == 6:
        try:
            r, g, b = (int(digits[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
            return (r, g, b)
        except ValueError:
            pass
    raise argparse.ArgumentTypeError(
        f"invalid colour {text!r}, use white, black or #rrggbb"
    )


def to_hex(rgb: Sequence[float]) -> str:
    r, g, b = (int(round(float(v) * 255.0)) for v in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def load_image(path: str, size: int, background: Sequence[float]) -> Image.Image:
    """Open an image, fix its orientation, flatten transparency, resize.

    The longest side becomes `size` pixels and the aspect ratio is preserved.
    """
    try:
        img = Image.open(path)
        img.load()
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise SystemExit(f"error: cannot read image {path!r}: {exc}")

    img = ImageOps.exif_transpose(img)
    has_alpha = img.mode in ("RGBA", "LA", "PA") or (
        img.mode == "P" and "transparency" in img.info
    )
    if has_alpha:
        rgba = img.convert("RGBA")
        base = Image.new(
            "RGBA", rgba.size, tuple(int(round(v * 255)) for v in background) + (255,)
        )
        img = Image.alpha_composite(base, rgba)
    img = img.convert("RGB")

    w, h = img.size
    scale = size / max(w, h)
    new_w = max(4, int(round(w * scale)))
    new_h = max(4, int(round(h * scale)))
    return img.resize((new_w, new_h), RESAMPLE)


def gaussian_blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur over axes 0 and 1 (edge padded), pure numpy."""
    radius = max(1, int(3.0 * sigma + 0.5))
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel = (kernel / kernel.sum()).astype(np.float32)
    out = a.astype(np.float32, copy=False)
    for axis in (0, 1):
        pad = [(0, 0)] * out.ndim
        pad[axis] = (radius, radius)
        padded = np.pad(out, pad, mode="edge")
        n = out.shape[axis]
        acc = np.zeros_like(out)
        for i, weight in enumerate(kernel):
            index = [slice(None)] * out.ndim
            index[axis] = slice(i, i + n)
            acc += weight * padded[tuple(index)]
        out = acc
    return out


def edge_angles(target: np.ndarray) -> np.ndarray:
    """Per-pixel angle (radians) of the direction that runs ALONG local edges.

    Uses the structure tensor, summed over colour channels.
    """
    smooth = gaussian_blur(target, 1.0)
    gy, gx = np.gradient(smooth, axis=(0, 1))
    jxx = gaussian_blur((gx * gx).sum(axis=2), 4.0)
    jyy = gaussian_blur((gy * gy).sum(axis=2), 4.0)
    jxy = gaussian_blur((gx * gy).sum(axis=2), 4.0)
    gradient_angle = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
    return (gradient_angle + 0.5 * math.pi).astype(np.float32)


# --------------------------------------------------------------------------
# Greedy thread placement
# --------------------------------------------------------------------------
class ThreadPainter:
    """Greedy thread placement.

    Every step proposes many random candidate threads, scores all of them at
    once (the best colour of each candidate is solved in closed form), polishes
    the best few with a small local search, then draws the winner on the canvas.
    Because each step simply adds the most useful thread, the first 5000
    threads of a run are exactly the 5000 thread image, the first 10000 are the
    10000 thread image, and so on.
    """

    SAMPLES = 48                 # points sampled along a candidate to score it
    INITIAL = 160                # random candidates per step
    PARENTS = 4                  # candidates polished by the local search
    CHILDREN = 12                # mutations tried per parent and round
    ROUNDS = (1.0, 0.5, 0.25)    # relative mutation size of each search round
    REFRESH = 8                  # steps between updates of the error map
    MAX_REJECTS = 40             # force a thread after this many rejections

    def __init__(
        self,
        target: np.ndarray,
        alpha: float,
        background: Sequence[float],
        width: float,
        seed: int,
    ) -> None:
        self.target = np.ascontiguousarray(target, dtype=np.float32)
        self.H, self.W, self.C = self.target.shape
        self.alpha = float(alpha)
        self.half_width = float(width) / 2.0

        self.canvas = np.empty((self.H, self.W, self.C), dtype=np.float32)
        self.canvas[:] = np.asarray(background, dtype=np.float32).reshape(1, 1, -1)
        self.t_flat = self.target.reshape(-1, self.C)
        self.c_flat = self.canvas.reshape(-1, self.C)
        if not np.shares_memory(self.canvas, self.c_flat):
            raise RuntimeError("canvas view is not backed by the canvas buffer")

        self.rng = np.random.default_rng(seed)
        self.theta = edge_angles(self.target)
        self.ramp = np.arange(self.SAMPLES, dtype=np.float32)
        self.phase = 0.5
        self.diag = math.hypot(self.W, self.H)
        self.min_len = 3.0
        self.max_len = self.diag
        self.log_len = math.log(0.3 * self.diag)   # running guess of useful length
        self.cdf = np.zeros(1, dtype=np.float64)
        self.refresh_error_map()

    # -- candidate generation ------------------------------------------------
    def refresh_error_map(self) -> None:
        """Rebuild the cumulative error map used to pick where to try threads."""
        err = ((self.target - self.canvas) ** 2).sum(axis=2).ravel()
        self.cdf = np.cumsum(err, dtype=np.float64)

    def clip_lines(self, lines: np.ndarray) -> np.ndarray:
        lines[:, 0::2] = np.clip(lines[:, 0::2], 0.0, self.W)
        lines[:, 1::2] = np.clip(lines[:, 1::2], 0.0, self.H)
        return lines

    def propose(self, n: int) -> np.ndarray:
        """Random candidate lines [x1, y1, x2, y2] in pixel units.

        Midpoints favour pixels that are still badly painted, directions often
        follow the local image edges, lengths follow what has been working.
        """
        rng = self.rng
        mx = rng.random(n) * self.W
        my = rng.random(n) * self.H
        total = float(self.cdf[-1])
        if total > 0.0:
            pick = np.searchsorted(self.cdf, rng.random(n) * total)
            np.minimum(pick, self.W * self.H - 1, out=pick)
            weighted = rng.random(n) >= 0.15
            mx = np.where(weighted, (pick % self.W) + rng.random(n), mx)
            my = np.where(weighted, (pick // self.W) + rng.random(n), my)

        iy = np.clip(my.astype(np.int64), 0, self.H - 1)
        ix = np.clip(mx.astype(np.int64), 0, self.W - 1)
        along_edge = self.theta[iy, ix] + rng.normal(0.0, 0.12, n)
        angle = np.where(rng.random(n) < 0.6, along_edge, rng.random(n) * math.pi)

        near_recent = rng.random(n) < 0.7
        log_len = np.where(
            near_recent,
            rng.normal(self.log_len, 0.6, n),
            rng.uniform(math.log(self.min_len), math.log(self.max_len), n),
        )
        half = 0.5 * np.clip(np.exp(log_len), self.min_len, self.max_len)
        dx, dy = np.cos(angle) * half, np.sin(angle) * half
        lines = np.stack([mx - dx, my - dy, mx + dx, my + dy], axis=1)
        return self.clip_lines(lines)

    # -- scoring -------------------------------------------------------------
    def evaluate(self, lines: np.ndarray) -> np.ndarray:
        """Estimated drop in squared error if each line were drawn.

        Each line is sampled at SAMPLES points. For a thread of opacity a, the
        best colour is c = mean(target - (1 - a) * canvas) / a, clipped to
        [0, 1], which gives the error after drawing in closed form.
        """
        k = self.SAMPLES
        a = self.alpha
        l32 = lines.astype(np.float32)
        x0, y0, x1, y1 = l32[:, 0:1], l32[:, 1:2], l32[:, 2:3], l32[:, 3:4]
        t = (self.ramp + np.float32(self.phase)) / np.float32(k)
        dx, dy = x1 - x0, y1 - y0
        ix = np.clip((x0 + t * dx).astype(np.int32), 0, self.W - 1)
        iy = np.clip((y0 + t * dy).astype(np.int32), 0, self.H - 1)
        idx = iy * self.W + ix

        tgt = self.t_flat[idx]                       # (n, k, C)
        cur = self.c_flat[idx]
        u = tgt - (1.0 - a) * cur
        r = tgt - cur
        su = u.sum(axis=1)                           # (n, C)
        qu = np.einsum("nkc,nkc->n", u, u)
        r2 = np.einsum("nkc,nkc->n", r, r)
        col = np.clip(su / (a * k), 0.0, 1.0)
        err_after = (
            qu - 2.0 * a * (col * su).sum(axis=1) + k * a * a * (col * col).sum(axis=1)
        )
        length = np.sqrt(dx * dx + dy * dy)[:, 0]
        return (r2 - err_after) * (np.maximum(length, 1.0) / k)

    def find_best(self) -> np.ndarray:
        """Random search followed by a short local search; returns one line."""
        self.phase = float(self.rng.random())
        lines = self.propose(self.INITIAL)
        gain = self.evaluate(lines)
        for scale in self.ROUNDS:
            top = np.argpartition(gain, -self.PARENTS)[-self.PARENTS:]
            parents = lines[top]
            length = np.hypot(parents[:, 2] - parents[:, 0], parents[:, 3] - parents[:, 1])
            sigma = np.maximum(1.5, 0.1 * length) * scale
            noise = self.rng.normal(size=(self.PARENTS, self.CHILDREN, 4))
            kids = (parents[:, None, :] + noise * sigma[:, None, None]).reshape(-1, 4)
            kids = self.clip_lines(kids)
            lines = np.concatenate([lines, kids])
            gain = np.concatenate([gain, self.evaluate(kids)])
        return lines[int(np.argmax(gain))]

    # -- drawing -------------------------------------------------------------
    def commit(self, line: np.ndarray, force: bool = False) -> np.ndarray | None:
        """Draw `line` on the canvas with its exact best colour.

        Returns the colour (length C) or None when the line would not reduce
        the error (unless `force` is set).
        """
        x0, y0, x1, y1 = (float(v) for v in line)
        reach = self.half_width + 1.0
        xa = max(int(math.floor(min(x0, x1) - reach)), 0)
        xb = min(int(math.ceil(max(x0, x1) + reach)), self.W)
        ya = max(int(math.floor(min(y0, y1) - reach)), 0)
        yb = min(int(math.ceil(max(y0, y1) + reach)), self.H)
        if xa >= xb or ya >= yb:
            return None

        px = (np.arange(xa, xb, dtype=np.float32) + 0.5)[None, :]
        py = (np.arange(ya, yb, dtype=np.float32) + 0.5)[:, None]
        dx, dy = x1 - x0, y1 - y0
        len2 = dx * dx + dy * dy
        if len2 > 1e-9:
            t = np.clip(((px - x0) * dx + (py - y0) * dy) / len2, 0.0, 1.0)
            ddx = px - (x0 + t * dx)
            ddy = py - (y0 + t * dy)
        else:
            ddx = px - x0
            ddy = py - y0
        dist = np.sqrt(ddx * ddx + ddy * ddy)
        coverage = np.clip(self.half_width + 0.5 - dist, 0.0, 1.0)

        m = (self.alpha * coverage)[:, :, None]          # per-pixel opacity
        den = float((m * m).sum(dtype=np.float64))
        if den < 1e-6:
            return None
        tp = self.target[ya:yb, xa:xb]
        cp = self.canvas[ya:yb, xa:xb]
        u = tp - (1.0 - m) * cp
        colour = np.clip((m * u).sum(axis=(0, 1)) / den, 0.0, 1.0)
        colour = np.round(colour * 255.0) / 255.0        # 8-bit thread colour
        new = cp * (1.0 - m) + m * colour

        before = ((tp - cp) ** 2).sum(dtype=np.float64)
        after = ((tp - new) ** 2).sum(dtype=np.float64)
        if after >= before and not force:
            return None
        self.canvas[ya:yb, xa:xb] = new
        return colour.astype(np.float32)

    # -- main loop -----------------------------------------------------------
    def run(
        self,
        counts: Sequence[int],
        on_snapshot: Callable[[int, np.ndarray, np.ndarray], None],
    ) -> np.ndarray:
        """Place threads until max(counts); call on_snapshot at every count."""
        wanted = set(counts)
        total = max(wanted)
        threads = np.zeros((total, 8), dtype=np.float32)
        scale = np.array([self.W, self.H, self.W, self.H], dtype=np.float64)
        n = steps = rejects = 0
        while n < total:
            if steps % self.REFRESH == 0:
                self.refresh_error_map()
            steps += 1
            line = self.find_best()
            colour = self.commit(line, force=rejects >= self.MAX_REJECTS)
            if colour is None:
                rejects += 1
                continue
            rejects = 0
            threads[n, 0:4] = line / scale
            threads[n, 4:7] = colour if self.C == 3 else colour[0]
            threads[n, 7] = self.alpha
            length = math.hypot(line[2] - line[0], line[3] - line[1])
            self.log_len = 0.97 * self.log_len + 0.03 * math.log(max(length, self.min_len))
            n += 1
            if n in wanted:
                on_snapshot(n, threads[:n], self.canvas)
        return threads


def render_threads(
    threads: np.ndarray,
    width_px: int,
    height_px: int,
    background: Sequence[float],
    thread_width: float,
    canvas: np.ndarray | None = None,
) -> np.ndarray:
    """Render an N x 8 thread matrix onto a background canvas.

    If `canvas` is given, the threads are drawn onto it in place and the same
    array is returned. Threads are drawn in order, so drawing threads 0..a and
    then a..b onto that canvas gives exactly the pixels of drawing 0..b at once.
    """
    c_channels = len(background)
    if canvas is None:
        canvas = np.empty((height_px, width_px, c_channels), dtype=np.float32)
        canvas[:] = np.asarray(background, dtype=np.float32).reshape(1, 1, -1)
    half_width = float(thread_width) / 2.0
    reach = half_width + 1.0

    scale = np.array([width_px, height_px, width_px, height_px], dtype=np.float32)
    lines = threads[:, :4].astype(np.float32) * scale
    colors = (
        threads[:, 4:4 + c_channels].astype(np.float32)
        if threads.shape[1] >= 4 + c_channels
        else threads[:, 4:7].astype(np.float32)
    )
    alphas = threads[:, 7].astype(np.float32)

    for i in range(len(threads)):
        x0, y0, x1, y1 = lines[i]
        xa = max(int(math.floor(min(x0, x1) - reach)), 0)
        xb = min(int(math.ceil(max(x0, x1) + reach)), width_px)
        ya = max(int(math.floor(min(y0, y1) - reach)), 0)
        yb = min(int(math.ceil(max(y0, y1) + reach)), height_px)
        if xa >= xb or ya >= yb:
            continue

        px = (np.arange(xa, xb, dtype=np.float32) + 0.5)[None, :]
        py = (np.arange(ya, yb, dtype=np.float32) + 0.5)[:, None]
        dx, dy = x1 - x0, y1 - y0
        len2 = dx * dx + dy * dy
        if len2 > 1e-9:
            t = np.clip(((px - x0) * dx + (py - y0) * dy) / len2, 0.0, 1.0)
            ddx = px - (x0 + t * dx)
            ddy = py - (y0 + t * dy)
        else:
            ddx = px - x0
            ddy = py - y0
        dist = np.sqrt(ddx * ddx + ddy * ddy)
        coverage = np.clip(half_width + 0.5 - dist, 0.0, 1.0)

        m = (alphas[i] * coverage)[:, :, None]
        canvas[ya:yb, xa:xb] = canvas[ya:yb, xa:xb] * (1.0 - m) + m * colors[i]

    return canvas


def save_png(path: str, canvas: np.ndarray) -> None:
    pixels = np.clip(canvas * 255.0 + 0.5, 0, 255).astype(np.uint8)
    if pixels.shape[2] == 1:
        pixels = np.repeat(pixels, 3, axis=2)
    Image.fromarray(pixels).save(path)


def save_svg(
    path: str,
    threads: np.ndarray,
    width_px: int,
    height_px: int,
    background: Sequence[float],
    thread_width: float,
) -> None:
    """Write threads as SVG lines, in order, on a plain background."""
    xy = (threads[:, :4].astype(np.float64) * np.array([width_px, height_px, width_px, height_px])).tolist()
    rgb = np.rint(threads[:, 4:7] * 255.0).astype(np.int64).tolist()
    alphas = threads[:, 7].astype(np.float64).tolist()
    same_alpha = all(a == alphas[0] for a in alphas)
    group_opacity = f' stroke-opacity="{alphas[0]:.3g}"' if same_alpha else ""

    rows = []
    for (a, b, c, d), (r, g, bl), alpha in zip(xy, rgb, alphas):
        opacity = "" if same_alpha else f' stroke-opacity="{alpha:.3g}"'
        rows.append(
            f'<line x1="{a:.2f}" y1="{b:.2f}" x2="{c:.2f}" y2="{d:.2f}" '
            f'stroke="#{r:02x}{g:02x}{bl:02x}"{opacity}/>'
        )

    header = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_px}" height="{height_px}" '
        f'viewBox="0 0 {width_px} {height_px}">\n'
        f'<rect width="{width_px}" height="{height_px}" fill="{to_hex(background)}"/>\n'
        f'<g fill="none" stroke-linecap="round" stroke-width="{thread_width:g}"{group_opacity}>\n'
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(header)
        handle.write("\n".join(rows))
        handle.write("\n</g>\n</svg>\n")


# --------------------------------------------------------------------------
# One (colour mode, alpha) run; executed in a worker process
# --------------------------------------------------------------------------
def run_job(job: dict[str, Any]) -> list[str]:
    mode: str = job["mode"]
    alpha: float = job["alpha"]
    target: np.ndarray = job["target"]
    counts: list[int] = job["counts"]
    out_dir: str = job["out_dir"]
    stem: str = job["stem"]
    background: tuple[float, float, float] = job["background"]
    height, width = target.shape[:2]
    pad = len(str(max(counts)))
    tag = f"{mode:<5} alpha={alpha:g}"

    if mode == "color":
        paint_bg: Sequence[float] = background
    else:
        paint_bg = (0.299 * background[0] + 0.587 * background[1] + 0.114 * background[2],)

    painter = ThreadPainter(target, alpha, paint_bg, job["thread_width"], job["seed"])
    written: list[str] = []
    started = time.time()

    def snapshot(n: int, threads: np.ndarray, canvas: np.ndarray) -> None:
        base = os.path.join(out_dir, f"{stem}_{mode}_{n:0{pad}d}threads_alpha{alpha:.1f}")
        save_png(base + ".png", canvas)
        save_svg(base + ".svg", threads, width, height, background, job["thread_width"])
        written.extend([base + ".png", base + ".svg"])
        if job["save_npy"]:
            dtype = job["npy_dtype"]
            if dtype == "uint4":
                q = np.clip(np.round(threads * 15.0), 0, 15).astype(np.uint8)
                packed = np.empty((len(threads), 4), dtype=np.uint8)
                for i in range(4):
                    packed[:, i] = (q[:, 2 * i] << 4) | (q[:, 2 * i + 1] & 0x0F)
                recon = q.astype(np.float32) / 15.0
                u4_canvas = render_threads(recon, width, height, paint_bg, job["thread_width"])
                save_png(base + "_uint4.png", u4_canvas)
                save_svg(base + "_uint4.svg", recon, width, height, background, job["thread_width"])
                written.extend([base + "_uint4.png", base + "_uint4.svg"])
                matrix = packed
            elif dtype == "uint8":
                matrix = np.clip(np.round(threads * 255.0), 0, 255).astype(np.uint8)
                recon = matrix.astype(np.float32) / 255.0
                u8_canvas = render_threads(recon, width, height, paint_bg, job["thread_width"])
                save_png(base + "_uint8.png", u8_canvas)
                save_svg(base + "_uint8.svg", recon, width, height, background, job["thread_width"])
                written.extend([base + "_uint8.png", base + "_uint8.svg"])
            else:
                matrix = threads.astype(np.float32)
            np.save(base + ".npy", matrix)
            written.append(base + ".npy")
        mse = float(((target - canvas) ** 2).mean())
        psnr = 10.0 * math.log10(1.0 / max(mse, 1e-12))
        log(f"[{tag}] {n:>{pad}d} threads  PSNR {psnr:5.2f} dB  ({time.time() - started:5.1f}s)")

    log(f"[{tag}] start")
    painter.run(counts, snapshot)
    return written


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------
def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an image into digital filography (straight threads)."
    )
    parser.add_argument("image", help="path to the input image")
    parser.add_argument(
        "-o", "--out-dir", default=None,
        help="output folder (default: <image name>_threads next to the input image)",
    )
    parser.add_argument(
        "--counts", type=int, nargs="+", default=list(DEFAULT_COUNTS),
        help="thread counts to export (default: 5000 10000 15000 20000)",
    )
    parser.add_argument(
        "--modes", nargs="+", choices=list(MODES), default=list(MODES),
        help=f"colour modes to run: color, bw (default: {' '.join(MODES)})",
    )
    parser.add_argument(
        "--alphas", type=float, nargs="+", default=list(ALPHAS),
        help=f"thread alphas to run (default: {' '.join(str(a) for a in ALPHAS)})",
    )
    parser.add_argument(
        "--size", type=int, default=640,
        help="length in px of the longest side of the output (default: 640)",
    )
    parser.add_argument(
        "--width", type=float, default=2.0,
        help="thread thickness in px at --size (default: 2.0)",
    )
    parser.add_argument(
        "--bg", type=parse_color, default=(1.0, 1.0, 1.0),
        help="background: white, black or #rrggbb (default: white)",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    parser.add_argument(
        "-j", "--jobs", type=int, default=None,
        help="parallel worker processes (default: up to 4, limited by CPU count)",
    )
    parser.add_argument(
        "--npy", nargs="?", const="float32", default=None, choices=["float32", "uint8", "uint4"],
        help="save thread matrices (N x 8) as .npy (format: float32, uint8, or uint4; default: float32)",
    )
    parser.add_argument(
        "--npy-dtype", choices=["float32", "uint8", "uint4"], default=None,
        help="explicit data type for .npy export: float32, uint8, or uint4",
    )
    args = parser.parse_args(argv)

    if any(c <= 0 for c in args.counts):
        parser.error("--counts must be positive integers")
    if args.size < 16:
        parser.error("--size must be at least 16")
    if args.width <= 0:
        parser.error("--width must be positive")
    args.counts = sorted(set(args.counts))

    npy_dtype = args.npy_dtype or args.npy
    args.save_npy = bool(npy_dtype)
    args.npy_dtype = npy_dtype or "float32"
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not os.path.isfile(args.image):
        log(f"error: no such file: {args.image}")
        return 2

    img = load_image(args.image, args.size, args.bg)
    width, height = img.size
    stem = os.path.splitext(os.path.basename(args.image))[0]
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.image)), f"{stem}_threads"
    )
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as exc:
        log(f"error: cannot create output folder {out_dir!r}: {exc}")
        return 2
    out_dir = os.path.abspath(out_dir)

    color = np.asarray(img, dtype=np.float32) / 255.0
    gray = (np.asarray(img.convert("L"), dtype=np.float32) / 255.0)[:, :, None]
    targets = {"color": color, "bw": gray}

    jobs = [
        {
            "mode": mode,
            "alpha": alpha,
            "target": targets[mode],
            "counts": args.counts,
            "out_dir": out_dir,
            "stem": stem,
            "background": args.bg,
            "thread_width": args.width,
            "seed": args.seed + 1000 * index,
            "save_npy": args.save_npy,
            "npy_dtype": args.npy_dtype,
        }
        for index, (mode, alpha) in enumerate(itertools.product(args.modes, args.alphas))
    ]

    workers = args.jobs if args.jobs else min(4, os.cpu_count() or 1)
    workers = max(1, min(workers, len(jobs)))
    log(f"{width}x{height} px, counts {args.counts}, {len(jobs)} runs, {workers} worker(s)")

    results: list[list[str]]
    if workers > 1:
        try:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(run_job, jobs))
        except (BrokenProcessPool, OSError, NotImplementedError) as exc:
            log(f"parallel run failed ({exc}); falling back to a single process")
            results = [run_job(job) for job in jobs]
    else:
        results = [run_job(job) for job in jobs]

    for paths in results:
        for path in paths:
            print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
