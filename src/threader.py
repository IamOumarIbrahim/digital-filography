#!/usr/bin/env python3
"""
threader.py : turn a bitmap image into "threads".

A thread is one straight line, 1 px wide, stored as a vector of 8 numbers:

    [Red, Green, Blue, Alpha, X1, Y1, X2, Y2]

    colours and alpha: integers 0..255
    coordinates: integers 0..639, measured at pixel centres
    background: always transparent (nothing is drawn except threads)

The input image is centre-cropped to a square and resized to 640 x 640.

Usage
-----
    python threader.py <path_to_image>
    python threader.py photo.jpg --out results --bg "#ffffff" --budget 1.5 --only 1,3 --npy
    python threader.py photo.jpg --threads 8000

For every algorithm you get  <name>.png  and  <name>.svg  in the output folder.

The nine algorithms (all of them are original designs)
------------------------------------------------------
 1 greedy_residual   analysis by synthesis. Thousands of random candidate lines
                     are scored, the best one is kept, then polished by local search.
 2 flow_streamlines  differential geometry. Lines follow the edge direction of a
                     structure-tensor field and stop when a local "ink budget" is paid.
 3 scanline_dashes   signal processing. Each pixel row is a 1D signal that is
                     converted to dashes with equal ink mass.
 4 radon_peel        transform domain. A Radon-style line-sum table is built, the
                     strongest chord is cut into coloured runs, ink is removed, repeat.
 5 stipple_tour      graph / path. Importance-sampled points joined by one
                     walking thread (greedy nearest neighbour).
 6 edge_tangent      gradient tangents. Edge-guided tangent tracing with density
                     weighted by gradient magnitude.
 7 greedy_error      tournament search. Greedy error-reduction ray evaluation
                     with spatial suppression.
 8 quadtree_hatch    structural division. Quadtree spatial decomposition with
                     directional cross-hatching.
 9 anchor_web        graph proximity. Structural anchor nodes connected in a
                     proximity web filtered by color coherence.

About prior work: no published method was copied, so there are no DOIs to list.
Only textbook tools are used as building blocks (Sobel filter, Gaussian blur,
structure tensor, line sums over an image, nearest neighbour search).

Tone model used by algorithms 2 to 5
------------------------------------
Because the background is transparent and a thread is only 1 px wide, tone is
produced by line density. For a target colour t over a background colour bg we
compute the smallest coverage q that can still reproduce t (the "ink need"), and
we draw threads in the most saturated colour on the ray bg -> t. Where the line
density is about q, the eye (or a downscaled image) sees t again.
Use --bg to say what colour you will look at the PNG/SVG on (default white).
"""

import argparse
import heapq
import math
import os
import sys
import time

import numpy as np
from PIL import Image, ImageOps

SIZE = 640
QMIN = 0.08  # ink need below this value is treated as "let the background show"


# ----------------------------------------------------------------------------
# basic helpers
# ----------------------------------------------------------------------------
def parse_color(text):
    text = text.strip().lstrip("#")
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    return np.array([int(text[i:i + 2], 16) for i in (0, 2, 4)], np.float64) / 255.0


def load_image(path, bg):
    im = ImageOps.exif_transpose(Image.open(path))
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        back = Image.new("RGBA", im.size, tuple(int(round(c * 255)) for c in bg) + (255,))
        im = Image.alpha_composite(back, im)
    im = im.convert("RGB")
    w, h = im.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    im = im.crop((left, top, left + s, top + s)).resize((SIZE, SIZE), Image.LANCZOS)
    return np.asarray(im, dtype=np.float32) / 255.0


def gauss_blur(a, sigma):
    """Separable Gaussian blur on axes 0 and 1 (works for 2D and 3D arrays)."""
    a = np.asarray(a, dtype=np.float32)
    if sigma <= 0:
        return a
    r = int(math.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k = (k / k.sum()).astype(np.float32)
    out = a
    for axis in (0, 1):
        pad = [(0, 0)] * a.ndim
        pad[axis] = (r, r)
        p = np.pad(out, pad, mode="edge")
        n = out.shape[axis]
        acc = np.zeros_like(out)
        for i, w in enumerate(k):
            sl = [slice(None)] * a.ndim
            sl[axis] = slice(i, i + n)
            acc += w * p[tuple(sl)]
        out = acc
    return out


def sobel(g):
    p = np.pad(g, 1, mode="edge")
    gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])
    gy = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])
    return gx / 8.0, gy / 8.0


def line_pixels(x1, y1, x2, y2):
    """Pixels covered by a 1 px thread. Returns (ys, xs). One pixel per major-axis step."""
    n = max(abs(x2 - x1), abs(y2 - y1)) + 1
    if n == 1:
        return np.array([y1], dtype=np.intp), np.array([x1], dtype=np.intp)
    t = np.linspace(0.0, 1.0, n)
    xs = np.floor(x1 + (x2 - x1) * t + 0.5).astype(np.intp)
    ys = np.floor(y1 + (y2 - y1) * t + 0.5).astype(np.intp)
    return ys, xs


def ink_need(t, bg):
    """Smallest coverage in [0,1] that can still reproduce colour t over background bg."""
    up = (t - bg) / np.maximum(1.0 - bg, 1e-6)
    dn = (bg - t) / np.maximum(bg, 1e-6)
    return np.clip(np.where(t > bg, up, dn), 0.0, 1.0).max(axis=-1)


def ink_colour(m, bg):
    """Most saturated colour on the ray bg -> m (works for one colour or an (n,3) array)."""
    q = np.maximum(ink_need(m, bg), 0.05)
    q = q[..., None] if np.ndim(q) else q
    return np.clip(bg + (m - bg) / q, 0.0, 1.0)


def rgb255(c):
    return [int(v) for v in np.rint(np.clip(c, 0, 1) * 255)]


def ink_thread(T, bg, x1, y1, x2, y2):
    ys, xs = line_pixels(x1, y1, x2, y2)
    c = ink_colour(T[ys, xs].mean(axis=0), bg)
    return (*rgb255(c), 255, int(x1), int(y1), int(x2), int(y2))


def tangent_field(T, hatch_deg=45.0, sigma_pre=1.0, sigma_tensor=4.0):
    """
    Unit vectors (vx, vy) that run ALONG edges (perpendicular to the gradient).
    A structure tensor is summed over R,G,B. A small constant tensor is added so that
    flat areas get a calm hatch direction (hatch_deg) instead of noise.
    """
    Tb = gauss_blur(T, sigma_pre)
    jxx = jxy = jyy = 0.0
    for ch in range(3):
        gx, gy = sobel(Tb[:, :, ch])
        jxx = jxx + gx * gx
        jxy = jxy + gx * gy
        jyy = jyy + gy * gy
    jxx, jxy, jyy = (gauss_blur(j, sigma_tensor) for j in (jxx, jxy, jyy))
    bias = 0.5 * float(np.mean(jxx + jyy)) + 1e-9
    phi_n = math.radians(hatch_deg + 90.0)
    nx, ny = math.cos(phi_n), math.sin(phi_n)
    jxx = jxx + bias * nx * nx
    jxy = jxy + bias * nx * ny
    jyy = jyy + bias * ny * ny
    phi = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)  # gradient direction
    return (-np.sin(phi)).astype(np.float32), np.cos(phi).astype(np.float32)


def split_colour(cols, tol=0.16, minlen=6):
    """Cut a run of pixel colours into pieces whose colour stays close to the piece start."""
    n = len(cols)
    out, i = [], 0
    while i < n:
        ref = cols[i:i + 4].mean(axis=0)
        bad = np.flatnonzero(np.abs(cols[i:] - ref).max(axis=1) > tol)
        j = i + (int(bad[0]) if bad.size else n - i)
        if j <= i:
            j = i + 1
        if j - i >= minlen:
            out.append((i, j - 1))
        i = j
    return out


def make_progress(name):
    state = {"t": 0.0}

    def log(done, total):
        now = time.time()
        if now - state["t"] > 0.4 or done >= total:
            state["t"] = now
            pct = min(100.0, 100.0 * done / max(total, 1))
            sys.stderr.write(f"\r    {name:<17} {pct:5.1f}%")
            sys.stderr.flush()

    return log


# ----------------------------------------------------------------------------
# ALGORITHM 1: greedy residual search (analysis by synthesis)
# ----------------------------------------------------------------------------
def algo_greedy_residual(T, bg, field, rng, budget, log, target_threads=None):
    """
    Keep a canvas (threads composited over bg). Repeat:
      1. draw 20 random candidate lines, centred where the error is large, half of them
         aligned with the local edge direction;
      2. for each line find the best colour for 3 alpha levels in closed form, and
         the error it would remove;
      3. keep the best, polish its endpoints with 8 random nudges, commit it.
    Lines start long and opaque and become short and faint (coarse to fine).
    """
    T = T.astype(np.float64)
    n_target = target_threads if target_threads is not None else int(5500 * budget)
    canvas = np.empty_like(T)
    canvas[:] = bg
    err = ((T - canvas) ** 2).sum(axis=2)
    ang = np.arctan2(field[1], field[0])
    ALPHAS = (0.3, 0.6, 1.0)
    threads = []

    def clip_i(v):
        return int(min(max(round(v), 0), SIZE - 1))

    def evaluate(x1, y1, x2, y2):
        ys, xs = line_pixels(x1, y1, x2, y2)
        n = ys.size
        if n < 4:
            return None
        V = canvas[ys, xs]
        Tp = T[ys, xs]
        D = V - Tp
        e_old = float((D * D).sum())
        sD = D.sum(axis=0)
        sDT = (D * Tp).sum(axis=0)
        sT = Tp.sum(axis=0)
        sTT = (Tp * Tp).sum(axis=0)
        mD = sD / n
        mV = (sD + sT) / n
        best = None
        for a in ALPHAS:
            c = np.clip(mV - mD / a, 0.0, 1.0)
            e_new = ((1 - a) ** 2 * e_old
                     + 2 * a * (1 - a) * float((c * sD - sDT).sum())
                     + a * a * float((n * c * c - 2 * c * sT + sTT).sum()))
            g = e_old - e_new
            if best is None or g > best[0]:
                best = (g, a, c)
        return best

    cdf = None
    for it in range(n_target * 2):
        if len(threads) >= n_target:
            break
        if it % 25 == 0:
            cdf = np.cumsum(err.ravel())
            if cdf[-1] < 1e-6:
                break
        progress = len(threads) / n_target
        Lmax = 360.0 * (0.12 ** progress)
        Lmin = max(8.0, 0.2 * Lmax)
        K = 20
        flat = np.minimum(np.searchsorted(cdf, rng.random(K) * cdf[-1]), cdf.size - 1)
        cy, cx = np.divmod(flat, SIZE)
        use_field = rng.random(K) < 0.6
        theta = np.where(use_field, ang[cy, cx] + rng.normal(0, 0.3, K), rng.uniform(0, np.pi, K))
        L = rng.uniform(Lmin, Lmax, K)
        dx, dy = np.cos(theta) * L / 2, np.sin(theta) * L / 2
        best = None
        for k in range(K):
            cand = (clip_i(cx[k] - dx[k]), clip_i(cy[k] - dy[k]), clip_i(cx[k] + dx[k]), clip_i(cy[k] + dy[k]))
            r = evaluate(*cand)
            if r is not None and (best is None or r[0] > best[0][0]):
                best = (r, cand)
        if best is None or best[0][0] <= 0:
            continue
        (g, a, c), (x1, y1, x2, y2) = best
        sig = max(2.0, 0.05 * math.hypot(x2 - x1, y2 - y1))
        for _ in range(8):
            mode = int(rng.integers(3))
            j = rng.normal(0, sig, 4)
            if mode == 0:
                j[2:] = 0
            elif mode == 1:
                j[:2] = 0
            else:
                j[2:] = j[:2]
            cand = (clip_i(x1 + j[0]), clip_i(y1 + j[1]), clip_i(x2 + j[2]), clip_i(y2 + j[3]))
            r = evaluate(*cand)
            if r is not None and r[0] > g:
                (g, a, c), (x1, y1, x2, y2) = r, cand
        if g <= 0:
            continue
        ys, xs = line_pixels(x1, y1, x2, y2)
        V = canvas[ys, xs]
        newV = V + a * (c - V)
        canvas[ys, xs] = newV
        err[ys, xs] = ((T[ys, xs] - newV) ** 2).sum(axis=1)
        threads.append((*rgb255(c), int(round(a * 255)), x1, y1, x2, y2))
        if len(threads) % 50 == 0:
            log(len(threads), n_target)
    log(n_target, n_target)
    return threads


# ----------------------------------------------------------------------------
# ALGORITHM 2: edge-tangent streamlines with a local ink budget
# ----------------------------------------------------------------------------
def greedy_chords(P, cols, tol=0.8, ctol=0.18):
    """Merge consecutive path points into long straight pieces (path error <= tol px)."""
    n = len(P)
    out, i = [], 0
    while i < n - 1:
        j = i + 1
        while j + 1 < n:
            a, b = P[i], P[j + 1]
            d = b - a
            L = math.hypot(d[0], d[1])
            if L < 1e-9:
                break
            seg = P[i + 1:j + 1]
            dev = np.abs((seg[:, 0] - a[0]) * d[1] - (seg[:, 1] - a[1]) * d[0]) / L
            if dev.max() > tol:
                break
            cc = cols[i:j + 2]
            if (cc.max(axis=0) - cc.min(axis=0)).max() > ctol:
                break
            j += 1
        out.append((i, j))
        i = j
    return out


def algo_flow_streamlines(T, bg, field, rng, budget, log, target_threads=None):
    """
    The image is cut into 4x4 cells. Each cell "needs" 16*q ink pixels (q = ink need).
    Seeds are drawn where the unpaid ink is largest. From a seed we walk both ways along
    the edge-tangent field in 3 px steps. We stop at a sharp turn, at the border or when
    the next cell is already paid. The path is merged into long straight threads, and
    every thread pays for the cells it crosses.
    """
    vx, vy = field
    q = ink_need(T, bg)
    q = np.where(q > QMIN, q, 0.0)
    C = 4
    nc = SIZE // C
    deficit = q.reshape(nc, C, nc, C).sum(axis=(1, 3)).astype(np.float64)
    max_threads = target_threads if target_threads is not None else int(14000 * budget)
    STEP, MAXSTEPS = 3.0, 70
    THR_SEED, THR_CONT = 2.0, 1.0
    inked = np.zeros((SIZE, SIZE), bool)   # pixels already covered by a thread
    threads = []

    def trace(x0, y0):
        halves = []
        for sign in (1.0, -1.0):
            x, y = x0, y0
            pdx = pdy = None
            side = []
            for _ in range(MAXSTEPS):
                dx = float(vx[int(y + 0.5), int(x + 0.5)])
                dy = float(vy[int(y + 0.5), int(x + 0.5)])
                if pdx is None:
                    dx, dy = dx * sign, dy * sign
                else:
                    dot = dx * pdx + dy * pdy
                    if dot < 0:
                        dx, dy, dot = -dx, -dy, -dot
                    if dot < 0.5:
                        break
                nx_, ny_ = x + STEP * dx, y + STEP * dy
                if nx_ < 0 or ny_ < 0 or nx_ > SIZE - 1 or ny_ > SIZE - 1:
                    break
                if deficit[int(ny_) // C, int(nx_) // C] < THR_CONT:
                    break
                side.append((nx_, ny_))
                x, y, pdx, pdy = nx_, ny_, dx, dy
            halves.append(side)
        return halves[1][::-1] + [(x0, y0)] + halves[0]

    cdf, since = None, 0
    max_attempts = max(80000, max_threads * 6)
    for attempt in range(max_attempts):
        if len(threads) >= max_threads:
            break
        if cdf is None or since >= 12:
            cdf = np.cumsum(np.maximum(deficit.ravel() - THR_SEED, 0.0))
            since = 0
            if cdf[-1] <= 0:
                break
        since += 1
        flat = min(int(np.searchsorted(cdf, rng.random() * cdf[-1])), cdf.size - 1)
        cy, cx = divmod(flat, nc)
        if deficit[cy, cx] <= THR_SEED:
            continue
        x0 = min((cx + rng.random()) * C, SIZE - 1)
        y0 = min((cy + rng.random()) * C, SIZE - 1)
        pts = trace(x0, y0)
        if len(pts) < 2:
            deficit[cy, cx] -= THR_SEED
            continue
        P = np.array(pts)
        cols = T[np.rint(P[:, 1]).astype(int), np.rint(P[:, 0]).astype(int)]
        added = 0
        for i, j in greedy_chords(P, cols):
            if target_threads is not None and len(threads) >= max_threads:
                break
            x1, y1 = int(round(P[i, 0])), int(round(P[i, 1]))
            x2, y2 = int(round(P[j, 0])), int(round(P[j, 1]))
            if x1 == x2 and y1 == y2:
                continue
            ys, xs = line_pixels(x1, y1, x2, y2)
            fresh = ~inked[ys, xs]
            if fresh.mean() < 0.5:      # mostly on top of older threads: useless
                continue
            inked[ys, xs] = True
            np.subtract.at(deficit, (ys[fresh] // C, xs[fresh] // C), 1.0)
            threads.append(ink_thread(T, bg, x1, y1, x2, y2))
            added += 1
        if added == 0:
            deficit[cy, cx] -= THR_SEED
        if attempt % 40 == 0:
            log(len(threads), max_threads)
    if target_threads is not None and len(threads) > max_threads:
        threads = threads[:max_threads]
    log(1, 1)
    return threads


# ----------------------------------------------------------------------------
# ALGORITHM 3: scanline dashes with equal ink mass
# ----------------------------------------------------------------------------
def _scanline_dashes_pass(T, bg, q, M, rng, log):
    MAXLEN, CTOL = 200, 24
    qT = q[:, :, None] * T
    phases = rng.random(SIZE)
    threads = []
    for y in range(SIZE):
        # each row starts with a different amount of "pre-paid" mass (random phase),
        # so the dashes of neighbouring rows do not line up into vertical stripes
        cum0 = np.cumsum(q[y])
        cum = cum0 + phases[y] * M
        nd = int(cum[-1] // M)
        if nd < 2:
            continue
        k = np.arange(1, nd)
        ctr = np.searchsorted(cum, (k + 0.5) * M)
        lo = np.searchsorted(cum, k * M, side="right")
        hi = np.minimum(np.searchsorted(cum, (k + 1) * M), SIZE - 1)
        cq = np.concatenate(([0.0], cum0))
        cw = np.vstack([np.zeros((1, 3)), np.cumsum(qT[y], axis=0)])
        mass = np.maximum(cq[hi + 1] - cq[lo], 1e-6)
        m = (cw[hi + 1] - cw[lo]) / mass[:, None]
        col = np.rint(ink_colour(m, bg) * 255).astype(int).tolist()
        x1 = np.clip(np.floor(ctr - (M - 1) / 2.0 + 0.5), 0, SIZE - 1).astype(int).tolist()
        x2 = np.clip(np.array(x1) + M - 1, 0, SIZE - 1).tolist()
        cur = None  # [start, end, [r,g,b]]
        for i in range(len(x1)):
            a, b, c = x1[i], x2[i], col[i]
            if (cur is not None and a <= cur[1] + 1 and b - cur[0] < MAXLEN
                    and max(abs(c[0] - cur[2][0]), abs(c[1] - cur[2][1]), abs(c[2] - cur[2][2])) <= CTOL):
                n0, n1 = cur[1] - cur[0] + 1, b - cur[1]
                cur[2] = [(cur[2][t] * n0 + c[t] * n1) / (n0 + n1) for t in range(3)]
                cur[1] = max(cur[1], b)
            else:
                if cur is not None and cur[1] > cur[0]:
                    threads.append((*[int(round(v)) for v in cur[2]], 255, cur[0], y, cur[1], y))
                cur = [a, b, [float(v) for v in c]]
        if cur is not None and cur[1] > cur[0]:
            threads.append((*[int(round(v)) for v in cur[2]], 255, cur[0], y, cur[1], y))
        if log is not None and y % 16 == 0:
            log(y, SIZE)
    if log is not None:
        log(1, 1)
    return threads


def algo_scanline_dashes(T, bg, field, rng, budget, log, target_threads=None):
    """
    Every pixel row is a 1D signal q(x) (ink need). We cut it into spans that each hold
    the same ink mass M. In each span one dash of length M is centred on the span's
    "centre of mass". Since q <= 1, a span is never shorter than M, so dashes never
    overlap, and the ink drawn equals the ink needed. Touching dashes of similar
    colour are merged into longer threads.
    """
    q = ink_need(T, bg)
    q = np.where(q > QMIN, q, 0.0)
    if target_threads is None:
        M = max(3, int(round(10.0 / budget)))
        return _scanline_dashes_pass(T, bg, q, M, rng, log)

    q_sum = float(q.sum())
    M = max(3, int(round(q_sum / (target_threads * 1.08))))
    seed_val = int(rng.integers(0, 1000000))
    th = _scanline_dashes_pass(T, bg, q, M, np.random.default_rng(seed_val), None)
    best_M = M
    best_th = th
    if len(th) > target_threads and M < SIZE:
        M2 = M + 1
        th2 = _scanline_dashes_pass(T, bg, q, M2, np.random.default_rng(seed_val), None)
        if abs(len(th2) - target_threads) < abs(len(best_th) - target_threads):
            best_M = M2
            best_th = th2
    elif len(th) < target_threads and M > 3:
        M2 = M - 1
        th2 = _scanline_dashes_pass(T, bg, q, M2, np.random.default_rng(seed_val), None)
        if abs(len(th2) - target_threads) < abs(len(best_th) - target_threads):
            best_M = M2
            best_th = th2

    log(1, 1)
    return best_th


# ----------------------------------------------------------------------------
# ALGORITHM 4: Radon-style chord peeling
# ----------------------------------------------------------------------------
def algo_radon_peel(T, bg, field, rng, budget, log, target_threads=None):
    """
    Build a table S[angle, offset] = total remaining ink need along every straight chord
    (120 angles, 1 px offset bins). Then repeat:
      1. take the chord with the largest total;
      2. walk along it, keep the places where ink is still needed, bridge gaps of
         <= 4 px, and cut where the colour drifts;
      3. every run becomes a thread; its ink (spread over a 5 px band) is removed from the
         image and S is updated incrementally.
    The result looks like string art, but every line is cut to where it is really needed.
    """
    q = ink_need(T, bg).astype(np.float32)
    TAU = QMIN
    H = 2                       # half width of the ink band that one thread pays for
    MINLEN, GAP = 8, 4
    if target_threads is not None:
        n_chords = max(int(target_threads), 4000)
    else:
        n_chords = int(8000 * budget)
    S_MIN = 30.0

    A = 120
    thetas = np.arange(A) * np.pi / A
    cs, sn = np.cos(thetas), np.sin(thetas)
    c0 = (SIZE - 1) / 2.0
    R = int(math.ceil(c0 * math.sqrt(2.0))) + 2
    NB = 2 * R + 1

    inked = np.zeros((SIZE, SIZE), bool)   # pixels already covered by a thread
    Dm = q.copy().ravel()       # remaining ink need, flat
    def eff(v):
        return np.where(v > TAU, v, 0.0)

    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    xc = (xx - c0).astype(np.float64).ravel()
    yc = (yy - c0).astype(np.float64).ravel()
    E = eff(Dm).astype(np.float64)
    S = np.zeros((A, NB), np.float64)
    for a in range(A):
        idx = np.floor(xc * cs[a] + yc * sn[a] + 0.5).astype(np.int64) + R
        S[a] = np.bincount(idx, weights=E, minlength=NB)
    Sf = S.reshape(-1)
    Dm2 = Dm.reshape(SIZE, SIZE)
    offs = (np.arange(A) * NB)[None, :]

    def chord(a, b):
        rho = b - R
        c_, s_ = cs[a], sn[a]
        lo, hi = -1e9, 1e9
        for p0, dd in ((c0 + rho * c_, -s_), (c0 + rho * s_, c_)):
            if abs(dd) < 1e-9:
                if p0 < -0.5 or p0 > SIZE - 0.5:
                    return None
            else:
                t1, t2 = (0.0 - p0) / dd, (SIZE - 1 - p0) / dd
                lo, hi = max(lo, min(t1, t2)), min(hi, max(t1, t2))
        if hi - lo < MINLEN:
            return None
        ds = 1.0 / max(abs(s_), abs(c_))
        t = np.arange(lo, hi + 1e-9, ds)
        x = np.floor(c0 + rho * c_ - t * s_ + 0.5).astype(np.int64)
        y = np.floor(c0 + rho * s_ + t * c_ + 0.5).astype(np.int64)
        ok = (x >= 0) & (x < SIZE) & (y >= 0) & (y < SIZE)
        return x[ok], y[ok]

    def deposit(x1, y1, x2, y2):
        ys, xs = line_pixels(x1, y1, x2, y2)
        horiz = abs(x2 - x1) >= abs(y2 - y1)
        fresh = ~inked[ys, xs]          # crossings of older threads cost nothing
        inked[ys, xs] = True
        ys, xs = ys[fresh], xs[fresh]
        if ys.size == 0:
            return
        fy, fx = [], []
        for k in range(-H, H + 1):
            if horiz:
                fy.append(ys + k); fx.append(xs)
            else:
                fy.append(ys); fx.append(xs + k)
        fy, fx = np.concatenate(fy), np.concatenate(fx)
        ok = (fy >= 0) & (fy < SIZE) & (fx >= 0) & (fx < SIZE)
        u, cnt = np.unique(fy[ok] * SIZE + fx[ok], return_counts=True)
        old = Dm[u]
        new = np.maximum(old - cnt / (2 * H + 1), 0.0)
        new[np.searchsorted(u, ys * SIZE + xs)] = 0.0   # the thread's own pixels are done
        Dm[u] = new
        delta = eff(new) - eff(old)
        nz = delta != 0
        if not nz.any():
            return
        u2, d2 = u[nz], delta[nz].astype(np.float64)
        px = (u2 % SIZE) - c0
        py = (u2 // SIZE) - c0
        idx = np.floor(px[:, None] * cs[None, :] + py[:, None] * sn[None, :] + 0.5).astype(np.int64) + R
        Sf[:] += np.bincount((idx + offs).ravel(), weights=np.repeat(d2, A), minlength=A * NB)

    threads = []
    chords_used = 0
    total_iters = n_chords * 4
    for it in range(total_iters):
        if target_threads is not None:
            if len(threads) >= target_threads:
                break
        else:
            if chords_used >= n_chords:
                break
        flat = int(np.argmax(Sf))
        if Sf[flat] < S_MIN:
            break
        a, b = divmod(flat, NB)
        ch = chord(a, b)
        made = False
        if ch is not None:
            cx_, cy_ = ch
            idx = np.flatnonzero(Dm2[cy_, cx_] > TAU)
            if idx.size >= MINLEN:
                brk = np.flatnonzero(np.diff(idx) > GAP + 1)
                starts = idx[np.concatenate(([0], brk + 1))]
                ends = idx[np.concatenate((brk, [idx.size - 1]))]
                for s, e in zip(starts, ends):
                    if target_threads is not None and len(threads) >= target_threads:
                        break
                    if e - s + 1 < MINLEN:
                        continue
                    cols = T[cy_[s:e + 1], cx_[s:e + 1]]
                    for i0, i1 in split_colour(cols, 0.16, MINLEN):
                        if target_threads is not None and len(threads) >= target_threads:
                            break
                        x1, y1 = int(cx_[s + i0]), int(cy_[s + i0])
                        x2, y2 = int(cx_[s + i1]), int(cy_[s + i1])
                        if x1 == x2 and y1 == y2:
                            continue
                        threads.append(ink_thread(T, bg, x1, y1, x2, y2))
                        deposit(x1, y1, x2, y2)
                        made = True
        if made:
            chords_used += 1
        else:
            Sf[flat] = 0.0
        if it % 20 == 0:
            if target_threads is not None:
                log(len(threads), target_threads)
            else:
                log(chords_used, n_chords)
    if target_threads is not None and len(threads) > target_threads:
        threads = threads[:target_threads]
    log(1, 1)
    return threads


# ----------------------------------------------------------------------------
# ALGORITHM 5: stipple tour (one walking thread)
# ----------------------------------------------------------------------------
def algo_stipple_tour(T, bg, field, rng, budget, log, target_threads=None):
    """
    1. Put one jittered point in every g x g cell and keep it with a probability equal
       to the (normalised, blurred) ink need there: dense points where the image is dark
       or colourful.
    2. Start at the top-left point and always walk to the nearest unvisited point.
       Every step is a thread. Steps longer than 3.2 g are not drawn (the walker
       "jumps" with the pen lifted).
    """
    q = ink_need(T, bg)
    qn = np.clip(q / max(float(np.percentile(q, 99)), 1e-3), 0.0, 1.0)
    qb = gauss_blur(qn.astype(np.float32), 1.5)
    if target_threads is not None:
        p_keep = float(np.mean((qb > QMIN) * (qb ** 0.8)))
        g = max(1.5, SIZE / math.sqrt((target_threads / 0.97) / max(p_keep, 1e-6)))
    else:
        g = 5.0 / math.sqrt(budget)
    centres = np.arange(g / 2.0, SIZE, g)
    GX, GY = np.meshgrid(centres, centres)
    px = np.clip(GX.ravel() + rng.uniform(-g / 2, g / 2, GX.size), 0, SIZE - 1)
    py = np.clip(GY.ravel() + rng.uniform(-g / 2, g / 2, GY.size), 0, SIZE - 1)
    dens = qb[np.rint(py).astype(int), np.rint(px).astype(int)]
    keep = (dens > QMIN) & (rng.random(px.size) < dens ** 0.8)
    P = np.stack([px[keep], py[keep]], axis=1)
    N = len(P)
    threads = []
    if N < 2:
        return threads
    alive = P.copy()
    cur = int(np.argmin(P[:, 0] + P[:, 1]))
    pos = P[cur].copy()
    alive[cur] = 1e9
    hop_max = 3.2 * g
    for step in range(N - 1):
        d = ((alive - pos) ** 2).sum(axis=1)
        nxt = int(np.argmin(d))
        new = P[nxt]
        if math.sqrt(d[nxt]) <= hop_max:
            x1, y1 = int(round(pos[0])), int(round(pos[1]))
            x2, y2 = int(round(new[0])), int(round(new[1]))
            if x1 != x2 or y1 != y2:
                threads.append(ink_thread(T, bg, x1, y1, x2, y2))
        pos = new.copy()
        alive[nxt] = 1e9
        if step % 200 == 0:
            log(step, N)
    log(1, 1)
    return threads


# ----------------------------------------------------------------------------
# output: thread array, PNG and SVG
# ----------------------------------------------------------------------------
def finalize(threads):
    if threads is None or len(threads) == 0:
        return np.zeros((0, 8), np.int32)
    a = np.asarray(threads, dtype=np.int32)
    a[:, 4:] = np.clip(a[:, 4:], 0, SIZE - 1)
    return a[(a[:, 4] != a[:, 6]) | (a[:, 5] != a[:, 7])]


def render_png(threads):
    """Own rasteriser: 1 px threads composited in order onto a transparent canvas."""
    prem = np.zeros((SIZE, SIZE, 3), np.float32)
    alpha = np.zeros((SIZE, SIZE), np.float32)
    for r, g, b, a, x1, y1, x2, y2 in threads.tolist():
        if a <= 0:
            continue
        ys, xs = line_pixels(x1, y1, x2, y2)
        af = a / 255.0
        col = np.array((r, g, b), np.float32) / 255.0
        prem[ys, xs] = prem[ys, xs] * (1 - af) + col * af
        alpha[ys, xs] = alpha[ys, xs] * (1 - af) + af
    rgb = np.where(alpha[..., None] > 0, prem / np.maximum(alpha[..., None], 1e-6), 0.0)
    out = np.dstack([rgb * 255.0, alpha * 255.0])
    return Image.fromarray(np.clip(np.rint(out), 0, 255).astype(np.uint8))


def write_svg(path, threads):
    # viewBox starts at -0.5 so that integer coordinates are pixel centres.
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SIZE}" height="{SIZE}" '
        f'viewBox="-0.5 -0.5 {SIZE} {SIZE}" shape-rendering="crispEdges">',
        '<g fill="none" stroke-width="1" stroke-linecap="square">',
    ]
    for r, g, b, a, x1, y1, x2, y2 in threads.tolist():
        op = "" if a >= 255 else f' stroke-opacity="{a / 255:.3f}"'
        parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#{r:02x}{g:02x}{b:02x}"{op}/>')
    parts.append("</g></svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


# =============================================================================
# Core Utilities for Algorithms 6-9
# =============================================================================

def compute_gradients(gray):
    """Computes horizontal and vertical gradients and magnitude using NumPy."""
    gx = np.zeros_like(gray, dtype=np.float32)
    gy = np.zeros_like(gray, dtype=np.float32)
    gx[:, 1:-1] = (gray[:, 2:] - gray[:, :-2]) * 0.5
    gy[1:-1, :] = (gray[2:, :] - gray[:-2, :]) * 0.5
    mag = np.hypot(gx, gy)
    return gx, gy, mag


def sample_line_color(img_arr, x1, y1, x2, y2, n_samples=5):
    """
    Vectorized sampling of RGB mean and color variance along segments.
    Accepts 1D arrays for x1, y1, x2, y2.
    Returns (R, G, B, A) arrays as uint8.
    """
    t = np.linspace(0.0, 1.0, n_samples, dtype=np.float32)[:, None]
    xs = np.clip(np.round(x1[None, :] + (x2 - x1)[None, :] * t).astype(np.int32), 0, img_arr.shape[1] - 1)
    ys = np.clip(np.round(y1[None, :] + (y2 - y1)[None, :] * t).astype(np.int32), 0, img_arr.shape[0] - 1)

    samples = img_arr[ys, xs]  # Shape: (n_samples, N, 4)
    rgb_mean = samples[:, :, :3].mean(axis=0)
    rgb_std = samples[:, :, :3].std(axis=0).mean(axis=1)
    alpha_mean = samples[:, :, 3].mean(axis=0)

    # Higher color consistency along the line = higher opacity
    coherence_alpha = np.clip(255.0 - rgb_std * 2.2, 70.0, 255.0)
    final_alpha = (coherence_alpha * (alpha_mean / 255.0)).astype(np.uint8)

    r = np.clip(np.round(rgb_mean[:, 0]), 0, 255).astype(np.uint8)
    g = np.clip(np.round(rgb_mean[:, 1]), 0, 255).astype(np.uint8)
    b = np.clip(np.round(rgb_mean[:, 2]), 0, 255).astype(np.uint8)
    return r, g, b, final_alpha


def make_thread_matrix(r, g, b, a, x1, y1, x2, y2):
    """Packs vectors into an (N, 8) int32 array: [R, G, B, A, X1, Y1, X2, Y2]."""
    return np.column_stack([
        r.astype(np.int32),
        g.astype(np.int32),
        b.astype(np.int32),
        a.astype(np.int32),
        x1.astype(np.int32),
        y1.astype(np.int32),
        x2.astype(np.int32),
        y2.astype(np.int32),
    ])


# =============================================================================
# ALGORITHM 6: Edge-Guided Tangent Tracing
# =============================================================================

def algo1_edge_tangent(img_arr, n_threads, rng):
    """
    Samples centers weighted by gradient magnitude and shoots straight threads
    along local edge tangents (perpendicular to gradient).
    """
    h, w, _ = img_arr.shape
    gray = img_arr[:, :, :3].mean(axis=2)
    gx, gy, mag = compute_gradients(gray)

    weights = (mag + 6.0) * (img_arr[:, :, 3] > 0)
    prob = weights.ravel()
    prob_sum = prob.sum()
    prob = prob / prob_sum if prob_sum > 0 else np.full_like(prob, 1.0 / prob.size)

    flat_idx = rng.choice(h * w, size=n_threads, replace=True, p=prob)
    yc, xc = np.divmod(flat_idx, w)

    # Tangent angle is perpendicular to gradient vector (gx, gy)
    angles = np.arctan2(gy[yc, xc], gx[yc, xc]) + (np.pi * 0.5)
    jitter = rng.normal(0.0, 0.12, size=n_threads)
    angles += jitter

    # Shorter lines on sharp edges for crisp detail; longer lines in smooth areas
    norm_mag = mag[yc, xc] / (mag.max() + 1e-5)
    base_scale = max(4.0, min(w, h) * 0.045)
    half_len = np.clip(base_scale * (1.25 - 0.85 * norm_mag) * rng.uniform(0.6, 1.4, size=n_threads), 2.0, base_scale * 2.0)

    dx = np.cos(angles) * half_len
    dy = np.sin(angles) * half_len

    x1 = np.clip(np.round(xc - dx), 0, w - 1).astype(np.int32)
    y1 = np.clip(np.round(yc - dy), 0, h - 1).astype(np.int32)
    x2 = np.clip(np.round(xc + dx), 0, w - 1).astype(np.int32)
    y2 = np.clip(np.round(yc + dy), 0, h - 1).astype(np.int32)

    r, g, b, a = sample_line_color(img_arr, x1, y1, x2, y2, n_samples=7)
    return make_thread_matrix(r, g, b, a, x1, y1, x2, y2)


# =============================================================================
# ALGORITHM 7: Greedy Error-Reduction Loom
# =============================================================================

def algo2_greedy_error(img_arr, n_threads, rng):
    """
    Iteratively picks high-importance pixels via tournament selection, evaluates
    8 candidate rays, chooses the ray with minimum color error, and suppresses
    visited regions so threads cover the whole canvas cleanly.
    """
    h, w, _ = img_arr.shape
    rgb = img_arr[:, :, :3].astype(np.float32)
    gray = rgb.mean(axis=2)
    _, _, mag = compute_gradients(gray)

    importance = (mag * 2.5 + 15.0) * (img_arr[:, :, 3] / 255.0)
    max_len = max(5, int(min(w, h) * 0.06))

    # 8 directional unit vectors
    theta = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    ux = np.cos(theta)
    uy = np.sin(theta)

    threads = np.zeros((n_threads, 8), dtype=np.int32)

    for i in range(n_threads):
        # Tournament selection of 24 random coordinates to find high priority spot
        ty = rng.integers(0, h, size=24)
        tx = rng.integers(0, w, size=24)
        best_t = np.argmax(importance[ty, tx])
        y1 = int(ty[best_t])
        x1 = int(tx[best_t])

        lengths = rng.integers(3, max_len + 1, size=8)
        cx2 = np.clip(np.round(x1 + ux * lengths).astype(np.int32), 0, w - 1)
        cy2 = np.clip(np.round(y1 + uy * lengths).astype(np.int32), 0, h - 1)
        cmx = (x1 + cx2) // 2
        cmy = (y1 + cy2) // 2

        c_start = rgb[y1, x1]
        c_mid = rgb[cmy, cmx]
        c_end = rgb[cy2, cx2]

        # Error is color difference across start, middle, and end of candidate ray
        err = np.abs(c_end - c_start).sum(axis=1) + np.abs(c_mid - c_start).sum(axis=1)
        best_dir = int(np.argmin(err))

        x2 = int(cx2[best_dir])
        y2 = int(cy2[best_dir])
        mx = int(cmx[best_dir])
        my = int(cmy[best_dir])

        mean_col = (c_start + c_mid[best_dir] + c_end[best_dir]) / 3.0
        line_err = float(err[best_dir]) / 6.0
        alpha = int(np.clip(255.0 - line_err * 1.5, 85.0, 255.0) * (img_arr[y1, x1, 3] / 255.0))

        threads[i] = [
            int(mean_col[0]), int(mean_col[1]), int(mean_col[2]), alpha,
            x1, y1, x2, y2
        ]

        # Reduce importance along the chosen thread so future threads spread out
        importance[y1, x1] *= 0.25
        importance[my, mx] *= 0.45
        importance[y2, x2] *= 0.45

    return threads


# =============================================================================
# ALGORITHM 8: Quadtree Structural Cross-Hatching
# =============================================================================

def algo3_quadtree_hatch(img_arr, n_threads, rng):
    """
    Recursively partitions the image into a Quadtree based on color variance,
    then fills each leaf cell with directional cross-hatch threads.
    """
    h, w, _ = img_arr.shape
    rgb = img_arr[:, :, :3].astype(np.float32)
    gray = rgb.mean(axis=2)
    gx, gy, _ = compute_gradients(gray)

    target_leaves = max(64, min(n_threads // 3, 2500))
    heap = []
    uid = 0

    def push_box(x0, y0, x1, y1):
        nonlocal uid
        sub = rgb[y0:y1, x0:x1]
        area = (x1 - x0) * (y1 - y0)
        if area <= 16 or (x1 - x0) < 4 or (y1 - y0) < 4:
            priority = 0.0
        else:
            var = float(sub.std(axis=(0, 1)).mean())
            priority = -(var * (area ** 0.65))
        heapq.heappush(heap, (priority, uid, x0, y0, x1, y1))
        uid += 1

    push_box(0, 0, w, h)

    while len(heap) < target_leaves:
        neg_p, _, x0, y0, x1, y1 = heapq.heappop(heap)
        if neg_p == 0.0 or (x1 - x0) < 6 or (y1 - y0) < 6:
            heapq.heappush(heap, (0.0, uid, x0, y0, x1, y1))
            uid += 1
            break
        mx = (x0 + x1) // 2
        my = (y0 + y1) // 2
        push_box(x0, y0, mx, my)
        push_box(mx, y0, x1, my)
        push_box(x0, my, mx, y1)
        push_box(mx, my, x1, y1)

    leaves = [(x0, y0, x1, y1) for _, _, x0, y0, x1, y1 in heap]
    n_leaves = len(leaves)

    # Distribute exact n_threads across leaves based on detail weight
    weights = np.zeros(n_leaves, dtype=np.float64)
    for i, (x0, y0, x1, y1) in enumerate(leaves):
        sub_std = float(rgb[y0:y1, x0:x1].std()) + 5.0
        weights[i] = sub_std * math.sqrt((x1 - x0) * (y1 - y0))

    weights /= weights.sum()
    counts = np.floor(weights * n_threads).astype(np.int32)
    remainder = n_threads - int(counts.sum())
    if remainder > 0:
        top_idx = np.argsort(-(weights * n_threads - counts))[:remainder]
        counts[top_idx] += 1

    x1_list, y1_list, x2_list, y2_list = [], [], [], []

    for (x0, y0, x1, y1), count in zip(leaves, counts):
        if count <= 0:
            continue
        bw = max(1, x1 - x0 - 1)
        bh = max(1, y1 - y0 - 1)
        mgx = float(gx[y0:y1, x0:x1].mean())
        mgy = float(gy[y0:y1, x0:x1].mean())
        angle = math.atan2(mgy, mgx) + math.pi * 0.5

        for k in range(count):
            # Alternate between primary tangent and cross-hatch angle
            ang = angle if (k % 3 != 2) else (angle + math.pi * 0.5)
            cx = rng.uniform(x0, x1 - 1)
            cy = rng.uniform(y0, y1 - 1)
            span = max(2.0, math.hypot(bw, bh) * 0.48)
            dx = math.cos(ang) * span
            dy = math.sin(ang) * span

            lx1 = int(np.clip(round(cx - dx), x0, x1 - 1))
            ly1 = int(np.clip(round(cy - dy), y0, y1 - 1))
            lx2 = int(np.clip(round(cx + dx), x0, x1 - 1))
            ly2 = int(np.clip(round(cy + dy), y0, y1 - 1))

            x1_list.append(lx1)
            y1_list.append(ly1)
            x2_list.append(lx2)
            y2_list.append(ly2)

    x1_arr = np.array(x1_list, dtype=np.int32)
    y1_arr = np.array(y1_list, dtype=np.int32)
    x2_arr = np.array(x2_list, dtype=np.int32)
    y2_arr = np.array(y2_list, dtype=np.int32)

    r, g, b, a = sample_line_color(img_arr, x1_arr, y1_arr, x2_arr, y2_arr, n_samples=5)
    return make_thread_matrix(r, g, b, a, x1_arr, y1_arr, x2_arr, y2_arr)


# =============================================================================
# ALGORITHM 9: Anchor-Graph Proximity Web
# =============================================================================

def algo4_anchor_web(img_arr, n_threads, rng):
    """
    Samples structural anchor nodes across the image, sorts them along spatial
    strips, builds a local proximity web of candidate edges, and picks the top
    N edges with the highest color coherence.
    """
    h, w, _ = img_arr.shape
    gray = img_arr[:, :, :3].mean(axis=2)
    _, _, mag = compute_gradients(gray)

    n_anchors = max(300, int(n_threads * 0.45))
    weights = (mag + 4.0) * (img_arr[:, :, 3] > 0)
    prob = weights.ravel()
    prob_sum = prob.sum()
    prob = prob / prob_sum if prob_sum > 0 else np.full_like(prob, 1.0 / prob.size)

    flat_idx = rng.choice(h * w, size=n_anchors, replace=False if n_anchors < h * w else True, p=prob)
    ay, ax = np.divmod(flat_idx, w)

    # Sort anchors by spatial strips so nearby indices are spatially close
    strip_w = max(8, w // 25)
    keys = (ax // strip_w) * h + ay
    order = np.argsort(keys)
    ax = ax[order]
    ay = ay[order]

    # Connect each anchor to several neighbors in the sorted spatial sequence
    cand_x1, cand_y1, cand_x2, cand_y2 = [], [], [], []
    max_dist_sq = (min(w, h) * 0.09) ** 2

    for offset in range(1, 9):
        dx = ax[offset:] - ax[:-offset]
        dy = ay[offset:] - ay[:-offset]
        dist_sq = dx * dx + dy * dy
        valid = (dist_sq >= 4) & (dist_sq <= max_dist_sq)

        cand_x1.append(ax[:-offset][valid])
        cand_y1.append(ay[:-offset][valid])
        cand_x2.append(ax[offset:][valid])
        cand_y2.append(ay[offset:][valid])

    x1 = np.concatenate(cand_x1)
    y1 = np.concatenate(cand_y1)
    x2 = np.concatenate(cand_x2)
    y2 = np.concatenate(cand_y2)

    # If fewer candidates than needed, pad with short local chords
    if len(x1) < n_threads:
        extra = n_threads - len(x1)
        idx = rng.integers(0, len(ax), size=extra)
        x1 = np.concatenate([x1, ax[idx]])
        y1 = np.concatenate([y1, ay[idx]])
        x2 = np.concatenate([x2, np.clip(ax[idx] + rng.integers(-12, 13, size=extra), 0, w - 1)])
        y2 = np.concatenate([y2, np.clip(ay[idx] + rng.integers(-12, 13, size=extra), 0, h - 1)])

    r, g, b, a = sample_line_color(img_arr, x1, y1, x2, y2, n_samples=5)

    # Rank candidate web edges by opacity/coherence and pick exactly n_threads
    if len(x1) > n_threads:
        scores = a.astype(np.float32) + rng.uniform(0, 15, size=len(a))
        keep = np.argsort(-scores)[:n_threads]
        r, g, b, a = r[keep], g[keep], b[keep], a[keep]
        x1, y1, x2, y2 = x1[keep], y1[keep], x2[keep], y2[keep]

    return make_thread_matrix(r, g, b, a, x1, y1, x2, y2)


# ----------------------------------------------------------------------------
# Adaptor wrappers for algorithms 6-9
# ----------------------------------------------------------------------------
def _to_img_arr(T):
    rgb = np.clip(np.rint(T * 255.0), 0, 255).astype(np.uint8)
    alpha = np.full((T.shape[0], T.shape[1], 1), 255, dtype=np.uint8)
    return np.concatenate([rgb, alpha], axis=2)


def algo_edge_tangent(T, bg, field, rng, budget, log, target_threads=None):
    n_threads = target_threads if target_threads is not None else int(round(8000 * budget))
    img_arr = _to_img_arr(T)
    res = algo1_edge_tangent(img_arr, max(1, n_threads), rng)
    log(1, 1)
    return res


def algo_greedy_error(T, bg, field, rng, budget, log, target_threads=None):
    n_threads = target_threads if target_threads is not None else int(round(8000 * budget))
    img_arr = _to_img_arr(T)
    res = algo2_greedy_error(img_arr, max(1, n_threads), rng)
    log(1, 1)
    return res


def algo_quadtree_hatch(T, bg, field, rng, budget, log, target_threads=None):
    n_threads = target_threads if target_threads is not None else int(round(8000 * budget))
    img_arr = _to_img_arr(T)
    res = algo3_quadtree_hatch(img_arr, max(1, n_threads), rng)
    log(1, 1)
    return res


def algo_anchor_web(T, bg, field, rng, budget, log, target_threads=None):
    n_threads = target_threads if target_threads is not None else int(round(8000 * budget))
    img_arr = _to_img_arr(T)
    res = algo4_anchor_web(img_arr, max(1, n_threads), rng)
    log(1, 1)
    return res


ALGORITHMS = [
    ("greedy_residual", algo_greedy_residual),
    ("flow_streamlines", algo_flow_streamlines),
    ("scanline_dashes", algo_scanline_dashes),
    ("radon_peel", algo_radon_peel),
    ("stipple_tour", algo_stipple_tour),
    ("edge_tangent", algo_edge_tangent),
    ("greedy_error", algo_greedy_error),
    ("quadtree_hatch", algo_quadtree_hatch),
    ("anchor_web", algo_anchor_web),
]


def main():
    ap = argparse.ArgumentParser(description="Convert an image into 1 px coloured threads (PNG + SVG per algorithm).")
    ap.add_argument("image", help="path to the input image (JPG, PNG, BMP, ...)")
    ap.add_argument("--out", default=None, help="output folder (default: <image name>_threads)")
    ap.add_argument("--bg", default="#ffffff", help="colour you will view the result on (default #ffffff)")
    ap.add_argument("--budget", type=float, default=1.0, help="more threads (>1) or fewer threads (<1)")
    ap.add_argument("--threads", type=int, default=None, help="target number of threads for all algorithms, e.g. 8000")
    ap.add_argument("--seed", type=int, default=7, help="random seed")
    ap.add_argument("--only", default=None, help="run only some algorithms, e.g. 1,3 or radon_peel")
    ap.add_argument("--npy", action="store_true", help="also save the (N, 8) thread arrays as .npy")
    args = ap.parse_args()

    if args.threads is not None and args.threads <= 0:
        ap.error("--threads must be a positive integer")

    bg = parse_color(args.bg)
    T = load_image(args.image, bg)
    stem = os.path.splitext(os.path.basename(args.image))[0]
    out_dir = args.out or f"{stem}_threads"
    os.makedirs(out_dir, exist_ok=True)

    wanted = None
    if args.only:
        wanted = {w.strip() for w in args.only.split(",")}

    field = tangent_field(T)
    target_info = f", target threads: ~{args.threads}" if args.threads is not None else ""
    print(f"Input {args.image} -> {SIZE}x{SIZE}, background {args.bg}{target_info}, output folder: {out_dir}")
    for number, (name, fn) in enumerate(ALGORITHMS, start=1):
        aliases = {name, str(number), f"algo_{name}", f"algo{number}_{name}"}
        if name == "edge_tangent":
            aliases.add("algo1_edge_tangent")
        elif name == "greedy_error":
            aliases.add("algo2_greedy_error")
        elif name == "quadtree_hatch":
            aliases.add("algo3_quadtree_hatch")
        elif name == "anchor_web":
            aliases.add("algo4_anchor_web")
        if wanted and not (wanted & aliases):
            continue
        rng = np.random.default_rng(args.seed + number)
        t0 = time.time()
        threads = finalize(fn(T, bg, field, rng, args.budget, make_progress(name), target_threads=args.threads))
        sys.stderr.write("\r" + " " * 40 + "\r")
        base = os.path.join(out_dir, f"{stem}_{name}")
        render_png(threads).save(base + ".png")
        write_svg(base + ".svg", threads)
        if args.npy:
            np.save(base + ".npy", threads)
        print(f"[{number}] {name:<17} {len(threads):>6} threads  {time.time() - t0:5.1f} s  -> {base}.png / .svg")


if __name__ == "__main__":
    main()