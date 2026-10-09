#!/usr/bin/env python3
"""generate_graphs.py: Publication-quality graph generator for Digital Filography benchmarks.

All figures are model-relative: a model's detections on the uncompressed photo
are the baseline, and the metrics describe how much of that survives.

  recovery_rate : matched baseline objects / baseline objects
  precision     : matched baseline objects / predicted objects (extra detections hurt)
  f1            : harmonic mean of the two

Figures written to the graphs directory:
1. Figure 1: Recovery rate vs. thread count (one panel per storage format)
2. Figure 2: Minimum thread count to reach the recovery level (grouped bars)
3. Figure 3: Confidence and IoU of recovered (matched) detections
4. Figure 4: Effect of storage format (threshold, plus recovery/precision at max N)
5. Figure 5: Storage vs. recovery (Pareto frontier, with an equal-storage callout)
6. Figure 6: Visual fidelity (PSNR vs. thread count)
7. Figure 7: Precision vs. thread count (one panel per storage format)
8. Appendix 1: Recovery-rate heatmaps (model x thread count)
9. Appendix 2: F1 vs. thread count (one panel per storage format)

Every figure above is also written a second time with "_normalized" added to the
file name. In the normalized version the best score is 100% and every other
score is shown as a share of it. The rule is the same everywhere: the reference
is the best value among everything the figure compares.

  Figures 1, 3, 6, 7, A1, A2 (value vs. thread count)
      At each thread count, the best model and format combination shown in the
      figure is 100%. Panels share one reference, so panels stay comparable.
  Figure 2 and Figure 4A (minimum thread count, lower is better)
      Score = best threshold / threshold, where the best threshold is the lowest
      one over all models and formats. The best bar is 100%, a bar at 50% needs
      twice as many threads, and "never" is 0%.
  Figure 4B
      Recovery and precision are each divided by the best format's value.
  Figure 5
      Mean recovery is divided by the best mean recovery in the figure.

A storage format is a packed layout (see packing.py): Uniform or Adjusted, at
FP16 or UINT8. UINT6 is not plotted (see PLOTTED_FORMATS), and any UINT6 rows in
the CSVs are ignored.

Tunable options (all have defaults, so eval_dataset.py can keep calling this
script without extra arguments):

  --recovery-level  Recovery fraction used by Figures 2 and 4A (default 0.40).
                    A format that never reaches the level gets a "never" bar in
                    Figure 2 and no point in Figure 4A. In the benchmark so far
                    adjusted_uint8 peaks below 50%, so at 0.50 it would show up
                    as "never" for every model.
  --min-matched     Figure 3 hides points built from fewer matched objects than
                    this (default 30), because the mean of a handful of matches
                    is mostly noise.
  --compare-kb      Storage size for the equal-storage callout in Figure 5
                    (default 60 KB).
  --norm-floor      Normalized figures hide a point when the best raw score at
                    that thread count is not above this fraction (default 0.01).
                    A ratio to a best score of almost zero is only noise.
  --no-normalized   Skip the normalized figures.

Not plotted on purpose: mAP (the old columns were not real mAP) and inference
time (workers competed for CPU/GPU, so timings were not meaningful).
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

from packing import FORMATS, bytes_per_thread

# Global style configuration
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "semibold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
    "figure.titleweight": "bold",
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 1.0,
    "grid.color": "#e5e5e5",
    "grid.linestyle": "--",
    "grid.linewidth": 0.7,
    "grid.alpha": 0.8,
})

MODEL_ORDER = ["yolov8n", "yolov10n", "yolo11n", "yolo12n", "yolo26n", "rtdetr-l", "yolov8s-worldv2"]

MODEL_COLORS = {
    "yolov8n": "#1f77b4",          # Classic Blue
    "yolov10n": "#ff7f0e",         # Amber Orange
    "yolo11n": "#2ca02c",          # Emerald Green
    "yolo12n": "#d62728",          # Crimson Red
    "yolo26n": "#9467bd",          # Royal Purple
    "rtdetr-l": "#17becf",         # Teal
    "yolov8s-worldv2": "#8c564b",  # Brown
}

MODEL_LABELS = {
    "yolov8n": "YOLOv8n",
    "yolov10n": "YOLOv10n",
    "yolo11n": "YOLO11n",
    "yolo12n": "YOLO12n",
    "yolo26n": "YOLO26n",
    "rtdetr-l": "RT-DETR-L",
    "yolov8s-worldv2": "YOLOv8s-WorldV2",
}

# One "tier" is one packed storage format (see packing.py). Blues are Uniform
# (all 8 values at one precision), oranges are Adjusted (coordinates at that
# precision, colour + alpha as RGBA4444). Darker = higher precision.
#
# Only these formats are plotted. The benchmark CSVs may contain more (the eval
# still runs every format in packing.FORMATS); rows for the rest are dropped when
# the data is loaded. To bring a format back, add it here and give it a colour.
PLOTTED_FORMATS = ("uniform_fp16", "uniform_uint8", "adjusted_fp16", "adjusted_uint8")
_unknown_formats = sorted(set(PLOTTED_FORMATS) - set(FORMATS))
if _unknown_formats:
    raise SystemExit(f"PLOTTED_FORMATS has formats that packing.py does not define: {_unknown_formats}")
TIER_ORDER = [t for t in FORMATS if t in PLOTTED_FORMATS]
TIER_BYTES = {t: bytes_per_thread(t) for t in TIER_ORDER}
TIER_COLORS = {
    "uniform_fp16": "#1f4e79",
    "uniform_uint8": "#2980b9",
    "adjusted_fp16": "#a04000",
    "adjusted_uint8": "#e67e22",
}
TIER_MARKERS = {"uniform": "o", "adjusted": "^"}

DEFAULT_RECOVERY_LEVEL = 0.40
DEFAULT_MIN_MATCHED = 30
DEFAULT_COMPARE_KB = 60.0
DEFAULT_NORM_FLOOR = 0.01

NORM_SUFFIX = "_normalized"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def save_figure(fig: plt.Figure, base_path: str, formats: Sequence[str], dpi: int = 300) -> None:
    """Save figure in specified formats with clean bounding box."""
    for fmt in formats:
        target = f"{base_path}.{fmt}"
        fig.savefig(target, format=fmt, dpi=dpi, bbox_inches="tight")
        print(f"  Saved: {target}")
    plt.close(fig)


def present_models(df: pd.DataFrame) -> list[str]:
    seen = set(df["model"].unique())
    return [m for m in MODEL_ORDER if m in seen] + sorted(seen - set(MODEL_ORDER))


def present_tiers(df: pd.DataFrame) -> list[str]:
    seen = set(df["quantization"].unique())
    return [t for t in TIER_ORDER if t in seen] + sorted(seen - set(TIER_ORDER))


def model_color(m: str) -> str:
    if m in MODEL_COLORS:
        return MODEL_COLORS[m]
    return plt.get_cmap("tab20")(sum(ord(c) for c in m) % 20)


def model_label(m: str) -> str:
    return MODEL_LABELS.get(m, m)


def tier_color(t: str) -> str:
    return TIER_COLORS.get(t, "#555555")


def tier_label(t: str) -> str:
    """'adjusted_uint8' -> 'Adjusted UINT8'."""
    family, _, precision = t.partition("_")
    return f"{family.capitalize()} {precision.upper()}" if precision else t


def tier_family(t: str) -> str:
    return t.partition("_")[0]


def level_label(level: float) -> str:
    """0.4 -> '40', 0.575 -> '57.5'."""
    return f"{level * 100:g}"


def first_threshold(sub: pd.DataFrame, level: float) -> int | None:
    """Smallest thread count whose recovery rate is >= level, or None if never reached."""
    hits = sub[sub["recovery_rate"] >= level]
    return int(hits["thread_count"].min()) if not hits.empty else None


def step_tag(step: int, normalize: bool) -> str:
    """Progress prefix: '[3/9]' for a normal figure, '[3/9 normalized]' for its normalized twin."""
    return f"[{step}/9{' normalized' if normalize else ''}]"


def out_name(base: str, normalize: bool) -> str:
    """File name without extension; the normalized twin gets a suffix."""
    return base + (NORM_SUFFIX if normalize else "")


def normalize_to_best(df: pd.DataFrame, metric: str, floor: float = 0.0) -> pd.Series:
    """Divide each row's metric by the best (highest) value at the same thread count.

    The best value is taken over every row in df, so pass only the rows the figure
    really shows (all models and formats in the figure). The best combination is
    exactly 1.0 at each thread count and no value can be above 1.0.

    A row becomes NaN (hidden) when the best value at its thread count is missing
    or not above `floor`, because a ratio to a best score near zero is only noise.
    """
    best = df.groupby("thread_count")[metric].transform("max")
    return (df[metric] / best).where(best > floor)


def add_footnote(fig: plt.Figure, text: str) -> None:
    """Small explanation under the figure. Sits just below the canvas; bbox_inches='tight' keeps it."""
    fig.text(0.5, -0.005, text, ha="center", va="top", fontsize=9, color="#34495e")


def norm_footnote(norm_floor: float, what: str = "model and format") -> str:
    return (
        f"100% = best {what} at that thread count. "
        f"Points are hidden where the best raw score is below {norm_floor * 100:g}%."
    )


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
def load_clean_data(bench_path: str, gen_path: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Load the summary CSV (one row per model x thread count x tier) and generation metrics."""
    df = pd.read_csv(bench_path, encoding="utf-8")
    if "image_type" in df.columns:
        df = df[df["image_type"] == "filography"].copy()

    required = ["model", "thread_count", "quantization", "storage_kb", "recovery_rate", "precision", "f1"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(
            f"Error: {bench_path} is missing columns {missing}.\n"
            "It looks like a summary CSV from the old schema (success_rate, mean_mAP50, ...). "
            "Re-run eval_dataset.py to regenerate it."
        )

    numeric_cols = [
        "recovery_rate", "precision", "f1", "mean_confidence", "mean_iou",
        "storage_kb", "mean_psnr_db", "recovered_objects",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")  # blanks stay NaN, they are not zeros
    df["confidence"] = df["mean_confidence"] if "mean_confidence" in df.columns else np.nan
    df["iou"] = df["mean_iou"] if "mean_iou" in df.columns else np.nan
    df["thread_count"] = pd.to_numeric(df["thread_count"], errors="coerce").astype(int)
    df["quantization"] = df["quantization"].astype(str).str.strip()

    n_before = len(df)
    df = df[df["quantization"].isin(TIER_ORDER)].copy()
    if df.empty:
        raise SystemExit(f"Error: {bench_path} has no rows for the plotted formats {list(TIER_ORDER)}.")
    if n_before > len(df):
        print(f"Ignoring {n_before - len(df)} summary rows for formats that are not plotted.")

    df_gen = None
    if gen_path and os.path.exists(gen_path):
        df_gen_raw = pd.read_csv(gen_path, encoding="utf-8")
        df_gen = df_gen_raw.rename(
            columns={c: c.replace("\u2193", "").replace("\u2191", "").strip() for c in df_gen_raw.columns}
        ).copy()

        def parse_num(val: object) -> float:
            if pd.isna(val):
                return 0.0
            clean = re.sub(r"[^\d.-]", "", str(val))
            try:
                return float(clean)
            except ValueError:
                return 0.0

        if "Thread Count (N)" in df_gen.columns:
            df_gen["thread_count"] = df_gen["Thread Count (N)"].astype(int)
        if "Storage (KB)" in df_gen.columns:
            df_gen["storage_kb"] = df_gen["Storage (KB)"].apply(parse_num)
        if "Generation Time" in df_gen.columns:
            df_gen["gen_time_s"] = df_gen["Generation Time"].apply(parse_num)
        if "PSNR (vs Original)" in df_gen.columns:
            df_gen["psnr_db"] = df_gen["PSNR (vs Original)"].apply(parse_num)
        if "Precision Tier" in df_gen.columns:
            df_gen["quantization"] = df_gen["Precision Tier"].str.strip()
            df_gen = df_gen[df_gen["quantization"].isin(TIER_ORDER)].copy()

    return df, df_gen


# -----------------------------------------------------------------------------
# Generic: metric vs thread count, one panel per tier
# -----------------------------------------------------------------------------
def plot_metric_by_tier(
    df: pd.DataFrame,
    metric: str,
    metric_name: str,
    title: str,
    basename: str,
    out_dir: str,
    formats: Sequence[str],
    dpi: int,
    step_msg: str,
    normalize: bool = False,
    norm_floor: float = 0.0,
) -> None:
    print(step_msg)
    tiers = present_tiers(df)
    models = present_models(df)
    # Four formats: 2 columns (FP16 / UINT8) x 2 rows (Uniform on top, Adjusted below).
    # A full set of six formats would use 3 columns.
    ncols = 3 if len(tiers) == 6 else (2 if len(tiers) > 1 else 1)
    nrows = math.ceil(len(tiers) / ncols)
    panel_w = 5.8 if ncols == 3 else 7.2
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_w * ncols, 4.8 * nrows), sharex=True, sharey=True, squeeze=False)
    axes_flat = axes.flatten()

    # In the normalized figure the reference is the best row at each thread count over
    # ALL models and formats, so the panels share one reference and stay comparable.
    plot_df = df.assign(**{metric: normalize_to_best(df, metric, norm_floor)}) if normalize else df

    for ax in axes_flat[len(tiers):]:
        ax.set_visible(False)

    for i, (ax, tier) in enumerate(zip(axes_flat, tiers)):
        if i + ncols >= len(tiers):
            # No visible panel below this one (odd panel count), so keep its x tick labels.
            ax.tick_params(labelbottom=True)
        sub = plot_df[plot_df["quantization"] == tier]
        for model in models:
            m_sub = sub[sub["model"] == model].sort_values("thread_count")
            ax.plot(
                m_sub["thread_count"],
                m_sub[metric] * 100.0,
                label=model_label(model),
                color=model_color(model),
                linewidth=2.0,
                marker="o",
                markersize=3.0,
                alpha=0.9,
            )

        ax.set_title(f"{tier_label(tier)} ({TIER_BYTES.get(tier, '?')} B/thread)", pad=8)
        ax.set_ylim(-5, 105)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(100))
        ax.grid(True)

        # Data-driven note: only shown if this tier really is zero everywhere (raw values).
        vals = df.loc[df["quantization"] == tier, metric].dropna()
        if vals.empty or vals.max() <= 0:
            ax.text(
                0.5, 0.5, f"{metric_name} is 0 at every\nthread count for this tier",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, fontweight="bold", color="#c0392b",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#fadbd8", edgecolor="#e74c3c", alpha=0.9),
            )

    fig.supxlabel("Thread Count (N)", fontweight="bold")
    fig.supylabel(f"{metric_name} (% of best)" if normalize else f"{metric_name} (%)", fontweight="bold")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.945),
        ncol=min(len(models), 7), frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9,
    )
    fig.suptitle(title, y=0.995)
    plt.tight_layout(rect=(0.03, 0.03, 1, 0.90))
    if normalize:
        add_footnote(fig, norm_footnote(norm_floor))
    save_figure(fig, os.path.join(out_dir, basename), formats, dpi)


# -----------------------------------------------------------------------------
# Figure 1: Recovery rate vs thread count
# -----------------------------------------------------------------------------
def plot_figure_1(
    df: pd.DataFrame, out_dir: str, formats: Sequence[str], dpi: int,
    normalize: bool = False, norm_floor: float = 0.0,
) -> None:
    title = (
        "Figure 1 (Normalized): Object Recovery as a Share of the Best Score at Each Thread Count"
        if normalize else
        "Figure 1: Model-Relative Object Recovery Rate Across Thread Counts and Packed Storage Formats"
    )
    plot_metric_by_tier(
        df, "recovery_rate", "Object Recovery Rate", title,
        out_name("fig1_detection_success_vs_threads", normalize), out_dir, formats, dpi,
        f"{step_tag(1, normalize)} Generating Figure 1: Model-Relative Recovery vs Thread Count...",
        normalize, norm_floor,
    )


# -----------------------------------------------------------------------------
# Figure 2: Minimum thread count to reach the recovery level
# -----------------------------------------------------------------------------
def plot_figure_2(
    df: pd.DataFrame, out_dir: str, formats: Sequence[str], dpi: int, level: float,
    normalize: bool = False,
) -> None:
    print(f"{step_tag(2, normalize)} Generating Figure 2: Recovery Threshold...")
    models = present_models(df)
    tiers = present_tiers(df)
    lvl = level_label(level)

    thresholds: dict[str, dict[str, int | None]] = {
        m: {t: first_threshold(df[(df["model"] == m) & (df["quantization"] == t)], level) for t in tiers}
        for m in models
    }

    # Best = the lowest threshold over all models and formats.
    reached = [(v, m, t) for m in models for t, v in thresholds[m].items() if v]
    if normalize and not reached:
        print(f"  Skipping normalized Figure 2: no model reaches {lvl}% recovery in any format.")
        return
    best_th, best_model, best_tier = min(reached) if reached else (None, None, None)

    def score(v: int | None) -> float:
        """Bar height: the threshold itself, or best/threshold in percent (0 = never)."""
        if not v:
            return 0.0
        return 100.0 * best_th / v if normalize else float(v)

    x = np.arange(len(models))
    width = 0.8 / max(len(tiers), 1)
    many = len(tiers) > 3
    fig, ax = plt.subplots(figsize=(max(10, (2.4 if many else 1.5) * len(models) + 3), 6.5))

    if normalize:
        ax.set_ylim(0, 132)   # bars stop at 100%; the rest is room for labels and the note
    else:
        all_vals = [v for m in models for v in thresholds[m].values() if v]
        top = max(all_vals) if all_vals else 1000
        ax.set_ylim(0, top * (1.45 if many else 1.35))

    for i, tier in enumerate(tiers):
        offset = (i - (len(tiers) - 1) / 2) * width
        vals = [score(thresholds[m][tier]) for m in models]
        bars = ax.bar(x + offset, vals, width, label=f"{tier_label(tier)} ({TIER_BYTES.get(tier, '?')} B)",
                      color=tier_color(tier), edgecolor="#2c3e50", linewidth=1.0)
        for bar, val in zip(bars, vals):
            if not val:
                label = "never"
            elif normalize:
                label = f"{val:.0f}%" if val >= 1 else f"{val:.1f}%"
            else:
                label = f"{int(val):,}"
            ax.annotate(label, xy=(bar.get_x() + bar.get_width() / 2, val), xytext=(0, 4),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=7 if many else 9, rotation=90 if many else 0,
                        fontweight="bold", color="#2c3e50")

    ax.set_xticks(x)
    ax.set_xticklabels([model_label(m) for m in models], fontsize=10, fontweight="bold", rotation=15)
    ax.grid(True, axis="y")

    if normalize:
        ax.set_ylabel("Threshold Efficiency (% of best)", fontsize=11, fontweight="bold")
        ax.set_yticks(range(0, 101, 20))
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(100))
        ax.text(
            0.5, 0.97,
            f"100% = fewest threads to reach {lvl}% recovery: {model_label(best_model)}, "
            f"{tier_label(best_tier)}, N = {best_th:,}.\n"
            "Efficiency = best threshold / threshold, so 50% means twice as many threads. "
            '"never" means the level is not reached.',
            transform=ax.transAxes, ha="center", va="top", fontsize=9, color="#34495e",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f4f6f7", edgecolor="#bdc3c7", alpha=0.95),
        )
        title = f"Figure 2 (Normalized): Threshold Efficiency Relative to the Best (>={lvl}% of Baseline Objects)"
    else:
        ax.set_ylabel(f"Minimum Thread Count for >={lvl}% Baseline Object Recovery", fontsize=11, fontweight="bold")
        # Resolution caveat instead of a model ranking: thresholds sit on the thread-count
        # grid and come from a single run, so neighbouring models are effectively tied.
        counts = sorted(df["thread_count"].unique())
        step = int(np.diff(counts).min()) if len(counts) > 1 else None
        if step:
            ax.text(
                0.5, 0.97,
                f"Resolution is one grid step ({step:,} threads) from a single run;\n"
                "treat gaps of one or two steps between models as ties.",
                transform=ax.transAxes, ha="center", va="top", fontsize=9, color="#34495e",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f4f6f7", edgecolor="#bdc3c7", alpha=0.95),
            )
        title = f"Figure 2: Minimum Thread Count for Reliable Recovery (>={lvl}% of Baseline Objects)"

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=min(max(len(tiers), 1), 4),
              frameon=True, facecolor="white", edgecolor="#cccccc")
    ax.set_title(title, pad=12)
    plt.tight_layout()
    save_figure(fig, os.path.join(out_dir, out_name("fig2_first_detection_threshold", normalize)), formats, dpi)


# -----------------------------------------------------------------------------
# Figure 3: Confidence and IoU of recovered detections
# -----------------------------------------------------------------------------
def plot_figure_3(
    df: pd.DataFrame, out_dir: str, formats: Sequence[str], dpi: int, min_matched: int,
    normalize: bool = False,
) -> None:
    print(f"{step_tag(3, normalize)} Generating Figure 3: Confidence & Localization IoU Trajectories...")
    models = present_models(df)
    # Two reference formats keep the figure readable: the biggest and the cheapest.
    present = set(df["quantization"])
    tiers = [t for t in ("uniform_fp16", "adjusted_uint8") if t in present] or present_tiers(df)[:2]
    styles = dict(zip(tiers, ["-", "--"]))

    can_mask = "recovered_objects" in df.columns
    if not can_mask:
        print("  Warning: no 'recovered_objects' column, so thin points cannot be hidden in Figure 3.")

    d = df[df["quantization"].isin(tiers)].copy()
    if can_mask:
        # The mean over a handful of matches is mostly noise, so hide those points.
        thin = d["recovered_objects"].fillna(0) < min_matched
        d.loc[thin, ["confidence", "iou"]] = np.nan
    if normalize:
        # Normalize after hiding thin points, so a noisy point can never be the reference.
        d["confidence"] = normalize_to_best(d, "confidence")
        d["iou"] = normalize_to_best(d, "iou")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    for model in models:
        for tier in tiers:
            m_sub = d[(d["model"] == model) & (d["quantization"] == tier)].sort_values("thread_count")
            ax1.plot(m_sub["thread_count"], m_sub["confidence"], color=model_color(model),
                     linestyle=styles[tier], linewidth=1.9, alpha=0.9)
            ax2.plot(m_sub["thread_count"], m_sub["iou"], color=model_color(model),
                     linestyle=styles[tier], linewidth=1.9, alpha=0.9)

    if can_mask and min_matched > 0:
        hidden_note = f"points with fewer than {min_matched} matched objects hidden"
    else:
        hidden_note = "all points shown, including those from very few matches"

    ax1.set_ylim(0.0, 1.02)
    ax1.grid(True)
    ax2.set_ylim(0.50, 1.005)
    ax2.set_xlabel("Thread Count (N)", fontweight="bold")
    ax2.grid(True)

    if normalize:
        for ax, col in ((ax1, "confidence"), (ax2, "iou")):
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
            # Ratios sit close to 100%, so zoom in a little instead of showing 0 to 100%.
            vals = d[col].dropna()
            low = max(0.0, math.floor((float(vals.min()) - 0.05) * 10) / 10) if not vals.empty else 0.0
            ax.set_ylim(low, 1.01)
        ax1.set_ylabel("Confidence (% of Best)", fontweight="bold")
        ax1.set_title(f"A. Confidence Relative to the Best at Each N ({hidden_note})", loc="left", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Bounding Box IoU (% of Best)", fontweight="bold")
        ax2.set_title("B. Localization Relative to the Best at Each N", loc="left", fontsize=11, fontweight="bold")
        suptitle = "Figure 3 (Normalized): Confidence and Localization Relative to the Best Score"
    else:
        ax1.set_ylabel("Mean Confidence of Recovered Detections", fontweight="bold")
        ax1.set_title(f"A. Confidence of Recovered Baseline Objects ({hidden_note})", loc="left", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Mean Bounding Box IoU vs. Baseline", fontweight="bold")
        ax2.set_title("B. Localization of Recovered Objects (IoU >= 0.50 by definition of a match)", loc="left", fontsize=11, fontweight="bold")
        suptitle = "Figure 3: Confidence and Localization of Recovered Baseline Objects"

    handles = [Line2D([0], [0], color=model_color(m), lw=2.2, label=model_label(m)) for m in models]
    handles += [Line2D([0], [0], color="#444444", lw=2.0, linestyle=styles[t], label=tier_label(t)) for t in tiers]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.97), ncol=5,
               frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)

    fig.suptitle(suptitle, y=1.01)
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    if normalize:
        add_footnote(fig, "100% = best model and format shown, at that thread count (thin points are hidden before normalizing).")
    save_figure(fig, os.path.join(out_dir, out_name("fig3_confidence_and_iou_trajectory", normalize)), formats, dpi)


# -----------------------------------------------------------------------------
# Figure 4: Effect of storage format
# -----------------------------------------------------------------------------
def plot_figure_4(
    df: pd.DataFrame, out_dir: str, formats: Sequence[str], dpi: int, level: float,
    normalize: bool = False,
) -> None:
    print(f"{step_tag(4, normalize)} Generating Figure 4: Effect of Storage Format...")
    models = present_models(df)
    tiers = present_tiers(df)
    sizes = sorted({TIER_BYTES[t] for t in tiers if t in TIER_BYTES})
    lvl = level_label(level)

    # Threshold for every (model, format); None means the level is never reached.
    th_map = {
        (m, t): first_threshold(df[(df["model"] == m) & (df["quantization"] == t)], level)
        for m in models for t in tiers
    }
    reached = [v for v in th_map.values() if v]
    if normalize and not reached:
        print(f"  Skipping normalized Figure 4: no model reaches {lvl}% recovery in any format.")
        return
    best_th = min(reached) if reached else None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))

    # Panel A: threads needed to reach the recovery level vs storage per thread.
    # One line per model and family: Uniform at 16 B (FP16) and 8 B (UINT8),
    # Adjusted at 10 B (FP16) and 6 B (UINT8).
    # Normalized: the same points as best threshold / threshold, so higher is better.
    for m in models:
        for family, linestyle in (("uniform", "-"), ("adjusted", "--")):
            pts = []
            for t in tiers:
                th = th_map[(m, t)]
                if tier_family(t) == family and t in TIER_BYTES and th is not None:
                    pts.append((TIER_BYTES[t], 100.0 * best_th / th if normalize else th))
            pts.sort()
            if pts:
                ax1.plot([p[0] for p in pts], [p[1] for p in pts], marker=TIER_MARKERS[family],
                         linestyle=linestyle, linewidth=2.0, color=model_color(m))

    ax1.set_xlabel("Storage per Thread (bytes)", fontweight="bold")
    if normalize:
        ax1.set_ylabel(f"Threshold Efficiency for >={lvl}% Recovery (% of best)", fontweight="bold")
        ax1.set_title("A. Threshold Efficiency vs. Storage per Thread (missing = never reached)", loc="left", fontsize=11, fontweight="bold")
        ax1.set_ylim(-5, 105)
        ax1.yaxis.set_major_formatter(ticker.PercentFormatter(100))
        ax1.text(
            0.02, 0.03, f"100% = {best_th:,} threads, the fewest in any model and format",
            transform=ax1.transAxes, ha="left", va="bottom", fontsize=9, color="#34495e",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f4f6f7", edgecolor="#bdc3c7", alpha=0.95),
        )
    else:
        ax1.set_ylabel(f"Minimum Thread Count for >={lvl}% Recovery", fontweight="bold")
        ax1.set_title("A. Recovery Threshold vs. Storage per Thread (missing = never reached)", loc="left", fontsize=11, fontweight="bold")
    if sizes:
        ax1.set_xticks(sizes)
        ax1.set_xlim(max(sizes) + 1, min(sizes) - 1)
    ax1.grid(True)
    handles_a = [Line2D([0], [0], color=model_color(m), lw=2.0, label=model_label(m)) for m in models]
    handles_a += [
        Line2D([0], [0], color="#444444", lw=2.0, linestyle="-", marker="o", label="Uniform"),
        Line2D([0], [0], color="#444444", lw=2.0, linestyle="--", marker="^", label="Adjusted"),
    ]
    ax1.legend(handles=handles_a, loc="best", frameon=True, facecolor="white", fontsize=9)

    # Panel B: recovery and precision at the largest thread count, mean over models
    n_final = int(df["thread_count"].max())
    at_final = df[df["thread_count"] == n_final]
    rec = [float(at_final[at_final["quantization"] == t]["recovery_rate"].mean() or 0.0) for t in tiers]
    prec = [float(at_final[at_final["quantization"] == t]["precision"].mean() or 0.0) for t in tiers]
    rec = [0.0 if np.isnan(v) else v * 100.0 for v in rec]
    prec = [0.0 if np.isnan(v) else v * 100.0 for v in prec]

    rec_label, prec_label = "Recovery", "Precision"
    if normalize:
        # Each metric is divided by the best format's value for that metric.
        best_rec, best_prec = max(rec, default=0.0), max(prec, default=0.0)
        rec_label = f"Recovery (100% = {best_rec:.1f}%)"
        prec_label = f"Precision (100% = {best_prec:.1f}%)"
        rec = [100.0 * v / best_rec if best_rec > 0 else 0.0 for v in rec]
        prec = [100.0 * v / best_prec if best_prec > 0 else 0.0 for v in prec]

    xb = np.arange(len(tiers))
    w = 0.36
    crowded = len(tiers) > 4   # with up to four formats the value labels fit horizontally
    b1 = ax2.bar(xb - w / 2, rec, w, label=rec_label, color="#2980b9", edgecolor="#2c3e50", linewidth=1.0)
    b2 = ax2.bar(xb + w / 2, prec, w, label=prec_label, color="#e67e22", edgecolor="#2c3e50", linewidth=1.0)
    for bars in (b1, b2):
        for bar in bars:
            v = bar.get_height()
            ax2.annotate(f"{v:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, v), xytext=(0, 3),
                         textcoords="offset points", ha="center", va="bottom", fontweight="bold",
                         fontsize=8 if crowded else 10, rotation=90 if crowded else 0)

    ax2.set_xticks(xb)
    ax2.set_xticklabels([f"{tier_label(t)}\n{TIER_BYTES.get(t, '?')} B" for t in tiers], fontsize=9)
    if normalize:
        ax2.set_ylabel(f"% of best format (mean over {len(models)} models)", fontweight="bold")
        ax2.set_title(f"B. Recovery and Precision at N = {n_final:,} Threads, Relative to the Best Format", loc="left", fontsize=11, fontweight="bold")
    else:
        ax2.set_ylabel(f"Mean over {len(models)} models (%)", fontweight="bold")
        ax2.set_title(f"B. Recovery and Precision at N = {n_final:,} Threads", loc="left", fontsize=11, fontweight="bold")
    ax2.set_xlabel(f"Storage Format at N = {n_final:,} Threads", fontweight="bold")
    ax2.set_ylim(0, 140 if crowded else 115)   # headroom for the value labels
    ax2.set_yticks(range(0, 101, 20))
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(100))
    ax2.grid(True, axis="y")
    ax2.legend(loc="upper right", frameon=True, facecolor="white")

    fig.suptitle(
        "Figure 4 (Normalized): Effect of Packed Storage Format Relative to the Best"
        if normalize else "Figure 4: Effect of Packed Storage Format on Baseline Recovery",
        y=1.02,
    )
    plt.tight_layout()
    save_figure(fig, os.path.join(out_dir, out_name("fig4_quantization_bitwidth_impact", normalize)), formats, dpi)


# -----------------------------------------------------------------------------
# Figure 5: Storage vs recovery (Pareto)
# -----------------------------------------------------------------------------
def plot_figure_5(
    df: pd.DataFrame, out_dir: str, formats: Sequence[str], dpi: int, compare_kb: float,
    normalize: bool = False,
) -> None:
    print(f"{step_tag(5, normalize)} Generating Figure 5: Storage vs Recovery Pareto...")

    # One point per (tier, thread count): recovery averaged over models
    pts = (
        df.groupby(["quantization", "thread_count"], as_index=False)
        .agg(storage_kb=("storage_kb", "first"), recovery=("recovery_rate", "mean"))
        .dropna(subset=["recovery"])
    )

    # Normalized: recovery is divided by the best mean recovery in the figure.
    best = float(pts["recovery"].max()) if not pts.empty else 0.0
    if normalize and best <= 0:
        print("  Skipping normalized Figure 5: mean recovery is 0 everywhere.")
        return
    scale = 100.0 / best if normalize else 100.0

    fig, ax = plt.subplots(figsize=(10, 6.5))

    for tier in present_tiers(pts):
        sub = pts[pts["quantization"] == tier]
        bpt = int(round(sub["storage_kb"].iloc[0] * 1000.0 / sub["thread_count"].iloc[0])) if not sub.empty else 0
        ax.scatter(
            sub["storage_kb"], sub["recovery"] * scale,
            s=np.clip(sub["thread_count"] / 100.0, 15, 120),
            color=tier_color(tier), alpha=0.7, edgecolors="none",
            marker=TIER_MARKERS.get(tier_family(tier), "o"),
            label=f"{tier_label(tier)} ({bpt} B/thread)",
        )

    # Pareto frontier: lowest storage for each new best recovery
    ordered = pts.sort_values(["storage_kb", "recovery"], ascending=[True, False])
    fx, fy, top = [], [], -1.0
    for s, r in zip(ordered["storage_kb"], ordered["recovery"]):
        if r > top:
            fx.append(s)
            fy.append(r * scale)
            top = r
    if fx:
        ax.step(fx, fy, where="post", color="#2c3e50", linestyle=":", linewidth=2.0, label="Pareto frontier")

    # Equal-storage comparison: every format at the same file size, so the callout
    # compares like with like (different formats need different thread counts).
    near = pts[np.isclose(pts["storage_kb"], compare_kb, atol=0.01)]
    if near.empty:
        print(f"  Note: no format has a grid point at {compare_kb:g} KB, so the equal-storage callout is skipped.")
    else:
        order = {t: i for i, t in enumerate(present_tiers(pts))}
        near = near.sort_values("quantization", key=lambda s: s.map(order))
        lines = [
            f"{tier_label(r.quantization)} (N={int(r.thread_count):,}): {r.recovery * scale:.1f}%"
            for r in near.itertuples()
        ]
        absent = [tier_label(t) for t in present_tiers(pts) if t not in set(near["quantization"])]
        if absent:
            lines.append("No grid point at this size: " + ", ".join(absent))
        ax.axvline(compare_kb, color="#999999", linestyle="--", linewidth=1.0, zorder=0)
        header = (
            f"Recovery relative to the best, at {compare_kb:g} KB"
            if normalize else f"Mean recovery over models at {compare_kb:g} KB"
        )
        ax.text(
            0.02, 0.97,
            header + "\n" + "\n".join(lines),
            transform=ax.transAxes, ha="left", va="top", fontsize=9.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#999999"),
        )

    ax.set_xlabel("Reconstruction File Size / Storage (KB)", fontweight="bold")
    if normalize:
        ax.set_ylabel(f"Recovery as % of Best (100% = {best * 100:.1f}% mean recovery)", fontweight="bold")
        ax.set_title("Figure 5 (Normalized): Storage Overhead vs. Recovery Relative to the Best Point", pad=12)
    else:
        ax.set_ylabel("Mean Object Recovery Rate over Models (%)", fontweight="bold")
        ax.set_title("Figure 5: Storage Overhead vs. Baseline Object Recovery", pad=12)
    ax.set_ylim(-5, 105)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(100))
    ax.grid(True)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cccccc")

    plt.tight_layout()
    save_figure(fig, os.path.join(out_dir, out_name("fig5_pareto_cost_vs_performance", normalize)), formats, dpi)


# -----------------------------------------------------------------------------
# Figure 6: Generation reconstruction quality (PSNR)
# -----------------------------------------------------------------------------
def plot_figure_6(
    df_gen: pd.DataFrame | None, out_dir: str, formats: Sequence[str], dpi: int,
    normalize: bool = False,
) -> None:
    print(f"{step_tag(6, normalize)} Generating Figure 6: Generation Quality...")
    if df_gen is None or df_gen.empty:
        print("  Skipping Figure 6: Generation CSV data unavailable.")
        return
    if "psnr_db" not in df_gen.columns:
        print("  Skipping Figure 6: the generation CSV has no 'PSNR (vs Original)' column.")
        return

    # Normalized: at each thread count the best format is 100%.
    plot_df = df_gen.assign(psnr_db=normalize_to_best(df_gen, "psnr_db")) if normalize else df_gen
    scale = 100.0 if normalize else 1.0

    fig, ax1 = plt.subplots(figsize=(10, 6))

    for tier in present_tiers(plot_df):
        sub = plot_df[plot_df["quantization"] == tier].sort_values("thread_count")
        if not sub.empty:
            ax1.plot(
                sub["thread_count"], sub["psnr_db"] * scale,
                label=tier_label(tier) if normalize else f"{tier_label(tier)} (PSNR)",
                color=tier_color(tier),
                linewidth=2.2,
                linestyle="-" if tier_family(tier) == "uniform" else "--",
            )

    ax1.set_xlabel("Thread Count (N)", fontweight="bold")
    if normalize:
        ax1.set_ylabel("PSNR as % of Best Format at Each Thread Count", fontweight="bold", color="#1c2833")
        ax1.set_title("Figure 6 (Normalized): Visual Fidelity (PSNR) Relative to the Best Format", pad=12)
        ax1.yaxis.set_major_formatter(ticker.PercentFormatter(100))
        ax1.set_ylim(top=100.5)
    else:
        ax1.set_ylabel("Reconstruction PSNR vs. Original COCO Photo (dB)", fontweight="bold", color="#1c2833")
        ax1.set_title("Figure 6: Digital Filography Visual Fidelity (Mean PSNR) vs. Thread Placement Density", pad=12)
    ax1.grid(True)
    # Normalized curves fall toward the lower right, so let matplotlib find a free corner there.
    ax1.legend(loc="best" if normalize else "lower right", frameon=True, facecolor="white", edgecolor="#cccccc")

    plt.tight_layout()
    if normalize:
        add_footnote(fig, "100% = best format at that thread count. PSNR is in dB (a log scale), so ratios compress real differences.")
    save_figure(fig, os.path.join(out_dir, out_name("fig6_generation_quality_psnr_storage", normalize)), formats, dpi)


# -----------------------------------------------------------------------------
# Figure 7: Precision vs thread count (replaces the latency figure)
# -----------------------------------------------------------------------------
def plot_figure_7(
    df: pd.DataFrame, out_dir: str, formats: Sequence[str], dpi: int,
    normalize: bool = False, norm_floor: float = 0.0,
) -> None:
    title = (
        "Figure 7 (Normalized): Precision as a Share of the Best Score at Each Thread Count"
        if normalize else
        "Figure 7: Precision of Reconstructions (Extra Detections Count as False Positives)"
    )
    plot_metric_by_tier(
        df, "precision", "Precision", title,
        out_name("fig7_precision_vs_threads", normalize), out_dir, formats, dpi,
        f"{step_tag(7, normalize)} Generating Figure 7: Precision vs Thread Count...",
        normalize, norm_floor,
    )


# -----------------------------------------------------------------------------
# Appendix Figure 1: Recovery heatmaps (model x thread count)
# -----------------------------------------------------------------------------
def plot_appendix_heatmap(
    df: pd.DataFrame, out_dir: str, formats: Sequence[str], dpi: int,
    normalize: bool = False, norm_floor: float = 0.0,
) -> None:
    print(f"{step_tag(8, normalize)} Generating Appendix Figure 1: Recovery Heatmaps...")
    models = present_models(df)
    tiers = present_tiers(df)
    model_labels = [model_label(m) for m in models]

    # Normalized: the best model and format at each thread count is 1.0 (one reference
    # for all panels). Hidden cells are NaN and are drawn in light grey.
    plot_df = df.assign(recovery_rate=normalize_to_best(df, "recovery_rate", norm_floor)) if normalize else df

    fig, axes = plt.subplots(len(tiers), 1, figsize=(14, 2.6 * len(tiers) + 1.5), sharex=True, squeeze=False)

    cbar_kws: dict[str, object] = {"label": "Recovery Rate", "shrink": 0.8}
    if normalize:
        cbar_kws = {"label": "Recovery (% of best)", "shrink": 0.8, "format": ticker.PercentFormatter(1.0)}

    for ax, tier in zip(axes.flatten(), tiers):
        sub = plot_df[plot_df["quantization"] == tier]
        # dropna=False keeps fully hidden columns, so every panel has the same x positions.
        pivot = sub.pivot_table(
            index="model", columns="thread_count", values="recovery_rate", aggfunc="mean", dropna=False
        ).reindex(models)
        if normalize:
            ax.set_facecolor("#e0e0e0")
        sns.heatmap(
            pivot,
            ax=ax,
            cmap="mako",
            vmin=0.0,
            vmax=1.0,
            cbar_kws=cbar_kws,
            linewidths=0.2,
            linecolor="#333333",
        )
        ax.set_yticklabels(model_labels, rotation=0, fontweight="bold")
        ax.set_ylabel("")
        ax.set_xlabel("")
        prefix = "Object Recovery Relative to Best" if normalize else "Object Recovery Rate"
        ax.set_title(f"{prefix}: {tier_label(tier)}", loc="left", fontsize=11, fontweight="bold")

    axes.flatten()[-1].set_xlabel("Thread Count (N)", fontweight="bold")
    fig.suptitle(
        "Appendix Figure 1 (Normalized): Object Recovery as a Share of the Best Score at Each Thread Count"
        if normalize else "Appendix Figure 1: Baseline Object Recovery Heatmaps across Thread Counts",
        y=1.0,
    )
    plt.tight_layout()
    if normalize:
        add_footnote(fig, norm_footnote(norm_floor) + " Grey cells are hidden.")
    save_figure(fig, os.path.join(out_dir, out_name("fig_appendix_heatmap_confidence", normalize)), formats, dpi)


# -----------------------------------------------------------------------------
# Appendix Figure 2: F1 vs thread count (replaces the mAP figure)
# -----------------------------------------------------------------------------
def plot_appendix_precision_dynamics(
    df: pd.DataFrame, out_dir: str, formats: Sequence[str], dpi: int,
    normalize: bool = False, norm_floor: float = 0.0,
) -> None:
    title = (
        "Appendix Figure 2 (Normalized): F1 as a Share of the Best Score at Each Thread Count"
        if normalize else
        "Appendix Figure 2: F1 Combining Recovery and Precision Across Thread Counts"
    )
    plot_metric_by_tier(
        df, "f1", "F1 (Recovery and Precision)", title,
        out_name("fig_appendix_precision_dynamics", normalize), out_dir, formats, dpi,
        f"{step_tag(9, normalize)} Generating Appendix Figure 2: F1 vs Thread Count...",
        normalize, norm_floor,
    )


# -----------------------------------------------------------------------------
# Main CLI runner
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate publication figures for Digital Filography benchmarks.")
    parser.add_argument("--benchmark-csv", type=str, default="output/dataset_evaluation_summary.csv", help="Path to summary CSV")
    parser.add_argument("--generation-csv", type=str, default="output/generation.csv", help="Path to Filography generation metrics CSV")
    parser.add_argument("--output-dir", type=str, default="graphs", help="Directory to save generated plots")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for raster output")
    parser.add_argument("--formats", nargs="+", default=["png", "svg"], help="Output formats (e.g. png svg pdf)")
    parser.add_argument(
        "--recovery-level", type=float, default=DEFAULT_RECOVERY_LEVEL,
        help=f"Recovery fraction for Figures 2 and 4A, in (0, 1] (default: {DEFAULT_RECOVERY_LEVEL})",
    )
    parser.add_argument(
        "--min-matched", type=int, default=DEFAULT_MIN_MATCHED,
        help=f"Figure 3 hides points with fewer matched objects than this (default: {DEFAULT_MIN_MATCHED})",
    )
    parser.add_argument(
        "--compare-kb", type=float, default=DEFAULT_COMPARE_KB,
        help=f"Storage size in KB for the Figure 5 equal-storage callout (default: {DEFAULT_COMPARE_KB:g})",
    )
    parser.add_argument(
        "--norm-floor", type=float, default=DEFAULT_NORM_FLOOR,
        help=(
            "Normalized figures hide a point when the best raw score at that thread count "
            f"is not above this fraction, in [0, 1) (default: {DEFAULT_NORM_FLOOR})"
        ),
    )
    parser.add_argument(
        "--no-normalized", action="store_true",
        help="Do not write the normalized (relative to best) version of each figure",
    )
    args = parser.parse_args()

    if not 0.0 < args.recovery_level <= 1.0:
        parser.error("--recovery-level must be in (0, 1]")
    if args.min_matched < 0:
        parser.error("--min-matched must be >= 0")
    if args.compare_kb <= 0:
        parser.error("--compare-kb must be > 0")
    if not 0.0 <= args.norm_floor < 1.0:
        parser.error("--norm-floor must be in [0, 1)")
    return args


def main() -> int:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("================================================================================")
    print("Generating Digital Filography Publication Visualizations")
    print(f"Benchmark CSV:    {args.benchmark_csv}")
    print(f"Generation CSV:   {args.generation_csv}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Formats:          {args.formats} (DPI={args.dpi})")
    print(f"Recovery level:   {level_label(args.recovery_level)}% (Figures 2 and 4A)")
    print(f"Min matched:      {args.min_matched} (Figure 3)")
    print(f"Compare at:       {args.compare_kb:g} KB (Figure 5)")
    if args.no_normalized:
        print("Normalized:       off")
    else:
        print(f"Normalized:       on (hide points when the best raw score is below {args.norm_floor * 100:g}%)")
    print("================================================================================")

    df_filo, df_gen = load_clean_data(args.benchmark_csv, args.generation_csv)
    print(f"Loaded {len(df_filo)} summary records for models: {present_models(df_filo)}")

    # First the nine original figures, then (unless switched off) their normalized twins.
    modes = [False] if args.no_normalized else [False, True]
    for normalize in modes:
        if normalize:
            print("\nNormalized figures (100% = best score)")
        plot_figure_1(df_filo, args.output_dir, args.formats, args.dpi, normalize, args.norm_floor)
        plot_figure_2(df_filo, args.output_dir, args.formats, args.dpi, args.recovery_level, normalize)
        plot_figure_3(df_filo, args.output_dir, args.formats, args.dpi, args.min_matched, normalize)
        plot_figure_4(df_filo, args.output_dir, args.formats, args.dpi, args.recovery_level, normalize)
        plot_figure_5(df_filo, args.output_dir, args.formats, args.dpi, args.compare_kb, normalize)
        plot_figure_6(df_gen, args.output_dir, args.formats, args.dpi, normalize)
        plot_figure_7(df_filo, args.output_dir, args.formats, args.dpi, normalize, args.norm_floor)
        plot_appendix_heatmap(df_filo, args.output_dir, args.formats, args.dpi, normalize, args.norm_floor)
        plot_appendix_precision_dynamics(df_filo, args.output_dir, args.formats, args.dpi, normalize, args.norm_floor)

    print(f"\nAll figures generated successfully in {args.output_dir}/!")
    return 0


if __name__ == "__main__":
    sys.exit(main())