#!/usr/bin/env python3
"""generate_graphs.py: Publication-quality graph generator for Digital Filography benchmarks.

All figures are model-relative: a model's detections on the uncompressed photo
are the baseline, and the metrics describe how much of that survives.

  recovery_rate : matched baseline objects / baseline objects
  precision     : matched baseline objects / predicted objects (extra detections hurt)
  f1            : harmonic mean of the two

Figures written to the graphs directory:
1. Figure 1: Recovery rate vs. thread count (one panel per precision tier)
2. Figure 2: Minimum thread count for >=50% recovery (grouped bars)
3. Figure 3: Confidence and IoU of recovered (matched) detections
4. Figure 4: Effect of bit-width (threshold, plus recovery/precision at max N)
5. Figure 5: Storage vs. recovery (Pareto frontier)
6. Figure 6: Visual fidelity (PSNR vs. thread count)
7. Figure 7: Precision vs. thread count (one panel per precision tier)
8. Appendix 1: Recovery-rate heatmaps (model x thread count)
9. Appendix 2: F1 vs. thread count (one panel per precision tier)

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

TIER_ORDER = ["float32", "float16", "uint8", "uint6", "uint5", "uint4"]
TIER_BITS = {"float32": 32, "float16": 16, "uint8": 8, "uint6": 6, "uint5": 5, "uint4": 4}
TIER_COLORS = {
    "float32": "#7f8c8d",
    "float16": "#2980b9",
    "uint8": "#27ae60",
    "uint6": "#f39c12",
    "uint5": "#d35400",
    "uint4": "#c0392b",
}
RECOVERY_LEVEL = 0.50


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


def first_threshold(sub: pd.DataFrame, level: float = RECOVERY_LEVEL) -> int | None:
    """Smallest thread count whose recovery rate is >= level, or None if never reached."""
    hits = sub[sub["recovery_rate"] >= level]
    return int(hits["thread_count"].min()) if not hits.empty else None


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

    for c in ["recovery_rate", "precision", "f1", "mean_confidence", "mean_iou", "storage_kb", "mean_psnr_db"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")  # blanks stay NaN, they are not zeros
    df["confidence"] = df["mean_confidence"] if "mean_confidence" in df.columns else np.nan
    df["iou"] = df["mean_iou"] if "mean_iou" in df.columns else np.nan
    df["thread_count"] = pd.to_numeric(df["thread_count"], errors="coerce").astype(int)
    df["quantization"] = df["quantization"].astype(str).str.strip()

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
) -> None:
    print(step_msg)
    tiers = present_tiers(df)
    models = present_models(df)
    ncols = 2 if len(tiers) > 1 else 1
    nrows = math.ceil(len(tiers) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2 * ncols, 4.8 * nrows), sharex=True, sharey=True, squeeze=False)
    axes_flat = axes.flatten()

    for ax in axes_flat[len(tiers):]:
        ax.set_visible(False)

    for i, (ax, tier) in enumerate(zip(axes_flat, tiers)):
        if i + ncols >= len(tiers):
            # No visible panel below this one (odd panel count), so keep its x tick labels.
            ax.tick_params(labelbottom=True)
        sub = df[df["quantization"] == tier]
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

        ax.set_title(f"Quantization: {tier}", pad=8)
        ax.set_ylim(-5, 105)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(100))
        ax.grid(True)

        # Data-driven note: only shown if this tier really is zero everywhere.
        vals = sub[metric].dropna()
        if vals.empty or vals.max() <= 0:
            ax.text(
                0.5, 0.5, f"{metric_name} is 0 at every\nthread count for this tier",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, fontweight="bold", color="#c0392b",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#fadbd8", edgecolor="#e74c3c", alpha=0.9),
            )

    fig.supxlabel("Thread Count (N)", fontweight="bold")
    fig.supylabel(f"{metric_name} (%)", fontweight="bold")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.945),
        ncol=min(len(models), 7), frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9,
    )
    fig.suptitle(title, y=0.995)
    plt.tight_layout(rect=(0.03, 0.03, 1, 0.90))
    save_figure(fig, os.path.join(out_dir, basename), formats, dpi)


# -----------------------------------------------------------------------------
# Figure 1: Recovery rate vs thread count
# -----------------------------------------------------------------------------
def plot_figure_1(df: pd.DataFrame, out_dir: str, formats: Sequence[str], dpi: int) -> None:
    plot_metric_by_tier(
        df, "recovery_rate", "Object Recovery Rate",
        "Figure 1: Model-Relative Object Recovery Rate Across Thread Counts and Quantization Tiers",
        "fig1_detection_success_vs_threads", out_dir, formats, dpi,
        "[1/9] Generating Figure 1: Model-Relative Recovery vs Thread Count...",
    )


# -----------------------------------------------------------------------------
# Figure 2: Minimum thread count for >=50% recovery
# -----------------------------------------------------------------------------
def plot_figure_2(df: pd.DataFrame, out_dir: str, formats: Sequence[str], dpi: int) -> None:
    print("[2/9] Generating Figure 2: Recovery Threshold...")
    models = present_models(df)
    tiers = present_tiers(df)

    thresholds: dict[str, dict[str, int | None]] = {
        m: {t: first_threshold(df[(df["model"] == m) & (df["quantization"] == t)]) for t in tiers}
        for m in models
    }

    x = np.arange(len(models))
    width = 0.8 / max(len(tiers), 1)
    fig, ax = plt.subplots(figsize=(max(10, 1.5 * len(models) + 3), 6.5))

    all_vals = [v for m in models for v in thresholds[m].values() if v]
    top = max(all_vals) if all_vals else 1000
    ax.set_ylim(0, top * 1.35)

    for i, tier in enumerate(tiers):
        offset = (i - (len(tiers) - 1) / 2) * width
        vals = [thresholds[m][tier] or 0 for m in models]
        bars = ax.bar(x + offset, vals, width, label=f"{tier} ({TIER_BITS.get(tier, '?')}-bit)",
                      color=tier_color(tier), edgecolor="#2c3e50", linewidth=1.0)
        for bar, val in zip(bars, vals):
            label = f"{val:,}" if val else "never"
            ax.annotate(label, xy=(bar.get_x() + bar.get_width() / 2, val), xytext=(0, 4),
                        textcoords="offset points", ha="center", va="bottom", fontsize=9,
                        fontweight="bold", color="#2c3e50")

    ax.set_xticks(x)
    ax.set_xticklabels([model_label(m) for m in models], fontsize=10, fontweight="bold", rotation=15)
    ax.set_ylabel("Minimum Thread Count for >=50% Baseline Object Recovery", fontsize=11, fontweight="bold")
    ax.grid(True, axis="y")

    # Dynamic ranking for the highest-precision tier present
    rank_tier = tiers[0] if tiers else None
    if rank_tier:
        ranked = sorted(models, key=lambda m: (thresholds[m][rank_tier] is None, thresholds[m][rank_tier] or 0))
        rank_str = " < ".join(
            f"{model_label(m)} ({thresholds[m][rank_tier]:,})" if thresholds[m][rank_tier] else f"{model_label(m)} (never)"
            for m in ranked
        )
        ax.text(
            0.5, 0.95, f"Fewest threads to reach 50% recovery ({rank_tier}):\n{rank_str}",
            transform=ax.transAxes, ha="center", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fef9e7", edgecolor="#f39c12", alpha=0.95),
        )

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=max(len(tiers), 1),
              frameon=True, facecolor="white", edgecolor="#cccccc")
    ax.set_title("Figure 2: Minimum Thread Count for Reliable Recovery (>=50% of Baseline Objects)", pad=12)
    plt.tight_layout()
    save_figure(fig, os.path.join(out_dir, "fig2_first_detection_threshold"), formats, dpi)


# -----------------------------------------------------------------------------
# Figure 3: Confidence and IoU of recovered detections
# -----------------------------------------------------------------------------
def plot_figure_3(df: pd.DataFrame, out_dir: str, formats: Sequence[str], dpi: int) -> None:
    print("[3/9] Generating Figure 3: Confidence & Localization IoU Trajectories...")
    models = present_models(df)
    tiers = [t for t in ("float16", "uint8") if t in set(df["quantization"])]
    styles = {"float16": "-", "uint8": "--"}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    for model in models:
        for tier in tiers:
            m_sub = df[(df["model"] == model) & (df["quantization"] == tier)].sort_values("thread_count")
            ax1.plot(m_sub["thread_count"], m_sub["confidence"], color=model_color(model),
                     linestyle=styles[tier], linewidth=1.9, alpha=0.9)
            ax2.plot(m_sub["thread_count"], m_sub["iou"], color=model_color(model),
                     linestyle=styles[tier], linewidth=1.9, alpha=0.9)

    ax1.set_ylabel("Mean Confidence of Recovered Detections", fontweight="bold")
    ax1.set_ylim(0.0, 1.02)
    ax1.grid(True)
    ax1.set_title("A. Confidence of Recovered Baseline Objects (gaps = nothing recovered)", loc="left", fontsize=11, fontweight="bold")

    ax2.set_ylabel("Mean Bounding Box IoU vs. Baseline", fontweight="bold")
    ax2.set_xlabel("Thread Count (N)", fontweight="bold")
    ax2.set_ylim(0.50, 1.005)
    ax2.grid(True)
    ax2.set_title("B. Localization of Recovered Objects (IoU >= 0.50 by definition of a match)", loc="left", fontsize=11, fontweight="bold")

    handles = [Line2D([0], [0], color=model_color(m), lw=2.2, label=model_label(m)) for m in models]
    handles += [Line2D([0], [0], color="#444444", lw=2.0, linestyle=styles[t], label=t) for t in tiers]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.97), ncol=5,
               frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)

    fig.suptitle("Figure 3: Confidence and Localization of Recovered Baseline Objects", y=1.01)
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, os.path.join(out_dir, "fig3_confidence_and_iou_trajectory"), formats, dpi)


# -----------------------------------------------------------------------------
# Figure 4: Effect of bit-width
# -----------------------------------------------------------------------------
def plot_figure_4(df: pd.DataFrame, out_dir: str, formats: Sequence[str], dpi: int) -> None:
    print("[4/9] Generating Figure 4: Effect of Bit-Width...")
    models = present_models(df)
    tiers = present_tiers(df)
    bit_widths = [TIER_BITS.get(t, 0) for t in tiers]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: threads needed for >=50% recovery vs bit-width
    for m in models:
        xs, ys = [], []
        for t, b in zip(tiers, bit_widths):
            th = first_threshold(df[(df["model"] == m) & (df["quantization"] == t)])
            if th is not None:
                xs.append(b)
                ys.append(th)
        if xs:
            ax1.plot(xs, ys, marker="o", linewidth=2.0, label=model_label(m), color=model_color(m))

    ax1.set_xlabel("Quantization Bit-Width (bits / thread attribute)", fontweight="bold")
    ax1.set_ylabel("Minimum Thread Count for >=50% Recovery", fontweight="bold")
    ax1.set_title("A. Recovery Threshold vs. Bit-Width (missing = never reached)", loc="left", fontsize=11, fontweight="bold")
    ax1.set_xticks(bit_widths)
    ax1.set_xlim(max(bit_widths) + 1, min(bit_widths) - 1)
    ax1.grid(True)
    if ax1.get_legend_handles_labels()[0]:
        ax1.legend(loc="best", frameon=True, facecolor="white", fontsize=9)

    # Panel B: recovery and precision at the largest thread count, mean over models
    n_final = int(df["thread_count"].max())
    at_final = df[df["thread_count"] == n_final]
    rec = [float(at_final[at_final["quantization"] == t]["recovery_rate"].mean() or 0.0) for t in tiers]
    prec = [float(at_final[at_final["quantization"] == t]["precision"].mean() or 0.0) for t in tiers]
    rec = [0.0 if np.isnan(v) else v * 100.0 for v in rec]
    prec = [0.0 if np.isnan(v) else v * 100.0 for v in prec]

    xb = np.arange(len(tiers))
    w = 0.36
    b1 = ax2.bar(xb - w / 2, rec, w, label="Recovery", color="#2980b9", edgecolor="#2c3e50", linewidth=1.0)
    b2 = ax2.bar(xb + w / 2, prec, w, label="Precision", color="#e67e22", edgecolor="#2c3e50", linewidth=1.0)
    for bars in (b1, b2):
        for bar in bars:
            v = bar.get_height()
            ax2.annotate(f"{v:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, v), xytext=(0, 3),
                         textcoords="offset points", ha="center", va="bottom", fontweight="bold", fontsize=10)

    ax2.set_xticks(xb)
    ax2.set_xticklabels([f"{TIER_BITS.get(t, '?')}-bit\n({t})" for t in tiers])
    ax2.set_ylabel(f"Mean over {len(models)} models (%)", fontweight="bold")
    ax2.set_xlabel(f"Precision Tier at N = {n_final:,} Threads", fontweight="bold")
    ax2.set_title(f"B. Recovery and Precision at N = {n_final:,} Threads", loc="left", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 115)
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(100))
    ax2.grid(True, axis="y")
    ax2.legend(loc="upper right", frameon=True, facecolor="white")

    fig.suptitle("Figure 4: Effect of Quantization Bit-Width on Baseline Recovery", y=1.02)
    plt.tight_layout()
    save_figure(fig, os.path.join(out_dir, "fig4_quantization_bitwidth_impact"), formats, dpi)


# -----------------------------------------------------------------------------
# Figure 5: Storage vs recovery (Pareto)
# -----------------------------------------------------------------------------
def plot_figure_5(df: pd.DataFrame, out_dir: str, formats: Sequence[str], dpi: int) -> None:
    print("[5/9] Generating Figure 5: Storage vs Recovery Pareto...")
    fig, ax = plt.subplots(figsize=(10, 6.5))

    # One point per (tier, thread count): recovery averaged over models
    pts = (
        df.groupby(["quantization", "thread_count"], as_index=False)
        .agg(storage_kb=("storage_kb", "first"), recovery=("recovery_rate", "mean"))
        .dropna(subset=["recovery"])
    )

    for tier in present_tiers(pts):
        sub = pts[pts["quantization"] == tier]
        bpt = int(round(sub["storage_kb"].iloc[0] * 1000.0 / sub["thread_count"].iloc[0])) if not sub.empty else 0
        ax.scatter(
            sub["storage_kb"], sub["recovery"] * 100.0,
            s=np.clip(sub["thread_count"] / 100.0, 15, 120),
            color=tier_color(tier), alpha=0.7, edgecolors="none",
            label=f"{tier} ({bpt} B/thread)",
        )

    # Pareto frontier: lowest storage for each new best recovery
    ordered = pts.sort_values(["storage_kb", "recovery"], ascending=[True, False])
    fx, fy, best = [], [], -1.0
    for s, r in zip(ordered["storage_kb"], ordered["recovery"]):
        if r > best:
            fx.append(s)
            fy.append(r * 100.0)
            best = r
    if fx:
        ax.step(fx, fy, where="post", color="#2c3e50", linestyle=":", linewidth=2.0, label="Pareto frontier")

    # Data-driven comparison note
    n_final = int(df["thread_count"].max())
    last = pts[pts["thread_count"] == n_final].set_index("quantization")["recovery"]
    if "float16" in last.index and "uint8" in last.index:
        ax.text(
            0.02, 0.97,
            f"At N = {n_final:,}: uint8 recovers {last['uint8'] * 100:.1f}% vs float16 {last['float16'] * 100:.1f}%\n"
            "at half the storage per thread (8 B vs 16 B)",
            transform=ax.transAxes, ha="left", va="top", fontsize=9.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#eafaf1", edgecolor="#27ae60"),
        )

    ax.set_xlabel("Reconstruction File Size / Storage (KB)", fontweight="bold")
    ax.set_ylabel("Mean Object Recovery Rate over Models (%)", fontweight="bold")
    ax.set_title("Figure 5: Storage Overhead vs. Baseline Object Recovery", pad=12)
    ax.set_ylim(-5, 105)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(100))
    ax.grid(True)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cccccc")

    plt.tight_layout()
    save_figure(fig, os.path.join(out_dir, "fig5_pareto_cost_vs_performance"), formats, dpi)


# -----------------------------------------------------------------------------
# Figure 6: Generation reconstruction quality (PSNR)
# -----------------------------------------------------------------------------
def plot_figure_6(df_gen: pd.DataFrame | None, out_dir: str, formats: Sequence[str], dpi: int) -> None:
    print("[6/9] Generating Figure 6: Generation Quality...")
    if df_gen is None or df_gen.empty:
        print("  Skipping Figure 6: Generation CSV data unavailable.")
        return

    fig, ax1 = plt.subplots(figsize=(10, 6))

    for tier in present_tiers(df_gen):
        sub = df_gen[df_gen["quantization"] == tier].sort_values("thread_count")
        if not sub.empty:
            ax1.plot(
                sub["thread_count"], sub["psnr_db"],
                label=f"{tier} (PSNR)",
                color=tier_color(tier),
                linewidth=2.2,
                linestyle="-" if tier in ("float32", "float16", "uint8") else "--",
            )

    ax1.set_xlabel("Thread Count (N)", fontweight="bold")
    ax1.set_ylabel("Reconstruction PSNR vs. Original COCO Photo (dB)", fontweight="bold", color="#1c2833")
    ax1.set_title("Figure 6: Digital Filography Visual Fidelity (Mean PSNR) vs. Thread Placement Density", pad=12)
    ax1.grid(True)
    ax1.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cccccc")

    plt.tight_layout()
    save_figure(fig, os.path.join(out_dir, "fig6_generation_quality_psnr_storage"), formats, dpi)


# -----------------------------------------------------------------------------
# Figure 7: Precision vs thread count (replaces the latency figure)
# -----------------------------------------------------------------------------
def plot_figure_7(df: pd.DataFrame, out_dir: str, formats: Sequence[str], dpi: int) -> None:
    plot_metric_by_tier(
        df, "precision", "Precision",
        "Figure 7: Precision of Reconstructions (Extra Detections Count as False Positives)",
        "fig7_precision_vs_threads", out_dir, formats, dpi,
        "[7/9] Generating Figure 7: Precision vs Thread Count...",
    )


# -----------------------------------------------------------------------------
# Appendix Figure 1: Recovery heatmaps (model x thread count)
# -----------------------------------------------------------------------------
def plot_appendix_heatmap(df: pd.DataFrame, out_dir: str, formats: Sequence[str], dpi: int) -> None:
    print("[8/9] Generating Appendix Figure 1: Recovery Heatmaps...")
    models = present_models(df)
    tiers = present_tiers(df)
    model_labels = [model_label(m) for m in models]

    fig, axes = plt.subplots(len(tiers), 1, figsize=(14, 2.6 * len(tiers) + 1.5), sharex=True, squeeze=False)

    for ax, tier in zip(axes.flatten(), tiers):
        sub = df[df["quantization"] == tier]
        pivot = sub.pivot_table(index="model", columns="thread_count", values="recovery_rate", aggfunc="mean").reindex(models)
        sns.heatmap(
            pivot,
            ax=ax,
            cmap="mako",
            vmin=0.0,
            vmax=1.0,
            cbar_kws={"label": "Recovery Rate", "shrink": 0.8},
            linewidths=0.2,
            linecolor="#333333",
        )
        ax.set_yticklabels(model_labels, rotation=0, fontweight="bold")
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_title(f"Object Recovery Rate: {tier}", loc="left", fontsize=11, fontweight="bold")

    axes.flatten()[-1].set_xlabel("Thread Count (N)", fontweight="bold")
    fig.suptitle("Appendix Figure 1: Baseline Object Recovery Heatmaps across Thread Counts", y=1.0)
    plt.tight_layout()
    save_figure(fig, os.path.join(out_dir, "fig_appendix_heatmap_confidence"), formats, dpi)


# -----------------------------------------------------------------------------
# Appendix Figure 2: F1 vs thread count (replaces the mAP figure)
# -----------------------------------------------------------------------------
def plot_appendix_precision_dynamics(df: pd.DataFrame, out_dir: str, formats: Sequence[str], dpi: int) -> None:
    plot_metric_by_tier(
        df, "f1", "F1 (Recovery and Precision)",
        "Appendix Figure 2: F1 Combining Recovery and Precision Across Thread Counts",
        "fig_appendix_precision_dynamics", out_dir, formats, dpi,
        "[9/9] Generating Appendix Figure 2: F1 vs Thread Count...",
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("================================================================================")
    print("Generating Digital Filography Publication Visualizations")
    print(f"Benchmark CSV:    {args.benchmark_csv}")
    print(f"Generation CSV:   {args.generation_csv}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Formats:          {args.formats} (DPI={args.dpi})")
    print("================================================================================")

    df_filo, df_gen = load_clean_data(args.benchmark_csv, args.generation_csv)
    print(f"Loaded {len(df_filo)} summary records for models: {present_models(df_filo)}")

    plot_figure_1(df_filo, args.output_dir, args.formats, args.dpi)
    plot_figure_2(df_filo, args.output_dir, args.formats, args.dpi)
    plot_figure_3(df_filo, args.output_dir, args.formats, args.dpi)
    plot_figure_4(df_filo, args.output_dir, args.formats, args.dpi)
    plot_figure_5(df_filo, args.output_dir, args.formats, args.dpi)
    plot_figure_6(df_gen, args.output_dir, args.formats, args.dpi)
    plot_figure_7(df_filo, args.output_dir, args.formats, args.dpi)
    plot_appendix_heatmap(df_filo, args.output_dir, args.formats, args.dpi)
    plot_appendix_precision_dynamics(df_filo, args.output_dir, args.formats, args.dpi)

    print(f"\nAll figures generated successfully in {args.output_dir}/!")
    return 0


if __name__ == "__main__":
    sys.exit(main())