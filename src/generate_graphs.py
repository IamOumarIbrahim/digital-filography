#!/usr/bin/env python3
"""generate_graphs.py: Publication-quality graph generator for Digital Filography & YOLO Benchmarks.

Generates comprehensive, aesthetic, model-relative figures saved into graphs/:
1. Figure 1: Model-Relative Detection Recovery Rate vs. Thread Count (2x2 grid by quantization tier)
2. Figure 2: First Successful Detection Recovery Threshold (Grouped bar chart & architectural ranking)
3. Figure 3: Confidence & Localization IoU Trajectories of Recovered Objects (Dual panel)
4. Figure 4: Effect of Bit-Width (Quantization Frontier & Recovery Breakdown)
5. Figure 5: Pareto Storage Overhead vs. Recovered Detection Confidence
6. Figure 6: Visual Fidelity (Mean PSNR vs. Thread Count & Storage)
7. Figure 7: Hardware Latency Distribution Across Architectures (RTX 4060 GPU)
8. Appendix 1: Model-Relative Recovery Confidence Heatmaps (float16 & uint8)
9. Appendix 2: Model-Relative Precision Dynamics (mAP@0.50 & mAP@0.50:0.95)
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

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

# Harmonious palettes
MODEL_COLORS = {
    "yolov8n": "#1f77b4",   # Classic Blue
    "yolov10n": "#ff7f0e",  # Amber Orange
    "yolo11n": "#2ca02c",   # Emerald Green
    "yolo12n": "#d62728",   # Crimson Red
    "yolo26n": "#9467bd",   # Royal Purple
}

MODEL_LABELS = {
    "yolov8n": "YOLOv8n",
    "yolov10n": "YOLOv10n",
    "yolo11n": "YOLO11n",
    "yolo12n": "YOLO12n",
    "yolo26n": "YOLO26n",
}

TIER_COLORS = {
    "float32": "#7f8c8d",
    "float16": "#2980b9",
    "uint8": "#27ae60",
    "uint6": "#f39c12",
    "uint5": "#d35400",
    "uint4": "#c0392b",
}


def save_figure(fig: plt.Figure, base_path: str, formats: Sequence[str], dpi: int = 300) -> None:
    """Save figure in specified formats with clean bounding box."""
    for fmt in formats:
        target = f"{base_path}.{fmt}"
        fig.savefig(target, format=fmt, dpi=dpi, bbox_inches="tight")
        print(f"  Saved: {target}")
    plt.close(fig)


def load_clean_data(bench_path: str, gen_path: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Load and clean benchmark and generation datasets."""
    df_bench = pd.read_csv(bench_path, encoding="utf-8")
    if "image_type" in df_bench.columns:
        df_filo = df_bench[df_bench["image_type"] == "filography"].copy()
    else:
        df_filo = df_bench.copy()

    # Normalize recovery / success rate
    if "success_rate" in df_filo.columns:
        df_filo["success_rate_pct"] = pd.to_numeric(df_filo["success_rate"], errors="coerce").fillna(0.0) * 100.0
        df_filo["target_detected"] = df_filo["success_rate_pct"] > 0.0
    elif "recovery_rate" in df_filo.columns:
        df_filo["success_rate_pct"] = pd.to_numeric(df_filo["recovery_rate"], errors="coerce").fillna(0.0) * 100.0
        df_filo["target_detected"] = df_filo["success_rate_pct"] > 0.0
    elif "detected" in df_filo.columns:
        df_filo["target_detected"] = df_filo["detected"].astype(bool)
        df_filo["success_rate_pct"] = df_filo["target_detected"].astype(float) * 100.0
    elif "cat_detected" in df_filo.columns:
        df_filo["target_detected"] = df_filo["cat_detected"].astype(bool)
        df_filo["success_rate_pct"] = df_filo["target_detected"].astype(float) * 100.0
    else:
        df_filo["target_detected"] = False
        df_filo["success_rate_pct"] = 0.0

    df_filo["cat_detected"] = df_filo["target_detected"]

    # Normalize confidence & IoU
    if "mean_confidence" in df_filo.columns:
        df_filo["confidence"] = pd.to_numeric(df_filo["mean_confidence"], errors="coerce").fillna(0.0)
    elif "confidence" in df_filo.columns:
        df_filo["confidence"] = pd.to_numeric(df_filo["confidence"], errors="coerce").fillna(0.0)
    elif "cat_confidence" in df_filo.columns:
        df_filo["confidence"] = pd.to_numeric(df_filo["cat_confidence"], errors="coerce").fillna(0.0)
    else:
        df_filo["confidence"] = 0.0

    df_filo["cat_confidence"] = df_filo["confidence"]

    if "mean_iou" in df_filo.columns:
        df_filo["iou"] = pd.to_numeric(df_filo["mean_iou"], errors="coerce").fillna(0.0)
    elif "iou" in df_filo.columns:
        df_filo["iou"] = pd.to_numeric(df_filo["iou"], errors="coerce").fillna(0.0)
    elif "cat_iou" in df_filo.columns:
        df_filo["iou"] = pd.to_numeric(df_filo["cat_iou"], errors="coerce").fillna(0.0)
    else:
        df_filo["iou"] = 0.0

    df_filo["cat_iou"] = df_filo["iou"]

    # Normalize mAP
    if "mean_mAP50" in df_filo.columns:
        df_filo["mAP50"] = pd.to_numeric(df_filo["mean_mAP50"], errors="coerce").fillna(0.0)
    elif "mAP50" in df_filo.columns:
        df_filo["mAP50"] = pd.to_numeric(df_filo["mAP50"], errors="coerce").fillna(0.0)
    else:
        df_filo["mAP50"] = 0.0

    if "mean_mAP50_95" in df_filo.columns:
        df_filo["mAP50_95"] = pd.to_numeric(df_filo["mean_mAP50_95"], errors="coerce").fillna(0.0)
    elif "mAP50_95" in df_filo.columns:
        df_filo["mAP50_95"] = pd.to_numeric(df_filo["mAP50_95"], errors="coerce").fillna(0.0)
    else:
        df_filo["mAP50_95"] = 0.0

    if "mean_inference_time_ms" in df_filo.columns:
        df_filo["inference_time_ms"] = pd.to_numeric(df_filo["mean_inference_time_ms"], errors="coerce").fillna(0.0)
    elif "inference_time_ms" in df_filo.columns:
        df_filo["inference_time_ms"] = pd.to_numeric(df_filo["inference_time_ms"], errors="coerce").fillna(0.0)

    df_filo["thread_count"] = pd.to_numeric(df_filo["thread_count"], errors="coerce").astype(int)
    df_filo["storage_kb"] = pd.to_numeric(df_filo["storage_kb"], errors="coerce").fillna(0.0)

    df_gen = None
    if gen_path and os.path.exists(gen_path):
        df_gen_raw = pd.read_csv(gen_path, encoding="utf-8")
        clean_cols = {}
        for c in df_gen_raw.columns:
            clean_c = c.replace("↓", "").replace("↑", "").strip()
            clean_cols[c] = clean_c
        df_gen = df_gen_raw.rename(columns=clean_cols).copy()

        def parse_num(val: str) -> float:
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

    return df_filo, df_gen


# -----------------------------------------------------------------------------
# Figure 1: Model-Relative Detection Recovery Rate vs Thread Count (2x2 Grid)
# -----------------------------------------------------------------------------
def plot_figure_1(df: pd.DataFrame, out_dir: str, formats: Sequence[str], dpi: int) -> None:
    print("[1/9] Generating Figure 1: Model-Relative Detection Recovery vs Thread Count...")
    tiers = ["float16", "uint8", "uint6", "uint4"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for idx, tier in enumerate(tiers):
        ax = axes_flat[idx]
        sub = df[df["quantization"] == tier]
        for model in ["yolov8n", "yolov10n", "yolo11n", "yolo12n", "yolo26n"]:
            m_sub = sub[sub["model"] == model].sort_values("thread_count")
            ax.plot(
                m_sub["thread_count"],
                m_sub["success_rate_pct"],
                label=MODEL_LABELS.get(model, model),
                color=MODEL_COLORS.get(model, "#333333"),
                linewidth=2.2,
                marker="o",
                markersize=3.5,
                alpha=0.9,
            )

        ax.set_title(f"Quantization: {tier}", pad=8)
        ax.set_ylim(-5, 105)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(100))
        ax.grid(True)

        if tier in ["uint6", "uint4"]:
            ax.text(
                0.5, 0.5, "Zero Baseline Detections Recovered Across All 10,000 Threads\n(Sub-8-Bit Quantization Step Noise Collapse)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, fontweight="bold", color="#c0392b",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#fadbd8", edgecolor="#e74c3c", alpha=0.9)
            )

    fig.text(0.5, 0.04, "Thread Count (N)", ha="center", fontsize=12, fontweight="bold")
    fig.text(0.04, 0.5, "Model-Relative Object Recovery Rate (%)", va="center", rotation="vertical", fontsize=12, fontweight="bold")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98),
        ncol=5, frameon=True, facecolor="white", edgecolor="#cccccc"
    )

    fig.suptitle("Figure 1: Model-Relative Detection Recovery Rate Across Thread Counts and Quantization Tiers", y=1.02)
    plt.tight_layout(rect=[0.05, 0.05, 0.98, 0.95])
    save_figure(fig, os.path.join(out_dir, "fig1_detection_success_vs_threads"), formats, dpi)


# -----------------------------------------------------------------------------
# Figure 2: First Successful Detection Recovery Threshold (Grouped Bar Chart)
# -----------------------------------------------------------------------------
def plot_figure_2(df: pd.DataFrame, out_dir: str, formats: Sequence[str], dpi: int) -> None:
    print("[2/9] Generating Figure 2: First Successful Detection Recovery Threshold...")
    models = ["yolov8n", "yolo12n", "yolo11n", "yolov10n", "yolo26n"]
    thresholds: dict[str, dict[str, int | None]] = {m: {"float16": None, "uint8": None} for m in models}

    for m in models:
        for t in ["float16", "uint8"]:
            sub_mt = df[(df["model"] == m) & (df["quantization"] == t)]
            hits_50 = sub_mt[sub_mt["success_rate_pct"] >= 50.0]
            hits_any = sub_mt[sub_mt["success_rate_pct"] > 0.0]
            if not hits_50.empty:
                thresholds[m][t] = int(hits_50["thread_count"].min())
            elif not hits_any.empty:
                thresholds[m][t] = int(hits_any["thread_count"].min())

    x = np.arange(len(models))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 6.5))
    vals_f16 = [thresholds[m]["float16"] or 0 for m in models]
    vals_u8 = [thresholds[m]["uint8"] or 0 for m in models]

    b1 = ax.bar(x - width / 2, vals_f16, width, label="float16 (16-bit)", color="#2980b9", edgecolor="#1a5276", linewidth=1.2)
    b2 = ax.bar(x + width / 2, vals_u8, width, label="uint8 (8-bit)", color="#27ae60", edgecolor="#196f3d", linewidth=1.2)

    for bar, val in zip(b1, vals_f16):
        if val > 0:
            ax.annotate(f"{val:,}", xy=(bar.get_x() + bar.get_width() / 2, val), xytext=(0, 4),
                        textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#1a5276")

    for bar, val in zip(b2, vals_u8):
        if val > 0:
            ax.annotate(f"{val:,}", xy=(bar.get_x() + bar.get_width() / 2, val), xytext=(0, 4),
                        textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#196f3d")

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models], fontsize=11, fontweight="bold")
    ax.set_ylabel("Minimum Thread Count for ≥50% Baseline Object Recovery", fontsize=11, fontweight="bold")
    max_val = max(max(vals_f16), max(vals_u8), 5000)
    ax.set_ylim(0, max_val * 1.25)
    ax.grid(True, axis="y")

    # Dynamic ranking annotation
    ranked = sorted([(m, thresholds[m]["float16"] or 99999) for m in models], key=lambda t: t[1])
    rank_str = " > ".join([f"{MODEL_LABELS[m]} ({th:,})" if th < 99999 else f"{MODEL_LABELS[m]} (N/A)" for m, th in ranked])

    ax.text(
        0.5, 0.88,
        f"Model Recovery Ranking (float16):\n{rank_str}\n"
        "*Note: uint6 (6-bit) and uint4 (4-bit) never recovered baseline objects across 10,000 threads.*",
        transform=ax.transAxes, ha="center", va="center", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#fef9e7", edgecolor="#f39c12", alpha=0.95)
    )

    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cccccc")
    ax.set_title("Figure 2: Minimum Thread Count for Reliable Detection Recovery (≥50% Baseline Match)", pad=12)
    plt.tight_layout()
    save_figure(fig, os.path.join(out_dir, "fig2_first_detection_threshold"), formats, dpi)


# -----------------------------------------------------------------------------
# Figure 3: Confidence & Localization IoU of Recovered Baseline Objects
# -----------------------------------------------------------------------------
def plot_figure_3(df: pd.DataFrame, out_dir: str, formats: Sequence[str], dpi: int) -> None:
    print("[3/9] Generating Figure 3: Confidence & Localization IoU Trajectories...")
    df_detected = df[df["target_detected"] & (df["quantization"].isin(["float16", "uint8"]))].copy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    for model in ["yolov8n", "yolov10n", "yolo11n", "yolo12n", "yolo26n"]:
        col = MODEL_COLORS.get(model, "#333333")
        label = MODEL_LABELS.get(model, model)
        m_f16 = df_detected[(df_detected["model"] == model) & (df_detected["quantization"] == "float16")].sort_values("thread_count")
        m_u8 = df_detected[(df_detected["model"] == model) & (df_detected["quantization"] == "uint8")].sort_values("thread_count")

        if not m_f16.empty:
            ax1.plot(m_f16["thread_count"], m_f16["confidence"], label=f"{label} (float16)", color=col, linewidth=2.0)
            ax2.plot(m_f16["thread_count"], m_f16["iou"], label=f"{label} (float16)", color=col, linewidth=2.0)

        if not m_u8.empty:
            ax1.plot(m_u8["thread_count"], m_u8["confidence"], label=f"{label} (uint8)", color=col, linestyle="--", linewidth=1.7, alpha=0.85)
            ax2.plot(m_u8["thread_count"], m_u8["iou"], label=f"{label} (uint8)", color=col, linestyle="--", linewidth=1.7, alpha=0.85)

    ax1.set_ylabel("Mean Confidence of Recovered Detections", fontweight="bold")
    ax1.set_ylim(0.2, 1.0)
    ax1.grid(True)
    ax1.set_title("A. Recovered Baseline Object Detection Confidence Progression", loc="left", fontsize=11, fontweight="bold")

    ax2.set_ylabel("Mean Bounding Box IoU vs. Baseline", fontweight="bold")
    ax2.set_xlabel("Thread Count (N)", fontweight="bold")
    ax2.set_ylim(0.50, 1.005)
    ax2.grid(True)
    ax2.set_title("B. Bounding Box Localization Fidelity (IoU vs. Uncompressed Original Baseline)", loc="left", fontsize=11, fontweight="bold")

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=5, frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)

    fig.suptitle("Figure 3: Confidence and Localization IoU Dynamics of Recovered Baseline Objects", y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, os.path.join(out_dir, "fig3_confidence_and_iou_trajectory"), formats, dpi)


# -----------------------------------------------------------------------------
# Figure 4: Effect of Bit-Width (Quantization Frontier)
# -----------------------------------------------------------------------------
def plot_figure_4(df: pd.DataFrame, out_dir: str, formats: Sequence[str], dpi: int) -> None:
    print("[4/9] Generating Figure 4: Effect of Bit-Width...")
    models = ["yolov8n", "yolov10n", "yolo11n", "yolo12n", "yolo26n"]
    tiers = ["float16", "uint8", "uint6", "uint4"]
    bit_widths = [16, 8, 6, 4]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Minimum threads needed vs Bit-Width
    for m in models:
        req_threads = []
        valid_bits = []
        for t, b in zip(tiers, bit_widths):
            hits = df[(df["model"] == m) & (df["quantization"] == t) & (df["target_detected"])]
            if not hits.empty:
                req_threads.append(hits["thread_count"].min())
                valid_bits.append(b)
        if req_threads:
            ax1.plot(valid_bits, req_threads, marker="o", linewidth=2.2, label=MODEL_LABELS[m], color=MODEL_COLORS[m])

    ax1.set_xlabel("Quantization Bit-Width (bits / thread attribute)", fontweight="bold")
    ax1.set_ylabel("Minimum Thread Count for Baseline Recovery", fontweight="bold")
    ax1.set_title("A. Recovery Thread Threshold vs. Quantization Bit-Width", loc="left", fontsize=11, fontweight="bold")
    ax1.set_xticks(bit_widths)
    ax1.set_xlim(17, 3)
    ax1.grid(True)
    ax1.axvspan(7.0, 3.5, color="#f9ebea", alpha=0.8, label="Sub-8-Bit Collapse Zone")
    ax1.legend(loc="upper right", frameon=True, facecolor="white")

    # Panel B: Success rate at 10,000 threads across bit-width
    at_10k = df[df["thread_count"] == 10000]
    success_rates = []
    for t in tiers:
        sub_t = at_10k[at_10k["quantization"] == t]
        rate = float(sub_t["success_rate_pct"].mean()) if len(sub_t) > 0 else 0.0
        success_rates.append(rate)

    bar_colors = ["#2980b9", "#27ae60", "#c0392b", "#c0392b"]
    bars = ax2.bar([str(b) + "-bit\n(" + t + ")" for b, t in zip(bit_widths, tiers)], success_rates, color=bar_colors, edgecolor="#2c3e50", linewidth=1.2, width=0.55)
    for bar, rate in zip(bars, success_rates):
        ax2.annotate(f"{rate:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, rate), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontweight="bold", fontsize=11)

    ax2.set_ylabel("Baseline Recovery Rate Across All 5 Architectures (%)", fontweight="bold")
    ax2.set_xlabel("Precision Tier at N = 10,000 Threads", fontweight="bold")
    ax2.set_title("B. High-Density Baseline Object Recovery (N = 10,000 Threads)", loc="left", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 115)
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(100))
    ax2.grid(True, axis="y")

    fig.suptitle("Figure 4: The Quantization Frontier — Absolute Collapse of Sub-8-Bit Representations", y=1.02)
    plt.tight_layout()
    save_figure(fig, os.path.join(out_dir, "fig4_quantization_bitwidth_impact"), formats, dpi)


# -----------------------------------------------------------------------------
# Figure 5: Pareto Cost vs Performance Trade-off
# -----------------------------------------------------------------------------
def plot_figure_5(df: pd.DataFrame, out_dir: str, formats: Sequence[str], dpi: int) -> None:
    print("[5/9] Generating Figure 5: Pareto Cost vs Performance Trade-off...")
    fig, ax = plt.subplots(figsize=(10, 6.5))

    scatter_tiers = ["float16", "uint8", "uint6", "uint4"]
    for tier in scatter_tiers:
        sub = df[df["quantization"] == tier]
        if not sub.empty:
            bpt = sub["bytes_per_thread"].iloc[0] if "bytes_per_thread" in sub.columns else (4 if tier == "uint8" else 8)
            ax.scatter(
                sub["storage_kb"],
                sub["confidence"],
                s=np.clip(sub["thread_count"] / 100.0, 15, 120),
                color=TIER_COLORS.get(tier, "#888"),
                alpha=0.65,
                edgecolors="none",
                label=f"{tier} ({bpt} B/thread)",
            )

    # Highlight Pareto optimal envelope (uint8 achieves 50% storage savings with equal confidence)
    u8_detected = df[(df["quantization"] == "uint8") & (df["target_detected"])].sort_values("storage_kb")
    if not u8_detected.empty:
        pareto_pts = u8_detected.groupby("storage_kb")["confidence"].max().reset_index()
        ax.plot(pareto_pts["storage_kb"], pareto_pts["confidence"], color="#27ae60", linestyle=":", linewidth=2.0, label="uint8 Efficiency Frontier")

    ax.set_xlabel("Reconstruction File Size / Storage (KB)", fontweight="bold")
    ax.set_ylabel("Mean Confidence of Recovered Detections", fontweight="bold")
    ax.set_title("Figure 5: Pareto Trade-off: Storage Overhead vs. Recovered Detection Confidence", pad=12)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cccccc")

    ax.annotate(
        "uint8 Sweet Spot:\n50% Storage Reduction\nMatches float16 Confidence",
        xy=(80, 0.85), xytext=(40, 0.60),
        arrowprops=dict(facecolor="#27ae60", shrink=0.08, width=1.5, headwidth=6),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#eafaf1", edgecolor="#27ae60"),
        fontsize=9.5, fontweight="bold"
    )

    plt.tight_layout()
    save_figure(fig, os.path.join(out_dir, "fig5_pareto_cost_vs_performance"), formats, dpi)


# -----------------------------------------------------------------------------
# Figure 6: Generation Reconstruction Quality (PSNR & Storage)
# -----------------------------------------------------------------------------
def plot_figure_6(df_gen: pd.DataFrame | None, out_dir: str, formats: Sequence[str], dpi: int) -> None:
    print("[6/9] Generating Figure 6: Generation Quality...")
    if df_gen is None or df_gen.empty:
        print("  Skipping Figure 6: Generation CSV data unavailable.")
        return

    fig, ax1 = plt.subplots(figsize=(10, 6))

    for tier in ["float32", "float16", "uint8", "uint6", "uint4"]:
        sub = df_gen[df_gen["quantization"] == tier].sort_values("thread_count")
        if not sub.empty:
            ax1.plot(
                sub["thread_count"], sub["psnr_db"],
                label=f"{tier} (PSNR)",
                color=TIER_COLORS.get(tier, "#333"),
                linewidth=2.2,
                linestyle="-" if tier in ["float32", "float16", "uint8"] else "--",
            )

    ax1.set_xlabel("Thread Count (N)", fontweight="bold")
    ax1.set_ylabel("Reconstruction PSNR vs. Original COCO Photo (dB)", fontweight="bold", color="#1c2833")
    ax1.set_title("Figure 6: Digital Filography Visual Fidelity (Mean PSNR) vs. Thread Placement Density", pad=12)
    ax1.grid(True)
    ax1.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cccccc")

    plt.tight_layout()
    save_figure(fig, os.path.join(out_dir, "fig6_generation_quality_psnr_storage"), formats, dpi)


# -----------------------------------------------------------------------------
# Figure 7: Inference Latency Distribution Across Models
# -----------------------------------------------------------------------------
def plot_figure_7(df: pd.DataFrame, out_dir: str, formats: Sequence[str], dpi: int) -> None:
    print("[7/9] Generating Figure 7: Inference Latency...")
    fig, ax = plt.subplots(figsize=(9, 5.5))

    models = ["yolov8n", "yolov10n", "yolo11n", "yolo12n", "yolo26n"]
    palette = [MODEL_COLORS[m] for m in models]

    sns.boxplot(
        data=df,
        x="model",
        y="inference_time_ms",
        hue="model",
        legend=False,
        order=models,
        palette=palette,
        ax=ax,
        fliersize=2,
        linewidth=1.2,
        boxprops=dict(alpha=0.8),
    )

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in models], fontweight="bold")
    ax.set_ylabel("Inference Time (ms) on RTX 4060 GPU", fontweight="bold")
    ax.set_xlabel("YOLO Model Architecture", fontweight="bold")
    ax.set_title("Figure 7: Hardware Latency Distribution Across Architectures (Batch Size = 1)", pad=12)
    ax.set_ylim(8, 45)
    ax.grid(True, axis="y")

    # Annotate median latency
    medians = df.groupby("model")["inference_time_ms"].median()
    for idx, m in enumerate(models):
        med = medians.get(m, 0.0)
        ax.annotate(f"{med:.1f} ms", xy=(idx, med), xytext=(0, 18), textcoords="offset points", ha="center", fontsize=9.5, fontweight="bold", color="#2c3e50")

    plt.tight_layout()
    save_figure(fig, os.path.join(out_dir, "fig7_inference_latency"), formats, dpi)


# -----------------------------------------------------------------------------
# Appendix Figure 1: Full Heatmaps (Model x Thread Count)
# -----------------------------------------------------------------------------
def plot_appendix_heatmap(df: pd.DataFrame, out_dir: str, formats: Sequence[str], dpi: int) -> None:
    print("[8/9] Generating Appendix Figure 1: Confidence Heatmaps...")
    models = ["yolov8n", "yolo12n", "yolo11n", "yolov10n", "yolo26n"]
    model_labels = [MODEL_LABELS[m] for m in models]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    for ax, tier in zip([ax1, ax2], ["float16", "uint8"]):
        sub = df[df["quantization"] == tier]
        pivot = sub.pivot_table(index="model", columns="thread_count", values="confidence", aggfunc="mean").reindex(models)
        sns.heatmap(
            pivot,
            ax=ax,
            cmap="mako",
            vmin=0.0,
            vmax=1.0,
            cbar_kws={"label": "Mean Confidence", "shrink": 0.8},
            linewidths=0.2,
            linecolor="#333333",
        )
        ax.set_yticklabels(model_labels, rotation=0, fontweight="bold")
        ax.set_ylabel(f"{tier}", fontweight="bold", fontsize=11)
        ax.set_title(f"Model-Relative Detection Confidence Progression: {tier}", loc="left", fontsize=11, fontweight="bold")

    ax2.set_xlabel("Thread Count (N)", fontweight="bold")
    fig.suptitle("Appendix Figure 1: Architectural Baseline Object Recovery Heatmaps across Thread Counts", y=1.02)
    plt.tight_layout()
    save_figure(fig, os.path.join(out_dir, "fig_appendix_heatmap_confidence"), formats, dpi)


# -----------------------------------------------------------------------------
# Appendix Figure 2: Model-Relative Precision Dynamics (mAP@0.50 & mAP@0.50:0.95)
# -----------------------------------------------------------------------------
def plot_appendix_precision_dynamics(df: pd.DataFrame, out_dir: str, formats: Sequence[str], dpi: int) -> None:
    print("[9/9] Generating Appendix Figure 2: Model-Relative Precision Dynamics...")
    models = ["yolov8n", "yolo12n", "yolo11n", "yolov10n", "yolo26n"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)

    sub_f16 = df[df["quantization"] == "float16"]

    for m in models:
        m_sub = sub_f16[sub_f16["model"] == m].sort_values("thread_count")
        col = MODEL_COLORS.get(m, "#333")
        lab = MODEL_LABELS.get(m, m)

        if "mAP50" in m_sub.columns and m_sub["mAP50"].max() > 0:
            ax1.plot(m_sub["thread_count"], m_sub["mAP50"], label=lab, color=col, linewidth=2.0, marker="o", markersize=3)
        if "mAP50_95" in m_sub.columns and m_sub["mAP50_95"].max() > 0:
            ax2.plot(m_sub["thread_count"], m_sub["mAP50_95"], label=lab, color=col, linewidth=2.0, marker="s", markersize=3)

    ax1.set_title("A. Mean mAP@0.50 vs. Baseline Ground Truth", loc="left", fontweight="bold")
    ax1.set_xlabel("Thread Count (N)", fontweight="bold")
    ax1.set_ylabel("mAP @ IoU 0.50", fontweight="bold")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True)
    ax1.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cccccc")

    ax2.set_title("B. Mean mAP@0.50:0.95 vs. Baseline Ground Truth", loc="left", fontweight="bold")
    ax2.set_xlabel("Thread Count (N)", fontweight="bold")
    ax2.set_ylabel("mAP @ IoU 0.50:0.95", fontweight="bold")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True)
    ax2.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cccccc")

    fig.suptitle("Appendix Figure 2: Model-Relative Precision Metrics Across Thread Counts (float16 Tier)", y=1.02)
    plt.tight_layout()
    save_figure(fig, os.path.join(out_dir, "fig_appendix_precision_dynamics"), formats, dpi)


# -----------------------------------------------------------------------------
# Main CLI Runner
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate publication figures for Digital Filography and YOLO benchmarks.")
    parser.add_argument("--benchmark-csv", type=str, default="output/dataset_evaluation_summary.csv", help="Path to YOLO benchmark results CSV")
    parser.add_argument("--generation-csv", type=str, default="output/generation.csv", help="Path to Filography generation metrics CSV")
    parser.add_argument("--output-dir", type=str, default="graphs", help="Directory to save generated plots")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for raster output")
    parser.add_argument("--formats", nargs="+", default=["png", "svg"], help="Output formats (e.g. png svg pdf)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("================================================================================")
    print("Generating Digital Filography & YOLO Detection Publication Visualizations")
    print(f"Benchmark CSV:    {args.benchmark_csv}")
    print(f"Generation CSV:   {args.generation_csv}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Formats:          {args.formats} (DPI={args.dpi})")
    print("================================================================================")

    df_filo, df_gen = load_clean_data(args.benchmark_csv, args.generation_csv)
    print(f"Loaded {len(df_filo)} filography benchmark records.")

    # Generate Figures
    plot_figure_1(df_filo, args.output_dir, args.formats, args.dpi)
    plot_figure_2(df_filo, args.output_dir, args.formats, args.dpi)
    plot_figure_3(df_filo, args.output_dir, args.formats, args.dpi)
    plot_figure_4(df_filo, args.output_dir, args.formats, args.dpi)
    plot_figure_5(df_filo, args.output_dir, args.formats, args.dpi)
    plot_figure_6(df_gen, args.output_dir, args.formats, args.dpi)
    plot_figure_7(df_filo, args.output_dir, args.formats, args.dpi)
    plot_appendix_heatmap(df_filo, args.output_dir, args.formats, args.dpi)
    plot_appendix_precision_dynamics(df_filo, args.output_dir, args.formats, args.dpi)

    print("\nAll figures generated successfully in graphs/!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
