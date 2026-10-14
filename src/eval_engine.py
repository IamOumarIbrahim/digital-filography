#!/usr/bin/env python3
"""Computational engine for YOLO dataset preparation, latency benchmarking, and testing.

Provides:
- Dataset discovery and COCO label formatting for YOLO evaluation
- Multi-seed validation loop calculating Mean ± Standard Deviation across seeds
- Latency and hardware throughput benchmarking
- Training session configuration and initialization
- Dataset downloading and cleanup actions
"""

import json
import math
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from config_manager import (
    ANNOTATIONS_CACHE,
    COCO_CLASSES,
    DATASET_DIR,
    MODELS_DIR,
    PROJECT_ROOT,
    RESULTS_DIR,
    RUNS_DIR,
    get_available_models,
    resolve_device,
    set_seed,
)

# ---------------------------------------------------------------------------
# Terminal Colors & Styling (Internal engine display)
# ---------------------------------------------------------------------------

USE_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
RED     = "\033[31m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
MAGENTA = "\033[35m"
CYAN    = "\033[36m"
WHITE   = "\033[37m"


def paint(text, *styles):
    if not USE_COLOR or not styles:
        return str(text)
    return "".join(styles) + str(text) + RESET


def say(kind, text):
    symbols = {
        "success": ("✔", GREEN),
        "info":    ("ℹ", CYAN),
        "warning": ("⚠", YELLOW),
        "error":   ("✖", RED),
    }
    symbol, color = symbols.get(kind, ("•", WHITE))
    print(paint(f"{symbol} ", BOLD, color) + text)


def confirm(question, default=False):
    hint = "Y/n" if default else "y/N"
    answer = input(paint("? ", BOLD, MAGENTA) + f"{question} ({hint}) ").strip().lower()
    if not answer:
        return default
    return answer.startswith("y")


# ---------------------------------------------------------------------------
# Statistical Helpers
# ---------------------------------------------------------------------------

def _mean_std(values: list[float]) -> tuple[float, float]:
    """Calculate mean and sample standard deviation (or 0.0 if values <= 1)."""
    if not values:
        return 0.0, 0.0
    n = len(values)
    mean_val = sum(values) / n
    if n <= 1:
        return mean_val, 0.0
    variance = sum((x - mean_val) ** 2 for x in values) / (n - 1)
    return mean_val, math.sqrt(max(0.0, variance))


# ---------------------------------------------------------------------------
# Results & Filesystem Helpers
# ---------------------------------------------------------------------------

def get_results_dir() -> Path:
    """Return the base results directory, creating it if needed."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def get_run_timestamp() -> str:
    """Format current time for folder names: YYYY-MM-DD_HH-MM-SS."""
    return datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")


def _count_all_images() -> int:
    """Count .jpg files in all subfolders of dataset/."""
    if not DATASET_DIR.exists():
        return 0
    return sum(1 for _ in DATASET_DIR.rglob("*.jpg"))


def _find_all_image_paths() -> list[Path]:
    """Return sorted list of absolute image paths inside dataset/."""
    if not DATASET_DIR.exists():
        return []
    return sorted(list(DATASET_DIR.rglob("*.jpg")))


def _list_result_runs() -> list[Path]:
    """Return sorted list of run directories in results/ and runs/."""
    runs = []
    if RESULTS_DIR.exists():
        for d in sorted(RESULTS_DIR.iterdir()):
            if d.is_dir():
                runs.append(d)
    if RUNS_DIR.exists():
        for d in sorted(RUNS_DIR.iterdir()):
            if d.is_dir() and d not in runs:
                runs.append(d)
    return runs


def _prepare_yolo_labels_and_dataset(images: list[Path], run_dir: Path, dataset_dir: Path, class_names: dict) -> Path:
    """Ensure every dataset image has a corresponding YOLO .txt annotation file and build test_data.yaml."""
    missing_labels = [p for p in images if not p.with_suffix(".txt").exists()]
    if missing_labels:
        say("info", f"Generating YOLO label files for {len(missing_labels):,} dataset images...")
        name_to_yolo_id = {name: int(idx) for idx, name in class_names.items()}
        coco_id_to_yolo_id = {}
        img_name_to_id = {}
        img_id_to_dim = {}
        img_id_to_annos = {}

        source_ann_files = list(dataset_dir.rglob("annotations.json"))
        if ANNOTATIONS_CACHE.exists() and ANNOTATIONS_CACHE not in source_ann_files:
            source_ann_files.append(ANNOTATIONS_CACHE)

        for af in source_ann_files:
            try:
                data = json.loads(af.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue

            for cat in data.get("categories", []):
                cname = cat.get("name")
                cid = cat.get("id")
                if cname in name_to_yolo_id and cid is not None:
                    coco_id_to_yolo_id[cid] = name_to_yolo_id[cname]

            for img in data.get("images", []):
                fname = img.get("file_name")
                iid = img.get("id")
                w = img.get("width")
                h = img.get("height")
                if fname and iid is not None:
                    img_name_to_id[fname] = iid
                    if w and h:
                        img_id_to_dim[iid] = (int(w), int(h))

            for a in data.get("annotations", []):
                iid = a.get("image_id")
                if iid is not None:
                    img_id_to_annos.setdefault(iid, []).append(a)

        from PIL import Image
        for img_path in missing_labels:
            txt_path = img_path.with_suffix(".txt")
            fname = img_path.name
            iid = img_name_to_id.get(fname)
            lines = []
            if iid is not None and iid in img_id_to_annos:
                dim = img_id_to_dim.get(iid)
                if not dim:
                    try:
                        with Image.open(img_path) as im:
                            dim = im.size
                    except OSError:
                        dim = (640, 640)
                w_img, h_img = dim
                for a in img_id_to_annos[iid]:
                    cid = a.get("category_id")
                    if cid in coco_id_to_yolo_id and "bbox" in a:
                        y_id = coco_id_to_yolo_id[cid]
                        bx, by, bw, bh = a["bbox"]
                        if bw >= 1 and bh >= 1:
                            xc = max(0.0, min(1.0, (bx + bw / 2.0) / w_img))
                            yc = max(0.0, min(1.0, (by + bh / 2.0) / h_img))
                            nw = max(0.0, min(1.0, bw / w_img))
                            nh = max(0.0, min(1.0, bh / h_img))
                            lines.append(f"{y_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
            txt_path.write_text("\n".join(lines), encoding="utf-8")

    val_txt_path = run_dir / "val_images.txt"
    val_lines = [str(p).replace("\\", "/") for p in images]
    val_txt_path.write_text("\n".join(val_lines) + "\n", encoding="utf-8")

    import yaml
    data_yaml_path = run_dir / "test_data.yaml"
    yaml_payload = {
        "path": str(dataset_dir).replace("\\", "/"),
        "train": str(val_txt_path).replace("\\", "/"),
        "val": str(val_txt_path).replace("\\", "/"),
        "test": str(val_txt_path).replace("\\", "/"),
        "names": class_names,
    }
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_payload, f)

    return data_yaml_path


# ---------------------------------------------------------------------------
# Dataset Management Actions
# ---------------------------------------------------------------------------

def _load_downloader():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "download_dataset",
        Path(__file__).resolve().parent / "download_dataset.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def action_start_download(app):
    count   = app.shared_settings["num_images"]
    classes = app.selected_classes or None
    workers = app.shared_settings.get("workers", 8)
    print()

    if classes:
        n   = len(classes)
        per = count // n
        say("info", f"Downloading ~{per:,} images × {n} class(es) with {workers} workers "
                    f"→ {', '.join('dataset/' + c for c in classes)}")
    else:
        say("info", f"Downloading {count:,} images with {workers} workers → dataset/ (no class filter)")

    try:
        mod     = _load_downloader()
        summary = mod.download_dataset(count, classes=classes, workers=workers)
        d, s, f = summary["downloaded"], summary["skipped"], len(summary["failed"])
        msg = f"Done — {d:,} downloaded, {s:,} skipped"
        if f:
            msg += f", {f:,} failed"
            app.notify("warning", msg)
        else:
            app.notify("success", msg)
    except KeyboardInterrupt:
        app.notify("warning", "Download interrupted. Run again to resume.")
    except Exception as exc:  # noqa: BLE001
        app.notify("error", f"Download failed: {exc}")


def action_clear_dataset(app):
    """Delete images and annotations from a chosen class folder (or everything)."""
    if not DATASET_DIR.exists():
        app.notify("info", "The dataset folder does not exist.")
        return

    subdirs = [d for d in sorted(DATASET_DIR.iterdir())
               if d.is_dir() and (any(d.rglob("*.jpg")) or (d / "annotations.json").exists())]
    all_images = list(DATASET_DIR.rglob("*.jpg"))

    if not all_images and not subdirs:
        app.notify("info", "The dataset folder is already empty.")
        return

    print()
    print(paint("Which folder do you want to clear?", BOLD, CYAN))
    print()
    choices = subdirs + [None]   # None = clear everything
    for i, d in enumerate(choices, start=1):
        if d is None:
            lbl = f"All folders ({len(all_images):,} images total)"
        else:
            n = sum(1 for _ in d.rglob("*.jpg"))
            lbl = f"dataset/{d.name}/  ({n:,} images)"
        print(f"  {paint(i, BOLD, YELLOW)}. {lbl}")
    print()

    raw = input(paint("> ", BOLD, MAGENTA)).strip()
    if not raw.isdigit() or not (1 <= int(raw) <= len(choices)):
        app.notify("info", "Cancelled.")
        return

    target = choices[int(raw) - 1]
    images = list(target.rglob("*.jpg")) if target else all_images
    desc   = f"all files in dataset/{target.name}/" if target else "all dataset files and subfolders"

    if not confirm(f"Delete {desc}? This cannot be undone."):
        app.notify("info", "Cancelled. Nothing was deleted.")
        return

    removed_count = len(images)
    if target:
        try:
            shutil.rmtree(target)
        except OSError as exc:
            app.notify("error", f"Could not remove folder dataset/{target.name}: {exc}")
            return
    else:
        for item in DATASET_DIR.iterdir():
            try:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            except OSError:
                pass

    msg = f"{removed_count:,} image(s) and folder data cleared."
    if ANNOTATIONS_CACHE.exists() and confirm(
            "Also delete the cached annotations (instances_val2017.json)?"):
        try:
            ANNOTATIONS_CACHE.unlink()
            msg += " Annotation cache cleared."
        except OSError:
            msg += " Could not delete the annotation cache."

    app.notify("success", msg)


def action_clear_results(app, auto_confirm: bool = False):
    """Delete previous run results from results/ or runs/."""
    run_dirs = _list_result_runs()
    if not run_dirs:
        app.notify("info", "The results folder is already empty.")
        return

    if auto_confirm:
        removed = 0
        for d in run_dirs:
            try:
                shutil.rmtree(d)
                removed += 1
            except OSError:
                pass
        app.notify("success", f"Cleared {removed} result folder(s).")
        say("success", f"Cleared {removed} previous result folder(s).")
        return

    print()
    print(paint("Which results do you want to clear?", BOLD, CYAN))
    print()
    choices = run_dirs + [None]   # None = clear everything
    for i, d in enumerate(choices, start=1):
        if d is None:
            lbl = f"All result folders ({len(run_dirs)} runs total)"
        else:
            try:
                rel = d.relative_to(PROJECT_ROOT)
            except ValueError:
                rel = d
            lbl = f"{rel}/"
        print(f"  {paint(i, BOLD, YELLOW)}. {lbl}")
    print()

    raw = input(paint("> ", BOLD, MAGENTA)).strip()
    if not raw.isdigit() or not (1 <= int(raw) <= len(choices)):
        app.notify("info", "Cancelled.")
        return

    target = choices[int(raw) - 1]
    targets = [target] if target else run_dirs
    desc = f"{target.name}/" if target else f"all {len(run_dirs)} run(s)"

    if not confirm(f"Delete {desc}? This cannot be undone."):
        app.notify("info", "Cancelled. Nothing was deleted.")
        return

    removed = 0
    for d in targets:
        try:
            shutil.rmtree(d)
            removed += 1
        except OSError:
            pass
    app.notify("success", f"Deleted {removed} result folder(s).")


# ---------------------------------------------------------------------------
# Model Selection Actions
# ---------------------------------------------------------------------------

def action_toggle_model(app, model_name: str):
    """Toggle a model in selected_models."""
    if model_name in app.selected_models:
        if len(app.selected_models) == 1:
            app.notify("warning", f"Cannot deselect '{model_name}'. At least one model must be active.")
            return
        app.selected_models.remove(model_name)
    else:
        app.selected_models.append(model_name)
    app.save()


def action_select_all_models(app):
    models = get_available_models()
    app.selected_models = list(models)
    app.save()
    app.notify("success", f"Selected all {len(models)} models.")


def action_clear_model_selection(app):
    models = get_available_models()
    default_m = models[0] if models else "yolov8n.pt"
    app.selected_models = [default_m]
    app.save()
    app.notify("info", f"Reset selection to single model: {default_m}")


def action_select_single_model(app, model_name: str):
    app.selected_models = [model_name]
    app.save()
    app.notify("success", f"Active model set to '{model_name}' exclusively.")


# ---------------------------------------------------------------------------
# Benchmark & Testing Actions
# ---------------------------------------------------------------------------

def action_test_latency(app):
    """Run real inference benchmark on selected models and save to results/[timestamp]/."""
    target_models = app.selected_models or [get_available_models()[0]]
    raw_device = app.shared_settings.get("device", "CUDA:0")
    seeds = app.shared_settings.get("seeds", [app.shared_settings.get("seed", 0)])
    set_seed(seeds[0])
    start_time = datetime.now().astimezone()
    timestamp_str = start_time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = get_results_dir() / timestamp_str

    iterations = app.metrics.get("iterations", 20)
    warmup_passes = app.metrics.get("warmup_passes", 5)
    track_speed = app.metrics.get("speed_breakdown", True)
    track_complexity = app.metrics.get("model_complexity", True)

    print()
    say("info", f"Initializing benchmark for {len(target_models)} model(s)...")
    say("info", f"Benchmark configuration: {iterations} timed iterations, {warmup_passes} warmup passes")
    if len(seeds) > 1:
        say("info", f"Seed(s) active: {paint(', '.join(str(s) for s in seeds), BOLD, GREEN)} ({len(seeds)} seeds)")
    else:
        say("info", f"Seed active: {paint(str(seeds[0]), BOLD, GREEN)}")

    try:
        resolved_device = resolve_device(raw_device)
    except Exception as exc:  # noqa: BLE001
        app.notify("error", f"Device resolution failed: {exc}")
        return

    cuda_active = resolved_device.startswith("cuda")
    track_vram = app.metrics.get("vram_usage", True) and cuda_active
    device_desc = resolved_device
    if cuda_active:
        import torch
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        device_desc = f"{raw_device} → {resolved_device} [{gpu_name}]"
        say("info", f"Compute target: {paint(device_desc, BOLD, GREEN)}")
        say("info", f"PyTorch CUDA active: {torch.version.cuda} | Device {torch.cuda.current_device()}")
    else:
        say("info", f"Compute target: {paint(resolved_device, BOLD, YELLOW)} (CPU)")

    say("info", f"Output directory: {paint(f'results/{timestamp_str}/', BOLD, CYAN)}")

    try:
        import numpy as np
        from ultralytics import YOLO
    except ImportError as exc:
        app.notify("error", f"Missing dependency: {exc}. Please ensure 'ultralytics' and 'numpy' are installed.")
        return

    results = []
    for model_name in target_models:
        model_path = MODELS_DIR / model_name
        if not model_path.exists():
            say("warning", f"Model file '{model_name}' not found in {MODELS_DIR}. Skipping.")
            continue

        say("info", f"Benchmarking {paint(model_name, BOLD, CYAN)}...")
        img_size = app.shared_settings.get("imgsz", 640)
        half_precision = app.shared_settings.get("half", False) and cuda_active
        conf = app.shared_settings.get("conf", 0.25)
        iou = app.shared_settings.get("iou", 0.70)

        try:
            model = YOLO(str(model_path))
            model.to(resolved_device)

            if cuda_active:
                is_on_cuda = next(model.model.parameters()).is_cuda
                if not is_on_cuda:
                    raise RuntimeError(f"Failed to place {model_name} onto CUDA.")

            params_m = None
            size_mb = None
            if track_complexity:
                if hasattr(model, "model") and hasattr(model.model, "parameters"):
                    params_m = sum(p.numel() for p in model.model.parameters()) / 1e6
                if model_path.exists():
                    size_mb = model_path.stat().st_size / (1024 * 1024)

            dummy_input = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            predict_kwargs = {
                "device": resolved_device,
                "conf": conf,
                "iou": iou,
                "verbose": False,
            }
            if half_precision:
                predict_kwargs["half"] = True

            if track_vram and cuda_active:
                import torch
                torch.cuda.reset_peak_memory_stats()

            for _ in range(warmup_passes):
                _ = model.predict(dummy_input, **predict_kwargs)

            if cuda_active:
                import torch
                torch.cuda.synchronize()

            latencies = []
            speed_samples = []
            for _ in range(iterations):
                t0 = time.perf_counter()
                preds = model.predict(dummy_input, **predict_kwargs)
                if cuda_active:
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)
                if track_speed and preds and hasattr(preds[0], "speed"):
                    speed_samples.append(preds[0].speed)

            avg_ms = sum(latencies) / len(latencies)
            min_ms = min(latencies)
            max_ms = max(latencies)
            fps = 1000.0 / avg_ms if avg_ms > 0 else 0

            pre_ms = sum(s.get("preprocess", 0) for s in speed_samples) / len(speed_samples) if speed_samples else 0.0
            inf_ms = sum(s.get("inference", 0) for s in speed_samples) / len(speed_samples) if speed_samples else 0.0
            post_ms = sum(s.get("postprocess", 0) for s in speed_samples) / len(speed_samples) if speed_samples else 0.0

            peak_vram = None
            res_vram = None
            if track_vram and cuda_active:
                import torch
                peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
                res_vram = torch.cuda.max_memory_reserved() / (1024 * 1024)

            entry = {
                "model": model_name,
                "avg_ms": avg_ms,
                "min_ms": min_ms,
                "max_ms": max_ms,
                "fps": fps,
                "precision": "FP16" if half_precision else "FP32",
                "imgsz": img_size,
            }
            if params_m is not None:
                entry["params_m"] = params_m
            if size_mb is not None:
                entry["size_mb"] = size_mb
            if track_speed:
                entry["preprocess_ms"] = pre_ms
                entry["inference_ms"] = inf_ms
                entry["postprocess_ms"] = post_ms
            if peak_vram is not None:
                entry["peak_vram_mb"] = peak_vram
                entry["reserved_vram_mb"] = res_vram

            results.append(entry)
            say("success", f"[{model_name}] Avg: {avg_ms:.2f} ms | FPS: {fps:.1f} | Min: {min_ms:.2f} ms | Max: {max_ms:.2f} ms")

        except Exception as exc:  # noqa: BLE001
            say("error", f"Error benchmarking {model_name}: {exc}")

    if not results:
        app.notify("warning", "No models were successfully benchmarked.")
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    benchmark_payload = {
        "timestamp_start": start_time.isoformat(),
        "folder": timestamp_str,
        "device_config": raw_device,
        "resolved_device": resolved_device,
        "device_description": device_desc,
        "seeds": seeds,
        "seed": seeds[0],
        "iterations": iterations,
        "warmup_passes": warmup_passes,
        "tracked_metrics": app.metrics,
        "results": results,
    }
    (run_dir / "latency_benchmark.json").write_text(json.dumps(benchmark_payload, indent=2) + "\n", encoding="utf-8")

    summary_lines = [
        "═" * 86,
        f" BENCHMARK & METRICS REPORT ({device_desc})",
        f" Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')} | Iterations: {iterations} | Warmup: {warmup_passes}",
        "═" * 86,
        f" {'Model':<14} {'Params':<8} {'Disk':<8} {'Prec.':<6} {'Latency (avg)':<15} {'Inference':<12} {'FPS':<7} {'Peak VRAM':<10}",
        "-" * 86,
    ]
    for r in results:
        m_str   = f" {r['model']:<13} "
        p_str   = f"{r.get('params_m', 0):.2f}M".ljust(8) if "params_m" in r else "—".ljust(8)
        sz_str  = f"{r.get('size_mb', 0):.1f}MB".ljust(8) if "size_mb" in r else "—".ljust(8)
        pr_str  = f"{r['precision']:<6} "
        lat_str = f"{r['avg_ms']:.2f} ms".ljust(15)
        inf_str = f"{r.get('inference_ms', r['avg_ms']):.2f} ms".ljust(12)
        fps_str = f"{r['fps']:.1f}".ljust(7)
        vr_str  = f"{r['peak_vram_mb']:.1f} MB".ljust(10) if "peak_vram_mb" in r else "—".ljust(10)
        summary_lines.append(m_str + p_str + sz_str + pr_str + lat_str + inf_str + fps_str + vr_str)

    if track_speed:
        summary_lines.append("-" * 86)
        summary_lines.append(" Detailed Speed Breakdown (Averages in ms):")
        for r in results:
            if "preprocess_ms" in r:
                summary_lines.append(
                    f"   • {r['model']:<14}: Preprocess={r['preprocess_ms']:.2f}ms | Forward={r['inference_ms']:.2f}ms | Postprocess={r['postprocess_ms']:.2f}ms | Total={r['avg_ms']:.2f}ms"
                )
    summary_lines.append("═" * 86)
    (run_dir / "latency_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print()
    print(paint("═" * 86, BOLD, CYAN))
    print(paint(f" BENCHMARK REPORT ({device_desc})", BOLD, WHITE))
    print(paint("═" * 86, BOLD, CYAN))
    print(f" {'Model':<14} {'Params':<8} {'Disk':<8} {'Prec.':<6} {'Latency (avg)':<15} {'Inference':<12} {'FPS':<7} {'Peak VRAM':<10}")
    print("-" * 86)
    for r in results:
        m_str   = f" {r['model']:<13} "
        p_str   = f"{r.get('params_m', 0):.2f}M".ljust(8) if "params_m" in r else "—".ljust(8)
        sz_str  = f"{r.get('size_mb', 0):.1f}MB".ljust(8) if "size_mb" in r else "—".ljust(8)
        pr_str  = f"{r['precision']:<6} "
        lat_str = f"{r['avg_ms']:.2f} ms".ljust(15)
        inf_str = f"{r.get('inference_ms', r['avg_ms']):.2f} ms".ljust(12)
        fps_str = f"{r['fps']:.1f}".ljust(7)
        vr_str  = f"{r['peak_vram_mb']:.1f} MB".ljust(10) if "peak_vram_mb" in r else "—".ljust(10)
        print(paint(m_str, BOLD, YELLOW) + p_str + sz_str + pr_str + paint(lat_str, BOLD, GREEN) + inf_str + fps_str + paint(vr_str, BOLD, CYAN))

    if track_speed:
        print("-" * 86)
        print(paint(" Detailed Speed Breakdown (ms):", BOLD, WHITE))
        for r in results:
            if "preprocess_ms" in r:
                print(
                    f"   • {paint(r['model'], BOLD, YELLOW)}: Preprocess={r['preprocess_ms']:.2f}ms | Forward={r['inference_ms']:.2f}ms | Postprocess={r['postprocess_ms']:.2f}ms | Total={r['avg_ms']:.2f}ms"
                )
    print(paint("═" * 86, BOLD, CYAN))
    print()
    say("success", f"Results saved to: {paint(f'results/{timestamp_str}/', BOLD, GREEN)}")
    say("info", f"  • {run_dir / 'latency_benchmark.json'}")
    say("info", f"  • {run_dir / 'latency_summary.txt'}")
    print()
    input(paint("Press Enter to return to menu... ", BOLD, MAGENTA))
    app.notify("success", f"Benchmark completed — saved to results/{timestamp_str}/")


def action_run_tests(app, auto_confirm: bool = False):
    """Run model validation and testing on dataset with multi-seed looping and Mean ± Std reporting."""
    target_models = app.selected_models or [get_available_models()[0]]
    raw_device = app.shared_settings.get("device", "CUDA:0")
    seeds = app.shared_settings.get("seeds", [app.shared_settings.get("seed", 0)])
    if not isinstance(seeds, list) or not seeds:
        seeds = [0]

    start_time = datetime.now().astimezone()
    timestamp_str = start_time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = get_results_dir() / timestamp_str

    images = _find_all_image_paths()
    if not images:
        app.notify("warning", "No images found in dataset/. Please run 'Download dataset' first.")
        return

    try:
        resolved_device = resolve_device(raw_device)
    except Exception as exc:  # noqa: BLE001
        app.notify("error", f"Device resolution failed: {exc}")
        return

    cuda_active = resolved_device.startswith("cuda")
    device_desc = resolved_device
    if cuda_active:
        import torch
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        device_desc = f"{raw_device} → {resolved_device} [{gpu_name}]"

    split = app.shared_settings.get("split", "val")
    track_speed = app.metrics.get("speed_breakdown", True)
    track_complexity = app.metrics.get("model_complexity", True)
    track_vram = app.metrics.get("vram_usage", True) and cuda_active

    seeds_str = ", ".join(str(s) for s in seeds)
    seed_desc = f"{seeds_str} ({len(seeds)} seed{'s' if len(seeds) > 1 else ''} — Mean ± Std mode)" if len(seeds) > 1 else str(seeds[0])

    print()
    print(paint("═" * 110, BOLD, CYAN))
    print(paint(" MODEL EVALUATION & TESTING (MULTI-SEED MEAN ± STD EVALUATION)", BOLD, WHITE))
    print(paint("═" * 110, BOLD, CYAN))
    print(f"  • Models to test:     {paint(', '.join(target_models), BOLD, YELLOW)} ({len(target_models)} models)")
    print(f"  • Target device:      {paint(device_desc, BOLD, GREEN)}")
    print(f"  • Dataset images:     {paint(f'{len(images):,} images', BOLD, GREEN)} in dataset/")
    print(f"  • Split:              {paint(split, BOLD, WHITE)}")
    print(f"  • Random seed(s):     {paint(seed_desc, BOLD, GREEN)}")
    print(f"  • Output directory:   {paint(f'results/{timestamp_str}/', BOLD, CYAN)}")
    print(paint("═" * 110, BOLD, CYAN))
    print()

    if not auto_confirm:
        confirm_msg = (
            f"Run test evaluation on {len(target_models)} model(s) across {len(seeds)} seed(s) "
            f"({len(target_models) * len(seeds)} total runs) with {resolved_device}?"
            if len(seeds) > 1
            else f"Run test evaluation on {len(target_models)} model(s) with {resolved_device}?"
        )
        if not confirm(confirm_msg):
            app.notify("info", "Test evaluation cancelled.")
            return

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        app.notify("error", f"Missing dependency: {exc}. Please ensure 'ultralytics' is installed.")
        return

    run_dir.mkdir(parents=True, exist_ok=True)

    # Load first model to retrieve COCO class names
    sample_model_name = target_models[0]
    sample_path = MODELS_DIR / sample_model_name
    if not sample_path.exists():
        app.notify("error", f"Model file '{sample_model_name}' not found in {MODELS_DIR}.")
        return

    say("info", "Preparing dataset and ground-truth labels for YOLO evaluation...")
    sample_model = YOLO(str(sample_path))
    class_names = getattr(sample_model, "names", {i: c for i, c in enumerate(COCO_CLASSES)})
    data_yaml_path = _prepare_yolo_labels_and_dataset(images, run_dir, DATASET_DIR, class_names)
    say("success", f"Dataset YAML created: {paint(str(data_yaml_path.name), BOLD, GREEN)}")

    all_runs = []
    aggregated_results = []

    for model_name in target_models:
        model_path = MODELS_DIR / model_name
        if not model_path.exists():
            say("warning", f"Model file '{model_name}' not found in {MODELS_DIR}. Skipping.")
            continue

        model_stem = model_name.replace(".pt", "")
        img_size = app.shared_settings.get("imgsz", 640)
        half_precision = app.shared_settings.get("half", False) and cuda_active
        conf = app.shared_settings.get("conf", 0.25)
        iou = app.shared_settings.get("iou", 0.70)
        max_det = app.shared_settings.get("max_det", 300)
        batch_size = app.shared_settings.get("batch_size", 32)
        workers = app.shared_settings.get("workers", 8)
        plots = app.shared_settings.get("plots", True)
        save_json = app.shared_settings.get("save_json", False)
        augment = app.shared_settings.get("augment", False)
        agnostic_nms = app.shared_settings.get("agnostic_nms", False)

        params_m = None
        size_mb = None
        try:
            model = YOLO(str(model_path))
            if track_complexity:
                if hasattr(model, "model") and hasattr(model.model, "parameters"):
                    params_m = sum(p.numel() for p in model.model.parameters()) / 1e6
                if model_path.exists():
                    size_mb = model_path.stat().st_size / (1024 * 1024)
        except Exception as exc:  # noqa: BLE001
            say("error", f"Error loading model weights for {model_name}: {exc}")
            continue

        model_seed_runs = []

        # Loop over every specified seed
        for seed_idx, seed in enumerate(seeds, 1):
            set_seed(seed)
            if len(seeds) > 1:
                say("info", f"Evaluating {paint(model_name, BOLD, CYAN)} [Seed {seed_idx}/{len(seeds)}: {paint(str(seed), BOLD, YELLOW)}] on {len(images):,} images...")
                run_name = f"{model_stem}_seed{seed}"
            else:
                say("info", f"Evaluating {paint(model_name, BOLD, CYAN)} on {len(images):,} images...")
                run_name = model_stem

            try:
                model = YOLO(str(model_path))

                if track_vram and cuda_active:
                    import torch
                    torch.cuda.reset_peak_memory_stats()

                # Execute model validation/test for this seed
                val_out = model.val(
                    data=str(data_yaml_path),
                    split=split,
                    batch=batch_size,
                    imgsz=img_size,
                    device=resolved_device,
                    conf=conf,
                    iou=iou,
                    half=half_precision,
                    max_det=max_det,
                    workers=workers,
                    plots=plots,
                    save_json=save_json,
                    augment=augment,
                    agnostic_nms=agnostic_nms,
                    project=str(run_dir),
                    name=run_name,
                    exist_ok=True,
                    verbose=False,
                )

                # Extract metrics
                map50_95 = float(val_out.box.map) if hasattr(val_out, "box") else 0.0
                map50 = float(val_out.box.map50) if hasattr(val_out, "box") else 0.0
                map75 = float(getattr(val_out.box, "map75", 0.0)) if hasattr(val_out, "box") else 0.0
                prec = float(val_out.box.mp) if hasattr(val_out, "box") else 0.0
                rec = float(val_out.box.mr) if hasattr(val_out, "box") else 0.0
                f1 = float(2 * prec * rec / (prec + rec + 1e-9))

                speed = getattr(val_out, "speed", {})
                inf_ms = float(speed.get("inference", 0.0))
                pre_ms = float(speed.get("preprocess", 0.0))
                post_ms = float(speed.get("postprocess", 0.0))
                tot_ms = pre_ms + inf_ms + post_ms
                fps = 1000.0 / tot_ms if tot_ms > 0 else 0.0

                peak_vram = None
                if track_vram and cuda_active:
                    import torch
                    peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)

                run_entry = {
                    "model": model_name,
                    "seed": seed,
                    "precision": "FP16" if half_precision else "FP32",
                    "map50_95": map50_95,
                    "map50": map50,
                    "map75": map75,
                    "precision_score": prec,
                    "recall_score": rec,
                    "f1_score": f1,
                    "inference_ms": inf_ms,
                    "preprocess_ms": pre_ms,
                    "postprocess_ms": post_ms,
                    "total_ms": tot_ms,
                    "fps": fps,
                    "imgsz": img_size,
                    "conf": conf,
                    "iou": iou,
                    "plots_saved": plots,
                    "run_folder": run_name,
                }
                if params_m is not None:
                    run_entry["params_m"] = params_m
                    run_entry["size_mb"] = size_mb
                if peak_vram is not None:
                    run_entry["peak_vram_mb"] = peak_vram

                model_seed_runs.append(run_entry)
                all_runs.append(run_entry)

                if len(seeds) > 1:
                    say("success", f"  ✔ [{model_name} | Seed {seed}] mAP50={map50:.3f} | mAP50-95={map50_95:.3f} | F1={f1:.3f} | Inf={inf_ms:.1f}ms ({fps:.1f} FPS)")
                else:
                    say("success", f"[{model_name}] mAP50={map50:.3f} | mAP50-95={map50_95:.3f} | F1={f1:.3f} | Inf={inf_ms:.1f}ms ({fps:.1f} FPS)")

            except KeyboardInterrupt:
                say("warning", f"Testing interrupted on {model_name} (seed {seed}).")
                break
            except Exception as exc:  # noqa: BLE001
                say("error", f"Error testing {model_name} (seed {seed}): {exc}")

        if not model_seed_runs:
            continue

        # Compute Mean ± Std across seeds for this model
        m50_mean, m50_std = _mean_std([r["map50"] for r in model_seed_runs])
        m95_mean, m95_std = _mean_std([r["map50_95"] for r in model_seed_runs])
        m75_mean, m75_std = _mean_std([r["map75"] for r in model_seed_runs])
        prec_mean, prec_std = _mean_std([r["precision_score"] for r in model_seed_runs])
        rec_mean, rec_std = _mean_std([r["recall_score"] for r in model_seed_runs])
        f1_mean, f1_std = _mean_std([r["f1_score"] for r in model_seed_runs])
        inf_mean, inf_std = _mean_std([r["inference_ms"] for r in model_seed_runs])
        pre_mean, pre_std = _mean_std([r["preprocess_ms"] for r in model_seed_runs])
        post_mean, post_std = _mean_std([r["postprocess_ms"] for r in model_seed_runs])
        tot_mean, tot_std = _mean_std([r["total_ms"] for r in model_seed_runs])
        fps_mean, fps_std = _mean_std([r["fps"] for r in model_seed_runs])
        vram_vals = [r["peak_vram_mb"] for r in model_seed_runs if "peak_vram_mb" in r]
        vram_mean, vram_std = _mean_std(vram_vals) if vram_vals else (0.0, 0.0)

        agg_entry = {
            "model": model_name,
            "precision": "FP16" if half_precision else "FP32",
            "num_seeds": len(model_seed_runs),
            "seeds": [r["seed"] for r in model_seed_runs],
            "map50": {"mean": m50_mean, "std": m50_std},
            "map50_95": {"mean": m95_mean, "std": m95_std},
            "map75": {"mean": m75_mean, "std": m75_std},
            "precision_score": {"mean": prec_mean, "std": prec_std},
            "recall_score": {"mean": rec_mean, "std": rec_std},
            "f1_score": {"mean": f1_mean, "std": f1_std},
            "inference_ms": {"mean": inf_mean, "std": inf_std},
            "preprocess_ms": {"mean": pre_mean, "std": pre_std},
            "postprocess_ms": {"mean": post_mean, "std": post_std},
            "total_ms": {"mean": tot_mean, "std": tot_std},
            "fps": {"mean": fps_mean, "std": fps_std},
            "map50_mean": m50_mean,
            "map50_95_mean": m95_mean,
            "f1_mean": f1_mean,
        }
        if params_m is not None:
            agg_entry["params_m"] = params_m
            agg_entry["size_mb"] = size_mb
        if vram_vals:
            agg_entry["peak_vram_mb"] = {"mean": vram_mean, "std": vram_std}

        aggregated_results.append(agg_entry)

        if len(seeds) > 1:
            say("info", f"★ [{paint(model_name, BOLD, CYAN)} Mean ± Std ({len(model_seed_runs)} seeds)]: "
                        f"mAP50={m50_mean:.3f}±{m50_std:.3f} | mAP50-95={m95_mean:.3f}±{m95_std:.3f} | "
                        f"F1={f1_mean:.3f}±{f1_std:.3f} | Inf={inf_mean:.1f}±{inf_std:.1f}ms ({fps_mean:.1f} FPS)")

    if not all_runs:
        app.notify("warning", "No models were successfully evaluated.")
        return

    # Build machine-readable test_results.json
    test_payload = {
        "timestamp_start": start_time.isoformat(),
        "folder": timestamp_str,
        "device_config": raw_device,
        "resolved_device": resolved_device,
        "device_description": device_desc,
        "dataset_images": len(images),
        "split": split,
        "seeds": seeds,
        "num_seeds": len(seeds),
        "shared_settings": app.shared_settings,
        "aggregated_results": aggregated_results,
        "results": all_runs,
    }
    (run_dir / "test_results.json").write_text(json.dumps(test_payload, indent=2) + "\n", encoding="utf-8")

    # Format human-readable test_summary.txt
    summary_lines = []
    width = 136 if len(seeds) > 1 else 98
    summary_lines.append("═" * width)
    summary_lines.append(f" MODEL EVALUATION & TEST REPORT ({device_desc})")
    seed_report_str = f"{len(seeds)} Seeds: [{seeds_str}]" if len(seeds) > 1 else f"Seed: {seeds[0]}"
    summary_lines.append(f" Evaluated on {len(images):,} images | Split: {split} | {seed_report_str} | Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append("═" * width)

    if len(seeds) > 1:
        summary_lines.append(f" {'Model':<13} {'Params':<8} {'Prec.':<6} {'mAP50 (mean±std)':<18} {'mAP50-95 (mean±std)':<20} {'P(box) (mean±std)':<18} {'R(box) (mean±std)':<18} {'F1 (mean±std)':<16} {'Inference (ms)':<16}")
        summary_lines.append("-" * width)
        for agg in aggregated_results:
            m_str   = f" {agg['model']:<12} "
            p_str   = f"{agg.get('params_m', 0):.2f}M".ljust(8) if "params_m" in agg else "—".ljust(8)
            pr_str  = f"{agg['precision']:<6} "
            m50_str = f"{agg['map50']['mean']:.3f} ± {agg['map50']['std']:.3f}".ljust(18)
            m95_str = f"{agg['map50_95']['mean']:.3f} ± {agg['map50_95']['std']:.3f}".ljust(20)
            pc_str  = f"{agg['precision_score']['mean']:.3f} ± {agg['precision_score']['std']:.3f}".ljust(18)
            rc_str  = f"{agg['recall_score']['mean']:.3f} ± {agg['recall_score']['std']:.3f}".ljust(18)
            f1_str  = f"{agg['f1_score']['mean']:.3f} ± {agg['f1_score']['std']:.3f}".ljust(16)
            inf_str = f"{agg['inference_ms']['mean']:.1f} ± {agg['inference_ms']['std']:.1f} ms".ljust(16)
            summary_lines.append(m_str + p_str + pr_str + m50_str + m95_str + pc_str + rc_str + f1_str + inf_str)
    else:
        summary_lines.append(f" {'Model':<13} {'Params':<8} {'Prec.':<6} {'mAP50':<8} {'mAP50-95':<10} {'P(box)':<8} {'R(box)':<8} {'F1':<7} {'Inference':<11} {'FPS':<7} {'Peak VRAM':<10}")
        summary_lines.append("-" * width)
        for r in all_runs:
            m_str   = f" {r['model']:<12} "
            p_str   = f"{r.get('params_m', 0):.2f}M".ljust(8) if "params_m" in r else "—".ljust(8)
            pr_str  = f"{r['precision']:<6} "
            m50_str = f"{r['map50']:.3f}".ljust(8)
            m_str95 = f"{r['map50_95']:.3f}".ljust(10)
            pc_str  = f"{r['precision_score']:.3f}".ljust(8)
            rc_str  = f"{r['recall_score']:.3f}".ljust(8)
            f1_str  = f"{r['f1_score']:.3f}".ljust(7)
            inf_str = f"{r['inference_ms']:.1f} ms".ljust(11)
            fps_str = f"{r['fps']:.1f}".ljust(7)
            vr_str  = f"{r['peak_vram_mb']:.1f} MB".ljust(10) if "peak_vram_mb" in r else "—".ljust(10)
            summary_lines.append(m_str + p_str + pr_str + m50_str + m_str95 + pc_str + rc_str + f1_str + inf_str + fps_str + vr_str)

    if track_speed:
        summary_lines.append("-" * width)
        summary_lines.append(" Detailed Speed Breakdown:")
        if len(seeds) > 1:
            for agg in aggregated_results:
                summary_lines.append(
                    f"   • {agg['model']:<14}: Preprocess={agg['preprocess_ms']['mean']:.2f}±{agg['preprocess_ms']['std']:.2f}ms | Forward={agg['inference_ms']['mean']:.2f}±{agg['inference_ms']['std']:.2f}ms | Postprocess={agg['postprocess_ms']['mean']:.2f}±{agg['postprocess_ms']['std']:.2f}ms | Total={agg['total_ms']['mean']:.2f}±{agg['total_ms']['std']:.2f}ms | FPS={agg['fps']['mean']:.1f}±{agg['fps']['std']:.1f}"
                )
        else:
            for r in all_runs:
                summary_lines.append(
                    f"   • {r['model']:<14}: Preprocess={r['preprocess_ms']:.2f}ms | Forward={r['inference_ms']:.2f}ms | Postprocess={r['postprocess_ms']:.2f}ms | Total={r['total_ms']:.2f}ms"
                )

    if len(seeds) > 1:
        summary_lines.append("-" * width)
        summary_lines.append(" Individual Seed Runs Breakdown:")
        for r in all_runs:
            summary_lines.append(
                f"   • {r['model']:<12} [Seed {r['seed']:<3}]: mAP50={r['map50']:.3f} | mAP50-95={r['map50_95']:.3f} | P={r['precision_score']:.3f} | R={r['recall_score']:.3f} | F1={r['f1_score']:.3f} | Inf={r['inference_ms']:.1f}ms ({r['fps']:.1f} FPS)"
            )

    summary_lines.append("═" * width)
    (run_dir / "test_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    # Print summary table to console
    print()
    print(paint("═" * width, BOLD, CYAN))
    print(paint(f" MODEL EVALUATION & TEST REPORT ({device_desc})", BOLD, WHITE))
    print(paint(f" Evaluated on {len(images):,} images | Split: {split} | {seed_report_str} | Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}", DIM))
    print(paint("═" * width, BOLD, CYAN))

    if len(seeds) > 1:
        print(f" {'Model':<13} {'Params':<8} {'Prec.':<6} {'mAP50 (mean±std)':<18} {'mAP50-95 (mean±std)':<20} {'P(box) (mean±std)':<18} {'R(box) (mean±std)':<18} {'F1 (mean±std)':<16} {'Inference (ms)':<16}")
        print("-" * width)
        for agg in aggregated_results:
            m_str   = f" {agg['model']:<12} "
            p_str   = f"{agg.get('params_m', 0):.2f}M".ljust(8) if "params_m" in agg else "—".ljust(8)
            pr_str  = f"{agg['precision']:<6} "
            m50_str = f"{agg['map50']['mean']:.3f} ± {agg['map50']['std']:.3f}".ljust(18)
            m95_str = f"{agg['map50_95']['mean']:.3f} ± {agg['map50_95']['std']:.3f}".ljust(20)
            pc_str  = f"{agg['precision_score']['mean']:.3f} ± {agg['precision_score']['std']:.3f}".ljust(18)
            rc_str  = f"{agg['recall_score']['mean']:.3f} ± {agg['recall_score']['std']:.3f}".ljust(18)
            f1_str  = f"{agg['f1_score']['mean']:.3f} ± {agg['f1_score']['std']:.3f}".ljust(16)
            inf_str = f"{agg['inference_ms']['mean']:.1f} ± {agg['inference_ms']['std']:.1f} ms".ljust(16)
            print(paint(m_str, BOLD, YELLOW) + p_str + pr_str + paint(m50_str, BOLD, GREEN) + paint(m95_str, BOLD, GREEN) + pc_str + rc_str + f1_str + inf_str)
    else:
        print(f" {'Model':<13} {'Params':<8} {'Prec.':<6} {'mAP50':<8} {'mAP50-95':<10} {'P(box)':<8} {'R(box)':<8} {'F1':<7} {'Inference':<11} {'FPS':<7} {'Peak VRAM':<10}")
        print("-" * width)
        for r in all_runs:
            m_str   = f" {r['model']:<12} "
            p_str   = f"{r.get('params_m', 0):.2f}M".ljust(8) if "params_m" in r else "—".ljust(8)
            pr_str  = f"{r['precision']:<6} "
            m50_str = f"{r['map50']:.3f}".ljust(8)
            m_str95 = f"{r['map50_95']:.3f}".ljust(10)
            pc_str  = f"{r['precision_score']:.3f}".ljust(8)
            rc_str  = f"{r['recall_score']:.3f}".ljust(8)
            f1_str  = f"{r['f1_score']:.3f}".ljust(7)
            inf_str = f"{r['inference_ms']:.1f} ms".ljust(11)
            fps_str = f"{r['fps']:.1f}".ljust(7)
            vr_str  = f"{r['peak_vram_mb']:.1f} MB".ljust(10) if "peak_vram_mb" in r else "—".ljust(10)
            print(paint(m_str, BOLD, YELLOW) + p_str + pr_str + paint(m50_str, BOLD, GREEN) + paint(m_str95, BOLD, GREEN) + pc_str + rc_str + f1_str + inf_str + fps_str + paint(vr_str, BOLD, CYAN))

    if track_speed:
        print("-" * width)
        print(paint(" Detailed Speed Breakdown (ms):", BOLD, WHITE))
        if len(seeds) > 1:
            for agg in aggregated_results:
                print(
                    f"   • {paint(agg['model'], BOLD, YELLOW)}: Preprocess={agg['preprocess_ms']['mean']:.2f}±{agg['preprocess_ms']['std']:.2f}ms | Forward={agg['inference_ms']['mean']:.2f}±{agg['inference_ms']['std']:.2f}ms | Postprocess={agg['postprocess_ms']['mean']:.2f}±{agg['postprocess_ms']['std']:.2f}ms | Total={agg['total_ms']['mean']:.2f}±{agg['total_ms']['std']:.2f}ms"
                )
        else:
            for r in all_runs:
                print(
                    f"   • {paint(r['model'], BOLD, YELLOW)}: Preprocess={r['preprocess_ms']:.2f}ms | Forward={r['inference_ms']:.2f}ms | Postprocess={r['postprocess_ms']:.2f}ms | Total={r['total_ms']:.2f}ms"
                )

    print(paint("═" * width, BOLD, CYAN))
    print()
    say("success", f"Results saved to: {paint(f'results/{timestamp_str}/', BOLD, GREEN)}")
    say("info", f"  • {run_dir / 'test_results.json'}")
    say("info", f"  • {run_dir / 'test_summary.txt'}")
    if any(r.get("plots_saved") for r in all_runs):
        say("info", f"  • Evaluation plots & PR curves saved in {run_dir}/<model>_seed*/")
    print()

    if not auto_confirm:
        input(paint("Press Enter to return to menu... ", BOLD, MAGENTA))
    app.notify("success", f"Test evaluation completed — saved to results/{timestamp_str}/")


def action_run_training(app):
    """Check dataset, device and trigger training pipeline, saving to results/[timestamp]/."""
    target_models = app.selected_models or [get_available_models()[0]]
    raw_device = app.shared_settings.get("device", "CUDA:0")
    seeds = app.shared_settings.get("seeds", [app.shared_settings.get("seed", 0)])
    set_seed(seeds[0])
    start_time = datetime.now().astimezone()
    timestamp_str = start_time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = get_results_dir() / timestamp_str

    try:
        resolved_device = resolve_device(raw_device)
    except Exception as exc:  # noqa: BLE001
        app.notify("error", f"Device resolution failed: {exc}")
        return

    cuda_active = resolved_device.startswith("cuda")
    if cuda_active:
        import torch
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        say("info", f"Using GPU device: {paint(gpu_name, BOLD, GREEN)} (device={resolved_device})")
    else:
        say("warning", f"Using CPU compute: {resolved_device}")

    n_images = _count_all_images()
    say("info", f"Dataset images available: {n_images:,}")
    if n_images == 0:
        app.notify("warning", "No images found in dataset/. Please run 'Download dataset' first.")
        return

    print()
    print(paint("Selected models for training / fine-tuning:", BOLD, CYAN))
    for m in target_models:
        ep = app.shared_settings["epochs"]
        bs = app.shared_settings["batch_size"]
        sz = app.shared_settings["imgsz"]
        print(f"  • {paint(m, BOLD, YELLOW)}: epochs={ep}, batch_size={bs}, imgsz={sz}")
    print()

    say("info", f"Output directory will be: {paint(f'results/{timestamp_str}/', BOLD, CYAN)}")
    print()

    if not confirm(f"Start training session on {len(target_models)} model(s) with {resolved_device}?"):
        app.notify("info", "Training cancelled.")
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    training_meta = {
        "timestamp_start": start_time.isoformat(),
        "folder": timestamp_str,
        "device": raw_device,
        "resolved_device": resolved_device,
        "seeds": seeds,
        "seed": seeds[0],
        "shared_settings": app.shared_settings,
        "selected_models": target_models,
        "dataset_images": n_images,
    }
    (run_dir / "training_config.json").write_text(
        json.dumps(training_meta, indent=2) + "\n", encoding="utf-8"
    )

    say("info", "Training pipeline initialized. Configuration saved.")
    say("success", f"Outputs mapped to: {paint(f'results/{timestamp_str}/', BOLD, GREEN)}")
    app.notify("success", f"Training configured — outputs saving to results/{timestamp_str}/")
