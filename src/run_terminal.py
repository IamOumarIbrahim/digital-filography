#!/usr/bin/env python3
"""Menu based terminal for dataset management and YOLO model workflows.

Start it from the project root:
    python src/run_terminal.py

Keys that work on every screen:
    1, 2, 3 ...  choose an option
    b            go back (like a browser)
    f            go forward
    h            go to the main menu
    q            quit

Settings are saved to config/config.json, with clean separation between
shared pipeline settings and model-specific parameters.
"""

import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT      = Path(__file__).resolve().parent.parent
CONFIG_PATH       = PROJECT_ROOT / "config" / "config.json"
DATASET_DIR       = PROJECT_ROOT / "dataset"
MODELS_DIR        = PROJECT_ROOT / "models"
RESULTS_DIR       = PROJECT_ROOT / "results"
RUNS_DIR          = PROJECT_ROOT / "runs"
ANNOTATIONS_CACHE = PROJECT_ROOT / "config" / "instances_val2017.json"

# ---------------------------------------------------------------------------
# YOLO Models catalogue
# ---------------------------------------------------------------------------

AVAILABLE_MODELS = (
    "yolo26n.pt",
    "yolo11n.pt",
    "yolov8n.pt",
    "yolov10n.pt",
    "yolo12n.pt",
)


def get_available_models() -> list[str]:
    """Return available models in models/ directory, keeping catalog order."""
    found = set()
    if MODELS_DIR.exists():
        found = {f.name for f in MODELS_DIR.glob("*.pt")}
    catalog = [m for m in AVAILABLE_MODELS if m in found or not found]
    extra = sorted(found - set(AVAILABLE_MODELS))
    return catalog + extra


# ---------------------------------------------------------------------------
# COCO class catalogue (val2017 - 80 categories, alphabetical)
# ---------------------------------------------------------------------------

COCO_CLASSES = (
    "airplane",      "apple",         "backpack",     "banana",        "baseball bat",
    "baseball glove","bear",           "bed",          "bench",         "bicycle",
    "bird",          "boat",          "book",          "bottle",        "bowl",
    "broccoli",      "bus",           "cake",          "car",           "carrot",
    "cat",           "cell phone",    "chair",         "clock",         "couch",
    "cow",           "cup",           "dining table",  "dog",           "donut",
    "elephant",      "fire hydrant",  "fork",          "frisbee",       "giraffe",
    "hair drier",    "handbag",       "horse",         "hot dog",       "keyboard",
    "kite",          "knife",         "laptop",        "microwave",     "motorcycle",
    "mouse",         "orange",        "oven",          "parking meter", "person",
    "pizza",         "potted plant",  "refrigerator",  "remote",        "sandwich",
    "scissors",      "sheep",         "sink",          "skateboard",    "skis",
    "snowboard",     "sports ball",   "spoon",         "stop sign",     "suitcase",
    "surfboard",     "teddy bear",    "tennis racket", "tie",           "toaster",
    "toilet",        "toothbrush",    "traffic light", "train",         "truck",
    "tv",            "umbrella",      "vase",          "wine glass",    "zebra",
)

# ---------------------------------------------------------------------------
# CUDA / Device resolution
# ---------------------------------------------------------------------------

def get_cuda_status():
    """Return (is_available, description, device_name)."""
    try:
        import torch
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            name = torch.cuda.get_device_name(0)
            return True, f"CUDA available ({count} GPU(s))", name
        return False, "CUDA not available (CPU only)", None
    except ImportError:
        return False, "PyTorch not installed", None
    except Exception as exc:  # noqa: BLE001
        return False, f"CUDA error: {exc}", None


def resolve_device(device_str: str = "CUDA:0") -> str:
    """Normalize device string and ensure CUDA:0 actually resolves to CUDA device.

    PyTorch requires lowercase 'cuda:0' or 'cuda' (uppercase 'CUDA:0' causes an error
    in torch.device). If CUDA:0 or any cuda device is requested:
      - Validates torch.cuda.is_available()
      - Selects the target GPU index with torch.cuda.set_device()
      - Returns lowercase 'cuda:<index>'
    """
    raw = str(device_str).strip()
    lower = raw.lower()
    if lower.startswith("cuda") or lower.isdigit():
        try:
            import torch
        except ImportError:
            raise RuntimeError("PyTorch is required to use CUDA device.")
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Device '{device_str}' requested, but CUDA is not available in PyTorch. "
                "Please verify NVIDIA drivers and CUDA PyTorch installation."
            )
        if lower.isdigit():
            idx = int(lower)
        else:
            parts = lower.split(":")
            idx = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        if idx >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {idx} requested ('{device_str}'), but only "
                f"{torch.cuda.device_count()} CUDA device(s) are available."
            )
        torch.cuda.set_device(idx)
        return f"cuda:{idx}"
    elif lower == "cpu":
        return "cpu"
    return lower


# ---------------------------------------------------------------------------
# Terminal setup
# ---------------------------------------------------------------------------

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

if os.name == "nt":
    os.system("")   # enable ANSI colors in the Windows console

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
GRAY    = "\033[90m"


def paint(text, *styles):
    """Wrap text in color/style codes (does nothing if colors are off)."""
    if not USE_COLOR or not styles:
        return str(text)
    return "".join(styles) + str(text) + RESET


def clear_screen():
    if sys.stdout.isatty():
        os.system("cls" if os.name == "nt" else "clear")


def banner(title, color=CYAN):
    """Print a title inside a box."""
    width = max(len(title) + 8, 36)
    print(paint("╔" + "═" * (width - 2) + "╗", BOLD, color))
    print(paint("║", BOLD, color) + paint(title.center(width - 2), BOLD, WHITE) + paint("║", BOLD, color))
    print(paint("╚" + "═" * (width - 2) + "╝", BOLD, color))


MESSAGE_STYLES = {
    "success": ("✔", GREEN),
    "info":    ("ℹ", CYAN),
    "warning": ("⚠", YELLOW),
    "error":   ("✖", RED),
}


def say(kind, text):
    """Print a message with a colored symbol. Kind: success, info, warning, error."""
    symbol, color = MESSAGE_STYLES[kind]
    print(paint(f"{symbol} ", BOLD, color) + text)


def confirm(question, default=False):
    """Ask a yes/no question."""
    hint   = "Y/n" if default else "y/N"
    answer = input(paint("? ", BOLD, MAGENTA) + f"{question} ({hint}) ").strip().lower()
    if not answer:
        return default
    return answer.startswith("y")


# ---------------------------------------------------------------------------
# Settings Schema
# ---------------------------------------------------------------------------

SHARED_SETTINGS_SCHEMA = {
    "device": {
        "group":   "Hardware",
        "label":   "Compute device",
        "type":    str,
        "default": "CUDA:0",
    },
    "conf": {
        "group":   "Testing & Inference",
        "label":   "Confidence threshold (conf)",
        "type":    float,
        "default": 0.25,
        "min":     0.01,
        "max":     1.0,
    },
    "iou": {
        "group":   "Testing & Inference",
        "label":   "NMS IoU threshold (iou)",
        "type":    float,
        "default": 0.70,
        "min":     0.01,
        "max":     1.0,
    },
    "half": {
        "group":   "Testing & Inference",
        "label":   "FP16 half-precision (half)",
        "type":    bool,
        "default": False,
    },
    "max_det": {
        "group":   "Testing & Inference",
        "label":   "Max detections per image (max_det)",
        "type":    int,
        "default": 300,
        "min":     1,
        "max":     1000,
    },
    "plots": {
        "group":   "Testing & Inference",
        "label":   "Save evaluation plots (plots)",
        "type":    bool,
        "default": True,
    },
    "save_json": {
        "group":   "Testing & Inference",
        "label":   "Save COCO format JSON (save_json)",
        "type":    bool,
        "default": False,
    },
    "agnostic_nms": {
        "group":   "Testing & Inference",
        "label":   "Class-agnostic NMS (agnostic_nms)",
        "type":    bool,
        "default": False,
    },
    "augment": {
        "group":   "Testing & Inference",
        "label":   "Test-time augmentation (augment)",
        "type":    bool,
        "default": False,
    },
    "split": {
        "group":   "Testing & Inference",
        "label":   "Dataset split (split)",
        "type":    str,
        "default": "val",
    },
    "epochs": {
        "group":   "Training",
        "label":   "Epochs",
        "type":    int,
        "default": 10,
        "min":     1,
        "max":     200,
    },
    "batch_size": {
        "group":   "Training",
        "label":   "Batch size",
        "type":    int,
        "default": 32,
        "min":     1,
        "max":     128,
    },
    "learning_rate": {
        "group":   "Training",
        "label":   "Learning rate",
        "type":    float,
        "default": 0.001,
        "min":     0.000001,
        "max":     1.0,
    },
    "imgsz": {
        "group":   "Pipeline",
        "label":   "Image size (imgsz)",
        "type":    int,
        "default": 640,
        "min":     128,
        "max":     1280,
    },
    "workers": {
        "group":   "Pipeline",
        "label":   "Worker threads",
        "type":    int,
        "default": 8,
        "min":     1,
        "max":     32,
    },
    "seeds": {
        "group":   "Pipeline",
        "label":   "Random seed(s)",
        "type":    list,
        "default": [0],
    },
    "num_images": {
        "group":   "Dataset",
        "label":   "Number of images",
        "type":    int,
        "default": 5000,
        "min":     1,
        "max":     5000,
    },
}

METRICS_SCHEMA = {
    "speed_breakdown": {
        "label":       "Speed breakdown (preprocess, inference, postprocess)",
        "description": "Measure separate timings for image prep, model forward, and NMS in ms",
        "type":        bool,
        "default":     True,
    },
    "model_complexity": {
        "label":       "Model complexity (parameter count & file size)",
        "description": "Extract total parameters (M) and model weights disk size (MB)",
        "type":        bool,
        "default":     True,
    },
    "vram_usage": {
        "label":       "GPU VRAM memory tracking (allocated & reserved)",
        "description": "Track peak CUDA memory allocated and reserved during inference (MB)",
        "type":        bool,
        "default":     True,
    },
    "detection_accuracy": {
        "label":       "Detection quality metrics (mAP50, mAP50-95, P, R)",
        "description": "Evaluate object detection precision, recall, and mAP on dataset",
        "type":        bool,
        "default":     True,
    },
    "iterations": {
        "label":       "Benchmark iterations",
        "description": "Number of timed iterations per model for latency / FPS benchmarking",
        "type":        int,
        "default":     20,
        "min":         5,
        "max":         500,
    },
    "warmup_passes": {
        "label":       "Warmup passes",
        "description": "Number of warmup inference passes before measuring latency",
        "type":        int,
        "default":     5,
        "min":         0,
        "max":         50,
    },
}


def format_value(spec, value):
    """Format setting value for display."""
    if spec["type"] is bool:
        return "True" if value else "False"
    if spec["type"] is list:
        if isinstance(value, list):
            if not value:
                return "none"
            if len(value) == 1:
                return str(value[0])
            if len(value) <= 3:
                return f"{', '.join(str(v) for v in value)} ({len(value)} seeds)"
            return f"{', '.join(str(v) for v in value[:3])} (+{len(value) - 3} more)"
        return str(value)
    if spec["type"] is int:
        return f"{value:,}"
    if spec["type"] is float:
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def parse_value(spec, text):
    """Turn text into a valid value. Returns (value, error_message)."""
    if spec["type"] is list:
        if isinstance(text, list):
            vals = [int(x) for x in text if str(x).replace("_", "").isdigit() and int(str(x).replace("_", "")) >= 0]
            return vals or [0], None
        cleaned = str(text).strip().strip("[]").replace(",", " ").split()
        vals = []
        for c in cleaned:
            c_clean = c.replace("_", "")
            if c_clean.isdigit() and int(c_clean) >= 0:
                vals.append(int(c_clean))
            else:
                return None, f"'{c}' is not a valid non-negative integer."
        return vals or [0], None

    text = str(text).strip()
    if spec["type"] is bool:
        lower = text.lower()
        if lower in ("true", "t", "1", "yes", "y", "on"):
            return True, None
        if lower in ("false", "f", "0", "no", "n", "off"):
            return False, None
        return None, "Please enter true/false (or y/n)."

    if spec["type"] is str:
        if not text:
            return None, "Value cannot be empty."
        return text, None

    clean = text.replace("_", "")
    if spec["type"] is int:
        clean = clean.replace(",", "")
    try:
        value = spec["type"](clean)
    except ValueError:
        kind = "a whole number" if spec["type"] is int else "a number"
        return None, f"Please enter {kind}."

    if "min" in spec and "max" in spec and not (spec["min"] <= value <= spec["max"]):
        low  = format_value(spec, spec["min"])
        high = format_value(spec, spec["max"])
        return None, f"The value must be between {low} and {high}."
    return value, None


# ---------------------------------------------------------------------------
# Config Load and Save
# ---------------------------------------------------------------------------

def load_settings():
    """Load settings from config/config.json."""
    shared = {k: spec["default"] for k, spec in SHARED_SETTINGS_SCHEMA.items()}
    shared["selected_classes"] = []
    selected_models = ["yolov8n.pt"]
    metrics = {k: spec["default"] for k, spec in METRICS_SCHEMA.items()}

    try:
        saved = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {
            "shared": shared,
            "selected_models": selected_models,
            "metrics": metrics,
        }

    if not isinstance(saved, dict):
        return {
            "shared": shared,
            "selected_models": selected_models,
            "metrics": metrics,
        }

    # 1. Load shared settings
    shared_src = saved.get("shared", saved)  # fallback to top-level for legacy config
    if isinstance(shared_src, dict):
        for key, spec in SHARED_SETTINGS_SCHEMA.items():
            if key in shared_src:
                val, err = parse_value(spec, shared_src[key] if isinstance(shared_src[key], list) else str(shared_src[key]))
                if err is None:
                    shared[key] = val
        # Sync seeds and seed
        raw_seeds = shared_src.get("seeds")
        if isinstance(raw_seeds, list) and raw_seeds:
            s_parsed = [int(s) for s in raw_seeds if str(s).replace("_", "").isdigit() and int(str(s).replace("_", "")) >= 0]
            if s_parsed:
                shared["seeds"] = s_parsed
                shared["seed"] = s_parsed[0]
        elif "seed" in shared_src:
            try:
                s_val = int(shared_src["seed"])
                if s_val >= 0:
                    shared["seeds"] = [s_val]
                    shared["seed"] = s_val
            except (ValueError, TypeError):
                pass
        raw_classes = shared_src.get("selected_classes", saved.get("selected_classes", []))
        if isinstance(raw_classes, list):
            shared["selected_classes"] = [c for c in raw_classes if isinstance(c, str) and c in COCO_CLASSES]

    # 2. Load selected models
    raw_models = saved.get("selected_models")
    if isinstance(raw_models, list) and raw_models:
        valid = [m for m in raw_models if isinstance(m, str)]
        if valid:
            selected_models = valid

    # 3. Load metrics settings
    metrics_src = saved.get("metrics")
    if isinstance(metrics_src, dict):
        for k, spec in METRICS_SCHEMA.items():
            if k in metrics_src:
                val, err = parse_value(spec, str(metrics_src[k]))
                if err is None:
                    metrics[k] = val

    return {
        "shared": shared,
        "selected_models": selected_models,
        "metrics": metrics,
    }


def save_settings(data):
    """Write settings dict to config/config.json. Returns True on success."""
    try:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Ensure seeds and seed stay in sync
        shared_seeds = data["shared"].get("seeds", [0])
        if not isinstance(shared_seeds, list) or not shared_seeds:
            shared_seeds = [data["shared"].get("seed", 0)]
        data["shared"]["seeds"] = shared_seeds
        data["shared"]["seed"] = shared_seeds[0]

        # Construct clean, organized JSON structure
        payload = {
            "selected_models": data["selected_models"],
            "shared": data["shared"],
            "metrics": data.get("metrics", {k: s["default"] for k, s in METRICS_SCHEMA.items()}),
        }
        CONFIG_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    except OSError:
        return False
    return True


# ---------------------------------------------------------------------------
# App State and Navigation
# ---------------------------------------------------------------------------

class App:
    """Holds the settings and the back/forward history."""

    def __init__(self):
        loaded                = load_settings()
        self.shared_settings  = loaded["shared"]
        self.selected_models  = list(loaded["selected_models"])
        self.metrics          = loaded.get("metrics", {k: s["default"] for k, s in METRICS_SCHEMA.items()})
        self.selected_classes = list(self.shared_settings.get("selected_classes", []))
        self.current          = "main"
        self.back_stack       = []
        self.forward_stack    = []
        self.message          = None
        self.running          = True

    def save(self):
        """Persist settings to config/config.json."""
        self.shared_settings["selected_classes"] = self.selected_classes
        payload = {
            "selected_models": self.selected_models,
            "shared": self.shared_settings,
            "metrics": self.metrics,
        }
        return save_settings(payload)

    def notify(self, kind, text):
        """Queue a message to show on the next render."""
        self.message = (kind, text)

    def go(self, screen_id):
        if screen_id == self.current:
            return
        self.back_stack.append(self.current)
        self.forward_stack.clear()
        self.current = screen_id

    def back(self):
        if not self.back_stack:
            self.notify("info", "There is nothing to go back to.")
            return
        self.forward_stack.append(self.current)
        self.current = self.back_stack.pop()

    def forward(self):
        if not self.forward_stack:
            self.notify("info", "There is nothing to go forward to.")
            return
        self.back_stack.append(self.current)
        self.current = self.forward_stack.pop()


@dataclass
class Item:
    label:  str
    action: Callable
    detail: str = ""


@dataclass
class Screen:
    title: str
    items: Callable
    body:  Callable = lambda app: []
    color: str = CYAN


# ---------------------------------------------------------------------------
# Shared Settings Editing
# ---------------------------------------------------------------------------

def prompt_multiple_seeds(current_seeds: list[int]) -> list[int] | None:
    """Interactive CLI to prompt for number of seeds and specific seed values."""
    print()
    print(paint("Random Seeds Configuration", BOLD, CYAN))
    current_str = ", ".join(str(s) for s in current_seeds) if current_seeds else "none"
    print(f"Current seed(s): {paint(current_str, BOLD, GREEN)} ({len(current_seeds)} active)")
    print()

    while True:
        raw_count = input(paint("Enter how many seeds you want (press Enter to cancel): ", BOLD, MAGENTA)).strip()
        if not raw_count:
            return None
        if not raw_count.isdigit() or int(raw_count) < 1:
            say("warning", "Please enter a whole number of at least 1 (e.g. 1, 2, 3).")
            continue
        count = int(raw_count)
        break

    print()
    if count == 1:
        print("Please enter the seed to use (whole number >= 0):")
    else:
        print(f"Please enter the {count} seeds that you want to use (e.g. 0, 42, 100):")
        print(paint("Hint: You can enter them one by one, or all at once separated by commas/spaces.", DIM))
    print()

    seeds: list[int] = []
    while len(seeds) < count:
        remaining = count - len(seeds)
        idx = len(seeds) + 1
        prompt_txt = f"Seed {idx} of {count}"
        if remaining > 1 and idx == 1:
            prompt_txt += f" (or enter all {count} separated by commas/spaces)"
        prompt_txt += " (press Enter to cancel): "

        raw = input(paint(prompt_txt, BOLD, MAGENTA)).strip()
        if not raw:
            return None

        # Check if user entered multiple values (comma or space separated)
        parts = raw.replace(",", " ").split()
        if len(parts) > 1:
            batch_vals = []
            parse_err = False
            for p in parts:
                clean = p.replace("_", "")
                if not clean.isdigit() or int(clean) < 0:
                    say("warning", f"'{p}' is not a valid non-negative integer. Please try again.")
                    parse_err = True
                    break
                batch_vals.append(int(clean))
            if parse_err:
                continue

            for v in batch_vals:
                if len(seeds) < count:
                    seeds.append(v)
            if len(seeds) < count:
                print(paint(f"Added {len(batch_vals)} seeds ({', '.join(str(s) for s in seeds)}). Still need {count - len(seeds)} more.", CYAN))
            continue

        clean = raw.replace("_", "")
        if not clean.isdigit() or int(clean) < 0:
            say("warning", f"'{raw}' is not a valid non-negative integer. Please enter a whole number >= 0.")
            continue
        seeds.append(int(clean))

    return seeds


def edit_shared_setting(app, key):
    spec = SHARED_SETTINGS_SCHEMA[key]
    current = app.shared_settings.get(key, spec["default"])
    print()
    print(paint(spec["label"], BOLD, CYAN))

    if key in ("seeds", "seed"):
        current_seeds = app.shared_settings.get("seeds", [app.shared_settings.get("seed", 0)])
        new_seeds = prompt_multiple_seeds(current_seeds)
        if new_seeds is None:
            app.notify("info", "No changes made.")
            return
        app.shared_settings["seeds"] = new_seeds
        app.shared_settings["seed"] = new_seeds[0]
        seeds_str = ", ".join(str(s) for s in new_seeds)
        if app.save():
            app.notify("success", f"Random seed(s) set to: {seeds_str} ({len(new_seeds)} active).")
        else:
            app.notify("warning", "Random seed(s) set, but could not save.")
        return

    if key == "device":
        cuda_ok, cuda_desc, gpu_name = get_cuda_status()
        print(f"Current device: {paint(current, BOLD, GREEN)}")
        if cuda_ok:
            print(f"Hardware detected: {paint(gpu_name, BOLD, YELLOW)} ({cuda_desc})")
        else:
            print(paint(f"Hardware note: {cuda_desc}", YELLOW))
        print()
        print("Quick options:")
        cuda_lbl = f"CUDA:0 ({gpu_name})" if cuda_ok else "CUDA:0 (Force CUDA)"
        print(f"  {paint('1', BOLD, YELLOW)}. {cuda_lbl}")
        print(f"  {paint('2', BOLD, YELLOW)}. cpu (Force CPU)")
        print(f"  {paint('3', BOLD, YELLOW)}. Custom device string")
        print()
        raw = input(paint("Select option [1-3] or enter custom device (press Enter to cancel): ", BOLD, MAGENTA)).strip()
        if not raw:
            app.notify("info", "No changes made.")
            return
        if raw == "1":
            new_val = "CUDA:0"
        elif raw == "2":
            new_val = "cpu"
        elif raw == "3":
            cust = input(paint("Enter device string (e.g. cuda:0, cuda:1, cpu): ", BOLD, MAGENTA)).strip()
            if not cust:
                app.notify("info", "No changes made.")
                return
            new_val = cust
        else:
            new_val = raw

        # Verify device
        try:
            resolved = resolve_device(new_val)
            app.shared_settings["device"] = new_val
            app.save()
            app.notify("success", f"Device set to '{new_val}' (verified: {resolved}).")
        except Exception as exc:  # noqa: BLE001
            app.notify("error", f"Device verification failed: {exc}")
        return

    if key == "split":
        print(f"Current split: {paint(current, BOLD, GREEN)}")
        print()
        print("Quick options:")
        print(f"  {paint('1', BOLD, YELLOW)}. val (Validation split - default)")
        print(f"  {paint('2', BOLD, YELLOW)}. test (Test split)")
        print(f"  {paint('3', BOLD, YELLOW)}. train (Train split)")
        print(f"  {paint('4', BOLD, YELLOW)}. Custom split name")
        print()
        raw = input(paint("Select option [1-4] or enter custom split (press Enter to cancel): ", BOLD, MAGENTA)).strip()
        if not raw:
            app.notify("info", "No changes made.")
            return
        if raw == "1":
            new_val = "val"
        elif raw == "2":
            new_val = "test"
        elif raw == "3":
            new_val = "train"
        elif raw == "4":
            cust = input(paint("Enter split name: ", BOLD, MAGENTA)).strip()
            if not cust:
                app.notify("info", "No changes made.")
                return
            new_val = cust
        else:
            new_val = raw
        app.shared_settings["split"] = new_val
        app.save()
        app.notify("success", f"Dataset split set to '{new_val}'.")
        return

    if spec["type"] is bool:
        app.shared_settings[key] = not current
        app.save()
        app.notify("success", f"{spec['label']} is now {app.shared_settings[key]}.")
        return

    low  = format_value(spec, spec.get("min", ""))
    high = format_value(spec, spec.get("max", ""))
    cur_fmt = format_value(spec, current)
    if "min" in spec and "max" in spec:
        print(f"Allowed range: {low} to {high}. Current value: {cur_fmt}")
    else:
        print(f"Current value: {cur_fmt}")

    while True:
        raw = input(paint("New value (press Enter to cancel): ", BOLD, MAGENTA))
        if not raw.strip():
            app.notify("info", "No changes made.")
            return
        value, error = parse_value(spec, raw)
        if error:
            say("warning", error)
            continue
        app.shared_settings[key] = value
        text = f"{spec['label']} is now {format_value(spec, value)}."
        if app.save():
            app.notify("success", text)
        else:
            app.notify("warning", text + f" Could not save to {CONFIG_PATH.name}.")
        return


def shared_setting_item(app, key, with_group=False):
    spec = SHARED_SETTINGS_SCHEMA[key]
    label = f"{spec['group']}: {spec['label']}" if with_group else f"Set {spec['label'].lower()}"
    val = app.shared_settings.get(key, spec["default"])
    detail = format_value(spec, val)
    if key == "device" and str(val).upper().startswith("CUDA"):
        cuda_ok, _, gpu_name = get_cuda_status()
        if cuda_ok and gpu_name:
            detail = f"{val} ({gpu_name})"
    return Item(
        label=label,
        action=lambda a, k=key: edit_shared_setting(a, k),
        detail=detail,
    )


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
    # Keep at least the first available model
    models = get_available_models()
    default_m = models[0] if models else "yolov8n.pt"
    app.selected_models = [default_m]
    app.save()
    app.notify("info", f"Reset selection to single model: {default_m}")


def action_select_single_model(app, model_name: str):
    app.selected_models = [model_name]
    app.save()
    app.notify("success", f"Active model set to '{model_name}' exclusively.")


def model_selection_items(app):
    models = get_available_models()
    items = []
    for m in models:
        exists = (MODELS_DIR / m).exists()
        tag = "✔" if m in app.selected_models else ""
        if not exists:
            tag += " [missing on disk]" if tag else "[missing]"
        items.append(
            Item(
                label=m,
                action=lambda a, name=m: action_toggle_model(a, name),
                detail=tag,
            )
        )
    items.append(Item("Select all models", action_select_all_models))
    items.append(Item("Reset to single model (yolov8n.pt)", lambda a: action_select_single_model(a, "yolov8n.pt")))
    return items


def model_selection_body(app):
    models = get_available_models()
    n_sel = len(app.selected_models)
    lines = [
        "Pick a number to toggle  ✔ on/off. You can select one or multiple models.",
    ]
    if n_sel == 1:
        lines.append(f"Active model: {paint(app.selected_models[0], BOLD, GREEN)} (single-model mode)")
    else:
        shown = ", ".join(app.selected_models)
        lines.append(f"Active models ({n_sel}/{len(models)}): {paint(shown, BOLD, GREEN)} (multi-model mode)")

    dev = app.shared_settings.get("device", "CUDA:0")
    cuda_ok, cuda_desc, gpu_name = get_cuda_status()
    if str(dev).upper().startswith("CUDA") and cuda_ok:
        lines.append(f"Target compute device: {paint(dev, BOLD, CYAN)} [{paint(gpu_name, BOLD, YELLOW)}]")
    else:
        lines.append(f"Target compute device: {paint(dev, BOLD, CYAN)} ({cuda_desc})")
    return lines


# ---------------------------------------------------------------------------
# Tracked Metrics Configuration
# ---------------------------------------------------------------------------

def action_toggle_metric(app, key: str):
    spec = METRICS_SCHEMA[key]
    if spec["type"] is bool:
        app.metrics[key] = not app.metrics.get(key, spec["default"])
        app.save()
        status_txt = "ON" if app.metrics[key] else "OFF"
        app.notify("success", f"Metric '{spec['label']}' is now {status_txt}.")
    else:
        edit_metric_setting(app, key)


def edit_metric_setting(app, key: str):
    spec = METRICS_SCHEMA[key]
    current = app.metrics.get(key, spec["default"])
    print()
    print(paint(spec["label"], BOLD, CYAN))
    if "description" in spec:
        print(paint(spec["description"], DIM))
    low = format_value(spec, spec["min"])
    high = format_value(spec, spec["max"])
    print(f"Allowed range: {low} to {high}. Current value: {current}")
    raw = input(paint("New value (press Enter to cancel): ", BOLD, MAGENTA))
    if not raw.strip():
        app.notify("info", "No changes made.")
        return
    value, error = parse_value(spec, raw)
    if error:
        say("warning", error)
        return
    app.metrics[key] = value
    app.save()
    app.notify("success", f"{spec['label']} set to {value}.")


def action_enable_all_metrics(app):
    for k, spec in METRICS_SCHEMA.items():
        if spec["type"] is bool:
            app.metrics[k] = True
    app.save()
    app.notify("success", "Enabled all tracking metrics.")


def action_reset_metrics(app):
    for k, spec in METRICS_SCHEMA.items():
        app.metrics[k] = spec["default"]
    app.save()
    app.notify("info", "Reset metrics configuration to defaults.")


def metrics_items(app):
    items = []
    for key, spec in METRICS_SCHEMA.items():
        val = app.metrics.get(key, spec["default"])
        if spec["type"] is bool:
            detail = "✔ ON" if val else "✘ OFF"
        else:
            detail = str(val)
        items.append(
            Item(
                label=spec["label"],
                action=lambda a, k=key: action_toggle_metric(a, k),
                detail=detail,
            )
        )
    items.append(Item("Enable all tracking metrics", action_enable_all_metrics))
    items.append(Item("Reset metrics to defaults", action_reset_metrics))
    return items


def metrics_body(app):
    active_count = sum(1 for k, s in METRICS_SCHEMA.items() if s["type"] is bool and app.metrics.get(k, s["default"]))
    total_bools = sum(1 for k, s in METRICS_SCHEMA.items() if s["type"] is bool)
    return [
        "Select what metrics to compute and display during benchmarking and evaluation.",
        f"Active metrics: {paint(f'{active_count}/{total_bools} enabled', BOLD, GREEN)} | Iterations: {app.metrics.get('iterations', 20)} | Warmup: {app.metrics.get('warmup_passes', 5)}",
        "Toggle boolean flags by number, or select numeric options to adjust counts.",
    ]


# ---------------------------------------------------------------------------
# Class selection
# ---------------------------------------------------------------------------

def action_toggle_class(app, class_name):
    """Add or remove class_name from selected_classes, then save."""
    if class_name in app.selected_classes:
        app.selected_classes.remove(class_name)
    else:
        app.selected_classes.append(class_name)
    app.save()


def action_clear_class_selection(app):
    if not app.selected_classes:
        app.notify("info", "No classes are selected.")
        return
    app.selected_classes.clear()
    app.save()
    app.notify("success", "All classes deselected.")


def class_items(app):
    """One toggleable item per COCO class, plus a Clear selection item."""
    items = [
        Item(
            label=cls,
            action=lambda a, c=cls: action_toggle_class(a, c),
            detail="✔" if cls in app.selected_classes else "",
        )
        for cls in COCO_CLASSES
    ]
    items.append(Item("Clear selection", action_clear_class_selection))
    return items


def class_body(app):
    if not app.selected_classes:
        return [
            f"All {len(COCO_CLASSES)} COCO classes listed below.",
            "Pick a number to toggle  ✔ on/off. Press 'b' when done.",
            "Selected: none",
        ]
    shown = ", ".join(app.selected_classes[:6])
    if len(app.selected_classes) > 6:
        shown += f" +{len(app.selected_classes) - 6} more"
    n   = len(app.selected_classes)
    per = app.shared_settings["num_images"] // n
    return [
        "Pick a number to toggle  ✔ on/off. Press 'b' when done.",
        f"Selected ({n}): {paint(shown, BOLD, GREEN)}",
        f"~{per:,} images per class  ({app.shared_settings['num_images']:,} total)",
    ]


# ---------------------------------------------------------------------------
# Dataset Actions
# ---------------------------------------------------------------------------

def _load_downloader():
    import importlib.util
    import pathlib
    spec = importlib.util.spec_from_file_location(
        "download_dataset",
        pathlib.Path(__file__).resolve().parent / "download_dataset.py",
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


def _count_all_images():
    """Count .jpg files in all subfolders of dataset/."""
    if not DATASET_DIR.exists():
        return 0
    return sum(1 for _ in DATASET_DIR.rglob("*.jpg"))


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

    import shutil
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


def _load_resizer():
    import importlib.util
    import pathlib
    spec = importlib.util.spec_from_file_location(
        "resize_dataset",
        pathlib.Path(__file__).resolve().parent / "resize_dataset.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def action_resize_dataset(app):
    """Crop and resize dataset to 640x640 using bilinear interpolation with bounded annotations."""
    if not DATASET_DIR.exists():
        app.notify("warning", "The dataset folder does not exist. Please download a dataset first.")
        return

    n_images = _count_all_images()
    if n_images == 0:
        app.notify("warning", "No images found in dataset/. Please download a dataset first.")
        return

    workers = app.shared_settings.get("workers", 16)
    target_size = 640

    print()
    print(paint("Crop & Resize Dataset (640x640 Bilinear)", BOLD, CYAN))
    print()
    print(f"  • Target resolution: {paint(f'{target_size}x{target_size}', BOLD, GREEN)} (square)")
    print(f"  • Resampling:        {paint('Bilinear interpolation (Image.Resampling.BILINEAR)', BOLD, WHITE)}")
    print(f"  • Crop type:         {paint('Central crop (no black bars, no distortion/stretching)', BOLD, WHITE)}")
    print(f"  • Annotations:       {paint('Bounded and clipped to the 640x640 square', BOLD, GREEN)}")
    print(f"  • Images found:      {paint(f'{n_images:,} images', BOLD, YELLOW)} across dataset folders")
    print(f"  • Worker threads:    {paint(str(workers), BOLD, WHITE)}")
    print()

    if not confirm("Proceed with central cropping and resizing dataset to 640x640?"):
        app.notify("info", "Resizing cancelled.")
        return

    try:
        mod = _load_resizer()
        summary = mod.resize_dataset(dataset_dir=DATASET_DIR, target_size=target_size, workers=workers)
        r, s, f = summary["images_resized"], summary["images_skipped"], len(summary["failed"])
        ann_stats = summary.get("annotations", {})
        k = ann_stats.get("kept", 0)
        d = ann_stats.get("dropped", 0)

        msg = f"Resized {r:,} image(s), {s:,} skipped, {k:,} annotations kept ({d:,} dropped)"
        if f:
            msg += f", {f:,} failed"
            app.notify("warning", msg)
        else:
            app.notify("success", msg)
    except KeyboardInterrupt:
        app.notify("warning", "Resizing interrupted.")
    except Exception as exc:  # noqa: BLE001
        app.notify("error", f"Resizing failed: {exc}")


# ---------------------------------------------------------------------------
# Results & Benchmarks Actions
# ---------------------------------------------------------------------------

def get_results_dir() -> Path:
    """Return the base results directory, creating it if needed."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def get_run_timestamp() -> str:
    """Format start time for folder names: YYYY-MM-DD_HH-MM-SS."""
    return datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")


def set_seed(seed: int = 0):
    """Set random seed across Python random, NumPy, and PyTorch for reproducible runs."""
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _list_result_runs():
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


def action_clear_results(app):
    """Delete previous run results from results/ or runs/."""
    run_dirs = _list_result_runs()
    if not run_dirs:
        app.notify("info", "The results folder is already empty.")
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

    import shutil
    removed = 0
    for d in targets:
        try:
            shutil.rmtree(d)
            removed += 1
        except OSError:
            pass
    app.notify("success", f"Deleted {removed} result folder(s).")


def action_test_latency(app):
    """Run real inference benchmark on selected models and save to results/[timeofstarting]/."""
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

    # Resolve device and verify CUDA usage
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

            # Explicit check to verify model parameters are on CUDA
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

            # Warmup passes
            for _ in range(warmup_passes):
                _ = model.predict(dummy_input, **predict_kwargs)

            if cuda_active:
                import torch
                torch.cuda.synchronize()

            # Timed iterations
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
            if track_speed and speed_samples:
                entry["preprocess_ms"] = pre_ms
                entry["inference_ms"] = inf_ms
                entry["postprocess_ms"] = post_ms
            if peak_vram is not None:
                entry["peak_vram_mb"] = peak_vram
                entry["reserved_vram_mb"] = res_vram

            results.append(entry)
        except Exception as exc:  # noqa: BLE001
            say("error", f"Error benchmarking {model_name}: {exc}")

    if not results:
        app.notify("warning", "No models were successfully benchmarked.")
        return

    # Create timestamped result directory
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save machine-readable JSON
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
    (run_dir / "latency_benchmark.json").write_text(
        json.dumps(benchmark_payload, indent=2) + "\n", encoding="utf-8"
    )

    # Save human-readable summary
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
    (run_dir / "latency_benchmark.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    # Print formatted benchmark summary table to console
    print()
    print(paint("═" * 86, BOLD, CYAN))
    print(paint(f" BENCHMARK & METRICS REPORT ({device_desc})", BOLD, WHITE))
    print(paint(f" Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')} | Iterations: {iterations} | Warmup: {warmup_passes}", DIM))
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
    say("info", f"  • {run_dir / 'latency_benchmark.txt'}")
    print()

    input(paint("Press Enter to return to menu... ", BOLD, MAGENTA))
    app.notify("success", f"Benchmark completed — saved to results/{timestamp_str}/")


def _find_all_image_paths() -> list[Path]:
    """Return all .jpg and .jpeg images in dataset/."""
    if not DATASET_DIR.exists():
        return []
    return sorted(set(list(DATASET_DIR.rglob("*.jpg")) + list(DATASET_DIR.rglob("*.jpeg"))))


def _prepare_yolo_labels_and_dataset(
    images: list[Path],
    run_dir: Path,
    dataset_dir: Path,
    class_names: dict[int, str],
) -> Path:
    """Ensure YOLO txt format labels exist for all images and create dataset YAML."""
    name_to_yolo_id = {v: k for k, v in class_names.items()}

    missing_labels = [img for img in images if not img.with_suffix(".txt").exists()]
    if missing_labels:
        say("info", f"Generating YOLO label files for {len(missing_labels):,} image(s)...")
        # Load COCO annotations: check dataset/annotations.json, subfolders, or cached COCO
        coco_sources = []
        if (dataset_dir / "annotations.json").exists():
            coco_sources.append(dataset_dir / "annotations.json")
        for sub in dataset_dir.iterdir():
            if sub.is_dir() and (sub / "annotations.json").exists() and (sub / "annotations.json") not in coco_sources:
                coco_sources.append(sub / "annotations.json")
        if not coco_sources and ANNOTATIONS_CACHE.exists():
            coco_sources.append(ANNOTATIONS_CACHE)

        img_name_to_id: dict[str, int] = {}
        img_id_to_dim: dict[int, tuple[int, int]] = {}
        img_id_to_annos: dict[int, list[dict]] = {}
        coco_id_to_yolo_id: dict[int, int] = {}

        for src in coco_sources:
            try:
                data = json.loads(src.read_text(encoding="utf-8"))
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


def action_run_tests(app):
    """Run model validation and testing on dataset without fine-tuning, saving to results/[timeofstarting]/."""
    target_models = app.selected_models or [get_available_models()[0]]
    raw_device = app.shared_settings.get("device", "CUDA:0")
    seeds = app.shared_settings.get("seeds", [app.shared_settings.get("seed", 0)])
    set_seed(seeds[0])
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

    print()
    print(paint("═" * 98, BOLD, CYAN))
    print(paint(" MODEL EVALUATION & TESTING (WITHOUT FINE-TUNING)", BOLD, WHITE))
    print(paint("═" * 98, BOLD, CYAN))
    print(f"  • Models to test:     {paint(', '.join(target_models), BOLD, YELLOW)} ({len(target_models)} models)")
    print(f"  • Target device:      {paint(device_desc, BOLD, GREEN)}")
    print(f"  • Dataset images:     {paint(f'{len(images):,} images', BOLD, GREEN)} in dataset/")
    print(f"  • Split:              {paint(split, BOLD, WHITE)}")
    print(f"  • Random seed:        {paint(str(seeds[0]), BOLD, GREEN)}")
    print(f"  • Output directory:   {paint(f'results/{timestamp_str}/', BOLD, CYAN)}")
    print(paint("═" * 98, BOLD, CYAN))
    print()

    if not confirm(f"Run test evaluation on {len(target_models)} model(s) with {resolved_device}?"):
        app.notify("info", "Test evaluation cancelled.")
        return

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        app.notify("error", f"Missing dependency: {exc}. Please ensure 'ultralytics' is installed.")
        return

    run_dir.mkdir(parents=True, exist_ok=True)

    # Load first model to retrieve class names
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

    results = []
    for model_name in target_models:
        model_path = MODELS_DIR / model_name
        if not model_path.exists():
            say("warning", f"Model file '{model_name}' not found in {MODELS_DIR}. Skipping.")
            continue

        model_stem = model_name.replace(".pt", "")
        say("info", f"Evaluating {paint(model_name, BOLD, CYAN)} on {len(images):,} images...")

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

        try:
            model = YOLO(str(model_path))

            params_m = None
            size_mb = None
            if track_complexity:
                if hasattr(model, "model") and hasattr(model.model, "parameters"):
                    params_m = sum(p.numel() for p in model.model.parameters()) / 1e6
                if model_path.exists():
                    size_mb = model_path.stat().st_size / (1024 * 1024)

            if track_vram and cuda_active:
                import torch
                torch.cuda.reset_peak_memory_stats()

            # Execute model validation/test
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
                name=model_stem,
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

            entry = {
                "model": model_name,
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
            }
            if params_m is not None:
                entry["params_m"] = params_m
                entry["size_mb"] = size_mb
            if peak_vram is not None:
                entry["peak_vram_mb"] = peak_vram

            results.append(entry)
            say("success", f"[{model_name}] mAP50={map50:.3f} | mAP50-95={map50_95:.3f} | F1={f1:.3f} | Inf={inf_ms:.1f}ms ({fps:.1f} FPS)")
        except KeyboardInterrupt:
            say("warning", f"Testing interrupted on {model_name}.")
            break
        except Exception as exc:  # noqa: BLE001
            say("error", f"Error testing {model_name}: {exc}")

    if not results:
        app.notify("warning", "No models were successfully evaluated.")
        return

    # Save test_results.json
    test_payload = {
        "timestamp_start": start_time.isoformat(),
        "folder": timestamp_str,
        "device_config": raw_device,
        "resolved_device": resolved_device,
        "device_description": device_desc,
        "dataset_images": len(images),
        "split": split,
        "seeds": seeds,
        "seed": seeds[0],
        "shared_settings": app.shared_settings,
        "results": results,
    }
    (run_dir / "test_results.json").write_text(json.dumps(test_payload, indent=2) + "\n", encoding="utf-8")

    # Save test_summary.txt
    summary_lines = [
        "═" * 98,
        f" MODEL EVALUATION & TEST REPORT ({device_desc})",
        f" Evaluated on {len(images):,} images | Split: {split} | Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}",
        "═" * 98,
        f" {'Model':<13} {'Params':<8} {'Prec.':<6} {'mAP50':<8} {'mAP50-95':<10} {'P(box)':<8} {'R(box)':<8} {'F1':<7} {'Inference':<11} {'FPS':<7} {'Peak VRAM':<10}",
        "-" * 98,
    ]
    for r in results:
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
        summary_lines.append("-" * 98)
        summary_lines.append(" Detailed Speed Breakdown (Averages in ms):")
        for r in results:
            summary_lines.append(
                f"   • {r['model']:<14}: Preprocess={r['preprocess_ms']:.2f}ms | Forward={r['inference_ms']:.2f}ms | Postprocess={r['postprocess_ms']:.2f}ms | Total={r['total_ms']:.2f}ms"
            )
    summary_lines.append("═" * 98)
    (run_dir / "test_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    # Print summary table to console
    print()
    print(paint("═" * 98, BOLD, CYAN))
    print(paint(f" MODEL EVALUATION & TEST REPORT ({device_desc})", BOLD, WHITE))
    print(paint(f" Evaluated on {len(images):,} images | Split: {split} | Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}", DIM))
    print(paint("═" * 98, BOLD, CYAN))
    print(f" {'Model':<13} {'Params':<8} {'Prec.':<6} {'mAP50':<8} {'mAP50-95':<10} {'P(box)':<8} {'R(box)':<8} {'F1':<7} {'Inference':<11} {'FPS':<7} {'Peak VRAM':<10}")
    print("-" * 98)
    for r in results:
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
        print("-" * 98)
        print(paint(" Detailed Speed Breakdown (ms):", BOLD, WHITE))
        for r in results:
            print(
                f"   • {paint(r['model'], BOLD, YELLOW)}: Preprocess={r['preprocess_ms']:.2f}ms | Forward={r['inference_ms']:.2f}ms | Postprocess={r['postprocess_ms']:.2f}ms | Total={r['total_ms']:.2f}ms"
            )
    print(paint("═" * 98, BOLD, CYAN))
    print()
    say("success", f"Results saved to: {paint(f'results/{timestamp_str}/', BOLD, GREEN)}")
    say("info", f"  • {run_dir / 'test_results.json'}")
    say("info", f"  • {run_dir / 'test_summary.txt'}")
    if any(r.get("plots_saved") for r in results):
        say("info", f"  • Evaluation plots & PR curves saved in {run_dir}/<model>/")
    print()

    input(paint("Press Enter to return to menu... ", BOLD, MAGENTA))
    app.notify("success", f"Test evaluation completed — saved to results/{timestamp_str}/")


def action_run_training(app):
    """Check dataset, device and trigger training pipeline, saving to results/[timeofstarting]/."""
    target_models = app.selected_models or [get_available_models()[0]]
    raw_device = app.shared_settings.get("device", "CUDA:0")
    seeds = app.shared_settings.get("seeds", [app.shared_settings.get("seed", 0)])
    set_seed(seeds[0])
    start_time = datetime.now().astimezone()
    timestamp_str = start_time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = get_results_dir() / timestamp_str

    print()
    say("info", "Checking training environment and parameters...")
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
    if cuda_active:
        import torch
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        say("info", f"Using GPU device: {paint(gpu_name, BOLD, GREEN)} (device={resolved_device})")
    else:
        say("warning", f"Using CPU compute: {resolved_device}")

    # Check dataset
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


# ---------------------------------------------------------------------------
# Screens Definition
# ---------------------------------------------------------------------------

def main_items(app):
    total = _count_all_images()
    n_models = len(app.selected_models)
    dev = app.shared_settings.get("device", "CUDA:0")
    cuda_ok, _, gpu_name = get_cuda_status()
    dev_str = f"{dev} ({gpu_name})" if (str(dev).upper().startswith("CUDA") and cuda_ok and gpu_name) else str(dev)
    n_runs = len(_list_result_runs())
    active_metrics = sum(1 for k, s in METRICS_SCHEMA.items() if s["type"] is bool and app.metrics.get(k, s["default"]))

    if n_models == 1:
        model_detail = app.selected_models[0]
    else:
        model_detail = f"{n_models} selected"

    return [
        Item("Download dataset",            lambda a: a.go("download_dataset")),
        Item("Resize dataset (640x640)",    action_resize_dataset,
             detail="central crop"),
        Item("Clear dataset",               lambda a: a.go("clear_dataset"),
             detail=f"{total:,} images" if total else "empty"),
        Item("Select model(s)",             lambda a: a.go("select_models"),
             detail=model_detail),
        Item("Configure shared settings",   lambda a: a.go("configure_shared"),
             detail=dev_str),
        Item("Configure tracked metrics",   lambda a: a.go("configure_metrics"),
             detail=f"{active_metrics} active"),
        Item("Test latency (benchmark)",    lambda a: a.go("test_latency"),
             detail=f"on {dev}"),
        Item("Run tests (eval without training)", lambda a: a.go("run_tests"),
             detail=f"{n_models} model(s)"),
        Item("Run training",                lambda a: a.go("run_training")),
        Item("Clear results",               lambda a: a.go("clear_results"),
             detail=f"{n_runs} run(s)" if n_runs else "empty"),
        Item("All settings",                lambda a: a.go("all_settings")),
    ]


def download_body(app):
    n   = len(app.selected_classes)
    total = app.shared_settings["num_images"]
    lines = []
    if n:
        per   = total // n
        shown = ", ".join(app.selected_classes[:4])
        if n > 4:
            shown += f" +{n - 4} more"
        lines += [
            f"Classes ({n}):  {paint(shown, BOLD, GREEN)}",
            f"~{per:,} images per class  ({total:,} total)",
        ]
        lines += [f"  → dataset/{c}/" for c in app.selected_classes]
    else:
        lines += [
            "Classes:  none — images will be downloaded by id into dataset/",
            f"Images:   {paint(format_value(SHARED_SETTINGS_SCHEMA['num_images'], total), BOLD, GREEN)}",
        ]
    return lines


def download_items(app):
    n = len(app.selected_classes)
    return [
        Item("Choose classes",          lambda a: a.go("select_classes"),
             detail=f"{n} selected" if n else "none (all images)"),
        shared_setting_item(app, "num_images"),
        Item("Start download",          action_start_download),
        Item("Resize dataset (640x640)", action_resize_dataset,
             detail="central crop"),
    ]


def run_training_body(app):
    seeds = app.shared_settings.get("seeds", [app.shared_settings.get("seed", 0)])
    seeds_fmt = ", ".join(str(s) for s in seeds)
    if len(seeds) > 1:
        seeds_fmt += f" ({len(seeds)} seeds)"

    lines = [
        "Training pipeline configuration:",
        f"  Compute device: {paint(app.shared_settings['device'], BOLD, GREEN)}",
        f"  Active model(s): {paint(', '.join(app.selected_models), BOLD, CYAN)}",
        f"  Random seed(s): {paint(seeds_fmt, BOLD, GREEN)}",
        f"  Epochs:         {paint(app.shared_settings['epochs'], BOLD, GREEN)}",
        f"  Batch size:     {paint(app.shared_settings['batch_size'], BOLD, GREEN)}",
        f"  Learning rate:  {paint(app.shared_settings['learning_rate'], BOLD, GREEN)}",
    ]
    return lines


def run_tests_body(app):
    n_models = len(app.selected_models)
    dev = app.shared_settings.get("device", "CUDA:0")
    cuda_ok, _, gpu_name = get_cuda_status()
    dev_str = f"{dev} [{gpu_name}]" if (str(dev).upper().startswith("CUDA") and cuda_ok and gpu_name) else str(dev)
    n_images = _count_all_images()
    split = app.shared_settings.get("split", "val")
    conf = app.shared_settings.get("conf", 0.25)
    iou = app.shared_settings.get("iou", 0.70)
    sz = app.shared_settings.get("imgsz", 640)
    half = app.shared_settings.get("half", False)
    plots = app.shared_settings.get("plots", True)
    seeds = app.shared_settings.get("seeds", [app.shared_settings.get("seed", 0)])

    lines = [
        "Evaluate object detection performance directly on dataset without fine-tuning.",
        f"  • Selected model(s) ({n_models}): {paint(', '.join(app.selected_models), BOLD, CYAN)}",
        f"  • Compute target:           {paint(dev_str, BOLD, GREEN)}",
        f"  • Dataset images:           {paint(f'{n_images:,} images', BOLD, YELLOW)} in dataset/",
        f"  • Global parameters:        imgsz={sz}, conf={conf}, iou={iou}, fp16={'Yes' if half else 'No'}, split={split}",
        f"  • Plots generation:         {'Enabled (confusion matrix, PR curves, batches)' if plots else 'Disabled'}",
        f"  • Random seed(s):           {', '.join(str(s) for s in seeds)}",
    ]
    return lines


def run_tests_items(app):
    return [
        Item("Start test evaluation now",   action_run_tests),
        Item("Select models to test",       lambda a: a.go("select_models")),
        Item("Configure global settings",   lambda a: a.go("configure_shared")),
    ]


def all_settings_items(app):
    items = []
    # Shared settings
    for key in SHARED_SETTINGS_SCHEMA:
        items.append(shared_setting_item(app, key, with_group=True))
    # Tracked metrics settings
    for key, spec in METRICS_SCHEMA.items():
        val = app.metrics.get(key, spec["default"])
        detail = ("✔ ON" if val else "✘ OFF") if spec["type"] is bool else str(val)
        items.append(
            Item(
                label=f"Metric: {spec['label']}",
                action=lambda a, k=key: action_toggle_metric(a, k),
                detail=detail,
            )
        )
    return items


SCREENS = {
    "main": Screen(
        title="Main menu",
        color=MAGENTA,
        body=lambda app: ["Choose an action or configure dataset & model parameters."],
        items=main_items,
    ),
    "download_dataset": Screen(
        title="Download dataset",
        body=download_body,
        items=download_items,
    ),
    "select_classes": Screen(
        title="Select COCO classes",
        body=class_body,
        items=class_items,
    ),
    "clear_dataset": Screen(
        title="Clear dataset",
        color=RED,
        body=lambda app: [
            "Removes downloaded images from dataset/ subfolders.",
            "You will choose which class folder to clear.",
        ],
        items=lambda app: [Item("Clear dataset now", action_clear_dataset)],
    ),
    "select_models": Screen(
        title="Select YOLO models",
        body=model_selection_body,
        items=model_selection_items,
    ),
    "configure_shared": Screen(
        title="Configure shared settings",
        body=lambda app: [
            "Global pipeline, model, and hardware settings.",
            "Model parameters (conf, iou, half, max_det, plots, etc.) apply to all models.",
        ],
        items=lambda app: [shared_setting_item(app, key) for key in SHARED_SETTINGS_SCHEMA],
    ),
    "configure_metrics": Screen(
        title="Configure tracked metrics",
        body=metrics_body,
        items=metrics_items,
    ),
    "clear_results": Screen(
        title="Clear results",
        body=lambda app: ["Removes saved runs and benchmark outputs."],
        items=lambda app: [Item("Clear results now", action_clear_results)],
    ),
    "test_latency": Screen(
        title="Test latency & metrics",
        body=lambda app: [
            f"Active models ({len(app.selected_models)}): {', '.join(app.selected_models)}",
            f"Target device: {app.shared_settings.get('device', 'CUDA:0')}",
            f"Benchmark parameters: {app.metrics.get('iterations', 20)} iterations, {app.metrics.get('warmup_passes', 5)} warmup passes",
            "Measures latency, throughput, speed breakdown, model complexity, and GPU VRAM.",
        ],
        items=lambda app: [
            Item("Run benchmark now", action_test_latency),
            Item("Configure tracked metrics", lambda a: a.go("configure_metrics")),
            Item("Select models to test", lambda a: a.go("select_models")),
            Item("Configure device / shared", lambda a: a.go("configure_shared")),
        ],
    ),
    "run_tests": Screen(
        title="Run tests (evaluation)",
        body=run_tests_body,
        items=run_tests_items,
    ),
    "run_training": Screen(
        title="Run training",
        body=run_training_body,
        items=lambda app: [
            Item("Start training session",   action_run_training),
            Item("Select models",            lambda a: a.go("select_models")),
            Item("Change shared settings",   lambda a: a.go("configure_shared")),
        ],
    ),
    "all_settings": Screen(
        title="All settings",
        body=lambda app: ["Every setting in one place. Choose one to modify it."],
        items=all_settings_items,
    ),
}


# ---------------------------------------------------------------------------
# Drawing and input
# ---------------------------------------------------------------------------

def key_hint(letter):
    return paint(f"[{letter}]", BOLD, YELLOW)


def nav_bar(app):
    def entry(letter, name, target):
        if target is None:
            return paint(f"[{letter}] {name} (none)", DIM)
        return f"{key_hint(letter)} {name} {paint('(' + SCREENS[target].title + ')', GRAY)}"

    back_target    = app.back_stack[-1]    if app.back_stack    else None
    forward_target = app.forward_stack[-1] if app.forward_stack else None
    parts = [
        entry("b", "Back",    back_target),
        entry("f", "Forward", forward_target),
        f"{key_hint('h')} Home",
        f"{key_hint('q')} Quit",
    ]
    return "   ".join(parts)


def render(app):
    """Draw the current screen and return its menu items."""
    screen = SCREENS[app.current]
    items  = screen.items(app)
    lines  = screen.body(app)

    clear_screen()
    banner(screen.title, screen.color)
    print()
    if lines:
        for line in lines:
            print(line)
        print()

    width = max((len(item.label) for item in items), default=20)
    for number, item in enumerate(items, start=1):
        detail = paint(item.detail, BOLD, GREEN) if item.detail else ""
        print(f"  {paint(number, BOLD, YELLOW)}. {item.label.ljust(width)}   {detail}".rstrip())
    print()

    if app.message:
        say(*app.message)
        app.message = None
        print()
    print(nav_bar(app))
    return items


def handle_input(app, raw, items):
    choice = raw.strip().lower()
    if choice in ("b", "back"):
        app.back()
    elif choice in ("f", "forward"):
        app.forward()
    elif choice in ("h", "home"):
        app.go("main")
    elif choice in ("q", "quit", "exit"):
        app.running = False
    elif choice.isdecimal() and 1 <= int(choice) <= len(items):
        items[int(choice) - 1].action(app)
    elif choice:
        app.notify("error", f"'{raw.strip()}' is not a valid choice. Use a number, b, f, h or q.")


def main():
    app = App()
    try:
        while app.running:
            items = render(app)
            handle_input(app, input(paint("> ", BOLD, MAGENTA)), items)
    except (KeyboardInterrupt, EOFError):
        print()
    print(paint("Goodbye!", BOLD, CYAN))


if __name__ == "__main__":
    main()