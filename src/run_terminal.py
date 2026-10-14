#!/usr/bin/env python3
"""Menu based terminal for the training project.

Start it from the project root:
    python src/run_terminal.py

Keys that work on every screen:
    1, 2, 3 ...  choose an option
    b            go back (like a browser)
    f            go forward
    h            go to the main menu
    q            quit

Settings are saved to config/config.json, so your other
scripts (download_dataset.py and so on) can read them.
"""

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT      = Path(__file__).resolve().parent.parent
CONFIG_PATH       = PROJECT_ROOT / "config" / "config.json"
DATASET_DIR       = PROJECT_ROOT / "dataset"
ANNOTATIONS_CACHE = PROJECT_ROOT / "config" / "instances_val2017.json"

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
# Settings
# ---------------------------------------------------------------------------
# To add a new numeric setting, add one entry here. It will appear on its
# group screen and on the "All settings" screen automatically.

SETTINGS = {
    "num_images": {
        "group":   "Dataset",
        "label":   "Number of images",
        "type":    int,
        "default": 1000,
        "min":     1,
        "max":     5_000,   # COCO val2017 has 5 000 images; matches MAX_IMAGES in download_dataset.py
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
        "max":     64,
    },
    "learning_rate": {
        "group":   "Training",
        "label":   "Learning rate",
        "type":    float,
        "default": 0.001,
        "min":     0.000001,
        "max":     1.0,
    },
}


def keys_in_group(group):
    return [key for key, spec in SETTINGS.items() if spec["group"] == group]


def format_value(spec, value):
    if spec["type"] is int:
        return f"{value:,}"
    return f"{value:.10f}".rstrip("0").rstrip(".")


def parse_value(spec, text):
    """Turn text into a valid value. Returns (value, error_message)."""
    text = text.strip().replace("_", "")
    if spec["type"] is int:
        text = text.replace(",", "")
    try:
        value = spec["type"](text)
    except ValueError:
        kind = "a whole number" if spec["type"] is int else "a number"
        return None, f"Please enter {kind}."
    if not spec["min"] <= value <= spec["max"]:
        low  = format_value(spec, spec["min"])
        high = format_value(spec, spec["max"])
        return None, f"The value must be between {low} and {high}."
    return value, None


def load_settings():
    """Load numeric settings + selected_classes from config/config.json."""
    values = {key: spec["default"] for key, spec in SETTINGS.items()}
    values["selected_classes"] = []
    try:
        saved = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return values
    if isinstance(saved, dict):
        for key, spec in SETTINGS.items():
            if key in saved:
                value, error = parse_value(spec, str(saved[key]))
                if error is None:
                    values[key] = value
        raw = saved.get("selected_classes", [])
        if isinstance(raw, list):
            values["selected_classes"] = [c for c in raw
                                          if isinstance(c, str) and c in COCO_CLASSES]
    return values


def save_settings(values):
    """Write settings dict to config/config.json. Returns True on success."""
    try:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
    except OSError:
        return False
    return True


# ---------------------------------------------------------------------------
# App state and navigation
# ---------------------------------------------------------------------------

class App:
    """Holds the settings and the back/forward history."""

    def __init__(self):
        loaded                = load_settings()
        self.settings         = {k: loaded[k] for k in SETTINGS}
        self.selected_classes = list(loaded["selected_classes"])
        self.current          = "main"
        self.back_stack       = []
        self.forward_stack    = []
        self.message          = None
        self.running          = True

    def save(self):
        """Persist numeric settings + selected_classes to config/config.json."""
        full = dict(self.settings)
        full["selected_classes"] = self.selected_classes
        return save_settings(full)

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
# Numeric settings editing
# ---------------------------------------------------------------------------

def setting_item(app, key, with_group=False):
    spec = SETTINGS[key]
    label = f"{spec['group']}: {spec['label']}" if with_group else f"Set {spec['label'].lower()}"
    return Item(
        label=label,
        action=lambda a, k=key: edit_setting(a, k),
        detail=format_value(spec, app.settings[key]),
    )


def edit_setting(app, key):
    spec    = SETTINGS[key]
    low     = format_value(spec, spec["min"])
    high    = format_value(spec, spec["max"])
    current = format_value(spec, app.settings[key])
    print()
    print(paint(spec["label"], BOLD, CYAN))
    print(f"Allowed range: {low} to {high}. Current value: {current}")
    while True:
        raw = input(paint("New value (press Enter to cancel): ", BOLD, MAGENTA))
        if not raw.strip():
            app.notify("info", "No changes made.")
            return
        value, error = parse_value(spec, raw)
        if error:
            say("warning", error)
            continue
        app.settings[key] = value
        text = f"{spec['label']} is now {format_value(spec, value)}."
        if app.save():
            app.notify("success", text)
        else:
            app.notify("warning", text + f" Could not save to {CONFIG_PATH.name}.")
        return


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
    per = app.settings["num_images"] // n
    return [
        "Pick a number to toggle  ✔ on/off. Press 'b' when done.",
        f"Selected ({n}): {paint(shown, BOLD, GREEN)}",
        f"~{per:,} images per class  ({app.settings['num_images']:,} total)",
    ]


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def _load_downloader():
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "download_dataset",
        pathlib.Path(__file__).resolve().parent / "download_dataset.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def action_start_download(app):
    count   = app.settings["num_images"]
    classes = app.selected_classes or None
    print()

    if classes:
        n   = len(classes)
        per = count // n
        say("info", f"Downloading ~{per:,} images × {n} class(es) "
                    f"→ {', '.join('dataset/' + c for c in classes)}")
    else:
        say("info", f"Downloading {count:,} images → dataset/ (no class filter)")

    try:
        mod     = _load_downloader()
        summary = mod.download_dataset(count, classes=classes)
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
    """Delete images from a chosen class folder (or everything)."""
    if not DATASET_DIR.exists():
        app.notify("info", "The dataset folder does not exist.")
        return

    subdirs = [d for d in sorted(DATASET_DIR.iterdir())
               if d.is_dir() and any(d.rglob("*.jpg"))]
    all_images = list(DATASET_DIR.rglob("*.jpg"))

    if not all_images:
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
    desc   = f"all images in dataset/{target.name}/" if target else "all dataset images"

    if not confirm(f"Delete {len(images):,} file(s) — {desc}? This cannot be undone."):
        app.notify("info", "Cancelled. Nothing was deleted.")
        return

    removed = 0
    for f in images:
        try:
            f.unlink()
            removed += 1
        except OSError:
            pass

    # Remove empty subdirectories
    roots = [target] if target else [DATASET_DIR]
    for root in roots:
        for d in sorted(root.rglob("*"), reverse=True):
            if d.is_dir():
                try:
                    d.rmdir()
                except OSError:
                    pass

    msg = f"{removed:,} image(s) deleted."

    if ANNOTATIONS_CACHE.exists() and confirm(
            "Also delete the cached annotations (instances_val2017.json)?"):
        try:
            ANNOTATIONS_CACHE.unlink()
            msg += " Annotation cache cleared."
        except OSError:
            msg += " Could not delete the annotation cache."

    app.notify("success", msg)


def action_clear_results(app):
    if not confirm("Delete all results? This cannot be undone."):
        app.notify("info", "Cancelled. Nothing was deleted.")
        return
    # TODO: delete your results files here.
    app.notify("warning", "Clearing results is not connected yet. Nothing was deleted.")


def action_test_latency(app):
    # TODO: run your latency test here.
    app.notify("warning", "The latency test is not connected yet.")


def action_run_training(app):
    # TODO: start your training here (read app.settings for the values).
    app.notify("warning", "Training is not connected yet.")


# ---------------------------------------------------------------------------
# Screens
# ---------------------------------------------------------------------------

def value_of(app, key):
    return format_value(SETTINGS[key], app.settings[key])


def main_items(app):
    total = _count_all_images()
    return [
        Item("Download dataset",   lambda a: a.go("download_dataset")),
        Item("Clear dataset",      lambda a: a.go("clear_dataset"),
             detail=f"{total:,} images" if total else "empty"),
        Item("Configure training", lambda a: a.go("configure_training")),
        Item("Clear results",      lambda a: a.go("clear_results")),
        Item("Test latency",       lambda a: a.go("test_latency")),
        Item("Run training",       lambda a: a.go("run_training")),
        Item("All settings",       lambda a: a.go("all_settings")),
    ]


def download_body(app):
    n   = len(app.selected_classes)
    total = app.settings["num_images"]
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
            f"Images:   {paint(value_of(app, 'num_images'), BOLD, GREEN)}",
        ]
    return lines


def download_items(app):
    n = len(app.selected_classes)
    return [
        Item("Choose classes",   lambda a: a.go("select_classes"),
             detail=f"{n} selected" if n else "none (all images)"),
        setting_item(app, "num_images"),
        Item("Start download",   action_start_download),
    ]


def run_training_body(app):
    lines = ["Training will use these settings:"]
    for key, spec in SETTINGS.items():
        lines.append(f"  {spec['label']}: {paint(value_of(app, key), BOLD, GREEN)}")
    return lines


SCREENS = {
    "main": Screen(
        title="Main menu",
        color=MAGENTA,
        body=lambda app: ["Choose what you want to do."],
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
    "configure_training": Screen(
        title="Configure training",
        body=lambda app: ["Settings used when you run training."],
        items=lambda app: [setting_item(app, key) for key in keys_in_group("Training")],
    ),
    "clear_results": Screen(
        title="Clear results",
        body=lambda app: ["Removes the results of previous runs."],
        items=lambda app: [Item("Clear results now", action_clear_results)],
    ),
    "test_latency": Screen(
        title="Test latency",
        body=lambda app: ["Measures how fast the model responds."],
        items=lambda app: [Item("Run latency test", action_test_latency)],
    ),
    "run_training": Screen(
        title="Run training",
        body=run_training_body,
        items=lambda app: [
            Item("Start training",           action_run_training),
            Item("Change training settings", lambda a: a.go("configure_training")),
        ],
    ),
    "all_settings": Screen(
        title="All settings",
        body=lambda app: ["Every setting in one place. Choose one to change it."],
        items=lambda app: [setting_item(app, key, with_group=True) for key in SETTINGS],
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

    width = max(len(item.label) for item in items)
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