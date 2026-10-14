#!/usr/bin/env python3
"""Download COCO images into per-class subfolders inside dataset/.

Usage (from the project root):
    python src/download_dataset.py                         # 1 000 images, no class filter
    python src/download_dataset.py -n 300                 # 300 images into dataset/
    python src/download_dataset.py -n 300 --classes dog person cat
                                                          # 100/class: dataset/dog/
                                                          #             dataset/person/
                                                          #             dataset/cat/
    python src/download_dataset.py --list-classes         # print all 80 COCO class names

When --classes is used, each class folder receives a filtered annotations.json.
Images come from COCO 2017 val (5 000 images total). Re-running skips files
already on disk and only fetches what is missing.

Only the Python standard library is needed.
"""

import argparse
import http.client
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

PROJECT_ROOT      = Path(__file__).resolve().parent.parent
CONFIG_PATH       = PROJECT_ROOT / "config" / "config.json"  # written by run_terminal.py
DEFAULT_OUT_DIR   = PROJECT_ROOT / "dataset"

ANNOTATIONS_URL   = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
ANNOTATION_FILE   = "instances_val2017.json"
ANNOTATIONS_CACHE = PROJECT_ROOT / "config" / ANNOTATION_FILE  # cached after first fetch

MAX_IMAGES      = 5_000  # images available in COCO val2017
DEFAULT_COUNT   = 1_000  # fallback when neither -n nor config/config.json is present
DEFAULT_WORKERS = 8      # parallel downloads
TIMEOUT         = 30     # seconds per request
RETRIES         = 3      # attempts before giving up

# ---------------------------------------------------------------------------
# Terminal colors
# ---------------------------------------------------------------------------

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

if os.name == "nt":
    os.system("")  # enable ANSI on Windows

USE_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")

RESET, BOLD = "\033[0m", "\033[1m"
RED, GREEN, YELLOW, CYAN, GRAY = "\033[31m", "\033[32m", "\033[33m", "\033[36m", "\033[90m"

MESSAGE_STYLES = {
    "success": ("✔", GREEN),
    "info":    ("ℹ", CYAN),
    "warning": ("⚠", YELLOW),
    "error":   ("✖", RED),
}


def paint(text, *styles):
    if not USE_COLOR or not styles:
        return str(text)
    return "".join(styles) + str(text) + RESET


def say(kind, text):
    symbol, color = MESSAGE_STYLES[kind]
    print(paint(f"{symbol} ", BOLD, color) + text)


def show_progress(done, total, width=30):
    filled  = int(width * done / total) if total else width
    bar     = paint("█" * filled, GREEN) + paint("░" * (width - filled), GRAY)
    percent = int(100 * done / total) if total else 100
    sys.stdout.write(f"\r{bar} {done:,}/{total:,} ({percent}%)")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------

def fetch(request):
    """Open a request and return (status, headers, body). Retries on network errors."""
    last_error = None
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                return response.status, response.headers, response.read()
        except urllib.error.HTTPError as error:
            if 400 <= error.code < 500:
                raise                      # a missing file will not appear on retry
            last_error = error
        except (urllib.error.URLError, http.client.HTTPException, ConnectionError, TimeoutError) as error:
            last_error = error
        time.sleep(1.5 * (attempt + 1))
    raise last_error


class RemoteFile(io.RawIOBase):
    """A read-only file fetched from a URL piece by piece.

    Lets us open the big COCO annotations zip and read only the one JSON file
    we need, without downloading the whole archive.
    """

    def __init__(self, url):
        super().__init__()
        self.url = url
        self.pos = 0
        status, headers, _ = fetch(self._request(0, 0))
        if status != 206:
            raise RuntimeError("The server does not support partial downloads.")
        self.size = int(headers["Content-Range"].split("/")[-1])

    def _request(self, start, end):
        return urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})

    def readable(self):  return True
    def seekable(self):  return True
    def tell(self):      return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        if   whence == io.SEEK_SET: target = offset
        elif whence == io.SEEK_CUR: target = self.pos + offset
        else:                       target = self.size + offset
        self.pos = max(0, target)
        return self.pos

    def readinto(self, buffer):
        wanted = min(len(buffer), self.size - self.pos)
        if wanted <= 0:
            return 0
        status, _, data = fetch(self._request(self.pos, self.pos + wanted - 1))
        if status != 206:
            raise RuntimeError("The server does not support partial downloads.")
        buffer[:len(data)] = data
        self.pos += len(data)
        return len(data)


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------

def load_annotations():
    """Load COCO val2017 annotations from local cache or remote archive.

    On the first call the annotation file is fetched from the COCO archive and
    saved to config/instances_val2017.json. Every subsequent call reads that
    local cache, skipping the remote download entirely.
    """
    if ANNOTATIONS_CACHE.exists():
        say("info", f"Reading annotations from cache ({ANNOTATIONS_CACHE.name})...")
        return json.loads(ANNOTATIONS_CACHE.read_text(encoding="utf-8"))
    say("info", "Fetching COCO annotations (one-time download, will be cached)...")
    remote = RemoteFile(ANNOTATIONS_URL)
    reader = io.BufferedReader(remote, buffer_size=1 << 20)
    with zipfile.ZipFile(reader) as archive:
        name = next((n for n in archive.namelist() if n.endswith(ANNOTATION_FILE)), None)
        if name is None:
            raise RuntimeError(f"{ANNOTATION_FILE} was not found in the COCO archive.")
        with archive.open(name) as handle:
            data = json.load(handle)
    ANNOTATIONS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    ANNOTATIONS_CACHE.write_text(json.dumps(data), encoding="utf-8")
    say("success", f"Annotations cached to {ANNOTATIONS_CACHE}")
    return data


def list_categories():
    """Return a sorted list of all 80 COCO category names."""
    return sorted(cat["name"] for cat in load_annotations()["categories"])


def _images_for_class(data, class_name, limit):
    """Return up to `limit` images containing `class_name`, sorted by image id."""
    name_to_id = {c["name"]: c["id"] for c in data["categories"]}
    cat_id = name_to_id.get(class_name)
    if cat_id is None:
        raise ValueError(
            f"'{class_name}' is not a COCO category. "
            "Run with --list-classes to see valid names."
        )
    img_ids    = {a["image_id"] for a in data["annotations"] if a["category_id"] == cat_id}
    img_lookup = {img["id"]: img for img in data["images"]}
    return sorted(
        (img_lookup[i] for i in img_ids if i in img_lookup),
        key=lambda x: x["id"],
    )[:limit]


def _class_annotations(data, images, class_name):
    """Return a COCO-format dict filtered to `images` and `class_name` only."""
    image_ids = {img["id"] for img in images}
    cat_id    = next(c["id"] for c in data["categories"] if c["name"] == class_name)
    return {
        "info":        data.get("info", {}),
        "licenses":    data.get("licenses", []),
        "images":      [img for img in data["images"] if img["id"] in image_ids],
        "annotations": [a   for a   in data["annotations"]
                        if a["image_id"] in image_ids and a["category_id"] == cat_id],
        "categories":  [c   for c   in data["categories"] if c["id"] == cat_id],
    }


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------

def download_one(image, out_dir):
    """Download one image. Returns ('downloaded'|'skipped'|'failed', error)."""
    path = out_dir / Path(image["file_name"]).name
    if path.exists() and path.stat().st_size > 0:
        return "skipped", None
    try:
        _, _, body = fetch(urllib.request.Request(image["coco_url"]))
        if not body.startswith(b"\xff\xd8"):
            raise ValueError("the file is not a JPEG image")
        temp = path.with_suffix(".part")
        temp.write_bytes(body)
        temp.replace(path)   # only finished files get the final name
    except (OSError, ValueError, http.client.HTTPException) as error:
        return "failed", str(error)
    return "downloaded", None


def _run_pool(image_dir_pairs, workers):
    """Download (image_info, out_dir) pairs in parallel. Returns summary dict."""
    total   = len(image_dir_pairs)
    summary = {"downloaded": 0, "skipped": 0, "failed": []}
    pool    = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = {pool.submit(download_one, img, d): img for img, d in image_dir_pairs}
        show_progress(0, total)
        for done, future in enumerate(as_completed(futures), start=1):
            status, error = future.result()
            if status == "failed":
                summary["failed"].append((futures[future]["file_name"], error))
            else:
                summary[status] += 1
            show_progress(done, total)
    except KeyboardInterrupt:
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown()
    print()
    return summary


def report(summary, out_dir):
    say("success", f"Downloaded {summary['downloaded']:,} new images to {out_dir}")
    if summary["skipped"]:
        say("info", f"{summary['skipped']:,} images were already there and were skipped")
    failed = summary["failed"]
    if failed:
        say("error", f"{len(failed):,} images failed. Run the script again to retry them.")
        for name, error in failed[:5]:
            print(paint(f"    {name}: {error}", GRAY))
        if len(failed) > 5:
            print(paint(f"    ...and {len(failed) - 5:,} more", GRAY))


def download_dataset(count, classes=None, workers=DEFAULT_WORKERS):
    """Download COCO val2017 images, optionally split into per-class subfolders.

    Parameters
    ----------
    count   : total images to download. When `classes` is given the count is
              divided equally: e.g. 300 images with 3 classes -> 100 per class.
    classes : list of COCO class names. Each class is saved in its own folder
                  dataset/<class>/
              with a matching annotations.json inside it.
              Pass None (or []) to download by image id into dataset/ directly.
    workers : parallel download threads.

    Returns
    -------
    dict  {"downloaded": int, "skipped": int, "failed": [(name, error), ...]}

    Disk layout with classes=["dog", "person"]
    -------------------------------------------
    dataset/
      dog/
        000001.jpg
        000002.jpg
        annotations.json    <- COCO-format, filtered to dog images only
      person/
        000003.jpg
        annotations.json    <- COCO-format, filtered to person images only
    """
    if count > MAX_IMAGES:
        say("warning", f"Capping at {MAX_IMAGES:,} (COCO val2017 limit).")
        count = MAX_IMAGES

    data = load_annotations()

    if classes:
        # ── per-class download ─────────────────────────────────────────────
        per_class = max(1, count // len(classes))
        combined  = {"downloaded": 0, "skipped": 0, "failed": []}

        for class_name in classes:
            say("info", f"[{class_name}] selecting up to {per_class:,} images...")
            try:
                images = _images_for_class(data, class_name, per_class)
            except ValueError as exc:
                say("error", str(exc))
                continue

            out_dir = DEFAULT_OUT_DIR / class_name
            out_dir.mkdir(parents=True, exist_ok=True)

            summary = _run_pool([(img, out_dir) for img in images], workers)
            report(summary, out_dir)

            # Filtered annotations.json inside the class folder
            ann_path = out_dir / "annotations.json"
            ann_path.write_text(
                json.dumps(_class_annotations(data, images, class_name), indent=2),
                encoding="utf-8",
            )
            say("info", f"Annotations saved to {ann_path}")

            for key in ("downloaded", "skipped"):
                combined[key] += summary[key]
            combined["failed"].extend(summary["failed"])

        return combined

    else:
        # ── plain download (no class filter) ──────────────────────────────
        images  = sorted(data["images"], key=lambda x: x["id"])[:count]
        out_dir = DEFAULT_OUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        say("info", f"Downloading {len(images):,} images to {out_dir}")
        summary = _run_pool([(img, out_dir) for img in images], workers)
        report(summary, out_dir)
        ann_path = out_dir / "annotations.json"
        ann_path.write_text(
            json.dumps(_plain_annotations(data, images), indent=2),
            encoding="utf-8",
        )
        say("info", f"Annotations saved to {ann_path}")
        return summary


def _plain_annotations(data, images):
    """Return a COCO-format dict filtered to `images` for unpartitioned downloads."""
    image_ids = {img["id"] for img in images}
    return {
        "info":        data.get("info", {}),
        "licenses":    data.get("licenses", []),
        "images":      [img for img in data["images"] if img["id"] in image_ids],
        "annotations": [a   for a   in data["annotations"] if a["image_id"] in image_ids],
        "categories":  data.get("categories", []),
    }


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def default_count():
    """The num_images value saved by run_terminal.py, or DEFAULT_COUNT."""
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            if "shared" in data and isinstance(data["shared"], dict) and "num_images" in data["shared"]:
                value = data["shared"]["num_images"]
            else:
                value = data.get("num_images")
        else:
            value = None
    except (OSError, ValueError, AttributeError):
        return DEFAULT_COUNT
    return value if isinstance(value, int) and value > 0 else DEFAULT_COUNT


def default_classes():
    """The selected_classes list saved by run_terminal.py, or None."""
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            if "shared" in data and isinstance(data["shared"], dict) and "selected_classes" in data["shared"]:
                classes = data["shared"]["selected_classes"]
            else:
                classes = data.get("selected_classes")
            if isinstance(classes, list) and classes:
                return classes
    except (OSError, ValueError, AttributeError):
        pass
    return None


def default_workers():
    """The workers value saved in config/config.json, or DEFAULT_WORKERS."""
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            if "shared" in data and isinstance(data["shared"], dict) and "workers" in data["shared"]:
                value = data["shared"]["workers"]
            else:
                value = data.get("workers")
            if isinstance(value, int) and value > 0:
                return value
    except (OSError, ValueError, AttributeError):
        pass
    return DEFAULT_WORKERS


def positive_int(text):
    value = int(text.replace(",", "").replace("_", ""))
    if value < 1:
        raise argparse.ArgumentTypeError("must be 1 or more")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Download COCO val2017 images into per-class subfolders inside dataset/.")
    parser.add_argument("-n", "--num-images", type=positive_int,
                        help="total images (split equally across classes if --classes is set)")
    parser.add_argument("--classes", nargs="+", metavar="CLASS",
                        help="COCO class names; each saved to dataset/<class>/")
    parser.add_argument("--workers", type=positive_int, default=None,
                        help=f"parallel downloads (default from config or {DEFAULT_WORKERS})")
    parser.add_argument("--list-classes", action="store_true",
                        help="print all 80 COCO class names and exit")
    args = parser.parse_args(argv)

    if args.list_classes:
        for name in list_categories():
            print(name)
        return 0

    count = args.num_images or default_count()
    chosen_classes = args.classes if args.classes is not None else default_classes()
    chosen_workers = args.workers if args.workers is not None else default_workers()
    try:
        summary = download_dataset(count, classes=chosen_classes, workers=chosen_workers)
    except KeyboardInterrupt:
        print()
        say("warning", "Stopped. Run the script again to continue where it stopped.")
        return 130
    except (RuntimeError, OSError, zipfile.BadZipFile, KeyError, ValueError) as error:
        print()
        say("error", f"The download stopped: {error}")
        return 1
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())