# Source Files

## `run_terminal.py`
Interactive menu-driven terminal for configuring and launching dataset downloads
and model training; run it from the project root with `python src/run_terminal.py`.

**Screens**
| Screen | Purpose |
|---|---|
| Download dataset | Pick COCO classes, set image count, start download |
| Select COCO classes | Toggle any of the 80 COCO val2017 classes on/off; images are split equally across chosen classes |
| Clear dataset | Delete images from a specific class folder (`dataset/<class>/`) or all folders |
| Configure training | Set epochs, batch size, learning rate |
| Run training | Review settings and launch training |
| Clear results | Remove previous run artefacts |
| Test latency | Measure model response time |
| All settings | Edit every setting from one place |

All settings (including selected classes) are persisted to `config/config.json`.

---

## `download_dataset.py`
Downloads COCO val2017 images into per-class subfolders and writes a filtered
`annotations.json` inside each one; importable as a library or run directly.

**CLI usage**
```bash
# Basic — downloads 1 000 images into dataset/ (falls back to config/config.json)
python src/download_dataset.py

# Fixed count, no class filter
python src/download_dataset.py -n 500

# Per-class subfolders — 100 images each into dataset/dog/, dataset/person/, dataset/cat/
python src/download_dataset.py -n 300 --classes dog person cat

# List all 80 valid class names
python src/download_dataset.py --list-classes
```

**Disk layout produced by `--classes`**
```
dataset/
  dog/
    000001.jpg
    annotations.json   ← COCO-format, filtered to dog only
  person/
    000002.jpg
    annotations.json
```

**Key functions**
| Function | Description |
|---|---|
| `download_dataset(count, classes, workers)` | Main entry point; call from other scripts |
| `load_annotations()` | Loads `config/instances_val2017.json` (cached) or fetches it once from the COCO archive |
| `list_categories()` | Returns a sorted list of all 80 COCO class names |

COCO annotations are cached in `config/instances_val2017.json` after the first
fetch and reused on every subsequent run.
