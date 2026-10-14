# Source Files

## `run_terminal.py`
Interactive menu-driven terminal for configuring and launching dataset downloads,
YOLO model selection, and CUDA hardware benchmarking; run it from the project root with `python src/run_terminal.py`.

### Screens
| Screen | Purpose |
|---|---|
| Download dataset | Pick COCO classes, set image count, start download |
| Clear dataset | Delete images from a specific class folder (`dataset/<class>/`) or all folders |
| Select model(s) | Choose one or multiple YOLO models (`yolo26n.pt`, `yolo11n.pt`, `yolov8n.pt`, `yolov10n.pt`, `yolo12n.pt`) |
| Configure shared settings | Manage shared pipeline settings: compute device (`CUDA:0`), random seed(s), epochs, batch size, learning rate, image size, workers |
| Configure model settings | Edit model-specific parameters (`conf`, `iou`, `imgsz`, `half` FP16, `max_det`, overrides) per model |
| Test latency (benchmark) | Run real-time inference latency benchmark (avg ms, FPS, min/max) on CUDA GPU or CPU |
| Run training | Review active models, shared settings, and initiate training pipeline |
| Clear results | Remove previous run artefacts (`runs/`, `results/`) |
| All settings | View and edit all shared and model-specific settings from a single place |

---

## Models Catalogue (`models/`)
Pretrained weights located in `models/`:
- `yolo26n.pt`
- `yolo11n.pt`
- `yolov8n.pt`
- `yolov10n.pt`
- `yolo12n.pt`

Users can select a single model or toggle multiple models simultaneously from the terminal.

---

## Configuration (`config/config.json`)
Settings are clearly separated into `selected_models`, `shared` pipeline settings, and `model_specific` overrides:

```json
{
  "selected_models": [
    "yolov8n.pt",
    "yolo11n.pt"
  ],
  "shared": {
    "device": "CUDA:0",
    "seeds": [0],
    "epochs": 10,
    "batch_size": 32,
    "imgsz": 640,
    "learning_rate": 0.001,
    "workers": 8,
    "num_images": 5000,
    "selected_classes": []
  },
  "model_specific": {
    "yolo26n.pt": {
      "conf": 0.25,
      "iou": 0.70,
      "imgsz": 640,
      "half": false,
      "max_det": 300,
      "batch_size": 32,
      "epochs": 10,
      "seeds": [0]
    },
    "yolo11n.pt": {
      "conf": 0.25,
      "iou": 0.70,
      "imgsz": 640,
      "half": false,
      "max_det": 300,
      "batch_size": 32,
      "epochs": 10,
      "seeds": [0]
    },
    "yolov8n.pt": {
      "conf": 0.25,
      "iou": 0.70,
      "imgsz": 640,
      "half": false,
      "max_det": 300,
      "batch_size": 32,
      "epochs": 10,
      "seeds": [0]
    },
    "yolov10n.pt": {
      "conf": 0.25,
      "iou": 0.70,
      "imgsz": 640,
      "half": false,
      "max_det": 300,
      "batch_size": 32,
      "epochs": 10,
      "seeds": [0]
    },
    "yolo12n.pt": {
      "conf": 0.25,
      "iou": 0.70,
      "imgsz": 640,
      "half": false,
      "max_det": 300,
      "batch_size": 32,
      "epochs": 10,
      "seeds": [0]
    }
  }
}
```

### CUDA:0 Hardware Handling
When `CUDA:0` is configured:
1. `resolve_device()` checks `torch.cuda.is_available()`.
2. Sets `torch.cuda.set_device(0)`.
3. Normalizes device string to lowercase `cuda:0` required by PyTorch and Ultralytics.
4. Verifies model tensor placement directly on the GPU.

---

## `download_dataset.py`
Downloads COCO val2017 images into per-class subfolders and writes a filtered
`annotations.json` inside each one; reads defaults (`num_images`, `selected_classes`, and `workers`) from `config/config.json`.

**CLI usage**
```bash
# Basic — downloads images into dataset/ (reads count and classes from config/config.json)
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

---

## Results Directory (`results/<timeofstarting>/`)
Whenever you execute a benchmark, latency test, or training run, results are automatically organized into:
```
results/
  YYYY-MM-DD_HH-MM-SS/
    latency_benchmark.json    ← Machine-readable benchmark metrics
    latency_benchmark.txt     ← Formatted benchmark summary table
    training_config.json      ← Full configuration snapshot for training runs
    train/                    ← Model checkpoints, curves, and confusion matrices
```
Previous runs can be selectively viewed and deleted through the **Clear results** menu in `run_terminal.py`.

