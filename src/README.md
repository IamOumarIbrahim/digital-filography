# Source Files

## `run_terminal.py`
Interactive menu-driven terminal for configuring and launching dataset downloads,
YOLO model selection, and CUDA hardware benchmarking; run it from the project root with `python src/run_terminal.py`.

### Screens
| Screen | Purpose |
|---|---|
| Download dataset | Pick COCO classes, set image count, start download |
| Resize dataset (640x640) | Central-crop and resize dataset to 640x640 via bilinear interpolation with bounded annotations |
| Clear dataset | Delete images from a specific class folder (`dataset/<class>/`) or all folders |
| Select model(s) | Choose one or multiple YOLO models (`yolo26n.pt`, `yolo11n.pt`, `yolov8n.pt`, `yolov10n.pt`, `yolo12n.pt`) |
| Configure shared settings | Manage global pipeline and model settings: compute device (`CUDA:0`), confidence threshold (`conf`), NMS threshold (`iou`), FP16 half-precision (`half`), max detections (`max_det`), evaluation plots (`plots`), COCO JSON export (`save_json`), class-agnostic NMS (`agnostic_nms`), test-time augmentation (`augment`), dataset split (`split`), random seed(s), epochs, batch size, learning rate, image size, workers |
| Configure tracked metrics | Toggle metrics tracking (speed breakdown, model complexity, GPU VRAM, accuracy) and benchmark iterations |
| Test latency (benchmark) | Run real-time inference latency benchmark (avg ms, FPS, min/max, VRAM, speed breakdown) on CUDA GPU or CPU |
| Run tests (evaluation) | Evaluate object detection performance (mAP50, mAP50-95, P, R, F1, speed, plots) directly on dataset without fine-tuning |
| Run training | Review active models, shared settings, and initiate training pipeline |
| Clear results | Remove previous run artefacts (`runs/`, `results/`) |
| All settings | View and edit all shared pipeline and metric settings from a single place |


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
Settings are organized into `selected_models`, `shared` pipeline settings, and `metrics` tracking preferences:

```json
{
  "selected_models": [
    "yolov8n.pt",
    "yolo11n.pt"
  ],
  "shared": {
    "device": "CUDA:0",
    "conf": 0.25,
    "iou": 0.70,
    "half": false,
    "max_det": 300,
    "plots": true,
    "save_json": false,
    "agnostic_nms": false,
    "augment": false,
    "split": "val",
    "seeds": [0],
    "epochs": 10,
    "batch_size": 32,
    "imgsz": 640,
    "learning_rate": 0.001,
    "workers": 8,
    "num_images": 5000,
    "selected_classes": []
  },
  "metrics": {
    "speed_breakdown": true,
    "model_complexity": true,
    "vram_usage": true,
    "detection_accuracy": true,
    "iterations": 20,
    "warmup_passes": 5
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

## `resize_dataset.py`
Crops and resizes dataset images to an exact $640 \times 640$ square using bilinear interpolation, with bounding boxes and polygon segmentations updated and clipped to match.

### Crop & Resample Mechanics
- **Central Crop**: For any image of dimensions $(W, H)$, an aspect-preserving square of side length $S = \min(W, H)$ is extracted from the center at coordinates:
  $$\text{crop\_x} = \frac{W - S}{2}, \quad \text{crop\_y} = \frac{H - S}{2}$$
- **Zero Distortion**: No black bars (letterboxing/padding), no stretching or non-uniform scaling.
- **Resampling**: The cropped square is resized to $640 \times 640$ using PIL Bilinear interpolation (`Image.Resampling.BILINEAR`).
- **Annotation Bounding**:
  - Bounding boxes $[x, y, w, h]$ are translated by $-(\text{crop\_x}, \text{crop\_y})$, scaled by $\frac{640}{S}$, and clamped to $[0, 640]$.
  - Width and height are recalculated; boxes that fall completely outside the crop or end up with dimensions $< 1\text{px}$ are discarded.
  - Segmentation polygons are transformed, scaled, clamped to $[0, 640]$, and updated.
  - Image metadata records in `annotations.json` are updated with `width: 640, height: 640`.
- **Atomic Writes**: Images are written to a `.part` temporary file and renamed atomically to prevent corrupted files if interrupted.

### CLI Usage
```bash
# Resize images in dataset/ to 640x640 using 16 worker threads
python src/resize_dataset.py

# Custom target size or custom thread pool
python src/resize_dataset.py --size 640 --workers 8

# Target an explicit dataset folder
python src/resize_dataset.py --dataset-dir dataset/
```

### Terminal Integration
Can be triggered directly from the interactive terminal (`python src/run_terminal.py`):
- From the **Main Menu**: choose `Resize dataset (640x640)`
- From the **Download dataset** screen: choose `Resize dataset (640x640)`

---

## Results Directory (`results/<timeofstarting>/`)
Whenever you execute a benchmark, test evaluation, or training run, results are automatically organized into:
```
results/
  YYYY-MM-DD_HH-MM-SS/
    latency_benchmark.json    ← Machine-readable benchmark metrics
    latency_benchmark.txt     ← Formatted benchmark summary table
    test_results.json         ← Machine-readable model test metrics (mAP50, mAP50-95, P, R, F1, speed)
    test_summary.txt          ← Formatted evaluation comparison table
    test_data.yaml            ← Generated dataset YAML configuration for testing
    val_images.txt            ← Manifest of dataset images evaluated
    <model_name>/             ← Evaluation plots (confusion matrices, PR curves, batch predictions)
    training_config.json      ← Full configuration snapshot for training runs
    train/                    ← Model checkpoints, curves, and confusion matrices
```
Previous runs can be selectively viewed and deleted through the **Clear results** menu in `run_terminal.py`.


