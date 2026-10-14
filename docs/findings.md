# Benchmark Findings: YOLO Nano Confidence Sweep

### Experimental Context
We benchmark five YOLO nano architectures (`yolo12n.pt`, `yolo11n.pt`, `yolov10n.pt`, `yolo26n.pt`, `yolov8n.pt`) across 11 confidence thresholds ($0.25 \le \text{conf} \le 0.75$, step $0.05$) on 5,000 uncropped COCO 2017 validation images using FP16 half-precision on an NVIDIA GeForce RTX 4060 (Seed 1, batch size 32).

---

## 1. Grand Average Across All Confidence Levels (0.25 to 0.75)

| Model | Params | Size | Prec. | mAP50 (Mean ± Std) | mAP50-95 (Mean ± Std) | Box Precision | Box Recall | F1 Score | Forward Latency | Throughput (FPS) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`yolo12n.pt`** | 2.60M | 5.34MB | FP16 | 0.361 ± 0.068 | 0.281 ± 0.044 | 0.815 ± 0.077 | 0.381 ± 0.081 | 0.509 ± 0.061 | 1.49 ± 0.02 ms | 339.9 ± 5.9 |
| **`yolo11n.pt`** | 2.62M | 5.35MB | FP16 | 0.348 ± 0.068 | 0.270 ± 0.043 | 0.803 ± 0.082 | 0.370 ± 0.081 | 0.496 ± 0.061 | 1.11 ± 0.02 ms | 379.8 ± 14.5 |
| **`yolov10n.pt`** | 2.78M | 5.59MB | FP16 | 0.342 ± 0.068 | 0.267 ± 0.045 | 0.813 ± 0.079 | 0.361 ± 0.081 | 0.490 ± 0.064 | 1.12 ± 0.02 ms | 389.5 ± 15.6 |
| **`yolo26n.pt`** | 2.57M | 5.29MB | FP16 | 0.341 ± 0.073 | 0.269 ± 0.048 | 0.836 ± 0.071 | 0.358 ± 0.084 | 0.491 ± 0.071 | 1.08 ± 0.06 ms | 391.0 ± 10.7 |
| **`yolov8n.pt`** | 3.16M | 6.25MB | FP16 | 0.329 ± 0.065 | 0.255 ± 0.042 | 0.797 ± 0.092 | 0.351 ± 0.079 | 0.476 ± 0.060 | 1.12 ± 0.02 ms | 380.6 ± 10.9 |

---

## 2. Summary
**Summary**: An equal-weighted average of mAP, Precision, and Recall establishes **`yolo12n.pt`** as the best model across all confidence thresholds (composite score: **0.459**) and **`conf = 0.25`** as the best confidence threshold across all models (composite score: **0.480**, or **`conf = 0.30`** at **0.513** when combining $\text{mAP}_{50}$ and $\text{mAP}_{50\text{–}95}$ into one term).
