# Confidence Threshold Sweep Report: YOLO Detection Benchmark (0.25 to 0.75)

> **Project**: Filography: A Revisited Technique for Digital Imagery  
> **Authors**: Oumar Ibrahim & Mohamad Khairy  
> **Affiliation**: Department of Computer Engineering, University of Sharjah, Sharjah, UAE  
> **Report Generated**: 2026-10-15 00:28:46  
> **Evaluation Scope**: 5 Models × 11 Confidence Thresholds (0.25 to 0.75, step 0.05, Seed 1) on 5,000 images under FP16  

---

## 1. Executive Summary & Sweep Overview

To rigorously evaluate model sensitivity to detection confidence filtering, this experiment sweeps the confidence threshold from **`conf = 0.25` to `conf = 0.75` in increments of 0.05** across all 5 YOLO nano models (`yolo26n.pt`, `yolo11n.pt`, `yolov8n.pt`, `yolov10n.pt`, `yolo12n.pt`). Each evaluation is conducted on 5,000 COCO 2017 validation images under FP16 half precision on CUDA hardware.

All metrics in Section 2 reflect the **comprehensive grand average across all 11 confidence levels** ($0.25 \le \text{conf} \le 0.75$). Detailed performance per individual confidence threshold is provided in Section 3.

---

## 2. Grand Average Across All Confidence Levels (0.25 to 0.75)

| Model | Params | Size | Prec. | mAP50 (Mean ± Std) | mAP50-95 (Mean ± Std) | Box Precision | Box Recall | F1 Score | Forward Latency | Throughput (FPS) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`yolo12n.pt`** | 2.60M | 5.34MB | FP16 | 0.361 ± 0.068 | 0.281 ± 0.044 | 0.815 ± 0.077 | 0.381 ± 0.081 | 0.509 ± 0.061 | 1.49 ± 0.02 ms | 339.9 ± 5.9 |
| **`yolo11n.pt`** | 2.62M | 5.35MB | FP16 | 0.348 ± 0.068 | 0.270 ± 0.043 | 0.803 ± 0.082 | 0.370 ± 0.081 | 0.496 ± 0.061 | 1.11 ± 0.02 ms | 379.8 ± 14.5 |
| **`yolov10n.pt`** | 2.78M | 5.59MB | FP16 | 0.342 ± 0.068 | 0.267 ± 0.045 | 0.813 ± 0.079 | 0.361 ± 0.081 | 0.490 ± 0.064 | 1.12 ± 0.02 ms | 389.5 ± 15.6 |
| **`yolo26n.pt`** | 2.57M | 5.29MB | FP16 | 0.341 ± 0.073 | 0.269 ± 0.048 | 0.836 ± 0.071 | 0.358 ± 0.084 | 0.491 ± 0.071 | 1.08 ± 0.06 ms | 391.0 ± 10.7 |
| **`yolov8n.pt`** | 3.16M | 6.25MB | FP16 | 0.329 ± 0.065 | 0.255 ± 0.042 | 0.797 ± 0.092 | 0.351 ± 0.079 | 0.476 ± 0.060 | 1.12 ± 0.02 ms | 380.6 ± 10.9 |

---

## 3. Detailed Confidence-by-Confidence Breakdown

The table below details how detection precision, recall, and mAP evolve as the confidence threshold increases from 0.25 to 0.75 (Seed 1).

### 3.1 `yolo26n.pt` Confidence Profile

| Confidence | mAP50 | mAP50-95 | Precision | Recall | F1 Score | Forward (ms) | FPS |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `0.25` | 0.4456 | 0.3348 | 0.6990 | 0.4844 | 0.5722 | 1.23 ms | 367.1 |
| `0.30` | 0.4257 | 0.3234 | 0.7404 | 0.4570 | 0.5652 | 1.07 ms | 395.2 |
| `0.35` | 0.4066 | 0.3119 | 0.7777 | 0.4320 | 0.5555 | 1.05 ms | 397.4 |
| `0.40` | 0.3882 | 0.3007 | 0.8146 | 0.4090 | 0.5446 | 1.12 ms | 382.2 |
| `0.45` | 0.3663 | 0.2871 | 0.8393 | 0.3839 | 0.5268 | 1.04 ms | 388.5 |
| `0.50` | 0.3461 | 0.2742 | 0.8600 | 0.3601 | 0.5076 | 1.05 ms | 396.9 |
| `0.55` | 0.3238 | 0.2596 | 0.8621 | 0.3349 | 0.4824 | 1.09 ms | 380.9 |
| `0.60` | 0.3026 | 0.2456 | 0.8792 | 0.3112 | 0.4597 | 1.08 ms | 398.2 |
| `0.65` | 0.2787 | 0.2292 | 0.8968 | 0.2854 | 0.4331 | 1.01 ms | 403.0 |
| `0.70` | 0.2511 | 0.2093 | 0.9104 | 0.2558 | 0.3994 | 1.08 ms | 391.0 |
| `0.75` | 0.2204 | 0.1870 | 0.9144 | 0.2241 | 0.3599 | 1.06 ms | 400.9 |

### 3.2 `yolo11n.pt` Confidence Profile

| Confidence | mAP50 | mAP50-95 | Precision | Recall | F1 Score | Forward (ms) | FPS |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `0.25` | 0.4436 | 0.3296 | 0.6539 | 0.4919 | 0.5614 | 1.08 ms | 392.7 |
| `0.30` | 0.4265 | 0.3197 | 0.6995 | 0.4665 | 0.5597 | 1.08 ms | 387.6 |
| `0.35` | 0.4099 | 0.3101 | 0.7332 | 0.4433 | 0.5525 | 1.14 ms | 372.1 |
| `0.40` | 0.3912 | 0.2987 | 0.7659 | 0.4184 | 0.5412 | 1.12 ms | 379.2 |
| `0.45` | 0.3731 | 0.2870 | 0.7927 | 0.3944 | 0.5267 | 1.13 ms | 363.0 |
| `0.50` | 0.3513 | 0.2737 | 0.8177 | 0.3696 | 0.5091 | 1.12 ms | 386.8 |
| `0.55` | 0.3330 | 0.2615 | 0.8381 | 0.3474 | 0.4912 | 1.09 ms | 356.1 |
| `0.60` | 0.3109 | 0.2473 | 0.8570 | 0.3228 | 0.4689 | 1.11 ms | 371.4 |
| `0.65` | 0.2906 | 0.2338 | 0.8770 | 0.2993 | 0.4463 | 1.10 ms | 377.8 |
| `0.70` | 0.2666 | 0.2176 | 0.8936 | 0.2735 | 0.4188 | 1.12 ms | 381.7 |
| `0.75` | 0.2355 | 0.1960 | 0.9037 | 0.2408 | 0.3803 | 1.10 ms | 409.1 |

### 3.3 `yolov8n.pt` Confidence Profile

| Confidence | mAP50 | mAP50-95 | Precision | Recall | F1 Score | Forward (ms) | FPS |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `0.25` | 0.4195 | 0.3111 | 0.6323 | 0.4687 | 0.5384 | 1.13 ms | 385.7 |
| `0.30` | 0.4034 | 0.3016 | 0.6775 | 0.4433 | 0.5359 | 1.09 ms | 383.0 |
| `0.35` | 0.3872 | 0.2919 | 0.7159 | 0.4200 | 0.5294 | 1.10 ms | 385.1 |
| `0.40` | 0.3711 | 0.2816 | 0.7556 | 0.3979 | 0.5213 | 1.16 ms | 365.4 |
| `0.45` | 0.3537 | 0.2709 | 0.7882 | 0.3757 | 0.5088 | 1.11 ms | 386.6 |
| `0.50` | 0.3339 | 0.2583 | 0.8131 | 0.3518 | 0.4911 | 1.13 ms | 392.5 |
| `0.55` | 0.3159 | 0.2470 | 0.8386 | 0.3304 | 0.4740 | 1.13 ms | 386.1 |
| `0.60` | 0.2952 | 0.2338 | 0.8617 | 0.3070 | 0.4528 | 1.10 ms | 378.8 |
| `0.65` | 0.2731 | 0.2186 | 0.8805 | 0.2821 | 0.4273 | 1.15 ms | 366.4 |
| `0.70` | 0.2488 | 0.2017 | 0.8948 | 0.2553 | 0.3972 | 1.10 ms | 362.9 |
| `0.75` | 0.2216 | 0.1832 | 0.9048 | 0.2265 | 0.3622 | 1.09 ms | 394.1 |

### 3.4 `yolov10n.pt` Confidence Profile

| Confidence | mAP50 | mAP50-95 | Precision | Recall | F1 Score | Forward (ms) | FPS |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `0.25` | 0.4386 | 0.3280 | 0.6686 | 0.4825 | 0.5605 | 1.15 ms | 361.1 |
| `0.30` | 0.4205 | 0.3174 | 0.7145 | 0.4557 | 0.5565 | 1.15 ms | 362.8 |
| `0.35` | 0.4041 | 0.3078 | 0.7469 | 0.4338 | 0.5489 | 1.11 ms | 377.2 |
| `0.40` | 0.3854 | 0.2959 | 0.7822 | 0.4090 | 0.5371 | 1.09 ms | 390.9 |
| `0.45` | 0.3657 | 0.2833 | 0.8100 | 0.3846 | 0.5216 | 1.10 ms | 395.3 |
| `0.50` | 0.3464 | 0.2709 | 0.8186 | 0.3615 | 0.5015 | 1.10 ms | 404.7 |
| `0.55` | 0.3283 | 0.2593 | 0.8398 | 0.3407 | 0.4847 | 1.10 ms | 403.9 |
| `0.60` | 0.3068 | 0.2456 | 0.8665 | 0.3172 | 0.4644 | 1.10 ms | 402.9 |
| `0.65` | 0.2827 | 0.2293 | 0.8886 | 0.2909 | 0.4383 | 1.13 ms | 394.5 |
| `0.70` | 0.2579 | 0.2122 | 0.8955 | 0.2640 | 0.4078 | 1.13 ms | 396.5 |
| `0.75` | 0.2277 | 0.1905 | 0.9111 | 0.2317 | 0.3695 | 1.13 ms | 394.3 |

### 3.5 `yolo12n.pt` Confidence Profile

| Confidence | mAP50 | mAP50-95 | Precision | Recall | F1 Score | Forward (ms) | FPS |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `0.25` | 0.4564 | 0.3406 | 0.6725 | 0.5004 | 0.5739 | 1.44 ms | 342.2 |
| `0.30` | 0.4392 | 0.3298 | 0.7165 | 0.4751 | 0.5713 | 1.49 ms | 343.4 |
| `0.35` | 0.4218 | 0.3203 | 0.7526 | 0.4522 | 0.5650 | 1.48 ms | 339.7 |
| `0.40` | 0.4030 | 0.3089 | 0.7849 | 0.4283 | 0.5542 | 1.49 ms | 335.7 |
| `0.45` | 0.3850 | 0.2979 | 0.8108 | 0.4057 | 0.5408 | 1.48 ms | 341.9 |
| `0.50` | 0.3660 | 0.2861 | 0.8363 | 0.3833 | 0.5256 | 1.53 ms | 330.6 |
| `0.55` | 0.3464 | 0.2735 | 0.8444 | 0.3606 | 0.5054 | 1.46 ms | 336.5 |
| `0.60` | 0.3248 | 0.2596 | 0.8626 | 0.3363 | 0.4839 | 1.50 ms | 338.2 |
| `0.65` | 0.3019 | 0.2440 | 0.8818 | 0.3106 | 0.4594 | 1.49 ms | 336.8 |
| `0.70` | 0.2758 | 0.2264 | 0.8980 | 0.2826 | 0.4299 | 1.50 ms | 339.7 |
| `0.75` | 0.2466 | 0.2056 | 0.9095 | 0.2516 | 0.3941 | 1.49 ms | 353.8 |

---

## 4. Key Scientific Insights from the Confidence Sweep

1. **Precision vs. Recall Inversion**:
   - As confidence filtering rises from 0.25 to 0.75, Precision monotonically increases while Recall drops.
   - F1-score typically peaks in the 0.25–0.40 range across all nano models.
2. **mAP Sensitivity**:
   - Because mAP is calculated by integrating the precision-recall envelope, higher confidence cutoffs truncate the recall domain, producing lower mAP values at higher `conf`.
3. **Model Resilience Across the Sweep**:
   - `yolo12n.pt` and `yolo26n.pt` consistently maintain the highest area-under-curve performance across the entire confidence spectrum.

---
*Report generated by `src/sweep_confidence.py` for the Filography research project.*
