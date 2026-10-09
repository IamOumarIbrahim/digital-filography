# Digital Filography

In this work, we study digital filography, a method that represents images using straight threads instead of pixel grids (bitmaps). Specifically, we evaluate the minimum thread count needed for accurate recognition by vision-language models (VLMs), traditional computer vision (CV) models, and human subjects.

Each thread is a vector with 8 elements:

$$T_i = [x_1, y_1, x_2, y_2, r, g, b, a]$$

Where:

* $(x_1, y_1)$ and $(x_2, y_2)$ are the start and end coordinates of the thread.
* $(r, g, b)$ sets the red, green, and blue (RGB) color values.
* $a$ is the alpha (opacity) value.

All values are normalized to the range $[0, 1]$.

An image with $N$ threads is stored as an $N \times 8$ matrix, with one thread per row:

$$I = \begin{bmatrix} T_1 \\ T_2 \\ \vdots \\ T_N \end{bmatrix} \in [0, 1]^{N \times 8}$$

Threads are rendered in sequential order from $1$ to $N$. Because recognition relies on dominant outlines rather than exact color mixing, changing thread draw order does not noticeably affect recognition.

## Phase 1: Recognition

Phase 1 asks one general question: **how many threads does an image need before object detectors still recognise what is in it, and how much does the way the threads are stored change the answer?**

It is a general recognition test. We are not deploying anything specific, so nothing is tuned to one class (for example people or vehicles). Phase 1 covers object detectors only. The VLM/classifier tests, the human study and a thread-based detector come in later phases.

### Test set: 500 images, no class quota

* **500 images in total**, taken from COCO 2017.
* **No per-class quota.** Images are drawn at random from all 80 COCO classes, so the class mix follows what real photos contain (some classes, such as person, are common and others are rare).
* **One safeguard:** every class that exists in the candidate pool appears in at least one image, so no class disappears by chance. This is a floor, not a quota.
* The set varies naturally in object class, number of objects per image, scale, occlusion, viewpoint, background complexity, lighting, image quality and original resolution.
* `dataset/manifest.json` and `dataset/manifest.csv` list every image. `dataset/class_summary.csv` shows how many images and objects each class received.

### How recognition is measured

Each detector's own detections on the original photo are its baseline ("model-relative"). A thread reconstruction recovers an object when the same detector finds the same class with IoU of at least 0.50. We report recovery, precision and F1 for thread counts from 250 to 10,000 (step 250), and the smallest thread count that reaches 50% recovery. Default detectors: YOLOv8n, YOLOv10n, YOLO11n, YOLO12n and YOLO26n.

### Storage formats compared

We compare two formats, each at three precisions:

* **Uniform:** all 8 values use FP16, UINT8, or packed UINT6.
* **Adjusted:** the coordinates use FP16, UINT8, or packed UINT6. Color uses RGBA4444, which is 4 bits per channel.

All six are run packed, meaning the threads are stored back to back with no padding bits. UINT6 values are bit-packed (four 6-bit values fill exactly three bytes), and every thread takes a whole number of bytes.

| Format | Coordinates $x_1, y_1, x_2, y_2$ | Color and alpha $r, g, b, a$ | Bytes per thread | Size at $N = 10{,}000$ |
|---|---|---|---:|---:|
| Uniform FP16 | 4 × 16 bit | 4 × 16 bit | 16 | 160 KB |
| Uniform UINT8 | 4 × 8 bit | 4 × 8 bit | 8 | 80 KB |
| Uniform UINT6 | 4 × 6 bit | 4 × 6 bit | 6 | 60 KB |
| Adjusted FP16 | 4 × 16 bit | RGBA4444 (16 bit) | 10 | 100 KB |
| Adjusted UINT8 | 4 × 8 bit | RGBA4444 (16 bit) | 6 | 60 KB |
| Adjusted UINT6 | 4 × 6 bit | RGBA4444 (16 bit) | 5 | 50 KB |

The Adjusted format tests whether color needs full precision when the coordinates keep theirs. Uniform UINT6 and Adjusted UINT8 both cost 6 bytes per thread, so they can be compared at the same storage budget.

In the benchmark every thread matrix is packed into real bytes and decoded again before it is drawn, so the storage numbers are the size of the packed bytes, not an estimate. The code for this is in `src/packing.py`.

### Run it (PowerShell)

```powershell
# 1. Download the 500-image test set (needs: pip install fiftyone pandas pillow)
python src\download_dataset.py

# 2. Run the benchmark: places the threads, packs them in all 6 formats,
#    runs the detectors, writes the CSV files to output\ and the figures to graphs\
python src\eval_dataset.py

# Optional: check the six packed formats on their own (fast, no dataset needed)
python src\packing.py
```

The benchmark saves a checkpoint (`output\checkpoint_rows.jsonl`) and resumes from it if it is stopped. To start again from scratch, for example after changing the formats, models or thread counts, clear the old data first:

```powershell
Remove-Item -Recurse -Force .\dataset, .\output, .\graphs -ErrorAction SilentlyContinue
```

### Earlier exploration: quantization vs. thread counts

![Quantization vs Thread Counts Grid](images/quantization_vs_counts_grid.png)

This image grid was made before the packed formats above. It compares reconstruction quality, storage, generation runtime and computational complexity across **10 thread counts** ($100$ to $5{,}000$) and **6 per-value precision tiers** (`float32`, `float16`, `uint8`, `uint6`, `uint5`, `uint4`).
