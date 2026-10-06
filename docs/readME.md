# Digital Filography

In this work, we study digital filography, a method that represents images using straight threads instead of pixel grids (bitmaps). Specifically, we evaluate the minimum thread count needed for accurate recognition by vision-language models (VLMs), traditional computer vision (CV) models, and human subjects.

Each thread is a vector with 8 elements:

$$T_i = [x_1, y_1, x_2, y_2, r, g, b, a]$$

Where:

* $(x_1, y_1)$ and $(x_2, y_2)$ are the start and end coordinates of the thread.
* $(r, g, b)$ sets the red, green, and blue (RGB) color values.
* $a$ is the alpha (opacity) value.

All values are normalized to the range $[0, 1]$.

An image with $N$ threads is stored as an $N \times 8$ matrix:

$$I = [T_1, T_2, \dots, T_N]$$

Threads are rendered in sequential order from $1$ to $N$. Because recognition relies on dominant outlines rather than exact color mixing, changing thread draw order does not noticeably affect recognition.

---

### Memory Footprint Analysis

The storage size of a filographic image depends entirely on the thread count $N$ ($O(N)$), not the canvas resolution. Each thread contains 8 normalized numbers. We compare this against an uncompressed bitmap of $640 \times 640$ pixels ($409{,}600$ total pixels).

Sizes are reported in kilobytes (KB) and megabytes (MB) using 32-bit floating-point numbers (`float32`) and 8-bit unsigned integers (`uint8`).

#### Table: Storage Footprint Relative to $640 \times 640$ RGB Bitmap Baseline

| Representation | Format | Dimensions | Raw Size (Bytes) | Storage | Size Compared to Baseline |
| --- | --- | --- | --- | --- | --- |
| **Raster Canvas** | Uncompressed RGBA (`uint8`) | $640 \times 640 \times 4$ | $1{,}638{,}400$ | $1.64\text{ MB}$ | **$133.33\%$** ($1.33\times$ larger) |
| **Raster Canvas** | Uncompressed RGB (`uint8`) | $640 \times 640 \times 3$ | $1{,}228{,}800$ | $1.23\text{ MB}$ | **$100.00\%$** ($1.00\times$ Baseline) |
| **Filography ($N = 4{,}000$)** | 32-bit Float (`float32`) | $4{,}000 \times 8$ | $128{,}000$ | $128.00\text{ KB}$ | **$10.42\%$** ($9.60\times$ smaller) |
| **Filography ($N = 4{,}000$)** | 8-bit Integer (`uint8`) | $4{,}000 \times 8$ | $32{,}000$ | $32.00\text{ KB}$ | **$2.60\%$** ($38.40\times$ smaller) |
| **Filography ($N = 12{,}000$)** | 32-bit Float (`float32`) | $12{,}000 \times 8$ | $384{,}000$ | $384.00\text{ KB}$ | **$31.25\%$** ($3.20\times$ smaller) |
| **Filography ($N = 12{,}000$)** | 8-bit Integer (`uint8`) | $12{,}000 \times 8$ | $96{,}000$ | $96.00\text{ KB}$ | **$7.81\%$** ($12.80\times$ smaller) |

---

### Key Observations

1. **Resolution Independence:** Bitmap file size grows with pixel resolution ($O(\text{height} \times \text{width})$). In contrast, filography file size depends only on thread count. It uses the exact same storage whether drawn at $640 \times 640$ or $2560 \times 2560$ pixels.
2. **Compact Byte Storage:** Because all values stay between $0$ and $1$, we can store each value as a standard byte (`uint8`) instead of a 4-byte float (`float32`). This cuts memory use by $4\times$ while keeping coordinate rounding error negligible ($\Delta = 1/255 \approx 0.0039$).