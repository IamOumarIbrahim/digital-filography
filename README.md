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

---

### Memory Footprint Analysis

The storage size of a filographic image depends entirely on the thread count $N$ ($O(N)$), not the canvas resolution. Each thread contains 8 normalized numbers, so an image occupies $8Nb$ bytes, where $b$ is the number of bytes per stored value: $b = 4$ for 32-bit floating-point numbers (`float32`) and $b = 1$ for 8-bit unsigned integers (`uint8`). We compare this against an uncompressed RGB bitmap of $640 \times 640$ pixels ($409{,}600$ total pixels at 3 bytes per pixel, i.e., $1{,}228{,}800$ bytes).

Sizes are reported in kilobytes (KB) and megabytes (MB), where $1\text{ KB} = 1{,}000$ bytes and $1\text{ MB} = 1{,}000{,}000$ bytes.

#### Table: Storage Footprint Relative to $640 \times 640$ RGB Bitmap Baseline

| Representation | Format | Dimensions | Raw Size (Bytes) | Storage | Size Compared to Baseline |
| --- | --- | --- | --- | --- | --- |
| **Raster Canvas** | Uncompressed RGBA (`uint8`) | $640 \times 640 \times 4$ | $1{,}638{,}400$ | $1.64\text{ MB}$ | **$133.33\%$** ($1.33\times$ larger) |
| **Raster Canvas** | Uncompressed RGB (`uint8`) | $640 \times 640 \times 3$ | $1{,}228{,}800$ | $1.23\text{ MB}$ | **$100.00\%$** ($1.00\times$ Baseline) |
| **Raster Canvas** | Compressed PNG (Lossless) | $640 \times 640$ | $\approx 250{,}000$ | $\approx 250.00\text{ KB}$ | **$\approx 20.35\%$** ($\approx 4.92\times$ smaller) |
| **Raster Canvas** | Compressed JPEG (Lossy, Q80) | $640 \times 640$ | $\approx 60{,}000$ | $\approx 60.00\text{ KB}$ | **$\approx 4.88\%$** ($\approx 20.48\times$ smaller) |
| **Raster Canvas** | Compressed WebP (Lossy, Q80) | $640 \times 640$ | $\approx 35{,}000$ | $\approx 35.00\text{ KB}$ | **$\approx 2.85\%$** ($\approx 35.11\times$ smaller) |
| **Filography ($N = 4{,}000$)** | 32-bit Float (`float32`) | $4{,}000 \times 8$ | $128{,}000$ | $128.00\text{ KB}$ | **$10.42\%$** ($9.60\times$ smaller) |
| **Filography ($N = 4{,}000$)** | 8-bit Integer (`uint8`) | $4{,}000 \times 8$ | $32{,}000$ | $32.00\text{ KB}$ | **$2.60\%$** ($38.40\times$ smaller) |
| **Filography ($N = 12{,}000$)** | 32-bit Float (`float32`) | $12{,}000 \times 8$ | $384{,}000$ | $384.00\text{ KB}$ | **$31.25\%$** ($3.20\times$ smaller) |
| **Filography ($N = 12{,}000$)** | 8-bit Integer (`uint8`) | $12{,}000 \times 8$ | $96{,}000$ | $96.00\text{ KB}$ | **$7.81\%$** ($12.80\times$ smaller) |