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




### Quantization & Precision Benchmark Table

![Quantization vs Thread Counts Grid](D:\Downloads\house-cat_MIZQ6V1ZJU_threads\house-cat_MIZQ6V1ZJU_quantization_vs_counts_grid.png)

The table below evaluates reconstruction quality, storage requirements, generation runtime, and computational complexity across **10 thread counts** ($100$ to $5{,}000$) and **6 precision tiers** (`float32`, `float16`, `uint8`, `uint6`, `uint5`, `uint4`).

> [!NOTE]
> **Benchmark Hardware & Execution Environment:**
> * **Compute Device:** CPU execution on an **12th Gen Intel® Core™ i5-12400F** (6 Cores / 12 Threads, base 2.50 GHz, max turbo 4.40 GHz).
> * **GPU:** NVIDIA GeForce RTX 4060 8GB present in testbed, but unutilized (the algorithmic placement search is CPU-bound via NumPy / CPython).
> * **Baseline Target:** High-resolution photo `house-cat_MIZQ6V1ZJU.jpg` normalized to a $640 \times 427$ canvas ($273{,}280$ total pixels).
> * **Complexity Metric:** Total FLOPs includes greedy candidate search ($304$ candidate evaluations per thread) plus rasterization and alpha compositing.

| Thread Count ($N$) | Precision Tier | Bits / Value | Bytes / Thread | Raw File Size | Storage (KB) | Generation Time | Total Operations (FLOPs) | MSE (vs Original) | PSNR (vs Original) | MSE (vs `float32`) | PSNR (vs `float32`) |
| ---: | :--- | :---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **100** | `float32` | 32 | 32 B | 3,200 B |   3.20 KB |  1.3 s |    85.9 MFLOPs | 0.126188 | **8.99 dB** | 0.000000 | Baseline |
| **100** | `float16` | 16 | 16 B | 1,600 B |   1.60 KB |  1.3 s |    85.9 MFLOPs | 0.126348 | **8.98 dB** | 0.000262 | 35.81 dB |
| **100** | `uint8` | 8 | 8 B | 800 B |   0.80 KB |  1.3 s |    85.9 MFLOPs | 0.126815 | **8.97 dB** | 0.020090 | 16.97 dB |
| **100** | `uint6` | 6 | 6 B | 600 B |   0.60 KB |  1.3 s |    85.9 MFLOPs | 0.145963 | **8.36 dB** | 0.080366 | 10.95 dB |
| **100** | `uint5` | 5 | 5 B | 500 B |   0.50 KB |  1.3 s |    85.9 MFLOPs | 0.163523 | **7.86 dB** | 0.101662 |  9.93 dB |
| **100** | `uint4` | 4 | 4 B | 400 B |   0.40 KB |  1.3 s |    85.9 MFLOPs | 0.182197 | **7.39 dB** | 0.122007 |  9.14 dB |
| **200** | `float32` | 32 | 32 B | 6,400 B |   6.40 KB |  2.8 s |   171.8 MFLOPs | 0.074610 | **11.27 dB** | 0.000000 | Baseline |
| **200** | `float16` | 16 | 16 B | 3,200 B |   3.20 KB |  2.8 s |   171.8 MFLOPs | 0.074589 | **11.27 dB** | 0.000239 | 36.22 dB |
| **200** | `uint8` | 8 | 8 B | 1,600 B |   1.60 KB |  2.8 s |   171.8 MFLOPs | 0.077791 | **11.09 dB** | 0.016422 | 17.85 dB |
| **200** | `uint6` | 6 | 6 B | 1,200 B |   1.20 KB |  2.8 s |   171.8 MFLOPs | 0.098649 | **10.06 dB** | 0.071930 | 11.43 dB |
| **200** | `uint5` | 5 | 5 B | 1,000 B |   1.00 KB |  2.8 s |   171.8 MFLOPs | 0.114187 | **9.42 dB** | 0.098403 | 10.07 dB |
| **200** | `uint4` | 4 | 4 B | 800 B |   0.80 KB |  2.8 s |   171.8 MFLOPs | 0.141253 | **8.50 dB** | 0.121652 |  9.15 dB |
| **300** | `float32` | 32 | 32 B | 9,600 B |   9.60 KB |  4.2 s |   257.7 MFLOPs | 0.051759 | **12.86 dB** | 0.000000 | Baseline |
| **300** | `float16` | 16 | 16 B | 4,800 B |   4.80 KB |  4.2 s |   257.7 MFLOPs | 0.051823 | **12.85 dB** | 0.000197 | 37.05 dB |
| **300** | `uint8` | 8 | 8 B | 2,400 B |   2.40 KB |  4.2 s |   257.7 MFLOPs | 0.055793 | **12.53 dB** | 0.013464 | 18.71 dB |
| **300** | `uint6` | 6 | 6 B | 1,800 B |   1.80 KB |  4.2 s |   257.7 MFLOPs | 0.076177 | **11.18 dB** | 0.062894 | 12.01 dB |
| **300** | `uint5` | 5 | 5 B | 1,500 B |   1.50 KB |  4.2 s |   257.7 MFLOPs | 0.091078 | **10.41 dB** | 0.084811 | 10.72 dB |
| **300** | `uint4` | 4 | 4 B | 1,200 B |   1.20 KB |  4.2 s |   257.7 MFLOPs | 0.115296 | **9.38 dB** | 0.106833 |  9.71 dB |
| **400** | `float32` | 32 | 32 B | 12,800 B |  12.80 KB |  5.3 s |   343.6 MFLOPs | 0.038241 | **14.17 dB** | 0.000000 | Baseline |
| **400** | `float16` | 16 | 16 B | 6,400 B |   6.40 KB |  5.3 s |   343.6 MFLOPs | 0.038242 | **14.17 dB** | 0.000169 | 37.73 dB |
| **400** | `uint8` | 8 | 8 B | 3,200 B |   3.20 KB |  5.3 s |   343.6 MFLOPs | 0.041760 | **13.79 dB** | 0.011747 | 19.30 dB |
| **400** | `uint6` | 6 | 6 B | 2,400 B |   2.40 KB |  5.3 s |   343.6 MFLOPs | 0.061063 | **12.14 dB** | 0.054057 | 12.67 dB |
| **400** | `uint5` | 5 | 5 B | 2,000 B |   2.00 KB |  5.3 s |   343.6 MFLOPs | 0.077061 | **11.13 dB** | 0.073904 | 11.31 dB |
| **400** | `uint4` | 4 | 4 B | 1,600 B |   1.60 KB |  5.3 s |   343.6 MFLOPs | 0.099863 | **10.01 dB** | 0.095367 | 10.21 dB |
| **500** | `float32` | 32 | 32 B | 16,000 B |  16.00 KB |  6.1 s |   429.5 MFLOPs | 0.029567 | **15.29 dB** | 0.000000 | Baseline |
| **500** | `float16` | 16 | 16 B | 8,000 B |   8.00 KB |  6.1 s |   429.5 MFLOPs | 0.029603 | **15.29 dB** | 0.000150 | 38.25 dB |
| **500** | `uint8` | 8 | 8 B | 4,000 B |   4.00 KB |  6.1 s |   429.5 MFLOPs | 0.033075 | **14.80 dB** | 0.010436 | 19.81 dB |
| **500** | `uint6` | 6 | 6 B | 3,000 B |   3.00 KB |  6.1 s |   429.5 MFLOPs | 0.052317 | **12.81 dB** | 0.048526 | 13.14 dB |
| **500** | `uint5` | 5 | 5 B | 2,500 B |   2.50 KB |  6.1 s |   429.5 MFLOPs | 0.069061 | **11.61 dB** | 0.067988 | 11.68 dB |
| **500** | `uint4` | 4 | 4 B | 2,000 B |   2.00 KB |  6.1 s |   429.5 MFLOPs | 0.091923 | **10.37 dB** | 0.089243 | 10.49 dB |
| **1,000** | `float32` | 32 | 32 B | 32,000 B |  32.00 KB |  8.7 s |   858.9 MFLOPs | 0.012116 | **19.17 dB** | 0.000000 | Baseline |
| **1,000** | `float16` | 16 | 16 B | 16,000 B |  16.00 KB |  8.7 s |   858.9 MFLOPs | 0.012152 | **19.15 dB** | 0.000082 | 40.84 dB |
| **1,000** | `uint8` | 8 | 8 B | 8,000 B |   8.00 KB |  8.7 s |   858.9 MFLOPs | 0.014393 | **18.42 dB** | 0.005463 | 22.63 dB |
| **1,000** | `uint6` | 6 | 6 B | 6,000 B |   6.00 KB |  8.7 s |   858.9 MFLOPs | 0.031144 | **15.07 dB** | 0.029885 | 15.25 dB |
| **1,000** | `uint5` | 5 | 5 B | 5,000 B |   5.00 KB |  8.7 s |   858.9 MFLOPs | 0.046014 | **13.37 dB** | 0.045871 | 13.38 dB |
| **1,000** | `uint4` | 4 | 4 B | 4,000 B |   4.00 KB |  8.7 s |   858.9 MFLOPs | 0.070148 | **11.54 dB** | 0.069662 | 11.57 dB |
| **2,000** | `float32` | 32 | 32 B | 64,000 B |  64.00 KB | 12.5 s |  1.72 GFLOPs | 0.004521 | **23.45 dB** | 0.000000 | Baseline |
| **2,000** | `float16` | 16 | 16 B | 32,000 B |  32.00 KB | 12.5 s |  1.72 GFLOPs | 0.004551 | **23.42 dB** | 0.000038 | 44.16 dB |
| **2,000** | `uint8` | 8 | 8 B | 16,000 B |  16.00 KB | 12.5 s |  1.72 GFLOPs | 0.006096 | **22.15 dB** | 0.002685 | 25.71 dB |
| **2,000** | `uint6` | 6 | 6 B | 12,000 B |  12.00 KB | 12.5 s |  1.72 GFLOPs | 0.019752 | **17.04 dB** | 0.019002 | 17.21 dB |
| **2,000** | `uint5` | 5 | 5 B | 10,000 B |  10.00 KB | 12.5 s |  1.72 GFLOPs | 0.035789 | **14.46 dB** | 0.034984 | 14.56 dB |
| **2,000** | `uint4` | 4 | 4 B | 8,000 B |   8.00 KB | 12.5 s |  1.72 GFLOPs | 0.062663 | **12.03 dB** | 0.061320 | 12.12 dB |
| **3,000** | `float32` | 32 | 32 B | 96,000 B |  96.00 KB | 15.9 s |  2.58 GFLOPs | 0.002538 | **25.96 dB** | 0.000000 | Baseline |
| **3,000** | `float16` | 16 | 16 B | 48,000 B |  48.00 KB | 15.9 s |  2.58 GFLOPs | 0.002556 | **25.92 dB** | 0.000024 | 46.22 dB |
| **3,000** | `uint8` | 8 | 8 B | 24,000 B |  24.00 KB | 15.9 s |  2.58 GFLOPs | 0.003732 | **24.28 dB** | 0.001786 | 27.48 dB |
| **3,000** | `uint6` | 6 | 6 B | 18,000 B |  18.00 KB | 15.9 s |  2.58 GFLOPs | 0.016442 | **17.84 dB** | 0.015744 | 18.03 dB |
| **3,000** | `uint5` | 5 | 5 B | 15,000 B |  15.00 KB | 15.9 s |  2.58 GFLOPs | 0.032172 | **14.93 dB** | 0.031239 | 15.05 dB |
| **3,000** | `uint4` | 4 | 4 B | 12,000 B |  12.00 KB | 15.9 s |  2.58 GFLOPs | 0.058769 | **12.31 dB** | 0.057545 | 12.40 dB |
| **4,000** | `float32` | 32 | 32 B | 128,000 B | 128.00 KB | 19.3 s |  3.44 GFLOPs | 0.001733 | **27.61 dB** | 0.000000 | Baseline |
| **4,000** | `float16` | 16 | 16 B | 64,000 B |  64.00 KB | 19.3 s |  3.44 GFLOPs | 0.001745 | **27.58 dB** | 0.000018 | 47.53 dB |
| **4,000** | `uint8` | 8 | 8 B | 32,000 B |  32.00 KB | 19.3 s |  3.44 GFLOPs | 0.002650 | **25.77 dB** | 0.001302 | 28.85 dB |
| **4,000** | `uint6` | 6 | 6 B | 24,000 B |  24.00 KB | 19.3 s |  3.44 GFLOPs | 0.014407 | **18.41 dB** | 0.013681 | 18.64 dB |
| **4,000** | `uint5` | 5 | 5 B | 20,000 B |  20.00 KB | 19.3 s |  3.44 GFLOPs | 0.029844 | **15.25 dB** | 0.028892 | 15.39 dB |
| **4,000** | `uint4` | 4 | 4 B | 16,000 B |  16.00 KB | 19.3 s |  3.44 GFLOPs | 0.057107 | **12.43 dB** | 0.055942 | 12.52 dB |
| **5,000** | `float32` | 32 | 32 B | 160,000 B | 160.00 KB | 22.9 s |  4.29 GFLOPs | 0.001312 | **28.82 dB** | 0.000000 | Baseline |
| **5,000** | `float16` | 16 | 16 B | 80,000 B |  80.00 KB | 22.9 s |  4.29 GFLOPs | 0.001324 | **28.78 dB** | 0.000015 | 48.35 dB |
| **5,000** | `uint8` | 8 | 8 B | 40,000 B |  40.00 KB | 22.9 s |  4.29 GFLOPs | 0.002086 | **26.81 dB** | 0.001049 | 29.79 dB |
| **5,000** | `uint6` | 6 | 6 B | 30,000 B |  30.00 KB | 22.9 s |  4.29 GFLOPs | 0.013499 | **18.70 dB** | 0.012828 | 18.92 dB |
| **5,000** | `uint5` | 5 | 5 B | 25,000 B |  25.00 KB | 22.9 s |  4.29 GFLOPs | 0.028847 | **15.40 dB** | 0.027981 | 15.53 dB |
| **5,000** | `uint4` | 4 | 4 B | 20,000 B |  20.00 KB | 22.9 s |  4.29 GFLOPs | 0.056463 | **12.48 dB** | 0.055360 | 12.57 dB |
