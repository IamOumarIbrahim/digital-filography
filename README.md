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

![Quantization vs Thread Counts Grid](images/quantization_vs_counts_grid.png)

The table below evaluates reconstruction quality, storage requirements, generation runtime, and computational complexity across **10 thread counts** ($100$ to $5{,}000$) and **6 precision tiers** (`float32`, `float16`, `uint8`, `uint6`, `uint5`, `uint4`).

> [!NOTE]
> **Benchmark Hardware & Execution Environment:**
> * **Compute Device:** CPU execution on an **12th Gen Intel® Core™ i5-12400F** (6 Cores / 12 Threads, base 2.50 GHz, max turbo 4.40 GHz).
> * **GPU:** NVIDIA GeForce RTX 4060 8GB present in testbed, but unutilized (the algorithmic placement search is CPU-bound via NumPy / CPython).
> * **Baseline Target:** High-resolution photo `house-cat_MIZQ6V1ZJU.jpg` normalized to a $640 \times 427$ canvas ($273{,}280$ total pixels).
> * **Complexity Metric:** Total FLOPs includes greedy candidate search ($304$ candidate evaluations per thread) plus rasterization and alpha compositing.
> * **Average Score:** Min-Max normalized index ($0$ to $100$) equally weighting Storage $\downarrow$, Generation Time $\downarrow$, FLOPs $\downarrow$, and PSNR $\uparrow$ ($25\%$ each). The optimal tier within each thread count is bolded, with **200 threads (`uint8`)** achieving the overall peak efficiency score of **76.88**.

| Thread Count ($N$) | Precision Tier | Bytes / Thread | Storage (KB) ↓ | Generation Time ↓ | Total Operations (FLOPs) ↓ | PSNR (vs Original) ↑ | Average ↑ |
| -----------------: | :------------- | :------------: | -------------: | ----------------: | -------------------------: | -------------------: | --------: |
|            **100** | `float32`      |      32 B      |        3.20 KB |             1.3 s |                85.9 MFLOPs |          **8.99 dB** |     76.43 |
|            **100** | `float16`      |      16 B      |        1.60 KB |             1.3 s |                85.9 MFLOPs |          **8.98 dB** |     76.67 |
|            **100** | `uint8`        |      8 B       |        0.80 KB |             1.3 s |                85.9 MFLOPs |          **8.97 dB** | **76.78** |
|            **100** | `uint6`        |      6 B       |        0.60 KB |             1.3 s |                85.9 MFLOPs |          **8.36 dB** |     76.10 |
|            **100** | `uint5`        |      5 B       |        0.50 KB |             1.3 s |                85.9 MFLOPs |          **7.86 dB** |     75.53 |
|            **100** | `uint4`        |      4 B       |        0.40 KB |             1.3 s |                85.9 MFLOPs |          **7.39 dB** |     75.00 |
|            **200** | `float32`      |      32 B      |        6.40 KB |             2.8 s |               171.8 MFLOPs |         **11.27 dB** |     76.34 |
|            **200** | `float16`      |      16 B      |        3.20 KB |             2.8 s |               171.8 MFLOPs |         **11.27 dB** |     76.84 |
|            **200** | `uint8`        |      8 B       |        1.60 KB |             2.8 s |               171.8 MFLOPs |         **11.09 dB** | **76.88** |
|            **200** | `uint6`        |      6 B       |        1.20 KB |             2.8 s |               171.8 MFLOPs |         **10.06 dB** |     75.74 |
|            **200** | `uint5`        |      5 B       |        1.00 KB |             2.8 s |               171.8 MFLOPs |          **9.42 dB** |     75.03 |
|            **200** | `uint4`        |      4 B       |        0.80 KB |             2.8 s |               171.8 MFLOPs |          **8.50 dB** |     73.99 |
|            **300** | `float32`      |      32 B      |        9.60 KB |             4.2 s |               257.7 MFLOPs |         **12.86 dB** |     75.56 |
|            **300** | `float16`      |      16 B      |        4.80 KB |             4.2 s |               257.7 MFLOPs |         **12.85 dB** | **76.30** |
|            **300** | `uint8`        |      8 B       |        2.40 KB |             4.2 s |               257.7 MFLOPs |         **12.53 dB** | **76.30** |
|            **300** | `uint6`        |      6 B       |        1.80 KB |             4.2 s |               257.7 MFLOPs |         **11.18 dB** |     74.82 |
|            **300** | `uint5`        |      5 B       |        1.50 KB |             4.2 s |               257.7 MFLOPs |         **10.41 dB** |     73.97 |
|            **300** | `uint4`        |      4 B       |        1.20 KB |             4.2 s |               257.7 MFLOPs |          **9.38 dB** |     72.82 |
|            **400** | `float32`      |      32 B      |       12.80 KB |             5.3 s |               343.6 MFLOPs |         **14.17 dB** |     74.81 |
|            **400** | `float16`      |      16 B      |        6.40 KB |             5.3 s |               343.6 MFLOPs |         **14.17 dB** |     75.81 |
|            **400** | `uint8`        |      8 B       |        3.20 KB |             5.3 s |               343.6 MFLOPs |         **13.79 dB** | **75.87** |
|            **400** | `uint6`        |      6 B       |        2.40 KB |             5.3 s |               343.6 MFLOPs |         **12.14 dB** |     74.07 |
|            **400** | `uint5`        |      5 B       |        2.00 KB |             5.3 s |               343.6 MFLOPs |         **11.13 dB** |     72.95 |
|            **400** | `uint4`        |      4 B       |        1.60 KB |             5.3 s |               343.6 MFLOPs |         **10.01 dB** |     71.71 |
|            **500** | `float32`      |      32 B      |       16.00 KB |             6.1 s |               429.5 MFLOPs |         **15.29 dB** |     74.17 |
|            **500** | `float16`      |      16 B      |        8.00 KB |             6.1 s |               429.5 MFLOPs |         **15.29 dB** |     75.43 |
|            **500** | `uint8`        |      8 B       |        4.00 KB |             6.1 s |               429.5 MFLOPs |         **14.80 dB** | **75.48** |
|            **500** | `uint6`        |      6 B       |        3.00 KB |             6.1 s |               429.5 MFLOPs |         **12.81 dB** |     73.32 |
|            **500** | `uint5`        |      5 B       |        2.50 KB |             6.1 s |               429.5 MFLOPs |         **11.61 dB** |     72.00 |
|            **500** | `uint4`        |      4 B       |        2.00 KB |             6.1 s |               429.5 MFLOPs |         **10.37 dB** |     70.63 |
|          **1,000** | `float32`      |      32 B      |       32.00 KB |             8.7 s |               858.9 MFLOPs |         **19.17 dB** |     70.63 |
|          **1,000** | `float16`      |      16 B      |       16.00 KB |             8.7 s |               858.9 MFLOPs |         **19.15 dB** |     73.11 |
|          **1,000** | `uint8`        |      8 B       |        8.00 KB |             8.7 s |               858.9 MFLOPs |         **18.42 dB** | **73.52** |
|          **1,000** | `uint6`        |      6 B       |        6.00 KB |             8.7 s |               858.9 MFLOPs |         **15.07 dB** |     69.92 |
|          **1,000** | `uint5`        |      5 B       |        5.00 KB |             8.7 s |               858.9 MFLOPs |         **13.37 dB** |     68.09 |
|          **1,000** | `uint4`        |      4 B       |        4.00 KB |             8.7 s |               858.9 MFLOPs |         **11.54 dB** |     66.12 |
|          **2,000** | `float32`      |      32 B      |       64.00 KB |            12.5 s |                1.72 GFLOPs |         **23.45 dB** |     61.09 |
|          **2,000** | `float16`      |      16 B      |       32.00 KB |            12.5 s |                1.72 GFLOPs |         **23.42 dB** |     66.07 |
|          **2,000** | `uint8`        |      8 B       |       16.00 KB |            12.5 s |                1.72 GFLOPs |         **22.15 dB** | **67.09** |
|          **2,000** | `uint6`        |      6 B       |       12.00 KB |            12.5 s |                1.72 GFLOPs |         **17.04 dB** |     61.76 |
|          **2,000** | `uint5`        |      5 B       |       10.00 KB |            12.5 s |                1.72 GFLOPs |         **14.46 dB** |     59.06 |
|          **2,000** | `uint4`        |      4 B       |        8.00 KB |            12.5 s |                1.72 GFLOPs |         **12.03 dB** |     56.54 |
|          **3,000** | `float32`      |      32 B      |       96.00 KB |            15.9 s |                2.58 GFLOPs |         **25.96 dB** |     49.96 |
|          **3,000** | `float16`      |      16 B      |       48.00 KB |            15.9 s |                2.58 GFLOPs |         **25.92 dB** |     57.43 |
|          **3,000** | `uint8`        |      8 B       |       24.00 KB |            15.9 s |                2.58 GFLOPs |         **24.28 dB** | **59.28** |
|          **3,000** | `uint6`        |      6 B       |       18.00 KB |            15.9 s |                2.58 GFLOPs |         **17.84 dB** |     52.70 |
|          **3,000** | `uint5`        |      5 B       |       15.00 KB |            15.9 s |                2.58 GFLOPs |         **14.93 dB** |     49.78 |
|          **3,000** | `uint4`        |      4 B       |       12.00 KB |            15.9 s |                2.58 GFLOPs |         **12.31 dB** |     47.19 |
|          **4,000** | `float32`      |      32 B      |      128.00 KB |            19.3 s |                3.44 GFLOPs |         **27.61 dB** |     37.82 |
|          **4,000** | `float16`      |      16 B      |       64.00 KB |            19.3 s |                3.44 GFLOPs |         **27.58 dB** |     47.81 |
|          **4,000** | `uint8`        |      8 B       |       32.00 KB |            19.3 s |                3.44 GFLOPs |         **25.77 dB** | **50.71** |
|          **4,000** | `uint6`        |      6 B       |       24.00 KB |            19.3 s |                3.44 GFLOPs |         **18.41 dB** |     43.38 |
|          **4,000** | `uint5`        |      5 B       |       20.00 KB |            19.3 s |                3.44 GFLOPs |         **15.25 dB** |     40.32 |
|          **4,000** | `uint4`        |      4 B       |       16.00 KB |            19.3 s |                3.44 GFLOPs |         **12.43 dB** |     37.66 |
|          **5,000** | `float32`      |      32 B      |      160.00 KB |            22.9 s |                4.29 GFLOPs |         **28.82 dB** |     25.00 |
|          **5,000** | `float16`      |      16 B      |       80.00 KB |            22.9 s |                4.29 GFLOPs |         **28.78 dB** |     37.48 |
|          **5,000** | `uint8`        |      8 B       |       40.00 KB |            22.9 s |                4.29 GFLOPs |         **26.81 dB** | **41.45** |
|          **5,000** | `uint6`        |      6 B       |       30.00 KB |            22.9 s |                4.29 GFLOPs |         **18.70 dB** |     33.56 |
|          **5,000** | `uint5`        |      5 B       |       25.00 KB |            22.9 s |                4.29 GFLOPs |         **15.40 dB** |     30.49 |
|          **5,000** | `uint4`        |      4 B       |       20.00 KB |            22.9 s |                4.29 GFLOPs |         **12.48 dB** |     27.87 |
