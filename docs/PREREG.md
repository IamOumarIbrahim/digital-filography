# Pre-registered protocol v1 (frozen before the 48 KB run)

DESIGN
Budget: 48,000 bytes. Seeds: 1 to 5. Images per seed: 500 COCO val2017 images.
Image subsets: shuffle the val2017 image ids with master seed 100. Seed k uses block k (500 images). Blocks do not overlap.
Encoder seed for seed k: 100 + k.
Methods: four thread formats (equal bytes) and five bitmap baselines.
Bitmap baselines: JPEG, WebP lossy, PNG-8, WebP lossless, PNG.
Lossy codecs: the highest quality that fits in 48,000 bytes.
Lossless codecs and PNG-8: the smallest downsampling that fits, then upsample to the detector input size, as in the 70 KB benchmark.
Score per seed and method = mean over the five detectors of recovery.
Paired gap = score of method A minus score of method B in the same seed. Unit: percentage points.
Top formats are ranked by their mean score over the five seeds.
Tie rule (block A3): choose in this order: Uniform UINT8, Uniform FP16, Adjusted UINT8, Adjusted FP16.

THRESHOLDS
D1: the standard deviation (n-1) of the paired gap between the top two thread formats is under 1.4 points.
D2: the 95% t-interval of that gap over seeds (t = 2.776, 4 degrees of freedom) excludes zero, and a paired bootstrap over images agrees. Otherwise NO.
D3: best bitmap minus chosen thread format is 15 points or less (off-the-shelf detectors).
D4: thread fine-tuning gain over off-the-shelf is 15 points or more.
D5: fine-tuned JPEG minus fine-tuned threads is 15 points or less.
D6 cost-matched test: threads beat JPEG by 5 recovery points or more at the same input size, with a 95% interval above zero.
D6 privacy test: the leakage ratio (threads / matched JPEG) is 0.5 or less, and the upper end of its 95% interval is below 1.0.
D7: human N_50 differs from the YOLO N_50 and from the VLM N_50, with non-overlapping 95% intervals.
