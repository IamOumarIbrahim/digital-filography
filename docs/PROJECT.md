# Filography: project context

IDEA
Rebuild an image from N straight colored line segments, called threads.
Thread vector: [x1, y1, x2, y2, r, g, b, a]. File size in bytes = N x b.

FORMATS (b = bytes per thread)
Uniform FP16: all 8 values FP16. b = 16.
Uniform UINT8: all 8 values UINT8. b = 8.
Adjusted FP16: coordinates FP16, color RGBA4444. b = 10.
Adjusted UINT8: coordinates UINT8, color RGBA4444. b = 6.

BUDGET UNDER TEST
48 KB = 48,000 bytes. Equal bytes means N = 48,000 / b.
N = 3,000 (Uniform FP16), 4,800 (Adjusted FP16), 6,000 (Uniform UINT8), 8,000 (Adjusted UINT8).
Thread files are N x b bytes, with no header. Codec files count their full size on disk.

DETECTORS
The benchmark evaluates five YOLO nano detectors:
1. yolov8n: Ultralytics YOLOv8 nano. Weights: yolov8n.pt (6.55 MB).
2. yolov10n: Ultralytics YOLOv10 nano. Weights: yolov10n.pt (5.86 MB).
3. yolo11n: Ultralytics YOLO11 nano. Weights: yolo11n.pt (5.61 MB).
4. yolo12n: Ultralytics YOLO12 nano. Weights: yolo12n.pt (5.60 MB).
5. yolo26n: Ultralytics YOLO26 nano. Weights: yolo26n.pt (5.54 MB).
All models run via Ultralytics 8.4.174 with PyTorch 2.6.0+cu124 on CUDA.
Input size: 640 pixels max dimension (aspect ratio preserved on white background canvas).
Confidence threshold: conf = 0.25.
NMS settings: Ultralytics default NMS (iou = 0.70, max_det = 300). yolov10n is architectural NMS-free.

METRICS
Exact definitions from src/eval_dataset.py:
- Baseline Ground Truth: Model-relative detections on the native uncompressed photo (size 640 px) with conf >= 0.25.
- Match Rule: A predicted box matches a baseline box if and only if:
  1. Class rule: p["cls_name"] == b["cls_name"] (exact string match).
  2. IoU threshold: compute_iou(p["box"], b["box"]) >= 0.50 (IOU_MATCH = 0.50).
  Matching is greedy by descending prediction confidence. Each baseline box matches at most once. Extra detections count as false positives.
- Recovery rate: matched baseline objects / baseline objects = len(matched_ious) / len(baseline).
- Precision: matched baseline objects / predicted objects = len(matched_ious) / len(preds).
- Mean IoU: arithmetic mean of matched IoU values = sum(matched_ious) / len(matched_ious) (0.0 if no matches).
- PSNR (dB): -10.0 * log10(max(MSE, 1e-10)), where MSE = mean((canvas - target_norm) ** 2) against the normalized target in [0, 1].
- Averaging rule:
  Image pooling per detector: counts are micro-averaged across images (sum of matched / sum of baseline).
  Detector averaging: the score for a method in a seed is the unweighted arithmetic mean of recovery rate across the five detectors:
  score = mean([recovery(yolov8n), recovery(yolov10n), recovery(yolo11n), recovery(yolo12n), recovery(yolo26n)]).

New metrics for this test: mAP50 and mAP50-95 against COCO labels (pycocotools, labels mapped to the detector input size), and recovery by size bucket.
Size bucket, from the reference box area at detector input size: under 32 x 32 pixels is small, 32 x 32 to 96 x 96 is medium, above 96 x 96 is large.

EARLIER NUMBERS (for sanity checks only. Never tune to match them.)
Mean recovery over 500 images and 5 detectors, at the equal-byte thread counts:
Uniform FP16 at 3,000 threads: 20.6%. Uniform UINT8 at 6,000 threads: 40.9%. Adjusted UINT8 at 8,000 threads: 42.7%.
Adjusted FP16 at 4,800 threads was not plotted.
At 70 KB: WebP lossy 92.21%, JPEG 90.25%, PNG-8 79.44%, WebP lossless 77.17%, PNG 73.26%, Filography 50.60%.
Warn if a new thread mean differs from the number above by more than 3 points.
Warn if a codec at 48 KB scores more than 1 point above its 70 KB number.

RESULTS FILES
Run id: results are in results/<run_id>/ (v1, v2, ...). Use the highest run id unless the task says otherwise.
results/<run_id>/raw/seed_<k>/per_object.csv: one row per reference object, method and detector.
results/<run_id>/raw/seed_<k>/per_image.csv: one row per image and method (bytes, encode seconds, PSNR).
results/<run_id>/summary/seed_scores.csv: one row per seed, method and detector.
results/<run_id>/payloads/seed_<k>/<method>/<image_id>.<ext>: the encoded files.
results/README.md: every column, defined.

HOUSE RULES
1. Never invent numbers, results or citations. Put missing facts under Open items in your report. If you need a fact only I know, ask one short list of questions, then wait.
2. Before you start, read docs/PROJECT.md, docs/PREREG.md and decision_log.md.
3. Never change a threshold in docs/PREREG.md. If you think one is wrong, say so and stop.
4. Fix all random seeds. For every run, log the seed, the git commit, the library versions and the exact command.
5. Save raw per-object outputs, not only summaries. Never overwrite earlier results. Write new runs to new folders.
6. Inside one seed, every method uses the same images (paired design).
7. Test images are for testing only. Never tune on them.
8. Reuse existing code. Change only what the task needs.
9. If a check fails, stop and report. Do not hide the problem.
10. When you finish, write reports/<block>.md in the standard report format, add a short entry to decision_log.md, and stop. Do not start the next block.
11. Write plain B2 English. Short sentences. No em dashes.

STANDARD REPORT FORMAT
Block:
Run id:
Verdict: YES, NO, DONE or FAILED
Key numbers: up to 6 lines, each with its source file
Files written:
Warnings:
Open items:
Next block:
