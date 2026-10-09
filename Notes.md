## Literature Review

Torralba’s “How many pixels make an image?” asks the same question with pixels instead of threads, so read it first. Geirhos et al. on human-vs-machine consistency are the other useful precedent.

## Recognition

Report the counts at which 50% and 90% of images are recognised

## Coding

You are my Python coding helper. I am a beginner, so explain things in simple
English (B1 level) and give me code I can copy and run.

## My project
I am building research about "digital filography": turning an image into
straight coloured threads. The goal is a journal paper. I want to find out
how many threads are needed before humans, AI classifiers and object
detectors can recognise something. Later I will build my own object detector
that works directly on threads.

## My computer
- Windows, PowerShell. Give commands for PowerShell only.
- Python with pip. Libraries: numpy, Pillow, torch, ultralytics.
- Use only Windows-friendly code (for example, set DataLoader workers=0 if
  training crashes).

## The two files below
1. threader.py: turns an image into threads.
   - A filographic image is an N x 8 matrix, one thread per row:
     [x1, y1, x2, y2, r, g, b, a], all values from 0 to 1.
   - Threads are drawn in order on a white background.
   - Threads are added one by one, so the first 1000 threads of a 10000-thread
     run are the 1000-thread image. This means I get all thread counts from one run.
   - Main pieces: ThreadPainter (places threads), render_threads (draws them),
     load_image (loads and resizes to 640 px on the longest side).
2. eval_dataset.py: my benchmark.
   - Tests many YOLO models on COCO images drawn with threads.
   - Uses TIERS = 6 packed formats from packing.py (uniform_ and adjusted_ with
     fp16, uint8, uint6). Threads are really packed into bytes and decoded
     again before they are drawn.
   - The test set is 500 COCO images with no per-class quota (download_dataset.py).
   - Ground truth is the model's own detections on the original image
     (a "model-relative baseline").
   - Saves a checkpoint file (checkpoint_rows.jsonl) so a run can resume.
   - Writes CSV files and then calls generate_graphs.py (I have not pasted this file).

## Important facts about my code
- If I change TIERS, the model list or the thread counts, I must delete the
  old checkpoint or use a new --output-dir. Remind me when this is needed.
- The summary at the end of eval_dataset.py needs the count 10000 to exist.
- The gen_time_s value in generation.csv is a formula, not a real measurement.
- I use these counts: 25 50 100 200 300 400 600 800 1200 1600 2400 3200 5000 10000.

## How I want you to answer
1. Before you write code, say in 2 or 3 short sentences what you will do.
2. Tell me exactly which file and which line to change. Use this format:
   "Find this:" (old code) and "Replace with:" (new code).
   For a new file, give the whole file.
3. Never say "the rest stays the same" inside a function. Show the full function.
4. After the code, give me the exact PowerShell command to run it and tell me
   what output I should see if it works.
5. If something can go wrong, tell me what the error will look like.
6. If you need a file you have not seen (for example generate_graphs.py), ask
   me for it. Do not guess what is inside it.
7. Do not change my existing results or file names without telling me.
8. Give one step at a time. After each step, wait for me to run it and
   report back.
9. If my idea has a problem (bad science, wrong metric), tell me directly.

## My task right now
(write your question here)

## Timeline

2.2 One thing I haven’t seen is generate_graphs.py. It may expect a straight x-axis, so tell me if the graphs look wrong.
 Done when output_log\dataset_evaluation_summary.csv exists.

Step 3: Find the threshold for each detector

 3.1 Define “recognised”: an image is recognised at N threads if the model detects it at N and at every larger count.
 3.2 Define N50: the smallest count where at least half of the images are recognised.
 3.3 Ask me for find_threshold.py. It reads yolo_benchmark_results.csv and prints N50 per model and precision.
 3.4 Ask me for the Faster R-CNN wrapper and add it as an eighth model.
 Done when you have one table: model × precision → N50.

Step 4: AI classifiers

A classifier gives one label for the whole image.

 4.1 Run pip install transformers.
 4.2 Start with CLIP (openai/clip-vit-base-patch32). You give it the 6 words directly, so you don’t need to map any labels.
 4.3 Use the same images and counts as Step 2. Ask me for the script.
 4.4 Later, add one ImageNet model, such as ResNet-50 or ViT, and one vision-language chatbot.
 Done when CLIP has an N50 for each class.

Step 5: Human test

 5.1 Ask your university whether you need ethics approval for people taking part. Do this first, because it can take weeks.
 5.2 Pick 100 images. Use 8 of the 14 counts: 50, 100, 200, 400, 800, 1600, 3200, 10000.
 5.3 Each person sees an image only once, at a random count. If they saw a clear version, they would remember it.
 5.4 Ask me for make_human_stimuli.py (it saves the images) and a small Python window app that shows an image and asks “What do you see?” with the 6 classes plus “nothing”.
 5.5 Test it on 3 friends first to find bugs. Then ask 20 to 30 people.
 Done when you have a CSV of answers and a human N50 per class.

Step 6: Compare everyone

 6.1 Plot recognition rate against thread count (log axis) for humans, CLIP and the detectors on one graph.
 6.2 Add a control: the same images as JPEG files with the same size in KB as each thread count. It shows whether threads are harder to recognise than normal compression.
 6.3 Check whether humans and models fail on the same images.

Step 7: Your own detector

 7.1 Time one image first, in PowerShell:
Measure-Command { python src\threader.py dataset\ONE_IMAGE.jpg --counts 10000 --modes color --alphas 1.0 --npy -o test_out }

Look at TotalSeconds. The number in generation.csv is a formula, not a real measurement. Total time ≈ images × seconds ÷ workers.

 7.2 Start making training threads for 5,000 to 10,000 COCO train images of people and vehicles. Do this early because it is slow. I’ll write a batch script for it.
 7.3 Baseline: fine-tune a YOLO on thread images. On Windows, set workers=0 in training if the data loader crashes.
 7.4 Quick test: do places where threads cross predict object locations better than plain thread density?
 7.5 Then build the thread-based detector (threads as nodes, crossings as links).


 TODO:
 Lock the format. Use 8-bit and retire uint6 and fp16 (except as the ceiling reference). Use adjusted below about 50 KB and uniform above about 60 KB.
Close the data gaps. Run uint8 beyond 10000 threads to find the real saturation point. Test 7-bit, and test which field you quantize (the one that causes the uint6 collapse).
Change the metric. Rank everything by recovery and F1 per KB, and add confidence intervals by bootstrapping over the 500 images.
Add the missing control. Compare against JPEG, WebP or AVIF at the same KB. Nothing in this data shows that threads beat a normal codec, and that decides whether the approach is worth continuing.
Test the upside. Fine-tune a detector on thread reconstructions, or add a detection-aware loss when fitting threads. This is a hypothesis, but it targets the main failure, which is missed objects.
Validate wider. Check results by object size and class, with yolo12n as the main detector and yolov8n as the weak check.