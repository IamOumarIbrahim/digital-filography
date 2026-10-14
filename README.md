<div align="center">

# Filography: A Revisited Technique for Digital Imagery

[![PDF](https://img.shields.io/badge/Paper-PDF-red?logo=adobeacrobatreader&logoColor=white)](docs/main.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)


**Oumar Ibrahim** and **Mohamad Khairy**  
*Department of Computer Engineering, University of Sharjah, Sharjah, UAE*  
`u22200741@sharjah.ac.ae` | `mishak@sharjah.ac.ae`

</div>


> [!TIP]
> **Work in progress**: The code will be updated as the research progresses.


## Project Structure

```
filography/
├── config/
│   ├── config.json               # Pipeline configuration (shared settings & metrics)
│   └── instances_val2017.json    # Cached COCO annotations
├── dataset/                      # Per-class subfolders with images & annotations
├── docs/
│   └── main.pdf                  # Research paper
├── models/                       # YOLO model weights (yolov8n, yolo11n, yolo12n, yolo26n, yolov10n)
├── results/                      # Timestamped experiment & benchmark outputs
└── src/
    ├── download_dataset.py       # COCO val2017 per-class dataset downloader
    ├── resize_dataset.py         # 640x640 bilinear central cropper & annotation bounder
    ├── run_terminal.py           # Interactive menu terminal
    └── README.md                 # Detailed documentation for source scripts
```

## Quickstart

Launch the interactive terminal:
```bash
python src/run_terminal.py
```

Download a dataset with specific COCO classes:
```bash
python src/download_dataset.py -n 300 --classes dog person cat
```
