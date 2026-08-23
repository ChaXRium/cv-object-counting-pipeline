# GPU-Accelerated Coin Counting — Local Setup (VS Code)

## Files
- `coin_counter.ipynb` — the full pipeline (preprocessing → classical CV baseline →
  auto-labelling → augmentation → YOLOv8 GPU training → evaluation).
- `requirements.txt` — Python dependencies.

## 1. Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

**GPU note (important for the "GPU-accelerated" part of the coursework):**
The `torch`/`torchvision` lines in `requirements.txt` install a working version,
but to actually get CUDA acceleration you should install the build matched to
your GPU driver instead. Check your CUDA version with `nvidia-smi`, then get the
right command from https://pytorch.org/get-started/locally/ — e.g. for CUDA 12.1:

```bash
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

If you don't have an NVIDIA GPU, everything still runs — training just falls
back to CPU automatically (the notebook prints which one it's using in the
first cell). Cut `epochs` and `imgsz` down in Section 7 if you're on CPU.

## 3. Add your data

Create a folder for your coin images, e.g.:

```
data/coin_images/
    IMG001.jpg
    IMG002.jpg
    ...
```

Update `DATA_DIR` in Section 1 of the notebook if you use a different path.

## 4. Open and run in VS Code

- Install the **Python** and **Jupyter** extensions in VS Code if you haven't already.
- Open `coin_counter.ipynb`, select your `venv` as the kernel (top-right kernel picker),
  and run the cells top to bottom.
- Section 7 (YOLO training) is the slow one — expect a few minutes on a GPU,
  much longer on CPU. `yolov8n.pt` pretrained weights download automatically
  on first run (needs internet once).

## 5. Before running Section 9 (evaluation)

Hand-count the coins in a handful of your images and fill them into the
`ground_truth` dict in Section 4 — this is what turns the final comparison
table and chart into real numbers for your report's evaluation section.

## Troubleshooting

- **`ModuleNotFoundError`** — make sure the VS Code kernel is your `venv`, not
  the system Python (check top-right of the notebook).
- **`nvidia-smi` not found / CUDA available: False`** — either you don't have
  an NVIDIA GPU, or the driver/CUDA toolkit isn't installed, or `torch` wasn't
  installed with CUDA support (see step 2). CPU fallback still works.
- **Very few/no detections in Section 3** — your images likely have a
  different lighting setup or coin/background contrast than the dataset this
  was tuned on. Adjust `min_radius`/`max_radius` in `detect_coins_classical`
  to roughly match coin size in pixels at the 1000px working resolution, and
  tweak the Hough `param2` (lower = more permissive, more false positives;
  higher = stricter, may miss faint coins).
