# LegoFitter

Take a photo of a pile of LEGO bricks → get a per-brick inventory → find out
which official sets you could build with them, and what's missing.

## How it works

Two models chained together, plus an optional lookup:

1. **Detector** finds every individual brick in a photo and draws a box
   around it (one class: "lego brick").
2. **Classifier** looks at each cropped box and names the brick (which part
   it is).
3. Counting up the classified crops gives an **inventory** (part → count)
   for the whole photo.
4. **(Bonus)** The inventory is sent to the Rebrickable API's
   `get_set_match` endpoint, which returns which sets can be built and what
   parts are still missing from each.

This mirrors the two-stage design from the Boiński "LEGO Sorter" paper
(detector finds bricks, a separate classifier names them) rather than one
end-to-end detector that both locates and names — see `project-status`
notes for why.

## Project layout

```
LegoFitter/
├── .env                 # local secrets (gitignored) — copy from .env.example
├── .env.example          # template, committed
├── .envrc                 # direnv: loads .env, sets PYTHONPATH
├── requirements.txt        # deps, grouped by phase
├── main_local.py            # CLI entry point, one subcommand per step below
├── app/
│   ├── config.py               # all paths/constants, read from .env
│   ├── classification/          # Objective 1
│   │   ├── dataset.py               # step 1: 50-class subset + DataLoader
│   │   ├── model.py                 # step 2a: our custom CNN
│   │   ├── train.py                 # step 2b: training loop
│   │   └── evaluate.py              # step 3: accuracy scoring
│   ├── detection/                # Objective 2
│   │   ├── prepare_data.py          # step 1: VOC XML -> YOLOv5 format
│   │   ├── train.py                 # step 2: YOLOv5 training wrapper
│   │   └── evaluate.py              # step 3: mAP scoring
│   ├── pipeline/                 # Objective 3
│   │   └── inference.py             # detect -> crop -> classify -> inventory
│   ├── rebrickable/               # Bonus
│   │   └── client.py                 # get_set_match wrapper
│   └── utils/
│       └── io.py
├── data/                  # gitignored — see below
│   ├── lego-dataset-classification/
│   └── lego-tagged-object_detection/
├── models/                # gitignored — trained weights land here
└── tests/
```

`data/` and `models/` are gitignored (large binary datasets/weights don't
belong in git). `.env` is gitignored too — copy `.env.example` to `.env` and
fill in `REBRICKABLE_API_KEY` when you reach the bonus objective.

## Setup

```bash
cd LegoFitter
pyenv virtualenv 3.12.9 legofitter
pyenv local legofitter
direnv allow                     # loads .env + PYTHONPATH automatically
pip install -r requirements.txt  # core + Phase 1 deps; Phase 2 adds a couple more, see below
```

## Roadmap

We're building this **one step at a time, in this order** — each objective
gates the next, so we don't move on until the one before it works.

### Objective 1 — Classification (up to 50 brick classes)

Goal: **≥70% top-1 accuracy** on a held-out test set before touching
Objective 2. Dataset: `data/lego-dataset-classification` (the Boiński
448-class dataset), downsized to 50 classes. Model: a CNN we design and
train ourselves with **tf.keras** (no pretrained backbone for this
objective).

1. **Prepare the data** (`app/classification/dataset.py`) — pick the 50
   classes to keep (e.g. the ones with the most real photos, so we're not
   training mostly on synthetic renders), build the working subset, and
   set up train/val/test splits + a PyTorch `Dataset`/`DataLoader`.
2. **Build & train the CNN** (`app/classification/model.py`,
   `app/classification/train.py`) — design the architecture, train it,
   checkpoint the best epoch.
3. **Evaluate** (`app/classification/evaluate.py`) — score top-1 accuracy
   on the test split.
   - **≥70%** → move on to Objective 2.
   - **<70%** → iterate (more real-photo weighting, augmentation,
     architecture tweaks, fewer/easier classes) before proceeding.

### Objective 2 — Object Detection (YOLOv5)

Goal: a YOLOv5 model that reliably draws a box around each brick in a
photo, regardless of what type it is (single class: "lego brick").
Dataset: `data/lego-tagged-object_detection` (VOC XML annotations).

1. **Prepare the data** (`app/detection/prepare_data.py`) — convert VOC
   XML annotations to YOLOv5's expected `images/` + `labels/*.txt` format,
   write the dataset YAML.
2. **Set up & train YOLOv5** (`app/detection/train.py`) — clone the
   `ultralytics/yolov5` repo (its own `train.py`/`val.py` are what we
   drive), train on our converted dataset. Note: YOLOv5 is **PyTorch**,
   while Objective 1's classifier is **tf.keras** — two different
   frameworks in one repo, which is fine; they only need to meet at
   inference time in Objective 3.
3. **Evaluate** (`app/detection/evaluate.py`) — score mAP via YOLOv5's
   `val.py`.

### Objective 3 — Chaining the two models

Goal: given one photo of *multiple* bricks, produce a full inventory.

1. **Wire up the pipeline** (`app/pipeline/inference.py`) — run the
   Objective 2 detector on the photo, crop every detected box, run the
   Objective 1 classifier on each crop, tally the results into a
   part → count inventory.
2. **Test end-to-end** on real multi-brick photos and sanity-check the
   counts.
3. **Expose it via `main_local.py`** (`pipeline-run <image>`).

### Bonus — Rebrickable integration

Goal: turn the inventory into "here's what you can build."

1. Get a free API key at rebrickable.com/api and add it to `.env`.
2. **Build the client** (`app/rebrickable/client.py`) — wrap the
   `get_set_match` endpoint.
3. **Connect it**: feed the Objective 3 inventory in, get back matched
   sets + missing pieces per set, surface it via `main_local.py
   match-sets`.

## Working agreement

Nothing beyond this scaffold is implemented yet — every module above is a
stub with a docstring describing what it will own. We'll fill them in
together step by step, in the order above, checking the gate (70% accuracy)
before leaving Objective 1.
