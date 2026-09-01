"""
Central params for LegoFitter, loaded from environment variables (see .env.example).

Every other module should read paths/constants from here rather than
hardcoding them, so the whole project stays configurable from one file + .env.
"""
from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()


def _env_flag(name: str, default: bool = False) -> bool:
    """Read a true/false switch from the environment.

    os.getenv always hands back a STRING, and bool("false") is True -- the
    trap this exists to avoid. Accepts 1 / true / yes / on in any case
    (quotes tolerated, since .env entries are often written MY_FLAG='true');
    anything else, including an empty value, is False.
    """
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().strip("\"'").lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    """Read a whole-number setting from the environment.

    Wrapped rather than inlined as int(os.getenv(...)) so a typo fails with a
    sentence instead of a bare ValueError raised while params.py is still
    importing -- which surfaces as every command on the machine breaking at
    once, with a traceback that never mentions the variable at fault.
    """
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value.strip().strip("\"'"))
    except ValueError:
        raise SystemExit(f"❌ {name}={value!r} is not a whole number")


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / os.getenv("DATA_DIR", "data")
MODELS_DIR = PROJECT_ROOT / os.getenv("MODELS_DIR", "models")

# Square input size for the CNN, in pixels. Read from the environment so the
# Mac and the VM can disagree (.env is gitignored) and so a single run can
# override it:  make run_vm IMG_SIZE=256
_img_px = int(os.getenv("IMG_SIZE", 128))
IMG_SIZE = (_img_px, _img_px)
BATCH_SIZE = 32

RANDOM_STATE = 42  # seed for every split / subsample, so runs are comparable

BUCKET_NAME = os.environ.get("BUCKET_NAME")

# --- Objective 1: Classification ---
# Which architecture to train: app/classification/models/model_<MODEL_NAME>.py.
# Defaults to christophe so a fresh clone (which has no .env -- it is
# gitignored) trains without anyone configuring anything. Override in .env for
# a machine that should always use one architecture, or per run on the command
# line:  make run_local MODEL_NAME=oriane
MODEL_NAME = os.getenv("MODEL_NAME", "christophe")

NUM_CLASSES = int(os.getenv("NUM_CLASSES", 50))
CLASSIFICATION_DATA_DIR = PROJECT_ROOT / os.getenv(
    "CLASSIFICATION_DATA_DIR", "data/lego-dataset-classification"
)
# Where dataset.ensure_local_data() fetches the images from when a machine
# (a fresh VM, a teammate's laptop) has no local copy. Kept here rather than
# in dataset.py so every path/bucket the project touches lives in one file.
# Where run folders live in the bucket, one prefix per objective. GCS has no
# real directories -- these are just name prefixes -- so nothing creates them.
GCS_CLASSIFICATION_MODELS = "models/classification"
GCS_DETECTION_MODELS = "models/detection"

GCS_CLASSIFICATION_DATA = os.getenv(
    "GCS_CLASSIFICATION_DATA",
    "gs://legofitter-datasets/lego-dataset-classification",
)
# The "current" folder is this machine's working state: whatever was trained
# or downloaded last. It is written during training, before a run has a name
# (ModelCheckpoint starts saving after epoch 1), which is why it needs a fixed
# location. registry.save_model() copies it into a named run folder afterwards,
# and registry.load_model() copies a run back over it. Keeping it in a folder
# means models/ holds nothing but folders -- one per run, plus this one.
# One folder per objective under models/, so a classifier run and a detector
# run can never be mistaken for each other -- and so "what runs do we have?"
# has an answer per objective rather than one mixed list.
CLASSIFICATION_MODELS_DIR = MODELS_DIR / "classification"

CURRENT_RUN_DIR = CLASSIFICATION_MODELS_DIR / "current"
CLASSIFICATION_MODEL_PATH = CURRENT_RUN_DIR / "classifier.keras"
CLASSIFICATION_ACCURACY_TARGET = 0.70  # gate before starting Objective 2

# Index -> part ID mapping, written at training time so a saved model can be
# used outside the session that trained it.
CLASS_NAMES_PATH = CURRENT_RUN_DIR / "class_names.json"

# The two image sources use different file extensions: globbing "*.jpg"
# against renders/ silently returns nothing.
SOURCE_PATTERNS = {
    "photos": "*.jpg",
    "renders": "*.jpeg",
}

# Variance of the Laplacian, used by dataset.py's is_blurry(). Sharp images
# have lots of high-frequency edge content (high variance); blurry ones are
# smoother (low variance) -- but so are sharp photos of plain, low-texture
# grey pieces, so this doesn't cleanly separate "blurry" from "sharp but
# plain": a 400-photo sample of the real dataset has a median score of ~21
# with no obvious gap, and 5 hand-picked blurry examples scored 4.9-14.8.
# 6 is the conservative choice -- catches only the most severely blurry
# photos. See notebooks/datascientist_deliverable.ipynb (Blur filtering
# section) to preview what a higher threshold would additionally flag
# before raising this.
BLUR_VARIANCE_THRESHOLD = 6

# Max renders kept per class, as a multiple of that class's photo count.
# 650/350 = 1.857 is the render:photo ratio Boiński et al. (Sci Data 2023)
# used for their baseline training set. Set CAP_RENDERS=false (.env, or per
# run: make train_classification_vm CAP_RENDERS=false) to skip the cap
# entirely and train on every render for every class -- the raw ~5:1
# distribution, i.e. the full photos+renders dataset.
CAP_RENDERS = _env_flag("CAP_RENDERS", default=True)
RENDER_PHOTO_RATIO = (650 / 350) if CAP_RENDERS else None

# Stratified split proportions: TEST_SIZE off the full set, then VAL_SIZE off
# what remains. (0.15, 0.15) gives roughly 72/13/15 — training-heavy, as in
# the paper, which held out only ~100 images per class.
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# --- Objective 1: model architecture (model.py) ---
DENSE_UNITS = 256       # width of the single hidden layer in the classifier head
DROPOUT_RATE = 0.4      # fraction of units randomly dropped during training
L2_REG = 1e-4           # weight-decay strength applied to Dense/Conv kernels

# --- Objective 1: augmentation (model.py) ---
# All are "how much", as a fraction. Bricks are photographed at arbitrary
# angles and lighting, so geometric + photometric jitter both help. No
# horizontal flip: LEGO has mirrored part pairs (left/right wedges) that are
# DIFFERENT part IDs, so flipping could teach the model to confuse two
# classes.
AUG_ROTATION = 0.15     # +/- 15% of a full turn (~54 degrees)
AUG_ZOOM = 0.15
AUG_TRANSLATION = 0.10
AUG_BRIGHTNESS = 0.20
AUG_CONTRAST = 0.20
# "nearest" extends the edge pixel into corners a rotation/zoom leaves empty.
# The Keras default, "reflect", mirrors the brick into those corners and
# invents phantom half-bricks — bad on our plain backgrounds.
AUG_FILL_MODE = "nearest"

# --- Objective 1: training (train.py) ---
LEARNING_RATE = 1e-3            # Adam's default; step_decay lowers it from here
# Upper bound — EarlyStopping normally stops us first. Overridable so a long
# run is a command-line change:  make train_classification_vm EPOCHS=300
EPOCHS = _env_int("EPOCHS", 100)
# step_decay's schedule: LEARNING_RATE * LR_DECAY_FACTOR ** (epoch // LR_DECAY_EVERY).
# The defaults suit a 100-epoch run. They must be RAISED for a longer one, or
# the LR collapses to nothing long before the last epoch: at the defaults it is
# already 8e-6 by epoch 90 and 6e-8 by epoch 180, so a 300-epoch run would
# spend its last 150 epochs not learning. Both are env-driven for that reason:
#   make train_classification_vm EPOCHS=300 LR_DECAY_EVERY=60 LR_DECAY_FACTOR=0.5
LR_DECAY_EVERY = _env_int("LR_DECAY_EVERY", 30)
LR_DECAY_FACTOR = float(os.getenv("LR_DECAY_FACTOR", 0.2))
EARLY_STOPPING_PATIENCE = _env_int("EARLY_STOPPING_PATIENCE", 30)
REDUCE_LR_PATIENCE = 5          # epochs without improvement before halving the LR
REDUCE_LR_FACTOR = 0.5          # new_lr = old_lr * this
MIN_LEARNING_RATE = 1e-4        # floor for ReduceLROnPlateau

# --- Objective 1: transfer learning (model_vgg16.py) ---
# Phase 2 only. How many layers at the TOP of the pretrained base to unfreeze:
# 4 is VGG16's last conv block plus its pool. Early layers hold generic edge
# and texture filters worth keeping exactly as ImageNet learned them.
VGG16_FINETUNE_LAYERS = 4
# 100x below LEARNING_RATE. Fine-tuning a pretrained base at the phase-1 rate
# destroys the filters in a few steps -- the classic "it got worse after
# unfreezing" failure.
FINETUNE_LEARNING_RATE = 1e-5
# Phase 2 runs for at most this many further epochs. Short on purpose: the
# base is already close to right, and a long fine-tune mostly overfits.
FINETUNE_EPOCHS = 30

# float16 compute on the GPU's tensor cores. Big speedup on the T4, no effect
# (or a slowdown) on CPU / Apple Silicon -- which is why this is an env switch
# and not a tracked constant: the Mac and the VM need different answers, and a
# constant means whoever commits it last breaks the other machine.
#
# `make run_vm` turns it on for you (see the Makefile), so nobody has to
# remember. Default off, so a laptop and a fresh clone are always safe.
USE_MIXED_PRECISION = _env_flag("USE_MIXED_PRECISION", default=False)

# Where train.py dumps history.history so the curves survive the process.
# A detached script has no notebook to hold `history` in memory.
HISTORY_PATH = CURRENT_RUN_DIR / "history.json"

# --- Objective 2: Object Detection ---
DETECTION_DATA_DIR = PROJECT_ROOT / os.getenv(
    "DETECTION_DATA_DIR", "data/lego-tagged-object_detection"
)
DETECTION_MODELS_DIR = MODELS_DIR / "detection"

# The detector's working state, mirroring models/classification/current/:
# whatever run was trained or loaded last. A detection run is three files that
# only mean anything together -- weights, the class map they were trained
# against, and the curves -- exactly like a classification run.
DETECTION_CURRENT_DIR = DETECTION_MODELS_DIR / "current"
DETECTION_MODEL_PATH = DETECTION_CURRENT_DIR / "best.pt"
DETECTION_DATA_YAML_PATH = DETECTION_CURRENT_DIR / "data.yaml"
DETECTION_RESULTS_PATH = DETECTION_CURRENT_DIR / "results.csv"

# The annotations call the class both "lego" and "legod" (a typo in the source
# dataset). Both mean the same thing, and YOLO trains on ONE class here: we
# only need to know WHERE a brick is; which brick it is, is Objective 1's job.
DETECTION_SOURCE_NAMES = {"lego", "legod"}
DETECTION_CLASS_NAME = "lego"
DETECTION_CLASS_ID = 0

# Extensions in the detection dataset -- the classification set is .jpg photos
# and .jpeg renders, but the tagged detection set also contains .png.
DETECTION_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# prepare_data.py rewrites the VOC dataset into YOLO's own layout here.
# Kept out of DETECTION_DATA_DIR: that folder is the input, and generated
# files do not belong in it.
YOLO_DATA_DIR = DATA_DIR / "lego-yolo-dataset"
YOLO_DATASET_YAML = YOLO_DATA_DIR / "data.yaml"

# --- Objective 2: YOLO training ---
# The three below are overridable per run, e.g.
#   make train_detection_vm YOLO_BASE_MODEL=yolo26s.pt YOLO_EPOCHS=100 YOLO_BATCH_SIZE=16
# YOLO_IMG_SIZE and YOLO_PATIENCE are deliberately not: 640 suits this dataset
# (median box ~91px at 640, only 4% COCO-"small"), and patience should track
# epochs rather than being tuned on its own.
# ultralytics downloads it
YOLO_BASE_MODEL = os.getenv("YOLO_BASE_MODEL", "yolo26n.pt")
YOLO_EPOCHS = _env_int("YOLO_EPOCHS", 50)
# YOLO's own input size, unrelated to IMG_SIZE
YOLO_IMG_SIZE = 640
YOLO_BATCH_SIZE = _env_int("YOLO_BATCH_SIZE", 8)
YOLO_PATIENCE = 10                      # epochs without improvement before stopping

# The gate for Objective 2, the detection twin of CLASSIFICATION_ACCURACY_TARGET.
# Higher than the classifier's 0.70 on purpose: this is a ONE-class problem --
# "is there a brick here" -- not a 50-way choice, so it should be a great deal
# easier. mAP50 is the headline (a box counts as correct at 50% overlap), but
# watch mAP50-95 too: Objective 3 crops each detected box and hands it to the
# classifier, so a loose box that swallows a neighbouring brick classifies badly
# even when mAP50 looks fine.
DETECTION_MAP_TARGET = 0.85

# --- Objective 2: prediction (predict.py) ---
# Minimum confidence for a box to count as a brick. 0.25 is ultralytics'
# default and a reasonable demo setting: low enough not to miss bricks in a
# crowded photo, high enough to keep the background out of the inventory.
DETECTION_CONFIDENCE = float(os.getenv("DETECTION_CONFIDENCE", 0.25))
# How much to grow each box before cropping it for the classifier, as a
# fraction of the box. The detector draws tight boxes; the classifier was
# trained on images where the brick sits inside a little margin, so feeding
# it an edge-to-edge crop is a small train/serve mismatch. 0.10 puts that
# margin back. Clamped to the image, so a brick at the edge is never padded
# out of bounds.
DETECTION_CROP_PADDING = float(os.getenv("DETECTION_CROP_PADDING", 0.10))

# --- Bonus: Rebrickable ---
REBRICKABLE_API_KEY = os.getenv("REBRICKABLE_API_KEY", "")
REBRICKABLE_BASE_URL = "https://rebrickable.com/api/v3"
