"""
Central params for LegoFitter, loaded from environment variables (see .env.example).

Every other module should read paths/constants from here rather than
hardcoding them, so the whole project stays configurable from one file + .env.
"""
from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()

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
CURRENT_RUN_DIR = MODELS_DIR / "current"
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
# used for their baseline training set. Set to None to train on the raw
# ~5:1 distribution instead.
RENDER_PHOTO_RATIO = 650 / 350

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
LEARNING_RATE = 1e-3            # Adam's default; ReduceLROnPlateau lowers it from here
EPOCHS = 100                    # upper bound — EarlyStopping normally stops us first
EARLY_STOPPING_PATIENCE = 30    # epochs without val_loss improvement before stopping
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
# (or a slowdown) on CPU / Apple Silicon — so leave False locally, True on the VM.
USE_MIXED_PRECISION = False

# Where train.py dumps history.history so the curves survive the process.
# A detached script has no notebook to hold `history` in memory.
HISTORY_PATH = CURRENT_RUN_DIR / "history.json"

# --- Objective 2: Object Detection ---
DETECTION_DATA_DIR = PROJECT_ROOT / os.getenv(
    "DETECTION_DATA_DIR", "data/lego-tagged-object_detection"
)
DETECTION_MODEL_PATH = MODELS_DIR / "detector.pt"

# --- Bonus: Rebrickable ---
REBRICKABLE_API_KEY = os.getenv("REBRICKABLE_API_KEY", "")
REBRICKABLE_BASE_URL = "https://rebrickable.com/api/v3"
