.PHONY: prepare_classification prepare_detection \
        train_classification_vm train_classification_local \
        train_detection_vm train_detection_local \
        test

# =====================================================================
#  Shared knobs
# =====================================================================

# Override per run:  make train_classification_vm IMG_SIZE=256
# Falls back to .env, then to 128 (see app/params.py).
IMG_SIZE ?= 128

# Which architecture to train:  make train_classification_local MODEL_NAME=oriane
# Deliberately NOT given a default here: setting one would export MODEL_NAME on
# every run, and an exported variable beats .env (load_dotenv does not override
# what is already in the environment). Left unset, .env decides -- and if there
# is no .env either, params.py falls back to christophe.
MODEL_NAME_ENV = $(if $(MODEL_NAME),MODEL_NAME=$(MODEL_NAME))

# float16 on the T4's tensor cores. ON for the _vm targets because the VM
# always has the GPU -- nobody should have to remember this. Turn it off for
# one run with  make train_classification_vm USE_MIXED_PRECISION=false
# The _local targets deliberately do NOT set it: a laptop has no tensor cores,
# so there it stays off (params.py defaults to False).
USE_MIXED_PRECISION ?= true

# =====================================================================
#  Objective 1 -- classification
# =====================================================================

# Build the splits and print their composition, without training.
prepare_classification:
	IMG_SIZE=$(IMG_SIZE) python -c "from app.classification.main import prepare_data; prepare_data()"

# The only way to train a classifier. Trains, archives the run to the GCS
# bucket, evaluates.
#
# There is deliberately no "train without saving" target:
# models/classification/current/ is overwritten by every run, so a training run
# that is not archived by save_model() is lost the moment the next one starts.
train_classification_vm:
	IMG_SIZE=$(IMG_SIZE) MACHINE=vm USE_MIXED_PRECISION=$(USE_MIXED_PRECISION) $(MODEL_NAME_ENV) python run_training.py

# Same run on a laptop. Archived as classifier_<model>_local_... so a run
# trained off the VM is never mistaken for one that came from it.
train_classification_local:
	IMG_SIZE=$(IMG_SIZE) MACHINE=local $(MODEL_NAME_ENV) python run_training.py

# =====================================================================
#  Objective 2 -- detection
# =====================================================================

# VOC XML -> YOLO layout. No GPU needed; writes ~6 GB into
# data/lego-yolo-dataset/, so run it once per machine, not once per run.
prepare_detection:
	python -m app.detection.prepare_data

# Trains YOLO, archives the run, evaluates against DETECTION_MAP_TARGET.
#
# No IMG_SIZE / MODEL_NAME here: detection has its own knobs in params.py, and
# passing the classifier's would silently mean nothing. Three of them can be
# overridden per run:
#
#   make train_detection_vm YOLO_BASE_MODEL=yolo26s.pt YOLO_EPOCHS=100 YOLO_BATCH_SIZE=16
#
# No plumbing needed for that: make exports command-line variables to the
# recipe's environment automatically, and params.py reads them. Left unset,
# .env decides, then params.py's defaults.
train_detection_vm:
	MACHINE=vm python -m app.detection.train

train_detection_local:
	MACHINE=local python -m app.detection.train

# =====================================================================
#  Development
# =====================================================================

# Smoke tests -- run before pushing. Same suite GitHub Actions runs on PRs.
test:
	pytest tests/ -v
