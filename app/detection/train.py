"""
Objective 2, Step 2 -- YOLO training.

Owns: training an ultralytics YOLO model on the dataset prepare_data.py built,
into a timestamped run folder, then publishing that run through
app.detection.registry.

Deliberately no argparse: every knob lives in params.py, the same as the
classifier. One place to look, and a run is reproducible from the repo alone
rather than from whatever flags someone typed.

Run it with:
    python -m app.detection.train
"""

import os
import time

from ultralytics import YOLO

from app.detection.evaluate import evaluate, summarise
from app.detection.registry import save_run
from app.params import (
    DETECTION_MODELS_DIR,
    YOLO_BASE_MODEL,
    YOLO_BATCH_SIZE,
    YOLO_DATASET_YAML,
    YOLO_EPOCHS,
    YOLO_IMG_SIZE,
    YOLO_PATIENCE,
)


def build_run_name() -> str:
    """detector_{machine}_{size}_{timestamp}.

    The detection twin of classifier_{model}_{machine}_{WxH}_{timestamp}.

    The timestamp is the point: without it every run lands in the same folder
    and silently overwrites the last one. MACHINE comes from the Makefile/env
    exactly as it does for the classifier, and defaults to "local" so a laptop
    run is never mislabelled as the VM's.
    """
    machine = os.environ.get("MACHINE", "local")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return f"detector_{machine}_{YOLO_IMG_SIZE}_{timestamp}"


def train_model():
    """Fine-tune YOLO on the LEGO dataset. Returns ultralytics' results object.

    YOLO_BASE_MODEL is pretrained on COCO, so this is transfer learning just
    like the VGG16 classifier: the model already knows what an object looks
    like, and only has to learn what a LEGO brick looks like.
    """
    if not YOLO_DATASET_YAML.exists():
        raise FileNotFoundError(
            f"{YOLO_DATASET_YAML} not found -- run this first:\n"
            f"    python -m app.detection.prepare_data"
        )

    run_name = build_run_name()
    DETECTION_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n🧱 LegoFitter -- YOLO detection, {YOLO_BASE_MODEL} at {YOLO_IMG_SIZE}px")
    print(f"   run: {run_name}\n")

    model = YOLO(YOLO_BASE_MODEL)

    # project + name point ultralytics straight at the run folder, so it writes
    # weights, plots and results.csv where the registry expects them. Without
    # this it writes models/runs/... and we would have to copy best.pt back out.
    #
    # exist_ok=False on purpose: a timestamped name should never collide, so if
    # it does, something is wrong and silently reusing the folder is the worst
    # possible answer.
    results = model.train(
        data=str(YOLO_DATASET_YAML.resolve()),
        epochs=YOLO_EPOCHS,
        imgsz=YOLO_IMG_SIZE,
        batch=YOLO_BATCH_SIZE,
        patience=YOLO_PATIENCE,
        project=str(DETECTION_MODELS_DIR.resolve()),
        name=run_name,
        exist_ok=False,
    )
    print("✅ Training finished")

    save_run(run_name)

    # Scored on the TEST split, which nothing in training ever saw -- the same
    # discipline as run_training.py's STEP 3/3 for the classifier.
    summarise(evaluate())

    print(f"\n✅ Run complete -- saved as: {run_name}")
    print(f'   reload it anywhere with: registry.load_detector("{run_name}")\n')

    return results


if __name__ == "__main__":
    train_model()
