"""
Training entry point -- for the VM and for a laptop.

Same idea as app/classification/main.py -- plain sequential calls, run top to
bottom like notebook cells -- with two additions:

  * the trained model is pushed to the GCS bucket via registry.save_model(),
    because the VM is disposable and its disk is not a place to keep anything;

  * a summary at the end puts train / val / test accuracy side by side, so one
    screenful tells you where the run landed.

Run it through the Makefile, which is also what labels the run:

    make run_vm IMG_SIZE=256    # on the GCP VM  -> classifier_christophe_vm_...
    make run_local              # on a laptop    -> classifier_christophe_local_...
    make run_local MODEL_NAME=oriane            -> classifier_oriane_local_...

The two targets differ only in the MACHINE label they pass; the training
itself is identical, so there is one script rather than two that drift.

Each run is saved as its own folder, locally and in the bucket:

    models/classifier_christophe_vm_256x256_20260827-154512/
        classifier.keras
        class_names.json
        history.json

The timestamp in the folder name means nothing is ever overwritten.
"""

import os
import time

from app.classification.evaluate import evaluate, summarise
from app.classification.registry import save_model
from app.classification.train import train_model
from app.params import IMG_SIZE, MODEL_NAME


def stage(number: int, title: str) -> None:
    """A visible marker between the run's three long phases.

    A training run prints thousands of lines; these are what let you scroll
    back and find where each phase started.
    """
    print(f"\n{'=' * 60}\n  STEP {number}/3 -- {title}\n{'=' * 60}\n")


if __name__ == "__main__":
    # Which machine this run happened on, for the archive name. Set by the
    # Makefile: `make run_vm` -> "vm", `make run_local` -> "local". Defaults
    # to "local" so a bare `python run_training.py` never mislabels a laptop
    # run as having come from the VM.
    machine = os.getenv("MACHINE", "local")

    print(f"\n🧱 LegoFitter -- {machine} training run, model_{MODEL_NAME} at {IMG_SIZE[0]}x{IMG_SIZE[1]}\n")

    # train_model() returns the BEST weights, not the last epoch's --
    # EarlyStopping(restore_best_weights=True) rolls them back before returning.
    # It also builds the splits and prints their composition, so there is no
    # need to call prepare_data() first: that would rebuild the entire dataset
    # a second time for nothing.
    stage(1, "TRAIN")
    model, history = train_model()

    stage(2, "ARCHIVE TO BUCKET")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = save_model(
        model,
        name=f"classifier_{MODEL_NAME}_{machine}_{IMG_SIZE[0]}x{IMG_SIZE[1]}_{timestamp}",
    )

    stage(3, "EVALUATE")
    test_all, test_photos = evaluate()
    summarise(test_all, test_photos, history)

    print(f"✅ Run complete -- saved as: {run_name}")
    print(f"   reload it anywhere with: registry.load_model(\"{run_name}\")\n")
