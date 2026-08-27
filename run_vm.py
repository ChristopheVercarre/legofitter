"""
Training entry point for the GCP VM.

Same idea as app/classification/main.py -- plain sequential calls, run top to
bottom like notebook cells -- with two additions:

  * the trained model is pushed to the GCS bucket via registry.save_model(),
    because the VM is disposable and its disk is not a place to keep anything;

  * a summary at the end puts train / val / test accuracy side by side, so one
    screenful tells you where the run landed.

Run it through the Makefile so the image size is explicit in the command:

    make run_vm                 # uses IMG_SIZE from .env, or 128
    make run_vm IMG_SIZE=256

Each run is saved as its own folder, locally and in the bucket:

    models/classifier_vm_256x256_20260827-154512/
        classifier.keras
        class_names.json
        history.json

The timestamp in the folder name means nothing is ever overwritten.
"""

import time

from app.classification.evaluate import evaluate, summarise
from app.classification.registry import save_model
from app.classification.train import train_model
from app.params import IMG_SIZE


if __name__ == "__main__":
    # train_model() returns the BEST weights, not the last epoch's --
    # EarlyStopping(restore_best_weights=True) rolls them back before returning.
    # It also builds the splits and prints their composition, so there is no
    # need to call prepare_data() first: that would rebuild the entire dataset
    # a second time for nothing.
    model, history = train_model()

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = save_model(model, name=f"classifier_vm_{IMG_SIZE[0]}x{IMG_SIZE[1]}_{timestamp}")

    test_all, test_photos = evaluate()
    summarise(test_all, test_photos, history)

    print(f"Saved as run: {run_name}")
    print(f"  reload it anywhere with: registry.load_model(\"{run_name}\")\n")
