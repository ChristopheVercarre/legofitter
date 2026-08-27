"""
Training entry point for the GCP VM.

Same idea as app/classification/main.py -- plain sequential calls, run top to
bottom like notebook cells -- with one addition: after training, the model is
pushed to the GCS bucket via registry.save_model(). Without that step the
weights only ever exist on the VM's disk, and the VM is disposable.

Run it through the Makefile so the image size is explicit in the command:

    make run_vm                 # uses IMG_SIZE from .env, or 128
    make run_vm IMG_SIZE=256

Saved to the bucket as models/classifier_vm_{size}_{timestamp}.keras -- the
timestamp keeps every run, so nothing is ever overwritten.
"""

import time

from app.classification.main import prepare_data
from app.classification.train import train_model
from app.classification.evaluate import evaluate
from app.classification.registry import save_model
from app.params import IMG_SIZE


if __name__ == "__main__":
    prepare_data()

    # train_model() returns the BEST weights, not the last epoch's --
    # EarlyStopping(restore_best_weights=True) rolls them back before returning.
    model, history = train_model()

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_model(model, name=f"classifier_vm_{IMG_SIZE[0]}x{IMG_SIZE[1]}_{timestamp}")

    evaluate()
