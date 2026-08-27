"""
Objective 1, Step 2 (part 2) — training loop.

Runs the whole training job end to end:

    dataset.py  ->  model.py  ->  model.fit()  ->  weights + history on disk

Designed to be run detached (`nohup python -m app.classification.train &`),
which is why two things happen here that a notebook gives you for free:

  * ModelCheckpoint writes the best weights to disk every time they improve.
    EarlyStopping(restore_best_weights=True) only keeps them in MEMORY, so a
    crash three hours in would otherwise lose everything.

  * The History object is dumped to JSON. In a notebook `history` survives in
    the kernel; in a script it dies with the process, and you cannot plot
    curves you no longer have.
"""

import json

from keras.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from app.classification.dataset import create_dataset, get_datasets
from app.classification.model import (
    compile_model,
    enable_mixed_precision,
    initialize_model,
)
from app.params import (
    CLASSIFICATION_ACCURACY_TARGET,
    CLASSIFICATION_MODEL_PATH,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    HISTORY_PATH,
    MIN_LEARNING_RATE,
    REDUCE_LR_FACTOR,
    REDUCE_LR_PATIENCE,
    USE_MIXED_PRECISION,
)


class PhotosOnlyMetric(Callback):
    """Report accuracy on real photos only, once per epoch.

    Why this exists: after the render/photo split, our validation set is still
    roughly 84% synthetic renders. So `val_accuracy` mostly answers "how well
    do we classify renders" — but the Objective 1 gate (70%) is about real
    photographs, because that is what the Objective 3 detector will feed us.

    Writing into `logs` rather than just printing means the metric lands in
    history.history alongside the built-in ones, so it gets saved and plotted
    like everything else — and could even be monitored by EarlyStopping.
    """

    def __init__(self, photos_dataset):
        super().__init__()
        self.photos_dataset = photos_dataset

    def on_epoch_end(self, epoch, logs=None):
        loss, accuracy = self.model.evaluate(self.photos_dataset, verbose=0)

        logs = logs if logs is not None else {}
        logs["val_photos_loss"] = loss
        logs["val_photos_accuracy"] = accuracy

        gate = " <-- clears the 70% gate" if accuracy >= CLASSIFICATION_ACCURACY_TARGET else ""
        print(f"    val_photos_accuracy: {accuracy:.4f}{gate}")


def build_callbacks(photos_val_dataset=None) -> list:
    """The four things that run between epochs."""
    CLASSIFICATION_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    callbacks = []

    # Deliberately FIRST: Keras runs callbacks in list order, so putting this
    # ahead of the others means val_photos_accuracy is already in `logs` by the
    # time they run — which is what would let EarlyStopping monitor it later
    # if you decide the photos metric is the one worth stopping on.
    if photos_val_dataset is not None:
        callbacks.append(PhotosOnlyMetric(photos_val_dataset))

    callbacks += [
        # Writes weights to disk whenever val_loss improves. This is our crash
        # insurance and our deliverable — CLASSIFICATION_MODEL_PATH is what
        # evaluate.py and the Objective 3 pipeline will load.
        ModelCheckpoint(
            filepath=str(CLASSIFICATION_MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),

        # Halve the learning rate when val_loss stalls. Early on, big steps
        # make fast progress; later they overshoot the minimum and val_loss
        # oscillates. Note this patience is deliberately SHORTER than
        # EarlyStopping's, so the model gets a chance to improve at a finer
        # step size before we give up on it entirely.
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=MIN_LEARNING_RATE,
            verbose=1,
        ),

        # Stop once val_loss has not improved for N epochs, and roll the
        # weights back to the best epoch rather than keeping whatever the
        # last (possibly worse) epoch produced.
        EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    print(f"✅ Callbacks ready ({len(callbacks)} active)")
    return callbacks


def save_history(history) -> None:
    """Dump history.history to JSON so the curves outlive the process.

    Keras stores metrics as numpy float32, which json cannot serialise, so
    everything is cast to plain Python floats on the way out.
    """
    serialisable = {
        metric: [float(value) for value in values]
        for metric, values in history.history.items()
    }

    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(serialisable, indent=2))

    print(f"✅ Training history saved: {HISTORY_PATH}")


def describe_splits(train_df, val_df, test_df) -> None:
    """Print how many photos vs renders ended up in each split.

    Worth seeing in the log before a long run: it is the quickest way to spot
    that the dataset did not copy properly, or that cap_renders() did not fire.
    """
    print("\nSplit composition (photos / renders):")
    for name, dataframe in (("train", train_df), ("val", val_df), ("test", test_df)):
        counts = dataframe["source"].value_counts()
        photos = int(counts.get("photos", 0))
        renders = int(counts.get("renders", 0))
        print(f"  {name:<6} {photos:>7,} photos  {renders:>7,} renders  ({len(dataframe):>7,} total)")
    print()


def train_model():
    """Build the datasets, build the model, fit it, save everything.

    Returns (model, history) so a notebook can plot straight away; a detached
    run just ignores the return value and reads the files from disk.
    """
    # Must happen before any layer is created, hence before initialize_model().
    if USE_MIXED_PRECISION:
        enable_mixed_precision()

    # get_datasets() applies cap_renders() to the training split only, so
    # train is ~1.84:1 renders:photos while val/test keep the natural ~5:1.
    train_dataset, val_dataset, _test_dataset, splits = get_datasets()
    train_df, val_df, test_df = splits
    describe_splits(train_df, val_df, test_df)

    # A photos-only view of the validation split, for the callback above.
    # Same underlying images, just filtered — no extra data is loaded.
    photos_val_dataset = create_dataset(val_df[val_df["source"] == "photos"])

    model = compile_model(initialize_model())
    model.summary()

    print(f"\nStarting training (up to {EPOCHS} epochs)...\n")

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=build_callbacks(photos_val_dataset),
    )

    epochs_run = len(history.history["loss"])
    print(f"\n✅ Training finished after {epochs_run} epoch(s)")

    save_history(history)
    print(f"✅ Best weights saved: {CLASSIFICATION_MODEL_PATH}")

    return model, history


if __name__ == "__main__":
    # Lets you run this file directly:
    #     python -m app.classification.train
    # from the project root, without going through main_local.py.
    train_model()
