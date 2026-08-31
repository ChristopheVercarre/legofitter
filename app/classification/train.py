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
import time

from keras.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    LearningRateScheduler
)

from app.classification.dataset import BLUR_STATS, create_dataset, get_datasets
from app.classification.models import load_architecture
from app.utils.format import format_duration
from app.params import (
    CLASSIFICATION_ACCURACY_TARGET,
    CLASSIFICATION_MODEL_PATH,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    FINETUNE_EPOCHS,
    FINETUNE_LEARNING_RATE,
    HISTORY_PATH,
    IMG_SIZE,
    LEARNING_RATE,
    LR_DECAY_EVERY,
    LR_DECAY_FACTOR,
    MIN_LEARNING_RATE,
    MODEL_NAME,
    NUM_CLASSES,
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

def step_decay(epoch: int, lr: float) -> float:
    """LR = LEARNING_RATE * LR_DECAY_FACTOR ** (epoch // LR_DECAY_EVERY).

    At the defaults (a fifth every 30 epochs, epoch being 0-based): 1e-3 for
    epochs 0-29, 2e-4 for 30-59, 4e-5 for 60-89, 8e-6 for 90-119.

    That is tuned for EPOCHS=100 and does NOT scale: the same schedule over 300
    epochs reaches 6e-8 by epoch 180, where training continues to burn GPU
    hours while learning nothing. Raise LR_DECAY_EVERY with EPOCHS -- roughly
    EPOCHS // 5 keeps the shape of the curve you already know.

    Recomputed from LEARNING_RATE rather than from `lr` on purpose, so the
    schedule is idempotent. Keras passes the CURRENT rate in as `lr`, and
    deriving the next value from it would compound the decay every epoch
    instead of every LR_DECAY_EVERY.
    """
    return LEARNING_RATE * (LR_DECAY_FACTOR ** (epoch // LR_DECAY_EVERY))


def finetune_lr(epoch: int, lr: float) -> float:
    """Phase 2's schedule: a flat, tiny rate.

    Deliberately NOT step_decay. Phase 2 continues the epoch count from where
    phase 1 stopped, so step_decay would read that high epoch number and apply
    a decay meant for a long run -- and, worse, it recomputes from
    LEARNING_RATE every epoch, which would silently undo the 100x drop that
    unfreeze_top() just set.
    """
    return FINETUNE_LEARNING_RATE

def build_callbacks(photos_val_dataset=None, finetuning=False, best_so_far=None) -> list:
    """The four things that run between epochs.

    `finetuning` swaps the learning-rate schedule for phase 2's flat rate.
    `best_so_far` is phase 1's best val_loss, handed to ModelCheckpoint so
    phase 2 can only overwrite the saved weights by actually beating phase 1.
    Without it the checkpoint's baseline resets on the second fit(), and a
    fine-tune that made things WORSE would still overwrite a better model.
    """
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
            initial_value_threshold=best_so_far,
            verbose=1,
        ),

        # Halve the learning rate when val_loss stalls. Early on, big steps
        # make fast progress; later they overshoot the minimum and val_loss
        # oscillates. Note this patience is deliberately SHORTER than
        # EarlyStopping's, so the model gets a chance to improve at a finer
        # step size before we give up on it entirely.
        # ReduceLROnPlateau(
        #     monitor="val_loss",
        #     factor=REDUCE_LR_FACTOR,
        #     patience=REDUCE_LR_PATIENCE,
        #     min_lr=MIN_LEARNING_RATE,
        #     verbose=1,
        # ),
        LearningRateScheduler(finetune_lr if finetuning else step_decay, verbose=1),

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


def merge_histories(first, second):
    """Glue phase 2's metrics onto phase 1's so the run has ONE history.

    Returns `first` with its lists extended, so everything downstream --
    save_history(), summarise(), the notebook curves -- keeps working on a
    single object and shows one continuous training curve.

    A metric present in only one phase is padded with None for the epochs it
    did not exist, so every list stays the same length as the epoch count.
    """
    epochs_first = len(next(iter(first.history.values())))
    epochs_second = len(next(iter(second.history.values())))

    for metric in set(first.history) | set(second.history):
        before = first.history.get(metric, [None] * epochs_first)
        after = second.history.get(metric, [None] * epochs_second)
        first.history[metric] = list(before) + list(after)

    return first


def save_history(history) -> None:
    """Dump history.history to JSON so the curves outlive the process.

    Keras stores metrics as numpy float32, which json cannot serialise, so
    everything is cast to plain Python floats on the way out.
    """
    serialisable = {
        metric: [float(value) for value in values]
        for metric, values in history.history.items()
    }

    # Split composition and wall-clock time, when train_model() attached them.
    # A separate key, not a metric: everything else here is a list of floats.
    run_info = getattr(history, "run_info", None)
    if run_info is not None:
        serialisable["run_info"] = run_info

    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(serialisable, indent=2))

    print(f"✅ Training history saved: {HISTORY_PATH}")


def describe_splits(train_df, val_df, test_df) -> dict:
    """Print how many photos vs renders ended up in each split.

    Worth seeing in the log before a long run: it is the quickest way to spot
    that the dataset did not copy properly, or that cap_renders() did not fire.
    """
    print("\nSplit composition (photos / renders):")
    composition = {}
    for name, dataframe in (("train", train_df), ("val", val_df), ("test", test_df)):
        counts = dataframe["source"].value_counts()
        photos = int(counts.get("photos", 0))
        renders = int(counts.get("renders", 0))
        composition[name] = {"photos": photos, "renders": renders, "total": len(dataframe)}
        print(f"  {name:<6} {photos:>7,} photos  {renders:>7,} renders  ({len(dataframe):>7,} total)")
    print()

    # Returned as well as printed so run_training's final summary can repeat it
    # hours later, at the bottom of a long log, without rebuilding the splits.
    return composition


def train_model():
    """Build the datasets, build the model, fit it, save everything.

    Returns (model, history) so a notebook can plot straight away; a detached
    run just ignores the return value and reads the files from disk.
    """
    # The architecture MODEL_NAME points at. Resolved here rather than at
    # import time so a bad MODEL_NAME fails when you start a run, with a clear
    # message, instead of when someone merely imports this module.
    architecture = load_architecture()

    # Must happen before any layer is created, hence before initialize_model().
    if USE_MIXED_PRECISION:
        architecture.enable_mixed_precision()
    else:
        # Printed rather than left silent: "did mixed precision actually turn
        # on?" is the first question when a GPU run is slower than expected.
        print("✅ Mixed precision OFF (float32 compute)")

    # get_datasets() applies cap_renders() to the training split only, so
    # train is ~1.84:1 renders:photos while val/test keep the natural ~5:1.
    train_dataset, val_dataset, _test_dataset, splits = get_datasets()
    train_df, val_df, test_df = splits
    composition = describe_splits(train_df, val_df, test_df)

    # A photos-only view of the validation split, for the callback above.
    # Same underlying images, just filtered — no extra data is loaded.
    photos_val_dataset = create_dataset(val_df[val_df["source"] == "photos"])

    model = architecture.compile_model(architecture.initialize_model())
    model.summary()

    print(f"\nStarting training (up to {EPOCHS} epochs)...\n")

    started = time.perf_counter()
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=build_callbacks(photos_val_dataset),
    )

    epochs_run = len(history.history["loss"])
    print(f"\n✅ Phase 1 finished after {epochs_run} epoch(s)")

    # --- Phase 2: fine-tuning -------------------------------------------
    # Only transfer-learning architectures offer unfreeze_top(). The custom
    # CNNs train every weight from the first step, so there is nothing to
    # unfreeze and this block is skipped entirely -- their behaviour is
    # unchanged.
    if hasattr(architecture, "unfreeze_top"):
        best_so_far = min(history.history["val_loss"])
        print(f"\nStarting phase 2 (up to {FINETUNE_EPOCHS} more epochs, "
              f"lr={FINETUNE_LEARNING_RATE})...\n")

        architecture.unfreeze_top(model)

        # initial_epoch continues the count so the two phases read as one run
        # in history.json rather than two overlapping curves.
        finetune_history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs_run + FINETUNE_EPOCHS,
            initial_epoch=epochs_run,
            callbacks=build_callbacks(
                photos_val_dataset, finetuning=True, best_so_far=best_so_far
            ),
        )

        history = merge_histories(history, finetune_history)
        best_now = min(history.history["val_loss"])
        verdict = "improved on" if best_now < best_so_far else "did NOT beat"
        print(f"\n✅ Phase 2 finished -- best val_loss {best_now:.4f}, "
              f"{verdict} phase 1's {best_so_far:.4f}")

    epochs_run = len(history.history["loss"])
    training_seconds = time.perf_counter() - started
    print(f"\n✅ Training finished after {epochs_run} epoch(s) in total, "
          f"{format_duration(training_seconds)}")

    # Facts about the run that the metric curves do not carry. Attached to the
    # History OBJECT rather than into history.history, so merge_histories() and
    # every notebook that iterates the curves keep seeing only lists of floats.
    history.run_info = {
        "model_name": MODEL_NAME,
        "img_size": list(IMG_SIZE),
        "num_classes": NUM_CLASSES,
        "blur": dict(BLUR_STATS),
        "splits": composition,
        "training_seconds": training_seconds,
        "epochs": epochs_run,
    }

    save_history(history)
    print(f"✅ Best weights saved: {CLASSIFICATION_MODEL_PATH}")

    return model, history


if __name__ == "__main__":
    # Lets you run this file directly:
    #     python -m app.classification.train
    # from the project root, without going through main_local.py.
    train_model()
