"""
Objective 1, Step 3 -- evaluation.

Owns: loading the trained Keras model -- either the local checkpoint
train.py just wrote, or the latest one saved to the GCS bucket via
registry.py -- and scoring top-1 accuracy on the held-out test split
(model.evaluate), reporting whether we've crossed
app/params.CLASSIFICATION_ACCURACY_TARGET (70%). This is the gate before
we move on to Objective 2.

Run directly:
    python -m app.classification.evaluate
Works either right after a local training run, or on a machine that never
trained anything itself but has a model in the bucket (see
load_trained_model() below).
"""

import json

from keras.models import load_model as load_keras_model

from app.classification import registry
from app.classification.dataset import create_dataset, get_datasets
from app.utils.format import format_duration
from app.classification.registry import model_input_size
from app.params import (
    CLASSIFICATION_ACCURACY_TARGET,
    CLASSIFICATION_MODEL_PATH,
    HISTORY_PATH,
)


def load_trained_model():
    """Load the model to evaluate: local checkpoint first, GCS bucket second.

    There are two places a trained model can live, and they answer
    different questions:

      - CLASSIFICATION_MODEL_PATH (models/current/classifier.keras) is the checkpoint
        THIS machine's train.py just wrote via ModelCheckpoint. Loading it is
        free (no network call) and it's the freshest thing this machine has
        trained.

      - registry.load_model() downloads the most recently uploaded model
        from the GCS bucket (see registry.save_model()). `models/` is
        gitignored, so a fresh clone -- or evaluating on your Mac after
        training happened on the VM -- has no local checkpoint at all;
        the bucket is what carries a trained model between machines.

    Local wins when both exist: it's guaranteed to be the most recent thing
    trained HERE, whereas the bucket only has whatever was last explicitly
    pushed with save_model() (train.py's ModelCheckpoint does not do that
    automatically -- pushing to the bucket is still a separate, manual step).
    """
    if CLASSIFICATION_MODEL_PATH.exists():
        print(f"✅ Loaded local checkpoint: {CLASSIFICATION_MODEL_PATH}")
        model = load_keras_model(CLASSIFICATION_MODEL_PATH)
        # Pair the model with the class list sitting NEXT TO IT, and carry the
        # pair as one object. predict_image() reads model.class_names, so the
        # mapping travels with the model instead of being looked up from
        # global state at prediction time -- see registry.load_model(), which
        # attaches it the same way.
        registry.attach_class_names(model, CLASSIFICATION_MODEL_PATH.parent)
        return model

    print(f"No local checkpoint at {CLASSIFICATION_MODEL_PATH} -- checking the GCS bucket instead.")
    model = registry.load_model()

    if model is None:
        # registry.load_model() already prints its own "no model in bucket"
        # message and returns None rather than raising -- see registry.py.
        # We turn that into a hard failure here because evaluate() has
        # nothing useful to do without a model.
        raise FileNotFoundError(
            "No trained model found locally or in the GCS bucket. "
            "Run train.py first (python -m app.classification.train), or "
            "confirm BUCKET_NAME in .env points at a bucket that has one."
        )

    return model


def evaluate_model(model, dataset, label: str = "test") -> tuple[float, float]:
    """Run model.evaluate() on `dataset`, print and return (loss, accuracy).

    `label` is only for the printout -- it's what tells "test (all)" apart
    from "test (photos only)" below, since they're two different numbers.
    """
    loss, accuracy = model.evaluate(dataset, verbose=0)
    print(f"  {label:<20} loss={loss:.4f}  accuracy={accuracy:.4f}")
    return loss, accuracy


def _load_curves(history=None) -> dict | None:
    """The training curves as a plain dict, from memory or from disk.

    `history` is whatever train_model() returned -- a Keras History object.
    Left as None (the usual case when evaluate.py is run on its own, hours or
    machines away from the training run) this falls back to the history.json
    that save_history() wrote, so a standalone evaluation can still report the
    train and val numbers rather than only the test ones.
    """
    if history is not None:
        curves = history.history if hasattr(history, "history") else dict(history)
        # run_info lives on the History OBJECT, not inside history.history --
        # merged in here (as a copy) so an in-memory summary shows the same
        # extra facts as one read back from history.json.
        run_info = getattr(history, "run_info", None)
        return {**curves, "run_info": run_info} if run_info else curves

    if HISTORY_PATH.exists():
        return json.loads(HISTORY_PATH.read_text())

    return None


def summarise(test_all: float, test_photos: float, history=None) -> bool:
    """Print train / val / test accuracy side by side, then the verdict.

    Returns True if the model clears CLASSIFICATION_ACCURACY_TARGET.

    Judged on photos only, not the mixed test set: cap_renders() thins renders
    out of the TRAINING split alone (see dataset.py), so the test split keeps
    the raw ~5:1 render:photo ratio. Accuracy on that mix mostly measures how
    well we classify clean synthetic renders, whereas the Objective 3 pipeline
    will only ever hand this model real photo crops from the detector.

    Which epoch's train/val numbers to quote is not obvious either:
    EarlyStopping runs with restore_best_weights=True, so the weights that got
    saved and evaluated are the BEST epoch's, not the last one's. Quoting the
    last epoch would describe weights that were thrown away -- flattering on
    train accuracy, worse on val. So the epoch that actually won (lowest
    val_loss) is the one reported, keeping all four numbers describing the
    same set of weights.
    """
    curves = _load_curves(history)

    print("\n" + "=" * 60)
    print("Run summary")
    print("=" * 60)

    # What this run was configured as. Read from run_info rather than params,
    # so summarising an OLD run reports the settings it was TRAINED with, not
    # whatever this machine happens to be set to now.
    config = (_load_curves(history) or {}).get("run_info") or {}
    if config.get("model_name"):
        width, height = config.get("img_size") or (None, None)
        print(f"  Model               model_{config['model_name']}.py")
        if width:
            print(f"  Input size          {width}x{height}")
        if config.get("num_classes"):
            print(f"  Classes             {config['num_classes']}")
        print("-" * 60)

    if curves and "val_loss" in curves:
        val_losses = curves["val_loss"]
        best = min(range(len(val_losses)), key=lambda i: val_losses[i])

        print(f"  Best epoch          {best + 1} of {len(val_losses)}")
        print(f"  Train accuracy      {curves['accuracy'][best]:.4f}")
        print(f"  Val accuracy        {curves['val_accuracy'][best]:.4f}")

        # The val split is ~84% renders, so the line above is mostly "how well
        # do we classify renders". This is the same split, photos only, and is
        # the one comparable to the photos-only test number below.
        if "val_photos_accuracy" in curves:
            print(f"  Val acc (photos)    {curves['val_photos_accuracy'][best]:.4f}")

        print("-" * 60)
    else:
        print(f"  (no training curves at {HISTORY_PATH} -- test scores only)")
        print("-" * 60)

    # Only present for runs trained since run_info was added; older runs and
    # hand-saved models simply skip these two blocks.
    run_info = (curves or {}).get("run_info")
    if run_info:
        splits = run_info.get("splits") or {}
        if splits:
            print(f"  {'Split':<8}{'photos':>10}{'renders':>10}{'total':>10}")
            totals = {"photos": 0, "renders": 0, "total": 0}
            for name in ("train", "val", "test"):
                counts = splits.get(name)
                if counts:
                    print(f"  {name:<8}{counts['photos']:>10,}"
                          f"{counts['renders']:>10,}{counts['total']:>10,}")
                    for key in totals:
                        totals[key] += counts[key]

            # The dataset the run actually saw, after select_classes(),
            # filter_blurry() and cap_renders() -- not the raw folder counts.
            print(f"  {'':<8}{'-' * 10}{'-' * 10}{'-' * 10}")
            print(f"  {'total':<8}{totals['photos']:>10,}"
                  f"{totals['renders']:>10,}{totals['total']:>10,}")

        # Photos the blur filter threw away before any of the above was split.
        blur = run_info.get("blur") or {}
        if blur.get("photos_checked"):
            share = blur["dropped"] / blur["photos_checked"] * 100
            print(f"  Blurry dropped      {blur['dropped']:,} of "
                  f"{blur['photos_checked']:,} photos ({share:.1f}%), "
                  f"threshold {blur['threshold']}")
        print("-" * 60)

        seconds = run_info.get("training_seconds")
        if seconds:
            epochs = run_info.get("epochs") or 0
            per_epoch = f", {format_duration(seconds / epochs)}/epoch" if epochs else ""
            print(f"  Training time       {format_duration(seconds)}"
                  f"  ({epochs} epochs{per_epoch})")
            print("-" * 60)

    print(f"  Test accuracy       {test_all:.4f}   (all: photos + renders)")
    print(f"  Test accuracy       {test_photos:.4f}   (real photos only)")
    print(f"  Target              {CLASSIFICATION_ACCURACY_TARGET:.4f}")
    print("-" * 60)

    passed = test_photos >= CLASSIFICATION_ACCURACY_TARGET
    margin = (test_photos - CLASSIFICATION_ACCURACY_TARGET) * 100
    print(f"  {'PASSED' if passed else 'FAILED'}  ({margin:+.2f} points on real photos)")
    print("=" * 60 + "\n")

    return passed


def evaluate() -> tuple[float, float]:
    """Load the trained model and score it on the held-out test split.

    Returns (all_accuracy, photos_accuracy) so the caller decides what to
    print and what to compare against the target -- see summarise(), which
    is what turns these two numbers into a report.
    """
    model = load_trained_model()

    # Resize the test images to what THIS model expects, not to whatever
    # IMG_SIZE this machine happens to be configured for. Scoring an old
    # 256x256 model on a 128-configured laptop then just works.
    img_size = model_input_size(model)

    # Rebuilding the splits here (rather than reusing train.py's in-memory
    # test_dataset) keeps evaluate.py runnable on its own, any time, without
    # depending on train.py's process still being alive. get_datasets() uses
    # the same RANDOM_STATE every call, so this reproduces the exact same
    # held-out test split train.py never trained on -- not a fresh random one.
    _train_ds, _val_ds, test_dataset, (_train_df, _val_df, test_df) = get_datasets(
        img_size=img_size
    )

    print(f"Evaluating on the held-out test split ({img_size[0]}x{img_size[1]}):")
    _loss, all_accuracy = evaluate_model(model, test_dataset, label="test (all)")

    photos_test_df = test_df[test_df["source"] == "photos"]
    photos_test_dataset = create_dataset(photos_test_df, img_size=img_size)
    _loss, photos_accuracy = evaluate_model(
        model, photos_test_dataset, label="test (photos only)"
    )

    print("✅ Evaluation complete")
    return all_accuracy, photos_accuracy


if __name__ == "__main__":
    # Lets you run this file directly:
    #     python -m app.classification.evaluate
    # from the project root, once a model exists locally or in the bucket.
    _all_accuracy, _photos_accuracy = evaluate()
    summarise(_all_accuracy, _photos_accuracy)
