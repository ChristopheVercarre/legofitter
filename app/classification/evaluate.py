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

from keras.models import load_model as load_keras_model

from app.classification import registry
from app.classification.dataset import create_dataset, get_datasets
from app.params import CLASSIFICATION_ACCURACY_TARGET, CLASSIFICATION_MODEL_PATH


def load_trained_model():
    """Load the model to evaluate: local checkpoint first, GCS bucket second.

    There are two places a trained model can live, and they answer
    different questions:

      - CLASSIFICATION_MODEL_PATH (models/classifier.keras) is the checkpoint
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
        print(f"Loading local checkpoint: {CLASSIFICATION_MODEL_PATH}")
        return load_keras_model(CLASSIFICATION_MODEL_PATH)

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


def check_gate(accuracy: float, target: float = CLASSIFICATION_ACCURACY_TARGET) -> bool:
    """Compare photos-only test accuracy against the Objective 1 gate.

    Gated on PHOTOS accuracy specifically, not the mixed test set. Same
    reasoning as train.py's PhotosOnlyMetric callback: cap_renders() only
    thins renders out of the TRAINING split (see dataset.py), so the test
    split still sits at the raw ~5:1 render:photo ratio. "Test accuracy"
    on that mix mostly measures how well we classify clean synthetic
    renders. The Objective 3 pipeline will only ever hand this model real
    photo crops from the detector, so the 70% gate has to be measured on
    real photos or it doesn't mean anything.
    """
    passed = accuracy >= target
    verdict = "PASSED" if passed else "FAILED"
    print(f"\nObjective 1 gate ({target:.0%} on real photos): {accuracy:.2%} -> {verdict}")
    if not passed:
        print("Do not start Objective 2 until this clears the gate.")
    return passed


def evaluate() -> bool:
    """Load the trained model, score it on the test split, check the gate.

    Returns the gate's pass/fail bool so a caller (main_local.py, a
    notebook) can branch on it without re-parsing the printed output.
    """
    model = load_trained_model()

    # Rebuilding the splits here (rather than reusing train.py's in-memory
    # test_dataset) keeps evaluate.py runnable on its own, any time, without
    # depending on train.py's process still being alive. get_datasets() uses
    # the same RANDOM_STATE every call, so this reproduces the exact same
    # held-out test split train.py never trained on -- not a fresh random one.
    _train_ds, _val_ds, test_dataset, (_train_df, _val_df, test_df) = get_datasets()

    print("Evaluating on the held-out test split:")
    evaluate_model(model, test_dataset, label="test (all)")

    photos_test_df = test_df[test_df["source"] == "photos"]
    photos_test_dataset = create_dataset(photos_test_df)
    _loss, photos_accuracy = evaluate_model(
        model, photos_test_dataset, label="test (photos only)"
    )

    return check_gate(photos_accuracy)


if __name__ == "__main__":
    # Lets you run this file directly:
    #     python -m app.classification.evaluate
    # from the project root, once a model exists locally or in the bucket.
    evaluate()
