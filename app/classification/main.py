"""
Objective 1 -- classification entry point.

Regroups the four classification modules in one place, run top to bottom
like a script rather than through a CLI framework -- edit the calls in the
`if __name__ == "__main__":` block below and run this file directly:

    python -m app.classification.main

Same idea as running notebook cells in order: comment out the steps you
don't want this run, change arguments in place, rerun.
"""

from app.classification.dataset import get_datasets
from app.classification.evaluate import evaluate
from app.classification.predict import predict_image
from app.classification.train import describe_splits, train_model


def prepare_data() -> None:
    """Build the train/val/test splits and print a summary, without training.

    Worth running on its own before committing to a long training run: are
    the 50 classes what you expect, is cap_renders() actually thinning the
    training split, does every split have a sensible photos/renders mix.

    Side effect worth knowing about: get_datasets() writes CLASS_NAMES_PATH
    (models/class_names.json) as it runs -- so this also happens to be what
    makes `predict` able to translate a class index back into a part ID,
    even before a model has been trained on this machine.
    """
    _train_ds, _val_ds, _test_ds, (train_df, val_df, test_df) = get_datasets()
    describe_splits(train_df, val_df, test_df)


def predict(image_path: str, top_k: int = 3) -> None:
    """Classify one image and print its top-k predicted part IDs."""
    print(f"Predictions for {image_path}:")
    for rank, (part_id, confidence) in enumerate(
        predict_image(image_path, top_k=top_k), start=1
    ):
        print(f"  {rank}. {part_id:<12} {confidence:.2%}")


if __name__ == "__main__":
    prepare_data()
    train_model()
    evaluate()
    # predict("path/to/a/brick/photo.jpg")  # uncomment once a model exists
