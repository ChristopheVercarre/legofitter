"""
Objective 2 -- detection entry point.

Regroups the detection modules in one place, run top to bottom like a script
rather than through a CLI framework -- edit the calls in the
`if __name__ == "__main__":` block below and run this file directly:

    python -m app.detection.main

Same idea as running notebook cells in order: comment out the steps you don't
want this run, change arguments in place, rerun.

The classification twin of this file is app/classification/main.py.
"""

from app.detection.evaluate import evaluate, summarise
from app.detection.prepare_data import prepare_data
from app.detection.predict import predict_boxes
from app.detection.train import train_model
from app.params import DETECTION_CONFIDENCE, YOLO_DATA_DIR, YOLO_DATASET_YAML


def describe_dataset() -> None:
    """Count what prepare_data() built, without rebuilding it.

    Worth a glance before committing to a long training run: do the image and
    label counts match in every split (a mismatch means pairs were dropped
    silently, and an image with no label teaches the detector that a picture
    full of bricks contains none), and is the split roughly 70/15/15.
    """
    if not YOLO_DATASET_YAML.exists():
        print(f"No dataset at {YOLO_DATA_DIR} -- run prepare_data() first")
        return

    print(f"\n{'split':<8}{'images':>9}{'labels':>9}{'boxes':>9}")
    totals = [0, 0, 0]
    for split in ("train", "val", "test"):
        images = list((YOLO_DATA_DIR / "images" / split).glob("*"))
        labels = list((YOLO_DATA_DIR / "labels" / split).glob("*.txt"))
        boxes = sum(
            len([line for line in path.read_text().splitlines() if line.strip()])
            for path in labels
        )
        flag = "" if len(images) == len(labels) else "   <-- MISMATCH"
        print(f"{split:<8}{len(images):>9,}{len(labels):>9,}{boxes:>9,}{flag}")
        totals = [a + b for a, b in zip(totals, (len(images), len(labels), boxes))]

    print(f"{'total':<8}{totals[0]:>9,}{totals[1]:>9,}{totals[2]:>9,}")
    if totals[0]:
        print(f"\n✅ {totals[2] / totals[0]:.2f} boxes per image on average")


def detect(image, confidence: float = DETECTION_CONFIDENCE) -> list[dict]:
    """Run the detector on one image, print what it found, and return it.

    A thin wrapper over predict.predict_boxes() -- the detection itself lives
    there, so the API and this file can never drift apart. This one adds the
    printing a notebook or a terminal wants, and hands the boxes back anyway
    so they can be fed straight to crop_boxes().
    """
    found = predict_boxes(image, confidence=confidence)

    print(f"\n{len(found)} brick(s) detected in {image}:")
    for rank, brick in enumerate(found, start=1):
        x_min, y_min, x_max, y_max = brick["box"]
        print(
            f"  {rank:>3}. confidence {brick['confidence']:.2%}   "
            f"box ({x_min}, {y_min}) -> ({x_max}, {y_max})"
        )
    return found


if __name__ == "__main__":
    prepare_data()
    describe_dataset()
    train_model()
    summarise(evaluate())
    # detect("path/to/a/photo/of/bricks.jpg")  # uncomment once a detector exists
