"""
Objective 2, Step 3 -- detection evaluation.

Owns: scoring the detector on the held-out test split and turning the numbers
into a verdict against DETECTION_MAP_TARGET.

The classification twin of this file, function for function:

    evaluate()                  <- evaluate()
    summarise(scores)           <- summarise(test_all, test_photos)

Run it with:
    python -m app.detection.evaluate
"""

import csv

from app.detection.registry import load_detector
from app.params import (
    DETECTION_MAP_TARGET,
    DETECTION_RESULTS_PATH,
    YOLO_DATASET_YAML,
)


def evaluate(detector=None, run_name: str = None) -> dict:
    """Score the detector on the held-out test split. Returns the four metrics.

    `detector` is optional: pass one already in memory (a notebook that loaded
    it once) to avoid reloading. Left as None, load_detector() applies the
    usual local-first-then-bucket lookup, and `run_name` picks a specific run.

    Two details that are easy to get wrong:

    split="test" -- ultralytics defaults to "val", which the model saw during
    training and which EarlyStopping tuned against. The test split is the only
    honest number.

    data=YOLO_DATASET_YAML -- THIS machine's dataset yaml, deliberately not the
    copy inside the run folder. prepare_data() writes an absolute `path:` into
    the yaml, so a run trained on the VM carries the VM's paths; using its own
    copy here would point at a directory that does not exist on this machine.
    The archived copy is kept for its class map, not for its paths.
    """
    if detector is None:
        detector = load_detector(run_name)

    print(f"Evaluating on the held-out test split ({YOLO_DATASET_YAML}):")
    metrics = detector.val(data=str(YOLO_DATASET_YAML), split="test", verbose=False)

    scores = {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }
    print("✅ Evaluation complete")
    return scores


def _training_curve() -> dict | None:
    """Epochs trained and best mAP50, read from the run's results.csv.

    The detection answer to history.json. Defensive on purpose: ultralytics
    renames its csv columns between versions, so a column we cannot find means
    we skip this block rather than crash an evaluation over a cosmetic line.
    """
    if not DETECTION_RESULTS_PATH.exists():
        return None

    with open(DETECTION_RESULTS_PATH, newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        return None

    column = next(
        (name for name in rows[0] if name and "mAP50(B)" in name and "95" not in name),
        None,
    )
    if column is None:
        return None

    values = [float(row[column]) for row in rows if row.get(column)]
    if not values:
        return None

    best = max(range(len(values)), key=lambda i: values[i])
    return {"epochs": len(values), "best_epoch": best + 1, "best_map50": values[best]}


def summarise(scores: dict) -> bool:
    """Print the run's verdict. Returns True if it clears DETECTION_MAP_TARGET.

    Judged on mAP50: a box counts as a hit when it overlaps the real one by at
    least half. The other three are printed rather than gated because they
    answer different questions, and a single number would hide them:

      mAP50-95   the same idea at stricter overlaps -- how TIGHT the boxes are.
                 The one to watch for Objective 3, which crops each box and
                 hands it to the classifier.
      precision  of what it found, how much was really a brick (false alarms).
      recall     of the bricks that were there, how many it found (misses).
                 Recall is what caps the inventory count: a brick never
                 detected can never be classified.
    """
    print("\n" + "=" * 60)
    print("Detection run summary")
    print("=" * 60)

    curve = _training_curve()
    if curve:
        print(f"  Best epoch          {curve['best_epoch']} of {curve['epochs']}")
        print(f"  Best val mAP50      {curve['best_map50']:.4f}")
        print("-" * 60)

    print(f"  Test mAP50          {scores['map50']:.4f}   (50% overlap -- the gate)")
    print(f"  Test mAP50-95       {scores['map50_95']:.4f}   (stricter: box tightness)")
    print(f"  Precision           {scores['precision']:.4f}   (of what it found, correct)")
    print(f"  Recall              {scores['recall']:.4f}   (of what was there, found)")
    print(f"  Target              {DETECTION_MAP_TARGET:.4f}")
    print("-" * 60)

    passed = scores["map50"] >= DETECTION_MAP_TARGET
    margin = (scores["map50"] - DETECTION_MAP_TARGET) * 100
    print(f"  {'PASSED' if passed else 'FAILED'}  ({margin:+.2f} points on mAP50)")
    print("=" * 60 + "\n")

    return passed


if __name__ == "__main__":
    summarise(evaluate())
