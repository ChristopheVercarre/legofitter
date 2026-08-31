"""
Objective 3 -- chaining detection + classification.

Given one photo of multiple bricks:
    1. run the detector and crop each detected brick
       (app.detection.predict.predict_boxes() + crop_boxes()),
    2. run the classifier on each crop
       (app.classification.predict.predict_image()),
    3. aggregate predictions into an inventory {part_id: count}
       -- or, in detailed mode, return every detection with its box and
       confidences, for debugging or a future API.

This file does NOT reimplement detection or classification -- it only
chains the two building blocks that already exist per-objective. Detection
crops already come EXIF-corrected and padded (see crop_boxes()'s docstring
for why the padding matters), and classification already accepts a crop
straight from memory (no temp files).

Requires Objectives 1 and 2 to both be done first:
    - a trained classifier, reachable via
      app.classification.evaluate.load_trained_model()
    - a trained detector, reachable via
      app.detection.registry.load_detector()

Run directly:
    python -m app.pipeline.inference path/to/photo.jpg
    python -m app.pipeline.inference path/to/photo.jpg --detailed
    python -m app.pipeline.inference path/to/photo.jpg --annotate out.jpg
"""

from collections import Counter

from app.classification.evaluate import load_trained_model
from app.classification.predict import predict_image
from app.detection.predict import crop_boxes, draw_boxes, predict_boxes
from app.detection.registry import load_detector
from app.params import DETECTION_CONFIDENCE


def detect_and_crop(image_path, detector=None, confidence: float = DETECTION_CONFIDENCE):
    """Run the detector on one photo, return (boxes, crops) -- parallel
    lists, so boxes[i] is the box crops[i] was cut from.

    Thin wrapper around app.detection.predict: predict_boxes() finds the
    bricks, crop_boxes() cuts them out (with EXIF correction and padding
    already handled there). Kept as its own function so the detections are
    inspectable on their own -- a notebook can call this alone to check
    what got found before trusting anything stacked on top of it (README
    Objective 3 Step 2: sanity-check detection and classification as two
    separate questions, not one blended one).

    detector=None loads the current detector via load_detector(). Pass one
    in when classifying many photos in a loop, so it's loaded once.
    """
    if detector is None:
        detector = load_detector()

    boxes = predict_boxes(image_path, confidence=confidence, detector=detector)
    crops = crop_boxes(image_path, boxes)
    return boxes, crops


def classify_detection(crop, classifier=None, top_k: int = 1) -> dict:
    """Classify one crop, return its top prediction plus the raw top-k list.

    Shared by build_inventory() (top_k=1 -- one brick is one part, runner-up
    guesses are discarded) and build_detailed_predictions() (top_k=3, to
    expose runner-up guesses for debugging).
    """
    if classifier is None:
        classifier = load_trained_model()

    predictions = predict_image(crop, top_k=top_k, model=classifier)
    (part_id, class_confidence), *_ = predictions

    return {
        "predicted_class": part_id,
        "classification_confidence": class_confidence,
        "top_k_predictions": predictions,
    }


def build_inventory(
    image_path,
    detector=None,
    classifier=None,
    detect_confidence: float = DETECTION_CONFIDENCE,
) -> dict[str, int]:
    """The full chain, business output: one photo in -> {part_id: count} out.

    detector / classifier default to None, which loads each via its own
    registry. Pass both in when processing a batch of photos so neither
    model is reloaded per photo -- that reload is the slow part.
    """
    if classifier is None:
        classifier = load_trained_model()

    _boxes, crops = detect_and_crop(image_path, detector=detector, confidence=detect_confidence)

    inventory = Counter()
    for crop in crops:
        classification = classify_detection(crop, classifier=classifier, top_k=1)
        inventory[classification["predicted_class"]] += 1

    return dict(inventory)


def build_detailed_predictions(
    image_path,
    detector=None,
    classifier=None,
    detect_confidence: float = DETECTION_CONFIDENCE,
) -> dict:
    """Debugging / future-API output: one photo in -> every detection out,
    each with its box, its detection confidence, and its top-3 classifier
    guesses -- everything build_inventory() throws away to get a clean
    count.
    """
    if classifier is None:
        classifier = load_trained_model()

    boxes, crops = detect_and_crop(image_path, detector=detector, confidence=detect_confidence)

    results = []
    for box, crop in zip(boxes, crops):
        classification = classify_detection(crop, classifier=classifier, top_k=3)
        results.append({
            "bbox": box["box"],
            "detection_confidence": box["confidence"],
            "predicted_class": classification["predicted_class"],
            "classification_confidence": classification["classification_confidence"],
            "top_k_predictions": classification["top_k_predictions"],
        })

    return {
        "image_path": str(image_path),
        "num_detections": len(results),
        "detections": results,
    }


def annotate_photo(
    image_path,
    detector=None,
    classifier=None,
    detect_confidence: float = DETECTION_CONFIDENCE,
):
    """Run the full chain and return the source photo with every detected
    brick's box AND its predicted part ID drawn on it.

    build_inventory() tells you if the total count looks right; this tells
    you WHICH brick the pipeline got wrong, and where on the photo --
    the picture equivalent of build_detailed_predictions(). Reuses
    app.detection.predict.draw_boxes() rather than drawing anything itself,
    just feeds it labels made from the classifier's output instead of raw
    detection confidences.

    Returns a PIL.Image -- save it yourself:
        annotate_photo("photo.jpg").save("photo_annotated.jpg")
    """
    detailed = build_detailed_predictions(
        image_path,
        detector=detector,
        classifier=classifier,
        detect_confidence=detect_confidence,
    )

    boxes = [
        {"box": d["bbox"], "confidence": d["detection_confidence"]}
        for d in detailed["detections"]
    ]
    labels = [
        f"{d['predicted_class']} {d['classification_confidence']:.0%}"
        for d in detailed["detections"]
    ]

    return draw_boxes(image_path, boxes, labels=labels)


def _print_inventory(inventory: dict, image_path) -> None:
    print(f"\n{sum(inventory.values())} brick(s) classified in {image_path}:")
    for part_id, count in sorted(inventory.items(), key=lambda kv: -kv[1]):
        print(f"  {count:>3} x {part_id}")


def main():
    # --detailed switches to build_detailed_predictions()'s JSON output.
    # --annotate <path> additionally saves the photo with boxes + predicted
    # part IDs drawn on it. Both need build_detailed_predictions() (boxes +
    # per-detection classification), so --annotate alone still runs that
    # path even without --detailed -- no point calling build_inventory()
    # separately and running detection+classification twice.
    import sys

    args = sys.argv[1:]
    want_detailed = "--detailed" in args

    annotate_path = None
    if "--annotate" in args:
        idx = args.index("--annotate")
        annotate_path = args[idx + 1]
        del args[idx:idx + 2]

    positional = [a for a in args if not a.startswith("--")]
    if len(positional) != 1:
        sys.exit(
            "Usage: python -m app.pipeline.inference <image_path> "
            "[--detailed] [--annotate output.jpg]"
        )

    image_path = positional[0]

    if want_detailed or annotate_path:
        detailed = build_detailed_predictions(image_path)

        if annotate_path:
            boxes = [
                {"box": d["bbox"], "confidence": d["detection_confidence"]}
                for d in detailed["detections"]
            ]
            labels = [
                f"{d['predicted_class']} {d['classification_confidence']:.0%}"
                for d in detailed["detections"]
            ]
            draw_boxes(image_path, boxes, labels=labels).save(annotate_path)
            print(f"✅ Annotated photo saved to {annotate_path}")

        if want_detailed:
            import json
            print(json.dumps(detailed, indent=2, default=str))
        else:
            inventory = Counter(d["predicted_class"] for d in detailed["detections"])
            _print_inventory(inventory, image_path)
        return

    _print_inventory(build_inventory(image_path), image_path)


if __name__ == "__main__":
    main()
