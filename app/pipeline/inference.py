"""
Objective 3 -- chaining detection + classification.

Owns: given one photo of multiple bricks, run the Objective 2 detector to
find each brick's bounding box, crop each box, run the Objective 1
classifier on each crop, and tally the results into an inventory
(part -> count).

Requires Objectives 1 and 2 to both be done first:
    - a trained classifier, reachable via
      app.classification.evaluate.load_trained_model()
    - a trained detector, reachable via
      app.detection.registry.load_detector()

Run directly:
    python -m app.pipeline.inference path/to/photo.jpg
"""

from collections import Counter

from PIL import Image

from app.classification.evaluate import load_trained_model
from app.classification.predict import predict_array
from app.detection.registry import load_detector


def detect_and_crop(image_path, detector=None, confidence: float = 0.25):
    """Run the detector on one photo, return one (crop, box, det_confidence)
    tuple per brick it finds.

    Kept separate from build_inventory() so the crops themselves -- not just
    the final counts -- are inspectable: a notebook can call this alone to
    look at what got detected (how many boxes, do they look like bricks)
    before trusting the classifier's counts stacked on top of it. That's
    exactly the README's Objective 3 Step 2 ("test end-to-end ... and
    sanity-check the counts") -- you sanity-check detection and
    classification as two separate questions, not one blended one.

    detector=None loads the current detector via load_detector(). Pass one
    in when classifying many photos in a loop, so it's loaded once.
    """
    if detector is None:
        detector = load_detector()

    result = detector.predict(str(image_path), conf=confidence, verbose=False)[0]
    image = Image.open(image_path).convert("RGB")

    crops = []
    for box in result.boxes:
        x_min, y_min, x_max, y_max = (int(v) for v in box.xyxy[0])
        crop = image.crop((x_min, y_min, x_max, y_max))
        crops.append((crop, (x_min, y_min, x_max, y_max), float(box.conf[0])))

    return crops


def build_inventory(
    image_path,
    detector=None,
    classifier=None,
    detect_confidence: float = 0.25,
) -> dict[str, int]:
    """The full chain: one photo in, {part_id: count} inventory out.

    For every brick the detector finds: crop it out of the source photo,
    run the crop through the classifier, keep only its TOP prediction (one
    brick is one part -- there is nothing to do with runner-up guesses
    here), and tally it.

    detector / classifier default to None, which loads each via its own
    registry (load_detector() / load_trained_model()). Pass both in when
    processing a batch of photos -- e.g. a folder of test images -- so
    neither model is reloaded per photo; that reload is the slow part.
    """
    if classifier is None:
        classifier = load_trained_model()

    crops = detect_and_crop(image_path, detector=detector, confidence=detect_confidence)

    inventory = Counter()
    for crop, _box, _det_confidence in crops:
        (part_id, _class_confidence), *_ = predict_array(crop, top_k=1, model=classifier)
        inventory[part_id] += 1

    return dict(inventory)


if __name__ == "__main__":
    # Lets you run this file directly:
    #     python -m app.pipeline.inference path/to/photo.jpg
    # from the project root, once both a detector and a classifier exist
    # locally or in the bucket.
    import sys

    if len(sys.argv) != 2:
        sys.exit("Usage: python -m app.pipeline.inference <image_path>")

    inventory = build_inventory(sys.argv[1])

    print(f"\n{sum(inventory.values())} brick(s) classified in {sys.argv[1]}:")
    for part_id, count in sorted(inventory.items(), key=lambda kv: -kv[1]):
        print(f"  {count:>3} x {part_id}")
