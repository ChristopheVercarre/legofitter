import tempfile
from collections import Counter
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.api.rebrickable import get_part
from app.classification.registry import load_model
from app.detection.registry import load_detector
from app.params import CLASSIFIER_CHOICES, CLASSIFIER_RUN, DETECTOR_RUN
from app.pipeline.inference import build_detailed_predictions
from app.recommendation.recommender import recommend_sets

app = FastAPI(title="LegoFitter API", version="0.1.0")

# Load both models ONCE, at service startup. build_detailed_predictions()
# would happily load them itself, but it does so PER CALL -- seconds of disk
# (or bucket) I/O added to every /predict. One slow cold start here buys
# fast requests forever after.
# DETECTOR_RUN (params.py, env var) pins the detector so a teammate uploading
# a new model to the bucket cannot change what the demo answers.
detector = load_detector(DETECTOR_RUN or None)

# Every classifier the frontend can pick from, keyed by display name, all
# loaded up front: a bucket download in the middle of a demo is not an
# option. CLASSIFIER_RUN (env var) narrows the menu to one run.
if CLASSIFIER_RUN:
    classifier_runs = {CLASSIFIER_RUN: CLASSIFIER_RUN}
else:
    classifier_runs = CLASSIFIER_CHOICES

classifiers = {}
for display_name, run_name in classifier_runs.items():
    model = load_model(run_name)
    if model is None:
        raise RuntimeError(f"Classifier run not found in the bucket: {run_name!r}")
    classifiers[display_name] = model
    print(f"✅ Classifier ready: {display_name} = {run_name}")

DEFAULT_CLASSIFIER = next(iter(classifiers))

LOADED_MODELS = {
    "detector": DETECTOR_RUN or "newest (unpinned)",
    "classifiers": dict(classifier_runs),
    "default_classifier": DEFAULT_CLASSIFIER,
}
print(f"✅ API ready: {LOADED_MODELS}")


@app.get("/")
def root():
    return {"status": "ok", "service": "LegoFitter API"}


@app.get("/ping")
def ping():
    # The model names let the frontend (and a nervous presenter) verify at a
    # glance which runs this instance is answering with.
    return {"message": "pong", "models": LOADED_MODELS}


@app.get("/parts/{part_id}")
def part_info(part_id: str):
    return get_part(part_id)


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    classifier: str | None = Form(None),
):
    # `classifier` is a display name from CLASSIFIER_CHOICES (see /ping).
    # Missing = the default; unknown = 400 rather than a silent fallback,
    # so a stale frontend can never claim it used a model it did not.
    classifier_name = classifier or DEFAULT_CLASSIFIER
    if classifier_name not in classifiers:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown classifier {classifier_name!r}. "
                f"Available: {list(classifiers)}"
            ),
        )

    # Extension check -- the pipeline decodes with PIL, which (with
    # pillow-heif registered in app/detection/predict.py) handles exactly
    # these. The temp file below keeps the suffix, so HEIC decodes too.
    allowed_extensions = {".jpg", ".jpeg", ".png", ".heic", ".heif"}
    extension = Path(file.filename or "").suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=415,
            detail="Unsupported image format. Use JPG, JPEG, PNG, HEIC or HEIF.",
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty.",
        )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False,
        ) as temp_file:
            temp_file.write(image_bytes)
            temp_path = Path(temp_file.name)

        detailed = build_detailed_predictions(
            temp_path,
            detector=detector,
            classifier=classifiers[classifier_name],
        )

        inventory = Counter(
            detection["predicted_class"]
            for detection in detailed["detections"]
        )

        # Human-readable inventory: nobody in an audience knows what part
        # ID "3001" is, so attach each part's name and photo from
        # Rebrickable. Same rule as recommendations below: a Rebrickable
        # failure (or missing key) must never take down the detections --
        # this degrades to bare part IDs, and the frontend knows to cope.
        inventory_details = []
        for part_id, count in inventory.most_common():
            entry = {
                "part_id": part_id,
                "count": count,
                "name": None,
                "img_url": None,
            }
            try:
                part = get_part(part_id)
                entry["name"] = part.get("name")
                entry["img_url"] = part.get("part_img_url")
            except Exception as error:
                print(f"⚠️ Part lookup skipped for {part_id}: {error}")
            inventory_details.append(entry)

        # Recommendations are the bonus, not the product. A missing API key,
        # a Rebrickable rate limit or a network blip must never take down
        # the detections we already computed -- degrade to "no
        # recommendation" instead of a 500.
        try:
            recommendations = recommend_sets(dict(inventory))
        except Exception as error:
            print(f"⚠️ Set recommendation skipped: {error}")
            recommendations = []

        detections = [
            {
                "part_id": detection["predicted_class"],
                "bbox": [int(x) for x in detection["bbox"]],
                "detection_confidence": float(
                    detection["detection_confidence"]
                ),
                "classification_confidence": float(
                    detection["classification_confidence"]
                ),
            }
            for detection in detailed["detections"]
        ]

        return {
            "status": "success",
            "filename": file.filename,
            "classifier": classifier_name,
            "inventory": dict(inventory),
            "inventory_details": inventory_details,
            "total_bricks": sum(inventory.values()),
            "detections": detections,
            "recommended_set": recommendations[0] if recommendations else None,
            "recommendations": recommendations,
        }

    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()
