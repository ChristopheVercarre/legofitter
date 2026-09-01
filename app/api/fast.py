import tempfile
from collections import Counter
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.api.rebrickable import get_part
from app.classification.evaluate import load_trained_model
from app.detection.registry import load_detector
from app.pipeline.inference import build_detailed_predictions
from app.recommendation.recommender import recommend_sets

app = FastAPI(title="LegoFitter API", version="0.1.0")

# Load both models ONCE, at service startup. build_detailed_predictions()
# would happily load them itself, but it does so PER CALL -- seconds of disk
# (or bucket) I/O added to every /predict. One slow cold start here buys
# fast requests forever after.
detector = load_detector()
classifier = load_trained_model()
print("✅ API ready: detector + classifier loaded")


@app.get("/")
def root():
    return {"status": "ok", "service": "LegoFitter API"}


@app.get("/ping")
def ping():
    return {"message": "pong"}


@app.get("/parts/{part_id}")
def part_info(part_id: str):
    return get_part(part_id)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

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
            classifier=classifier,
        )

        inventory = Counter(
            detection["predicted_class"]
            for detection in detailed["detections"]
        )

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
            "inventory": dict(inventory),
            "total_bricks": sum(inventory.values()),
            "detections": detections,
            "recommended_set": recommendations[0] if recommendations else None,
            "recommendations": recommendations,
        }

    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()
