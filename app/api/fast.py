import tempfile
from collections import Counter
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.api.rebrickable import get_part
from app.pipeline.inference import build_detailed_predictions
from app.recommendation.recommender import recommend_sets

app = FastAPI(title="LegoFitter API", version="0.1.0")


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

    # Vérification du format
    allowed_extensions = {".jpg", ".jpeg", ".png", ".heic", ".heif"}

    extension = Path(file.filename or "").suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=415,
            detail="Unsupported image format. Use JPG, JPEG, PNG, HEIC or HEIF.",
        )

    # FastAPI lit simplement le fichier en bytes
    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty.",
        )

    # Plus tard :
    # result = predict_pipeline(image_bytes)
    # return result

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False,
        ) as temp_file:
            temp_file.write(image_bytes)
            temp_path = Path(temp_file.name)

        detailed = build_detailed_predictions(temp_path)

        inventory = Counter(
            detection["predicted_class"]
            for detection in detailed["detections"]
        )

        recommendations = recommend_sets(
        dict(inventory),
        candidate_sets_per_part=10,
        max_candidates=5,
    )

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
