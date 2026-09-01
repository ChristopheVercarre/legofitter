from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from app.api.rebrickable import get_part

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

    return {
        "status": "received",
        "filename": file.filename,
        "content_type": file.content_type,
        "size_bytes": len(image_bytes),
        "message": "Image received successfully. Prediction pipeline not connected yet.",
    }
