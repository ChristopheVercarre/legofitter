
from app.params import *
import os
import time
import sys
from google.cloud import storage
from tensorflow import keras
# TODO: `sys` is no longer used anywhere in this file now that
# `sys.path.append("..")` was removed — worth dropping this import too.


def save_model(model: keras.Model = None) -> None:
    """
    Save model locally and in your bucket on GCS at "models/{timestamp}.h5"
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally

    model_path = os.path.join(MODELS_DIR, f"{timestamp}.keras")

    model.save(model_path)

    print("✅ Model saved locally")

    model_filename = os.path.basename(model_path)

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"models/{model_filename}")
    blob.upload_from_filename(model_path)

    print("✅ Model saved to GCS")

    return None


def load_model(model_name: str = None) -> keras.Model:
    """
    Return a saved model from GCS.

        registry.load_model()                          -> the most recently
                                                            updated model in
                                                            the bucket
        registry.load_model("20260826-143731.keras")   -> that exact model,
                                                            by name

    Return None (but do not Raise) if no model is found.
    """
    client = storage.Client()
    bucket = client.get_bucket(BUCKET_NAME)

    if model_name is not None:
        # Accept either "20260826-143731.keras" or the full
        # "models/20260826-143731.keras" (what save_model() prints/uploads).
        blob_name = model_name if model_name.startswith(
            "models/") else f"models/{model_name}"
        blob = bucket.blob(blob_name)

        if not blob.exists():
            print(
                f"\n❌ No model named {blob_name} found in GCS bucket {BUCKET_NAME}")
            return None
    else:
        blobs = list(bucket.list_blobs(prefix="model"))

        if not blobs:
            print(f"\n❌ No model found in GCS bucket {BUCKET_NAME}")
            return None

        blob = max(blobs, key=lambda x: x.updated)

    model_path_to_save = os.path.join(PROJECT_ROOT, blob.name)
    blob.download_to_filename(model_path_to_save)

    model = keras.models.load_model(model_path_to_save)

    print(f"✅ Model downloaded from cloud storage ({blob.name})")

    return model
