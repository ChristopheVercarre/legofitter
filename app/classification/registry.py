
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


def load_model() -> keras.Model:
    """
    Return a saved model from GCS (most recent one)
    Return None (but do not Raise) if no model is found
    """

    client = storage.Client()
    blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="model"))

    try:
        latest_blob = max(blobs, key=lambda x: x.updated)
        latest_model_path_to_save = os.path.join(
            PROJECT_ROOT, latest_blob.name)
        latest_blob.download_to_filename(latest_model_path_to_save)

        latest_model = keras.models.load_model(latest_model_path_to_save)

        print("✅ Latest model downloaded from cloud storage")

        return latest_model
    except:

        print(f"\n❌ No model found in GCS bucket {BUCKET_NAME}")

        return None
