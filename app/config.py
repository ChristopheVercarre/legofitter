"""
Central config for LegoFitter, loaded from environment variables (see .env.example).

Every other module should read paths/constants from here rather than
hardcoding them, so the whole project stays configurable from one file + .env.
"""
from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / os.getenv("DATA_DIR", "data")
MODELS_DIR = PROJECT_ROOT / os.getenv("MODELS_DIR", "models")

# --- Objective 1: Classification ---
NUM_CLASSES = int(os.getenv("NUM_CLASSES", 50))
CLASSIFICATION_DATA_DIR = PROJECT_ROOT / os.getenv(
    "CLASSIFICATION_DATA_DIR", "data/lego-dataset-classification"
)
CLASSIFICATION_MODEL_PATH = MODELS_DIR / "classifier.keras"
CLASSIFICATION_ACCURACY_TARGET = 0.70  # gate before starting Objective 2

# --- Objective 2: Object Detection ---
DETECTION_DATA_DIR = PROJECT_ROOT / os.getenv(
    "DETECTION_DATA_DIR", "data/lego-tagged-object_detection"
)
DETECTION_MODEL_PATH = MODELS_DIR / "detector.pt"

# --- Bonus: Rebrickable ---
REBRICKABLE_API_KEY = os.getenv("REBRICKABLE_API_KEY", "")
REBRICKABLE_BASE_URL = "https://rebrickable.com/api/v3"
