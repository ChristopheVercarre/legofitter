"""
Objective 2 -- the detector run registry.

The mirror of app/classification/registry.py, function for function, so that
learning one teaches you the other:

    save_run(name)                  <- save_model(model, name)
    load_detector(name)             <- load_model(name)
    is_run_complete(run_dir)        <- is_run_complete(run_dir)

A detection RUN is one folder holding three things that only mean anything
together:

    models/detection/<name>/weights/best.pt   the weights
    models/detection/<name>/data.yaml         which classes they were trained on
    models/detection/<name>/results.csv       the training curves

best.pt on its own emits boxes with a class index and no idea what index 0
means -- data.yaml is that mapping. Same trap as pairing a classifier with the
wrong class_names.json, and the same fix: you copy a run, never a file.

One difference from the classification registry: ultralytics writes the run
folder itself (we point its `project` and `name` at it), so save_run() uploads
and promotes an existing folder rather than saving a model.
"""

import shutil

from google.cloud import storage
from ultralytics import YOLO

from app.params import (
    BUCKET_NAME,
    DETECTION_CURRENT_DIR,
    DETECTION_DATA_YAML_PATH,
    DETECTION_MODEL_PATH,
    DETECTION_MODELS_DIR,
    DETECTION_RESULTS_PATH,
    GCS_DETECTION_MODELS,
)

# What a run folder holds, and where each file lands in current/.
# weights/best.pt keeps its subfolder because that is where ultralytics writes
# it; flattening it in current/ makes DETECTION_MODEL_PATH a plain path.
RUN_FILES = (
    ("weights/best.pt", DETECTION_MODEL_PATH),
    ("data.yaml", DETECTION_DATA_YAML_PATH),
    ("results.csv", DETECTION_RESULTS_PATH),
)


def is_run_complete(run_dir) -> bool:
    """True if this folder holds a usable detector.

    Requires BOTH the weights and data.yaml: a half-finished download that had
    only best.pt would otherwise pass, and load as a detector whose class
    indices mean nothing.
    """
    return (run_dir / "weights" / "best.pt").exists() and (run_dir / "data.yaml").exists()


def save_run(name: str) -> str:
    """Upload one run folder to the bucket and promote it to current/.

    ultralytics has already written models/detection/<name>/ by the time this
    is called -- train.py points it there -- so there is nothing to save, only
    to publish. Returns the run name.
    """
    run_dir = DETECTION_MODELS_DIR / name

    if not is_run_complete(run_dir):
        print(f"⚠️  {run_dir} has no best.pt + data.yaml -- nothing uploaded")
        return name

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    for relative_path, _ in RUN_FILES:
        source = run_dir / relative_path
        if not source.exists():
            print(f"⚠️  {relative_path} missing -- not uploaded")
            continue
        blob = bucket.blob(f"{GCS_DETECTION_MODELS}/{name}/{relative_path}")
        blob.upload_from_filename(source)
        print(f"   uploaded {blob.name}")

    print(f"✅ Run uploaded to gs://{BUCKET_NAME}/{GCS_DETECTION_MODELS}/{name}/")

    _finalise_load(name, run_dir)
    return name


def _latest_run_name(bucket) -> str:
    """The newest run in the bucket, judged by its best.pt.

    Judged by best.pt rather than by whatever blob is newest: uploads finish in
    arbitrary order, so "the newest blob" would routinely be a .csv.
    """
    blobs = [
        blob for blob in bucket.list_blobs(prefix=f"{GCS_DETECTION_MODELS}/")
        if blob.name.endswith("/weights/best.pt")
    ]
    if not blobs:
        raise FileNotFoundError(
            f"No detector runs in gs://{BUCKET_NAME}/{GCS_DETECTION_MODELS}/"
        )

    latest = max(blobs, key=lambda blob: blob.updated)
    # "models/detection/<name>/weights/best.pt" -> "<name>"
    return latest.name.split("/")[-3]


def load_detector(name: str = None, force_download: bool = False) -> YOLO:
    """Load a detector run by name, or the newest one if no name is given.

    Local-first, like load_model(): a complete run folder already on disk is
    served without touching the network. Safe because run folders are
    immutable -- the same shortcut on current/ would be a bug, since current/
    is rewritten by every run.

    force_download=True re-fetches, for when a local copy is suspect.
    """
    run_dir = DETECTION_MODELS_DIR / name if name else None

    if not force_download and run_dir is not None and is_run_complete(run_dir):
        print(f"✅ Found run {name} locally -- no download needed")
        return _finalise_load(name, run_dir)

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    if name is None:
        name = _latest_run_name(bucket)
        print(f"✅ Newest run in the bucket: {name}")
        run_dir = DETECTION_MODELS_DIR / name

        # Checked again now that the name is known: a bare load_detector() has
        # to ask the bucket WHICH run is newest, but if we already hold it
        # there is still nothing to download.
        if not force_download and is_run_complete(run_dir):
            print(f"✅ Found run {name} locally -- no download needed")
            return _finalise_load(name, run_dir)

    prefix = f"{GCS_DETECTION_MODELS}/{name}/"
    blobs = list(bucket.list_blobs(prefix=prefix))
    if not blobs:
        raise FileNotFoundError(f"No run named {name} in gs://{BUCKET_NAME}/{prefix}")

    for blob in blobs:
        destination = run_dir / blob.name[len(prefix):]
        destination.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(destination)
        print(f"   downloaded {blob.name}")

    return _finalise_load(name, run_dir)


def _finalise_load(name: str, run_dir) -> YOLO:
    """Copy the run's files into current/ and return the loaded detector.

    Shared tail of both load paths, so current/ is always promoted as a SET.
    Promoting the weights without data.yaml would leave current/ describing one
    run with another run's class map.
    """
    DETECTION_CURRENT_DIR.mkdir(parents=True, exist_ok=True)

    for relative_path, destination in RUN_FILES:
        source = run_dir / relative_path
        if source.exists():
            shutil.copy2(source, destination)
        else:
            print(f"⚠️  {relative_path} not in the run -- current/ keeps the old one")

    print(f"✅ Run {name} loaded and set as the current detector")
    return YOLO(str(DETECTION_MODEL_PATH))
