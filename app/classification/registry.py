
import json
import os
import shutil
import time
from pathlib import Path

from google.cloud import storage
from tensorflow import keras

# Imported for its SIDE EFFECT: defining the custom layers in model.py runs
# their @register_keras_serializable decorators, so keras.models.load_model()
# below can reconstruct any run that uses one. Without this, loading a
# teammate's model fails with "Could not locate class 'ColorAugmentation'".
from app.classification.models import model as _model_layers  # noqa: F401
from app.params import (
    BUCKET_NAME,
    CLASS_NAMES_PATH,
    CLASSIFICATION_MODEL_PATH,
    HISTORY_PATH,
    MODELS_DIR,
)


def save_model(model: keras.Model = None, name: str = None) -> str:
    """Save one training run as a self-contained folder, locally and on GCS.

    A run is three files that only mean anything together:

        models/<name>/classifier.keras      the architecture + weights
        models/<name>/class_names.json      index -> part ID, from dataset.py
        models/<name>/history.json          the loss/accuracy curves

    The model alone is not a usable artifact. It emits 50 probabilities and
    has no idea that slot 37 means part 3001 -- that mapping lives in
    class_names.json, and pairing a model with the WRONG one gives confident,
    silently incorrect predictions rather than an error. Keeping the three in
    one folder is what makes that mistake hard to make: you copy a run, not a
    file.

    Filenames inside the folder are fixed, so nothing downstream has to parse
    a name apart; the run's identity lives entirely in the folder name.

        registry.save_model(model)              -> "{timestamp}_{username}"
        registry.save_model(model, "baseline")  -> "baseline"

    Returns the run name.
    """
    if name is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        name = f"{timestamp}_{os.environ.get('USER', 'unknown')}"

    run_dir = MODELS_DIR / name
    run_dir.mkdir(parents=True, exist_ok=True)

    model.save(run_dir / "classifier.keras")

    # Copied rather than moved: the flat models/class_names.json is the
    # working state that predict.py and evaluate.py read with no arguments,
    # and it has to stay put for those to keep working after this call.
    for source, filename in (
        (CLASS_NAMES_PATH, "class_names.json"),
        (HISTORY_PATH, "history.json"),
    ):
        if source.exists():
            shutil.copy2(source, run_dir / filename)
        else:
            print(
                f"⚠️  {source.name} not found -- not included in the run folder")

    print(f"✅ Run saved locally to {run_dir}")

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    for path in sorted(run_dir.iterdir()):
        blob = bucket.blob(f"models/{name}/{path.name}")
        blob.upload_from_filename(path)
        print(f"   uploaded {blob.name}")

    print(f"✅ Run uploaded to gs://{BUCKET_NAME}/models/{name}/")

    return name


def model_input_size(model: keras.Model) -> tuple[int, int]:
    """The (height, width) this model was built for, read off the model itself.

    Every saved .keras file carries its own input shape, so inference never
    has to be told what size to use -- it asks. IMG_SIZE is the knob that
    decides what size you TRAIN at; once a model exists, the model is the
    authority. That is what lets a teammate load a 256x256 model and predict
    with it on a machine configured for 128, with nothing to set and nothing
    to restart.
    """
    height, width = model.input_shape[1:3]  # (batch, H, W, C)
    return int(height), int(width)


def attach_class_names(model: keras.Model, run_dir) -> None:
    """Attach the class list sitting in `run_dir` to the model object itself.

    This is what makes model/label pairing waterproof: the list is read from
    the SAME folder as the classifier.keras that was just loaded -- the one
    pairing save_model() guarantees -- and rides along on the model object,
    so predict_image() never has to guess which class_names.json on disk
    belongs to the model it was handed.

    A plain attribute does not survive model.save(); it does not need to.
    It only needs to live as long as the loaded model object, and every load
    path (here and in evaluate.load_trained_model()) re-attaches it from the
    folder it loaded from.
    """
    class_names_path = Path(run_dir) / "class_names.json"
    if class_names_path.exists():
        with open(class_names_path) as f:
            model.class_names = json.load(f)
        print(
            f"✅ Class names attached to model ({len(model.class_names)} classes)")
    else:
        model.class_names = None
        print(f"⚠️  No class_names.json next to the model in {run_dir}")


def _latest_run_name(bucket) -> str:
    """Name of the most recently uploaded run folder, or None if there is none.

    Judged by each run's classifier.keras rather than by whatever blob is
    newest: the JSON sidecars upload a moment AFTER the model, so "newest
    blob" would routinely be a .json -- which keras.models.load_model()
    cannot open.
    """
    blobs = [
        blob for blob in bucket.list_blobs(prefix="models/")
        if blob.name.endswith("/classifier.keras")
    ]

    if not blobs:
        return None

    latest = max(blobs, key=lambda blob: blob.updated)

    # "models/<name>/classifier.keras" -> "<name>"
    return latest.name.split("/")[1]


def is_run_complete(run_dir) -> bool:
    """True if `run_dir` holds a usable run: the model AND its class names.

    "The folder exists" is not good enough. An interrupted download leaves a
    partial folder behind, and a classifier.keras without its class_names.json
    is exactly the pairing that produces confident, silently wrong part IDs.
    history.json is not required -- losing the curves costs you a chart, not
    correctness.
    """
    run_dir = Path(run_dir)
    return (run_dir / "classifier.keras").exists() and (
        run_dir / "class_names.json"
    ).exists()


def load_model(model_name: str = None, force_download: bool = False) -> keras.Model:
    """Load a run and make it this machine's current model.

        registry.load_model()                  -> the most recent run (bucket)
        registry.load_model("baseline")        -> that run, local copy if there
        registry.load_model("baseline", force_download=True)  -> re-fetch it

    Named runs are served from disk when a complete copy is already in
    models/<name>/, and downloaded from GCS otherwise. That local shortcut is
    safe precisely because run folders are IMMUTABLE: the name carries a
    timestamp and save_model() never overwrites one, so a local copy cannot
    have drifted from the bucket's. (The same shortcut would be wrong for
    models/current/, which every run overwrites.) It also means a named run
    loads with no network at all.

    Passing no name means "whatever is newest in the bucket", which only the
    bucket can answer -- that path always goes to the network.

    Either way the run is then promoted to the working state, so predict.py
    and evaluate.py pick up THIS model and THIS class mapping together on
    their next no-argument call. Without that step a freshly loaded model
    would be scored against whatever class_names.json was lying around from
    an earlier run, and the part IDs would be quietly wrong.

    Return None (but do not raise) if no run is found.
    """
    run_dir = MODELS_DIR / model_name if model_name else None

    # --- Fast path: a complete local copy of a named run ------------------
    if model_name and not force_download and is_run_complete(run_dir):
        print(f"✅ Found run {model_name} locally -- no download needed")

    # --- Otherwise go to the bucket ---------------------------------------
    else:
        client = storage.Client()
        bucket = client.get_bucket(BUCKET_NAME)

        if model_name is None:
            model_name = _latest_run_name(bucket)

            if model_name is None:
                print(f"\n❌ No run found in GCS bucket {BUCKET_NAME}")
                return None

            run_dir = MODELS_DIR / model_name

            # The newest run may already be here from a previous call.
            if not force_download and is_run_complete(run_dir):
                print(f"✅ Newest run is {model_name}, already here locally")
                return _finalise_load(model_name, run_dir)

        prefix = f"models/{model_name}/"
        blobs = list(bucket.list_blobs(prefix=prefix))

        if not blobs:
            print(
                f"\n❌ No run named {model_name} found in GCS bucket {BUCKET_NAME}")
            return None

        run_dir.mkdir(parents=True, exist_ok=True)

        for blob in blobs:
            destination = run_dir / Path(blob.name).name
            blob.download_to_filename(destination)
            print(f"   downloaded {blob.name}")

    return _finalise_load(model_name, run_dir)


def _finalise_load(model_name: str, run_dir) -> keras.Model:
    """Promote a run folder to the working state and load the model from it.

    Shared by every path through load_model() -- local or downloaded -- so
    the model, its class names and its history are always promoted together.
    """
    model_path = run_dir / "classifier.keras"

    if not model_path.exists():
        print(f"\n❌ Run {model_name} has no classifier.keras")
        return None

    # Promote this run to the working state, so predict.py / evaluate.py pick
    # up THIS model and THIS class mapping together on their next no-argument
    # call. Overwriting is intended: the run you just asked for is the one
    # you want to be current.
    #
    # models/current/ is normally created by training (build_callbacks and
    # save_class_names both mkdir it). On a machine that has NEVER trained --
    # a teammate who clones the repo and downloads a model straight from the
    # bucket -- it does not exist yet, and these copies would fail. So create
    # it here too.
    CLASSIFICATION_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(model_path, CLASSIFICATION_MODEL_PATH)

    history = run_dir / "history.json"
    if history.exists():
        # So evaluate.py's summarise() can still report this run's train/val
        # curves on a machine that never trained it.
        shutil.copy2(history, HISTORY_PATH)

    class_names = run_dir / "class_names.json"
    if class_names.exists():
        shutil.copy2(class_names, CLASS_NAMES_PATH)
    else:
        print(
            "⚠️  This run has no class_names.json -- predictions will use "
            f"whatever is already at {CLASS_NAMES_PATH}, which may not match "
            "this model. Check before trusting any part IDs."
        )

    model = keras.models.load_model(model_path)
    attach_class_names(model, run_dir)

    print(f"✅ Run {model_name} loaded and set as the current model")

    return model
