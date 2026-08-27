"""
Objective 1, Step 1 — dataset prep.

Turns `data/lego-dataset-classification/{photos,renders}/<part_id>/*` into the
three `tf.data.Dataset` objects that train.py / evaluate.py consume.

Pipeline, in order:
    select_classes()   -> the NUM_CLASSES part IDs with the most real photos
    build_dataframe()  -> one row per image (image_path, label, source)
    filter_blurry()    -> drop photos below the sharpness threshold
    encode_labels()    -> label column becomes int codes; class names persisted
    split_dataframe()  -> stratified train / val / test DataFrames
    create_dataset()   -> DataFrame -> batched, prefetched tf.data.Dataset

`get_datasets()` runs the whole chain and is the single entry point for
train.py. `load_class_names()` is the inverse used at inference time
(Objective 3) to map a predicted class index back to a LEGO part ID.
"""

import cv2
import json

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import subprocess
from app.params import (
    BATCH_SIZE,
    BLUR_VARIANCE_THRESHOLD,
    CLASS_NAMES_PATH,
    CLASSIFICATION_DATA_DIR,
    IMG_SIZE,
    NUM_CLASSES,
    RANDOM_STATE,
    RENDER_PHOTO_RATIO,
    SOURCE_PATTERNS,
    TEST_SIZE,
    VAL_SIZE,
)

GCS_CLASSIFICATION_DATA = (
    "gs://legofitter-datasets/lego-dataset-classification"
)


def ensure_local_data():
    """
    Vérifie si les données de classification existent en local.

    - Si elles existent et ne sont pas vides -> ne fait rien
    - Sinon -> télécharge les données depuis GCS
    """

    data_exists = (
        CLASSIFICATION_DATA_DIR.exists()
        and any(CLASSIFICATION_DATA_DIR.iterdir())
    )

    if data_exists:
        print(f"✅ Dataset déjà présent : {CLASSIFICATION_DATA_DIR}")
        return

    print("⬇️ Dataset absent en local.")
    print("Téléchargement depuis GCS...")

    # S'assurer que le dossier data existe
    CLASSIFICATION_DATA_DIR.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    subprocess.run(
        [
            "gcloud",
            "storage",
            "cp",
            "--recursive",
            GCS_CLASSIFICATION_DATA,
            str(CLASSIFICATION_DATA_DIR.parent),
        ],
        check=True
    )

    print("✅ Dataset téléchargé avec succès.")

def select_classes(num_classes: int = NUM_CLASSES) -> list[str]:
    """Return the `num_classes` part IDs that have the most real photos.

    Ranking uses photos only (never renders) — the classifier's real job is
    real photographs, so class choice follows where the real data is.
    """
    photos_dir = CLASSIFICATION_DATA_DIR / "photos"

    counts = {}
    for class_dir in photos_dir.iterdir():
        if class_dir.is_dir():
            counts[class_dir.name] = len(
                list(class_dir.glob(SOURCE_PATTERNS["photos"])))

    ranked = sorted(counts, key=counts.get, reverse=True)
    return ranked[:num_classes]


def build_dataframe(classes: list[str] | None = None) -> pd.DataFrame:
    """One row per image: image_path, label (part ID), source (photos/renders)."""
    if classes is None:
        classes = select_classes()

    rows = []
    for label in classes:
        for source, pattern in SOURCE_PATTERNS.items():
            for img in (CLASSIFICATION_DATA_DIR / source / label).glob(pattern):
                rows.append(
                    {
                        "image_path": str(img),
                        "label": label,
                        "source": source,
                    }
                )

    return pd.DataFrame(rows)


def is_blurry(image_path: str, threshold: float = BLUR_VARIANCE_THRESHOLD) -> bool:
    """True if image_path's sharpness falls below threshold.

    Uses the variance of the Laplacian: a crisp image has lots of
    high-frequency edge content and a high variance, a blurry one is
    smoother and scores lower. See BLUR_VARIANCE_THRESHOLD in params.py.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        # Unreadable/corrupt file: treat as blurry so it gets dropped here
        # instead of crashing tf.io.decode_jpeg() later in create_dataset().
        return True
    return cv2.Laplacian(img, cv2.CV_64F).var() < threshold


def filter_blurry(
    dataframe: pd.DataFrame,
    threshold: float = BLUR_VARIANCE_THRESHOLD,
) -> pd.DataFrame:
    """Drop rows whose image is blurry (see is_blurry()).

    Only photos are checked -- renders are computer-generated and never
    blurry. Files stay on disk untouched; only the returned DataFrame
    excludes the blurry rows, so this is safe to re-run with a different
    threshold at any time.
    """
    dataframe = dataframe.reset_index(drop=True)
    is_photo = dataframe["source"] == "photos"

    # .loc (not plain bracket assignment) on both sides: assigning through a
    # boolean-mask __setitem__ silently upcasts `blurry` away from bool,
    # which then breaks the `~blurry` negation below.
    blurry = pd.Series(False, index=dataframe.index, dtype=bool)
    blurry.loc[is_photo] = dataframe.loc[is_photo, "image_path"].apply(
        lambda path: is_blurry(path, threshold)
    ).astype(bool)

    dropped = int(blurry.sum())
    if dropped:
        print(
            f"filter_blurry: dropping {dropped} blurry photo(s) "
            f"(threshold={threshold}) out of {int(is_photo.sum())} photos"
        )

    return dataframe[~blurry].reset_index(drop=True)


def encode_labels(dataframe: pd.DataFrame, persist: bool = True):
    """Replace the part-ID label column with integer codes.

    Returns (dataframe, encoder). `encoder.classes_[i]` is the part ID the
    model means when it predicts class `i` — persisted to CLASS_NAMES_PATH so
    inference does not depend on re-running this function over the same data.
    """
    encoder = LabelEncoder()
    dataframe = dataframe.copy()
    dataframe["label"] = encoder.fit_transform(dataframe["label"].astype(str))

    if persist:
        save_class_names(encoder)

    return dataframe, encoder


def save_class_names(encoder: LabelEncoder) -> None:
    """Write the encoder's class order to disk (index -> part ID)."""
    CLASS_NAMES_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLASS_NAMES_PATH.write_text(json.dumps(list(encoder.classes_)))


def load_class_names() -> list[str]:
    """Read back the class order saved at training time.

    The inverse of the encoder: `load_class_names()[predicted_index]` is the
    LEGO part ID. Used by the Objective 3 inference pipeline.
    """
    return json.loads(CLASS_NAMES_PATH.read_text())


def cap_renders(
    dataframe: pd.DataFrame,
    ratio: float = RENDER_PHOTO_RATIO,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Keep at most `ratio` x (that class's photo count) renders per class.

    Every photo is kept; only renders are subsampled. Apply to the TRAINING
    split only — val/test should keep the natural distribution so photos-only
    slices of them stay meaningful.

    Why a ratio rather than the paper's flat 650-renders-per-class: their
    smallest class still had 350+ photos, so a fixed budget gave them exactly
    equal class sizes. Our photo counts run 254-899 per class, so a flat
    render budget would leave class frequency tracking render availability
    (which is arbitrary) instead of photo availability (which is the target
    domain). Scaling to each class's photo count keeps the paper's source
    ratio while letting class size follow the real data.
    """
    photos = dataframe[dataframe["source"] == "photos"]
    renders = dataframe[dataframe["source"] == "renders"]
    photo_counts = photos.groupby("label").size()

    kept = [
        group.sample(
            n=min(len(group), int(photo_counts.get(label, 0) * ratio)),
            random_state=random_state,
        )
        for label, group in renders.groupby("label", sort=False)
    ]

    return (
        pd.concat([photos, *kept])
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )


def split_dataframe(
    dataframe: pd.DataFrame,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_STATE,
):
    """Stratified train / val / test split.

    `test_size` is carved off the full set first; `val_size` is then carved
    off what remains — see params.py for the values and what they give.

    Stratifying on label keeps every class proportionally represented in all
    three splits.
    """
    train_val_df, test_df = train_test_split(
        dataframe,
        test_size=test_size,
        random_state=random_state,
        stratify=dataframe["label"],
    )

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        random_state=random_state,
        stratify=train_val_df["label"],
    )

    return train_df, val_df, test_df


def load_and_preprocess(file_path, label):
    """Read one image file, decode it, resize to IMG_SIZE, scale to [0, 1]."""
    img = tf.io.read_file(file_path)
    # Both sources are JPEG — .jpg and .jpeg differ only in extension.
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img, label


def create_dataset(dataframe: pd.DataFrame, training: bool = False) -> tf.data.Dataset:
    """Turn a split DataFrame into a batched, prefetched tf.data.Dataset."""
    image_paths = dataframe["image_path"].astype(str).to_numpy()
    labels = dataframe["label"].astype("int32").to_numpy()

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    # Shuffle BEFORE decoding: the buffer then holds file paths (a few MB)
    # rather than decoded images (buffer_size * 128*128*3 floats — gigabytes).
    if training:
        dataset = dataset.shuffle(
            buffer_size=len(dataframe),
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(load_and_preprocess,
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def get_datasets(
    num_classes: int = NUM_CLASSES,
    render_ratio: float | None = RENDER_PHOTO_RATIO,
):


    """Run the whole chain: disk -> (train, val, test) tf.data.Datasets.

    `render_ratio` caps renders per class in the training split (None = keep
    the raw ~5:1 distribution). val/test are left untouched.

    Returns (train_dataset, val_dataset, test_dataset, splits) where `splits`
    is the (train_df, val_df, test_df) tuple — keep it if you want to evaluate
    on a subset, e.g. photos only:

        _, _, test_ds, (_, _, test_df) = get_datasets()
        photos_ds = create_dataset(test_df[test_df["source"] == "photos"])
    """

    # Vérifie automatiquement les données avant de continuer
    ensure_local_data()

    dataframe = build_dataframe(select_classes(num_classes))
    dataframe = filter_blurry(dataframe)
    dataframe, _encoder = encode_labels(dataframe)

    train_df, val_df, test_df = split_dataframe(dataframe)

    if render_ratio is not None:
        train_df = cap_renders(train_df, ratio=render_ratio)

    train_dataset = create_dataset(train_df, training=True)
    val_dataset = create_dataset(val_df, training=False)
    test_dataset = create_dataset(test_df, training=False)

    return train_dataset, val_dataset, test_dataset, (train_df, val_df, test_df)
