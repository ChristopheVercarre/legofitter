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

gg    return train_dataset, val_dataset, test_dataset, (train_df, val_df, test_df)
