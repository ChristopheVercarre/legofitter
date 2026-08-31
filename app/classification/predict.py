"""
Objective 1, Step 4 -- single-image prediction.

Owns: loading one photo, running it through the trained classifier (via
evaluate.py's load_trained_model(), so it gets the same local-checkpoint-
then-bucket fallback for free), and turning the model's raw softmax output
back into LEGO part IDs using the class order dataset.py saved at training
time.

This is the "one photo in, part ID out" building block Objective 3's
pipeline.py will call once per detected brick crop -- but it's useful on its
own right now too, to spot-check the classifier the moment train.py produces
a model, without waiting for Objective 2's detector to exist.

Run directly:
    python -m app.classification.predict path/to/brick.jpg
"""

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from keras import Model

from app.classification.dataset import load_class_names
from app.classification.evaluate import load_trained_model
from app.classification.registry import model_input_size


def load_image_for_prediction(image, img_size) -> tf.Tensor:
    """Read one image and preprocess it exactly like dataset.py does.

    `image` is a path (str or Path), the raw bytes of an image file, or an
    already-decoded image (a PIL image or a numpy array). Three shapes
    because three callers need them:

      path   -- a notebook spot-checking test_sample/
      bytes  -- a FastAPI upload, `await file.read()`; writing those to a
                temp file just to have a path to read back would be pure
                ceremony, plus a file to clean up in a container
      PIL    -- Objective 3's pipeline, handing over a crop that
                detection.predict.crop_boxes() cut out of a photo. That crop
                only exists in memory, and round-tripping it through a PNG
                encode just to decode it again would be silly.

    `img_size` is required and comes from the model itself (see predict_image
    below), never from IMG_SIZE: a photo has to be resized to what the model
    expects, and only the model knows that.

    Decoding stays on TensorFlow rather than PIL on purpose. dataset.py
    decodes with tf.io during training, and a classifier is sensitive to the
    exact pixels it is fed -- swapping decoders at prediction time would
    introduce a difference between training and serving for no gain. Note
    that this also means EXIF rotation is ignored, exactly as it was ignored
    when the model was trained.

    decode_image rather than decode_jpeg: the API will be handed PNGs sooner
    or later, and decode_jpeg raises on them. expand_animations=False keeps
    the result 3-D -- without it an animated GIF would decode to 4-D and the
    resize below would fail.

    Deliberately not reusing dataset.load_and_preprocess() directly: that
    function's signature is (file_path, label) because it's built for
    tf.data.Dataset.map(), and a single prediction has no label to pass it.
    Same three steps either way -- decode, resize, scale to [0, 1] -- just
    without the label plumbing.
    """
    if isinstance(image, (bytes, bytearray)):
        img = tf.io.decode_image(image, channels=3, expand_animations=False)
    elif isinstance(image, (str, Path)):
        img = tf.io.decode_image(
            tf.io.read_file(str(image)), channels=3, expand_animations=False
        )
    else:
        # Already decoded: a PIL image or a numpy array. np.asarray() handles
        # both, and RGBA is trimmed to RGB so a crop from a PNG still works.
        img = tf.convert_to_tensor(np.asarray(image.convert("RGB"))
                                   if hasattr(image, "convert") else image)
        img = img[..., :3]

    img = tf.image.resize(img, tuple(img_size))
    img = tf.cast(img, tf.float32) / 255.0
    return img


def predict_image(image, top_k: int = 3, model: Model = None) -> list[tuple[str, float]]:
    """Classify one image, return its top-k (part_id, confidence) pairs.

    `image` is a path (str or Path) or raw image bytes, so the same function
    serves a notebook spot-checking test_sample/ and a FastAPI endpoint
    serving an upload.

    confidence is the model's softmax probability for that class -- not a
    calibrated "percent chance correct", but it's what tells a confident
    prediction apart from the model guessing between two or three parts.

    `model` is optional. Pass one in (e.g. loaded once in a notebook with
    registry.load_model()) to reuse it across many predict_image() calls --
    handy since loading a model is the slow part, and a notebook predicting
    on a whole folder of test photos shouldn't reload it every single time.
    Left as None, this falls back to load_trained_model()'s usual
    local-checkpoint-then-bucket lookup.
    """
    if model is None:
        model = load_trained_model()

    # The class list attached to the model at load time is the one that was
    # sitting in the same folder as its .keras file -- the pairing
    # save_model()/load_model() guarantee. Only a model that arrived here
    # WITHOUT that attribute (loaded by hand, not through load_trained_model
    # or registry.load_model) falls back to the current working-state file,
    # which is a guess about which run it belongs to.
    # The model carries its own input size, so a 256x256 model works here
    # whatever IMG_SIZE this machine is set to -- nothing to configure.
    img = load_image_for_prediction(image, model_input_size(model))

    class_names = getattr(model, "class_names", None)
    if class_names is None:
        print(
            "⚠️  Model has no attached class names -- falling back to the "
            "current working-state class_names.json, which may not match "
            "this model. Load models via registry.load_model() or "
            "evaluate.load_trained_model() to avoid this."
        )
        class_names = load_class_names()

    # model.predict() expects a BATCH of images, i.e. shape (batch, H, W, 3).
    # tf.expand_dims adds that batch dimension of 1 for our single image.
    batch = tf.expand_dims(img, axis=0)
    probabilities = model.predict(batch, verbose=0)[0]

    # argsort is ascending; the last top_k entries are the highest
    # probabilities, so [::-1] puts the best prediction first.
    top_indices = np.argsort(probabilities)[-top_k:][::-1]

    return [(class_names[i], float(probabilities[i])) for i in top_indices]


if __name__ == "__main__":
    # Lets you run this file directly:
    #     python -m app.classification.predict path/to/brick.jpg
    # from the project root, once a model exists locally or in the bucket.
    if len(sys.argv) != 2:
        sys.exit("Usage: python -m app.classification.predict <image_path>")

    for rank, (part_id, confidence) in enumerate(predict_image(sys.argv[1]), start=1):
        print(f"  {rank}. {part_id:<12} {confidence:.2%}")
