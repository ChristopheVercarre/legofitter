"""Custom Keras layers, shared by every architecture.

These live here rather than in a model_<name>.py file on purpose. A .keras
file stores a layer's NAME and config, never its code, so Keras can only
rebuild a model whose custom classes are importable at load time -- and
registry.py imports THIS module (never an architecture) so that loading a
teammate's model works whatever MODEL_NAME this machine is set to. Putting a
custom layer in one person's architecture file would make their saved runs
unloadable for anyone training a different architecture.

A layer defined only in a notebook has the same problem, one step worse: it is
unloadable for everyone, including its author on another machine.
"""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="LegoFitter")
class ColorAugmentation(tf.keras.layers.Layer):
    """Random brightness / saturation / hue jitter, training only.

    Written by Jules for his JV1_silu_img-size224 runs; kept here so his
    models (and anyone else's that use it) load on every machine.

    Note the `if not training` guard: like Keras's own augmentation layers,
    this is a no-op during predict and evaluate, so it changes what the model
    LEARNS but never what it PREDICTS.
    """

    def call(self, images, training=None):
        # No augmentation during predict / evaluate
        if not training:
            return images

        images = tf.image.random_brightness(images, max_delta=0.10)
        images = tf.image.random_saturation(images, lower=0.7, upper=1.3)
        images = tf.image.random_hue(images, max_delta=0.08)

        return tf.clip_by_value(images, 0.0, 1.0)


@tf.keras.utils.register_keras_serializable(package="LegoFitter")
class VGGPreprocess(tf.keras.layers.Layer):
    """Turn dataset.py's 0-1 RGB into what VGG16's ImageNet weights expect.

    Keras's vgg16.preprocess_input wants 0-255 RGB and returns BGR with the
    ImageNet channel means subtracted. dataset.py hands every model 0-1, so we
    multiply by 255 first. That round trip is exact in float32 -- all 256 byte
    values map back to themselves -- so nothing is lost, and dataset.py stays
    identical for every architecture instead of growing a per-model flag.

    Living in the model rather than the input pipeline is what lets
    predict.py and evaluate.py stay unchanged: they feed 0-1 as always and the
    model converts on the way in, exactly like it carries its own input size.
    """

    def call(self, images):
        return tf.keras.applications.vgg16.preprocess_input(images * 255.0)
