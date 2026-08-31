"""
Objective 1, Step 2 (part 1) — custom CNN architecture.

Owns the from-scratch CNN we train ourselves (the project brief forbids a
pretrained backbone for this objective), sized for NUM_CLASSES.

The model has three parts, in order:

    1. Augmentation   — random rotation/zoom/translation/brightness/contrast.
                        These are LAYERS, not a preprocessing step: Keras runs
                        them only during training and switches them off
                        automatically for validation, evaluation and
                        prediction. That is why they live inside the model
                        rather than in dataset.py.

    2. Conv blocks    — the feature extractor, written out explicitly below
                        as Block 1 / 2 / 3 (32 -> 64 -> 128 filters), matching
                        notebooks/datascientist_deliverable.ipynb. Each block
                        halves the image and doubles the filter count.

    3. Head           — GlobalAveragePooling + one Dense layer + softmax.
                        This is the actual classifier; everything before it
                        just turns pixels into features.

Most numbers here come from app/params.py so experiments are a one-line edit
in one file rather than a hunt through this module.
"""

from keras import Input, Sequential, mixed_precision, regularizers
from keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    MaxPooling2D,
    RandomBrightness,
    RandomContrast,
    RandomRotation,
    RandomTranslation,
    RandomZoom,
    Activation,
)
from keras.optimizers import Adam

from app.classification.models.layers import ColorAugmentation
from app.params import (
    AUG_BRIGHTNESS,
    AUG_CONTRAST,
    AUG_FILL_MODE,
    AUG_ROTATION,
    AUG_TRANSLATION,
    AUG_ZOOM,
    DENSE_UNITS,
    DROPOUT_RATE,
    IMG_SIZE,
    L2_REG,
    LEARNING_RATE,
    NUM_CLASSES,
)


def enable_mixed_precision() -> None:
    """Switch Keras to float16 compute with float32 weights.

    On the VM's T4 this lets the GPU's tensor cores do the convolutions, which
    is typically a large speedup. Weights stay float32 so training stays
    numerically stable, and Keras handles the loss scaling for us.

    MUST be called BEFORE initialize_model() — the policy is read when each
    layer is constructed, so calling it afterwards does nothing.
    """
    mixed_precision.set_global_policy("mixed_float16")
    print("✅ Mixed precision enabled (float16 compute, float32 weights)")


def build_augmentation() -> Sequential:
    """The random image transforms applied to training images only.

    Why this matters here: after cap_renders(), roughly two thirds of our
    training images are synthetic renders — clean, evenly lit, centred. Real
    photos are none of those things. Augmentation is what stops the model
    learning "render-ness" instead of brick shape.

    No horizontal flip: LEGO has mirrored part pairs (left/right wedges)
    that are DIFFERENT part IDs, so flipping could teach the model to
    confuse two classes.
    """
    layers = [
        # Geometric jitter — a brick can be photographed at any angle, any
        # distance, anywhere in frame. fill_mode decides what goes in the
        # corners these transforms leave empty; see AUG_FILL_MODE in params.py.
        RandomRotation(AUG_ROTATION, fill_mode=AUG_FILL_MODE),
        RandomZoom(AUG_ZOOM, fill_mode=AUG_FILL_MODE),
        RandomTranslation(AUG_TRANSLATION, AUG_TRANSLATION,
                          fill_mode=AUG_FILL_MODE),
        # Photometric jitter — renders have studio lighting, phone photos do not.
        RandomBrightness(AUG_BRIGHTNESS, value_range=(0.0, 1.0)),
        RandomContrast(AUG_CONTRAST),
        # Saturation and hue jitter (Jules's layer, shared from layers.py).
        # RandomBrightness/RandomContrast only move lightness; this is what
        # varies COLOUR. The same part ID exists in red, blue, grey and more,
        # and the renders are far more uniformly coloured than the photos, so
        # jittering hue is what teaches "shape matters, colour does not".
        ColorAugmentation(),
    ]

#     return Sequential(layers, name="data_augmentation")


def initialize_model() -> Sequential:
    """Build (but do not compile) the classifier."""
    model = Sequential(name="lego_classifier")

    # Declaring the input shape up front means model.summary() shows real
    # numbers immediately instead of "unbuilt".
    model.add(Input(shape=(*IMG_SIZE, 3)))

    # model.add(build_augmentation())

    # --- Block 1 ---------------------------------------------------------
    model.add(
        Conv2D(
            32,
            (3, 3),
            padding="same",
            kernel_regularizer=regularizers.l2(L2_REG),
        )
    )
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(
        Conv2D(
            32,
            (3, 3),
            padding="same",
            kernel_regularizer=regularizers.l2(L2_REG),
        )
    )
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D((2, 2)))

    # --- Block 2 ---------------------------------------------------------
    model.add(
        Conv2D(
            64,
            (3, 3),
            padding="same",
            kernel_regularizer=regularizers.l2(L2_REG),
        )
    )
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(
        Conv2D(
            64,
            (3, 3),
            padding="same",
            kernel_regularizer=regularizers.l2(L2_REG),
        )
    )
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D((2, 2)))

    # --- Block 3 ---------------------------------------------------------
    model.add(
        Conv2D(
            128,
            (3, 3),
            padding="same",
            kernel_regularizer=regularizers.l2(L2_REG),
        )
    )
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(
        Conv2D(
            128,
            (3, 3),
            padding="same",
            kernel_regularizer=regularizers.l2(L2_REG),
        )
    )
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D((2, 2)))

    # --- Block 4 ---------------------------------------------------------
    model.add(
        Conv2D(
            256,
            (3, 3),
            padding="same",
            kernel_regularizer=regularizers.l2(L2_REG),
        )
    )
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(
        Conv2D(
            256,
            (3, 3),
            padding="same",
            kernel_regularizer=regularizers.l2(L2_REG),
        )
    )
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D((2, 2)))

    # --- Head --------------------------------------------------------------
    # GlobalAveragePooling2D collapses each of Block 3's feature maps to a
    # single number (its spatial average), turning 16x16x128 into a
    # 128-vector. Flatten() would produce 32768 values instead, making the
    # next Dense layer ~250x larger than the entire feature extractor — which
    # is how the first version of this model ended up overfitting so hard.
    model.add(GlobalAveragePooling2D())

    model.add(
        Dense(
            DENSE_UNITS,
            activation="SiLu",
            kernel_regularizer=regularizers.l2(L2_REG),
        )
    )
    model.add(Dropout(DROPOUT_RATE))

    # dtype="float32" is only relevant under mixed precision: softmax over 50
    # classes can underflow in float16, so we force the last layer back to
    # full precision. Harmless when mixed precision is off.
    model.add(Dense(NUM_CLASSES, activation="softmax", dtype="float32"))

    trainable = sum(int(w.numpy().size) for w in model.trainable_weights)
    print(
        f"✅ Model built: {IMG_SIZE[0]}x{IMG_SIZE[1]} input -> "
        f"{NUM_CLASSES} classes, {trainable:,} trainable parameters"
    )
    return model


def compile_model(model: Sequential, learning_rate: float = LEARNING_RATE) -> Sequential:
    """Attach loss, optimizer and metrics.

    sparse_categorical_crossentropy (rather than plain categorical) because
    dataset.py hands us integer labels — 17 — not one-hot vectors
    [0,...,1,...,0]. Same maths, no one-hot matrix to build.
    """
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=Adam(learning_rate=learning_rate),
        metrics=["accuracy"],
    )
    print(f"✅ Model compiled (Adam, lr={learning_rate}, sparse CE loss)")
    return model
