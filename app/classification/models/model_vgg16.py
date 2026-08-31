"""VGG16 transfer learning -- the architecture the Boinski paper found best.

Boinski, Zawora & Szymanski (2022), "How to Sort Them? A Network for LEGO
Bricks Classification", compared 28 topologies on this exact dataset. VGG16
scored highest (94.56% top-1, 99.21% top-5); ResNet50 came within a point on a
fifth of the parameters. They train in two phases, which is what the two
functions below are for:

    Phase 1  base frozen, only the new head learns.     initialize_model()
    Phase 2  top conv layers unfrozen, tiny LR.         unfreeze_top()

Phase 1 first is not optional. A brand-new head outputs noise, and its large
early gradients would wreck the pretrained filters if they were trainable from
step one -- the thing transfer learning exists to avoid.

Interface note: this module exposes the same four functions as every other
model_<name>.py, so train.py can use it unchanged (see models/__init__.py).
unfreeze_top() is extra -- phase 2 needs a caller, see the module TODO below.
"""

from keras import Sequential, regularizers
from keras.applications import VGG16
from keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    RandomBrightness,
    RandomContrast,
    RandomRotation,
    RandomTranslation,
    RandomZoom,
)
from keras.optimizers import Adam
from keras import mixed_precision

from app.classification.models.layers import VGGPreprocess
from app.params import (
    AUG_BRIGHTNESS,
    AUG_CONTRAST,
    AUG_FILL_MODE,
    AUG_ROTATION,
    AUG_TRANSLATION,
    AUG_ZOOM,
    DENSE_UNITS,
    DROPOUT_RATE,
    FINETUNE_LEARNING_RATE,
    IMG_SIZE,
    L2_REG,
    LEARNING_RATE,
    NUM_CLASSES,
    VGG16_FINETUNE_LAYERS,
)


def enable_mixed_precision() -> None:
    """Switch Keras to float16 compute with float32 weights.

    Same as the other architectures: worth it on the VM's T4, pointless or
    slower on a laptop. The output Dense below is pinned to float32 either way.
    """
    mixed_precision.set_global_policy("mixed_float16")
    print("✅ Mixed precision enabled (float16 compute, float32 weights)")


def build_augmentation() -> Sequential:
    """The random image transforms applied to training images only.

    Identical to the custom CNN's: same dataset, same reasons. No horizontal
    flip -- LEGO has mirrored part pairs that are DIFFERENT part IDs.
    """
    layers = [
        RandomRotation(AUG_ROTATION, fill_mode=AUG_FILL_MODE),
        RandomZoom(AUG_ZOOM, fill_mode=AUG_FILL_MODE),
        RandomTranslation(AUG_TRANSLATION, AUG_TRANSLATION, fill_mode=AUG_FILL_MODE),
        RandomBrightness(AUG_BRIGHTNESS, value_range=(0.0, 1.0)),
        RandomContrast(AUG_CONTRAST),
    ]

    return Sequential(layers, name="data_augmentation")


def initialize_model() -> Sequential:
    """Build (but do not compile) VGG16 + a fresh head, base frozen.

    Layer order matters and is easy to get subtly wrong:

      1. augmentation   -- on 0-1 RGB, training only. RandomBrightness needs
                           value_range=(0,1), so it MUST come before the
                           rescale back to 0-255.
      2. VGGPreprocess  -- undoes dataset.py's /255 and applies Keras's own
                           vgg16.preprocess_input (RGB->BGR, ImageNet means).
                           Baked in here so predict.py and evaluate.py keep
                           feeding 0-1 like they do for every other model.
      3. VGG16          -- ImageNet weights, no classifier head (include_top
                           False), frozen for phase 1.
      4. head           -- GlobalAveragePooling2D, not Flatten: flattening
                           7x7x512 gives 25,088 values and a Dense layer
                           bigger than the entire feature extractor.

    IMG_SIZE should be 224 for this model -- what the ImageNet weights were
    trained at. It will build at other sizes (the conv base is fully
    convolutional) but the pretrained filters are tuned for 224.
    """
    base = VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3),
    )
    base.trainable = False  # phase 1

    model = Sequential(name="lego_vgg16")
    model.add(Input(shape=(*IMG_SIZE, 3)))
    model.add(build_augmentation())
    model.add(VGGPreprocess())
    model.add(base)

    # --- Head ---------------------------------------------------------------
    model.add(GlobalAveragePooling2D())
    model.add(
        Dense(
            DENSE_UNITS,
            activation="relu",
            kernel_regularizer=regularizers.l2(L2_REG),
        )
    )
    model.add(Dropout(DROPOUT_RATE))
    model.add(Dense(NUM_CLASSES, activation="softmax", dtype="float32"))

    trainable = sum(int(w.numpy().size) for w in model.trainable_weights)
    frozen = sum(int(w.numpy().size) for w in model.non_trainable_weights)
    print(
        f"✅ Model built: VGG16 {IMG_SIZE[0]}x{IMG_SIZE[1]} -> {NUM_CLASSES} classes, "
        f"{trainable:,} trainable / {frozen:,} frozen parameters (phase 1)"
    )
    if IMG_SIZE[0] != 224:
        print(
            f"⚠️  VGG16's ImageNet weights were trained at 224x224, not "
            f"{IMG_SIZE[0]}x{IMG_SIZE[1]} -- run with IMG_SIZE=224 for the paper's setup."
        )
    return model


def compile_model(model: Sequential, learning_rate: float = LEARNING_RATE) -> Sequential:
    """Attach loss, optimizer and metrics.

    sparse_categorical_crossentropy because dataset.py hands us integer
    labels, not one-hot vectors.
    """
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=Adam(learning_rate=learning_rate),
        metrics=["accuracy"],
    )
    print(f"✅ Model compiled (Adam, lr={learning_rate}, sparse CE loss)")
    return model


def unfreeze_top(
    model: Sequential,
    n_layers: int = VGG16_FINETUNE_LAYERS,
    learning_rate: float = FINETUNE_LEARNING_RATE,
) -> Sequential:
    """Phase 2: unfreeze the last `n_layers` of the VGG16 base and recompile.

    Only the TOP of the base is unfrozen. Early conv layers hold generic edge
    and texture filters that transfer to anything; the late ones hold
    ImageNet-specific shapes (dog faces, car wheels) that are worth re-learning
    as brick shapes. Unfreezing everything mostly costs time and overfits.

    The learning rate must drop sharply here -- FINETUNE_LEARNING_RATE is 100x
    below LEARNING_RATE. At the phase-1 rate the pretrained filters are
    destroyed in a handful of steps, which looks like the model "suddenly
    getting worse" for no visible reason.

    Recompiling is required, not optional: Keras reads layer.trainable when
    the train function is built, so a trainable flag flipped without a
    recompile has no effect at all.
    """
    base = next(layer for layer in model.layers if layer.name == "vgg16")
    base.trainable = True

    for layer in base.layers[:-n_layers]:
        layer.trainable = False

    compile_model(model, learning_rate=learning_rate)

    trainable = sum(int(w.numpy().size) for w in model.trainable_weights)
    unfrozen = [layer.name for layer in base.layers if layer.trainable]
    print(
        f"✅ Phase 2: unfroze {len(unfrozen)} VGG16 layers ({', '.join(unfrozen)}) "
        f"-> {trainable:,} trainable parameters, lr={learning_rate}"
    )
    return model
