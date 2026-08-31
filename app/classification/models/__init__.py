"""The classifier architectures -- one file per person, model_<name>.py.

train.py never imports an architecture directly. It asks for the one MODEL_NAME
points at (see app/params.py), so switching architecture is a one-word change
in .env or on the make command line rather than an edit to train.py:

    make run_local MODEL_NAME=oriane

Every model_<name>.py must expose the same four functions -- enable_mixed_precision,
build_augmentation, initialize_model, compile_model -- because that is what
train.py calls. Adding an architecture means adding a file here; nothing else
in the project needs to know about it.

Custom Keras layers do NOT belong in these files. They go in layers.py -- see
the docstring there for why.
"""

from importlib import import_module
from pathlib import Path

from app.params import MODEL_NAME

# The folder holding the model_<name>.py files -- NOT params.MODELS_DIR,
# which is where trained runs are stored. Named apart so the two cannot be
# confused now that models/ is split into classification/ and detection/.
ARCHITECTURES_DIR = Path(__file__).parent
PREFIX = "model_"


def available_models() -> list[str]:
    """Every name MODEL_NAME accepts: the model_<name>.py files in this folder."""
    return sorted(p.stem[len(PREFIX):] for p in ARCHITECTURES_DIR.glob(f"{PREFIX}*.py"))


def load_architecture(name: str = None):
    """Import the architecture module MODEL_NAME points at, and return it.

    `name` defaults to MODEL_NAME; pass one explicitly to load a specific
    architecture (a notebook comparing two of them, say).

    Only the requested module is imported, never all of them: a teammate's
    broken architecture file would otherwise break everyone's training run,
    which is exactly what an eager "import every model" would do.

    An unknown name stops the run here, listing what IS available, rather than
    letting a typo surface later as a confusing ImportError mid-training.
    """
    name = name if name is not None else MODEL_NAME

    if name not in available_models():
        raise SystemExit(
            f"❌ MODEL_NAME={name!r} does not exist "
            f"(expected app/classification/models/{PREFIX}{name}.py).\n"
            f"   Available: {', '.join(available_models())}"
        )

    architecture = import_module(f"app.classification.models.{PREFIX}{name}")
    print(f"✅ Architecture: {PREFIX}{name}.py")
    return architecture
