"""
Smoke tests -- the guard against a broken file reaching master.

Written to run WITHOUT importing the project, and therefore without
TensorFlow, OpenCV, or a GCS connection. They parse the source files instead
(Python's own `ast` module), which means:

  * they run in CI in seconds, with no dependencies beyond pytest -- important
    because requirements.txt resolves to tensorflow[and-cuda] on Linux, and no
    one wants a 2GB download to check that a file is not empty;
  * they catch the two failures that have actually happened here: a file
    getting truncated/emptied and pushed, and a name being used that nothing
    imports or defines (`subprocess` in dataset.py).

What they deliberately do NOT catch: anything that only shows up at runtime.
That is what the (skipped-in-CI) import test below and, eventually, real unit
tests are for.

Run:
    pytest tests/ -q
"""

import ast
import builtins
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The public API each module is expected to expose. This is the actual
# contract: if a function disappears -- deleted, renamed, or lost to a bad
# merge -- the test naming it fails and says which one.
EXPECTED_API = {
    "app/params.py": [],  # constants only; checked separately below
    "app/classification/dataset.py": [
        "ensure_local_data",
        "select_classes",
        "build_dataframe",
        "is_blurry",
        "filter_blurry",
        "encode_labels",
        "save_class_names",
        "load_class_names",
        "cap_renders",
        "split_dataframe",
        "load_and_preprocess",
        "create_dataset",
        "get_datasets",
    ],
    "app/classification/models/__init__.py": [
        "available_models",
        "load_architecture",
    ],
    "app/classification/models/layers.py": [],  # custom layers are classes, checked below
    "app/classification/models/model_vgg16.py": [
        "enable_mixed_precision",
        "build_augmentation",
        "initialize_model",
        "compile_model",
        "unfreeze_top",
    ],
    "app/classification/models/model_christophe.py": [
        "enable_mixed_precision",
        "build_augmentation",
        "initialize_model",
        "compile_model",
    ],
    "app/classification/train.py": [
        "build_callbacks",
        "save_history",
        "describe_splits",
        "train_model",
    ],
    "app/utils/format.py": [
        "format_duration",
    ],
    "app/classification/evaluate.py": [
        "load_trained_model",
        "evaluate_model",
        "summarise",
        "evaluate",
    ],
    "app/classification/predict.py": [
        "load_image_for_prediction",
        "predict_image",
    ],
    "app/classification/registry.py": [
        "model_input_size",
        "attach_class_names",
        "is_run_complete",
        "save_model",
        "load_model",
    ],
    "run_training.py": ["stage"],
}

# Constants the rest of the project reads from params.py. A missing one is a
# broken import everywhere at once, so they are worth pinning by name.
EXPECTED_PARAMS = [
    "IMG_SIZE",
    "BATCH_SIZE",
    "NUM_CLASSES",
    "RANDOM_STATE",
    "BUCKET_NAME",
    "MODELS_DIR",
    "CURRENT_RUN_DIR",
    "CLASSIFICATION_DATA_DIR",
    "GCS_CLASSIFICATION_DATA",
    "CLASSIFICATION_MODEL_PATH",
    "CLASSIFICATION_ACCURACY_TARGET",
    "CLASS_NAMES_PATH",
    "HISTORY_PATH",
    "BLUR_VARIANCE_THRESHOLD",
    "RENDER_PHOTO_RATIO",
    "TEST_SIZE",
    "VAL_SIZE",
    "EPOCHS",
    "USE_MIXED_PRECISION",
]

MODULE_PATHS = sorted(EXPECTED_API)


def parse(relative_path: str) -> ast.Module:
    """Parse one project file into an AST, failing loudly if it is unusable."""
    path = PROJECT_ROOT / relative_path
    assert path.exists(), f"{relative_path} does not exist"
    source = path.read_text()
    assert source.strip(), f"{relative_path} is empty"
    return ast.parse(source, filename=str(path))


def defined_names(tree: ast.Module) -> set[str]:
    """Every name a module binds: imports, defs, assignments, args, handlers.

    Used by the undefined-name test below. Deliberately generous -- the goal
    is to never fail on correct code, while still catching a name that
    nothing in the file provides.
    """
    # Module-level dunders Python provides but that are not in dir(builtins).
    names = set(dir(builtins)) | {
        "__file__",
        "__name__",
        "__doc__",
        "__package__",
        "__spec__",
        "__loader__",
        "__builtins__",
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.asname or a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.update(a.asname or a.name for a in node.names)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.Global):
            names.update(node.names)

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            a = node.args
            names.update(arg.arg for arg in a.args + a.posonlyargs + a.kwonlyargs)
            if a.vararg:
                names.add(a.vararg.arg)
            if a.kwarg:
                names.add(a.kwarg.arg)

    return names


def used_names(tree: ast.Module) -> set[str]:
    """Every name a module reads."""
    return {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }


@pytest.mark.parametrize("relative_path", MODULE_PATHS)
def test_file_is_present_and_not_empty(relative_path):
    """The disaster test: a truncated or emptied file fails here.

    Also a rough size floor -- every one of these modules is well over 500
    bytes, so a file that survives `ast.parse` but has clearly lost its
    contents still gets caught.
    """
    path = PROJECT_ROOT / relative_path
    assert path.exists(), f"{relative_path} is missing"
    assert path.stat().st_size > 500, (
        f"{relative_path} is only {path.stat().st_size} bytes -- "
        "it looks truncated or emptied"
    )


@pytest.mark.parametrize("relative_path", MODULE_PATHS)
def test_file_is_valid_python(relative_path):
    """Syntax errors never reach master."""
    parse(relative_path)


@pytest.mark.parametrize("relative_path", MODULE_PATHS)
def test_expected_functions_exist(relative_path):
    """Every function the rest of the project calls is still defined."""
    tree = parse(relative_path)
    defined = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    missing = [name for name in EXPECTED_API[relative_path] if name not in defined]
    assert not missing, f"{relative_path} is missing: {', '.join(missing)}"


@pytest.mark.parametrize("relative_path", MODULE_PATHS)
def test_no_undefined_names(relative_path):
    """Catches a name used but never imported or defined.

    This is the `import subprocess` bug: ensure_local_data() called
    subprocess.run() while nothing imported subprocess, which Python only
    complains about at call time -- long after a push.
    """
    tree = parse(relative_path)
    undefined = sorted(used_names(tree) - defined_names(tree))
    assert not undefined, (
        f"{relative_path} uses names nothing defines or imports: "
        f"{', '.join(undefined)}"
    )


def test_params_defines_expected_constants():
    """params.py is imported by everything; a missing constant breaks it all."""
    tree = parse("app/params.py")
    assigned = {
        target.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }

    missing = [name for name in EXPECTED_PARAMS if name not in assigned]
    assert not missing, f"app/params.py is missing: {', '.join(missing)}"


def test_dataset_calls_ensure_local_data_from_get_datasets():
    """get_datasets() must actually CALL ensure_local_data(), not just define it.

    Regression test: the function was once defined but orphaned, so a machine
    without the dataset silently failed later instead of downloading it.
    """
    tree = parse("app/classification/dataset.py")
    get_datasets = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "get_datasets"
    )

    called = {
        node.func.id
        for node in ast.walk(get_datasets)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "ensure_local_data" in called, (
        "get_datasets() no longer calls ensure_local_data() -- a machine "
        "without the dataset will fail instead of downloading it"
    )


# --- The one test that really imports the project -------------------------
# Skipped where TensorFlow is absent (CI), so it costs nothing there but still
# runs on the Mac and the VM, where a real import is the stronger check.

@pytest.mark.parametrize("module_name", [
    "app.params",
    "app.classification.dataset",
    "app.utils.format",
    "app.classification.models",
    "app.classification.models.layers",
    "app.classification.models.model_christophe",
    "app.classification.models.model_vgg16",
    "app.classification.train",
    "app.classification.evaluate",
    "app.classification.predict",
    "app.classification.registry",
])
def test_modules_import(module_name):
    """Actually import each module, where the heavy dependencies exist."""
    pytest.importorskip("tensorflow", reason="TensorFlow not installed (CI)")
    __import__(module_name)
