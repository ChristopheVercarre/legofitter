"""
Packaging config for LegoFitter.

The point of this file is `pip install -e .` from the project root. That does
two useful things:

  1. It puts `app` on Python's import path permanently, so
     `from app.params import IMG_SIZE` works from anywhere — a notebook in
     notebooks/, a script in the repo root, a python shell in /tmp. That is
     what lets us delete the `sys.path.append("..")` hack at the top of
     datascientist_deliverable.ipynb.

  2. The `-e` (editable) flag means pip installs a LINK to this directory
     rather than a copy. Edit app/classification/model.py and the change is
     live immediately — no reinstall. Without -e you would be running a stale
     snapshot of the code every time, which is a genuinely confusing bug to
     chase.

For a teammate setting up on the VM:

    pyenv virtualenv 3.12.9 legofitter
    pyenv local legofitter
    pip install -e .

That single command also installs everything in requirements.txt, so there is
no separate `pip install -r requirements.txt` step.

Note on modern packaging: `pyproject.toml` is the current standard and
setup.py is technically legacy, but setup.py is still fully supported by pip
and is what most tutorials use. Nothing here needs changing unless you start
publishing to PyPI.
"""

from pathlib import Path

from setuptools import find_packages, setup

PROJECT_ROOT = Path(__file__).parent


def read_requirements() -> list[str]:
    """Parse requirements.txt into a list setuptools can consume.

    We strip blank lines and comments (our requirements.txt is heavily
    commented, grouped by project phase) and skip `-e` / `-r` lines, which
    are pip flags rather than package specifiers and would break the install.

    A note on the purist objection: requirements.txt is meant for *pinned,
    reproducible* environments while install_requires is meant for *abstract*
    dependencies ("needs pandas, any version"). Keeping them separate matters
    for a library published to PyPI. For an application like this one, where
    we actively WANT everyone on identical pinned versions, reading one from
    the other is the simpler and safer choice — one file to maintain, no
    chance of the two drifting apart.
    """
    requirements_path = PROJECT_ROOT / "requirements.txt"

    if not requirements_path.exists():
        return []

    lines = requirements_path.read_text().splitlines()

    return [
        line.strip()
        for line in lines
        if line.strip()                      # drop blank lines
        and not line.strip().startswith("#") # drop comment lines
        and not line.strip().startswith("-") # drop pip flags like -e / -r
    ]


setup(
    name="legofitter",
    version="0.1.0",
    description=(
        "Classify LEGO bricks from a photo, detect them in a pile, and "
        "suggest sets that can be built from the result."
    ),
    author="Christophe Vercarre",

    # find_packages() walks the tree and picks up every directory containing
    # an __init__.py — here that means app, app.classification, app.detection,
    # app.pipeline, app.rebrickable and app.utils.
    #
    # The exclude list keeps tests and build artefacts out of the install. If
    # you ever add a package and it mysteriously fails to import after a
    # reinstall, the first thing to check is a missing __init__.py.
    packages=find_packages(exclude=["tests", "tests.*", "notebooks", "notebooks.*"]),

    install_requires=read_requirements(),

    # No console_scripts entry point for now: main_local.py (the project-level
    # CLI it would have called into) was removed while we're keeping things
    # simple. Re-add one here if/when a project-level entry point comes back.

    # Informational only — pip warns rather than refuses if you are on an
    # older interpreter. Our .python-version pins the actual environment.
    python_requires=">=3.10",

    include_package_data=True,
    zip_safe=False,
)
