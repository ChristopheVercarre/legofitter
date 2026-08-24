"""
LegoFitter — local CLI entry point.

This wires up one subcommand per project step so we can run each phase in
isolation as we build it out. Nothing is implemented yet — each subcommand
is a placeholder we'll fill in together, one step at a time, in this order:

  Objective 1 — Classification
    classify-prepare-data   Downsize the dataset to 50 classes
    classify-train           Train the custom CNN
    classify-evaluate        Score accuracy (gate: >=70% before Objective 2)

  Objective 2 — Object Detection
    detect-prepare-data      Convert annotations to YOLOv5 format
    detect-train              Train YOLOv5 on the single "lego brick" class
    detect-evaluate           Score mAP

  Objective 3 — Chaining
    pipeline-run              Detect -> crop -> classify -> inventory, on one image

  Bonus — Rebrickable
    match-sets                Call get_set_match with the inventory

See README.md for the full roadmap and what each step needs before it can run.
"""
import argparse


def main():
    parser = argparse.ArgumentParser(description="LegoFitter local CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("classify-prepare-data", help="Objective 1, step 1")
    subparsers.add_parser("classify-train", help="Objective 1, step 2")
    subparsers.add_parser("classify-evaluate", help="Objective 1, step 3")

    subparsers.add_parser("detect-prepare-data", help="Objective 2, step 1")
    subparsers.add_parser("detect-train", help="Objective 2, step 2")
    subparsers.add_parser("detect-evaluate", help="Objective 2, step 3")

    pipeline_parser = subparsers.add_parser("pipeline-run", help="Objective 3")
    pipeline_parser.add_argument("image", help="Path to a photo of multiple bricks")

    subparsers.add_parser("match-sets", help="Bonus: Rebrickable get_set_match")

    args = parser.parse_args()

    dispatch = {
        "classify-prepare-data": _not_implemented,
        "classify-train": _not_implemented,
        "classify-evaluate": _not_implemented,
        "detect-prepare-data": _not_implemented,
        "detect-train": _not_implemented,
        "detect-evaluate": _not_implemented,
        "pipeline-run": _not_implemented,
        "match-sets": _not_implemented,
    }
    dispatch[args.command]()


def _not_implemented():
    raise NotImplementedError(
        "Not built yet — we're taking the roadmap in README.md one step at a "
        "time. Ask Claude to implement the next step when you're ready."
    )


if __name__ == "__main__":
    main()
