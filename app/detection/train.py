"""
Objective 2, Step 2 — YOLOv5 training wrapper.

Owns: kicking off YOLOv5's own train.py (cloned per README Phase 2 setup)
against the dataset YAML produced by prepare_data.py, and copying the
resulting best.pt to app/params.DETECTION_MODEL_PATH.

Not implemented yet — see README.md Phase 2, Step 2.
"""
from pathlib import Path
import argparse

from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_YAML = ROOT_DIR / "data" / "lego-yolo-dataset" / "data.yaml"
DEFAULT_PROJECT_DIR = ROOT_DIR / "models" / "runs"
DEFAULT_MODEL = "yolo26n.pt"


def train_model(
    data_yaml=DEFAULT_DATASET_YAML,
    model_name=DEFAULT_MODEL,
    epochs=50,
    imgsz=640,
    batch=8,
    patience=10,
    run_name="lego_yolo_v1",
):
    data_yaml = Path(data_yaml)

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Fichier data.yaml introuvable : {data_yaml}"
        )

    DEFAULT_PROJECT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_name)

    results = model.train(
        data=str(data_yaml.resolve()),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        project=str(DEFAULT_PROJECT_DIR.resolve()),
        name=run_name,
        exist_ok=True,
    )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Entraînement YOLO sur le dataset LEGO."
    )

    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATASET_YAML),
        help="Chemin vers data.yaml.",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Modèle YOLO à utiliser.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Nombre d'epochs.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Taille des images.",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Taille du batch.",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Nombre d'epochs sans amélioration avant arrêt anticipé.",
    )

    parser.add_argument(
        "--name",
        default="lego_yolo_v1",
        help="Nom de l'expérience.",
    )

    args = parser.parse_args()

    train_model(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        run_name=args.name,
    )


if __name__ == "__main__":
    main()
