"""
Objective 2, Step 1 — dataset prep for YOLOv5.

Owns: converting data/lego-tagged-object_detection (VOC XML annotations)
into YOLOv5's expected format (images/ + labels/*.txt, single class
"lego brick") and writing the dataset YAML YOLOv5 needs.

Not implemented yet — see README.md Phase 2, Step 1.
"""
from pathlib import Path
import random
import shutil
import xml.etree.ElementTree as ET


ROOT_DIR = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT_DIR / "data" / "lego-tagged-object_detection"
TARGET_DIR = ROOT_DIR / "data" / "lego-yolo-dataset"

PHOTOS_DIR = SOURCE_DIR / "photos"
RENDERS_DIR = SOURCE_DIR / "renders"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VALID_CLASS_NAMES = {"lego", "legod"}
TARGET_CLASS_NAME = "lego"
TARGET_CLASS_ID = 0
RANDOM_SEED = 42


def find_xml_files():
    xml_files = []

    for base_dir in [PHOTOS_DIR, RENDERS_DIR]:
        if base_dir.exists():
            xml_files.extend(
                path for path in base_dir.rglob("*.xml")
                if path.is_file()
            )

    return xml_files


def normalize_class_name(class_name):
    class_name = class_name.strip().lower()

    if class_name in VALID_CLASS_NAMES:
        return TARGET_CLASS_NAME

    return None


def convert_box_to_yolo(xmin, ymin, xmax, ymax, width, height):
    x_center = ((xmin + xmax) / 2) / width
    y_center = ((ymin + ymax) / 2) / height
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height

    x_center = min(max(x_center, 0.0), 1.0)
    y_center = min(max(y_center, 0.0), 1.0)
    box_width = min(max(box_width, 0.0), 1.0)
    box_height = min(max(box_height, 0.0), 1.0)

    return x_center, y_center, box_width, box_height


def convert_xml_to_yolo_txt(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    size = root.find("size")
    if size is None:
        return 0

    width = float(size.find("width").text)
    height = float(size.find("height").text)

    yolo_annotations = []

    for obj in root.findall("object"):
        name_element = obj.find("name")
        if name_element is None or name_element.text is None:
            continue

        normalized_name = normalize_class_name(name_element.text)
        if normalized_name is None:
            continue

        box = obj.find("bndbox")
        if box is None:
            continue

        xmin = float(box.find("xmin").text)
        ymin = float(box.find("ymin").text)
        xmax = float(box.find("xmax").text)
        ymax = float(box.find("ymax").text)

        x_center, y_center, box_width, box_height = convert_box_to_yolo(
            xmin, ymin, xmax, ymax, width, height
        )

        yolo_annotations.append(
            f"{TARGET_CLASS_ID} {x_center:.6f} {y_center:.6f} "
            f"{box_width:.6f} {box_height:.6f}"
        )

    txt_file = xml_file.with_suffix(".txt")
    txt_file.write_text("\n".join(yolo_annotations), encoding="utf-8")

    return len(yolo_annotations)


def find_image_label_pairs():
    image_files = []

    for base_dir in [PHOTOS_DIR, RENDERS_DIR]:
        if base_dir.exists():
            image_files.extend(
                path for path in base_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )

    pairs = []

    for image_file in image_files:
        label_file = image_file.with_suffix(".txt")
        if label_file.exists():
            pairs.append((image_file, label_file))

    return pairs


def reset_target_dirs():
    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)

    for split in ["train", "val", "test"]:
        (TARGET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (TARGET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def split_pairs(pairs):
    random.seed(RANDOM_SEED)
    random.shuffle(pairs)

    n = len(pairs)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)

    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]
    test_pairs = pairs[n_train + n_val:]

    return train_pairs, val_pairs, test_pairs


def copy_split(split_name, split_pairs):
    for image_file, label_file in split_pairs:
        shutil.copy2(image_file, TARGET_DIR / "images" / split_name / image_file.name)
        shutil.copy2(label_file, TARGET_DIR / "labels" / split_name / label_file.name)


def write_dataset_yaml():
    yaml_lines = [
        f"path: {TARGET_DIR}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "names:",
        "  0: lego",
    ]

    yaml_path = TARGET_DIR / "data.yaml"
    yaml_path.write_text("\n".join(yaml_lines), encoding="utf-8")


def main():
    xml_files = find_xml_files()
    if not xml_files:
        raise FileNotFoundError("Aucun fichier XML trouvé dans data/lego-tagged-object_detection.")

    total_annotations = 0
    for xml_file in xml_files:
        total_annotations += convert_xml_to_yolo_txt(xml_file)

    pairs = find_image_label_pairs()
    if not pairs:
        raise FileNotFoundError("Aucune paire image/label trouvée après conversion XML -> TXT.")

    reset_target_dirs()

    train_pairs, val_pairs, test_pairs = split_pairs(pairs)

    copy_split("train", train_pairs)
    copy_split("val", val_pairs)
    copy_split("test", test_pairs)

    write_dataset_yaml()

    print(f"XML trouvés : {len(xml_files)}")
    print(f"Annotations YOLO générées : {total_annotations}")
    print(f"Paires image/label : {len(pairs)}")
    print(f"Train : {len(train_pairs)}")
    print(f"Val : {len(val_pairs)}")
    print(f"Test : {len(test_pairs)}")
    print(f"Dataset YAML : {TARGET_DIR / 'data.yaml'}")


if __name__ == "__main__":
    main()
