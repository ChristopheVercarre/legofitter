"""
Objective 2, Step 1 -- dataset prep for YOLO.

Owns: converting data/lego-tagged-object_detection (VOC XML annotations) into
the layout YOLO expects -- images/<split>/ + labels/<split>/*.txt, one class
"lego brick" -- and writing the dataset YAML that YOLO reads.

Single class on purpose: the detector only has to answer WHERE a brick is.
WHICH brick it is, is Objective 1's classifier, and Objective 3 chains the two.

Run it with:
    python -m app.detection.prepare_data
"""

import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from app.params import (
    DETECTION_CLASS_ID,
    DETECTION_CLASS_NAME,
    DETECTION_DATA_DIR,
    DETECTION_IMAGE_EXTENSIONS,
    DETECTION_SOURCE_NAMES,
    RANDOM_STATE,
    TEST_SIZE,
    VAL_SIZE,
    YOLO_DATA_DIR,
    YOLO_DATASET_YAML,
)

SOURCE_DIRS = (DETECTION_DATA_DIR / "photos", DETECTION_DATA_DIR / "renders")
SPLITS = ("train", "val", "test")


def find_source_files(pattern: str = "*") -> list[Path]:
    """Every file under photos/ and renders/ matching `pattern`, sorted.

    sorted() is what makes the split reproducible across machines. rglob
    returns filesystem order, which differs between the Mac (APFS) and the VM
    (ext4) -- and random.seed(RANDOM_STATE) only reproduces a shuffle if the
    list going INTO it is in the same order. Without this, a detector trained
    on the VM would be evaluated on the Mac against a test split that overlaps
    its own training data. Same fix as dataset.py's build_dataframe().
    """
    files = []
    for base_dir in SOURCE_DIRS:
        if base_dir.exists():
            files.extend(sorted(path for path in base_dir.rglob(pattern) if path.is_file()))
    return files


def convert_box_to_yolo(xmin, ymin, xmax, ymax, width, height):
    """VOC corner pixels -> YOLO centre/size, as fractions of the image.

    VOC says "the box runs from pixel 30 to pixel 210". YOLO wants "the box is
    centred 40% across, 60% down, and is 25% wide" -- so the label survives the
    image being resized, which is exactly what happens during training.

    Clamped to 0..1: a few boxes in the source data run a pixel or two past the
    image edge, and YOLO rejects labels outside that range.
    """
    x_center = ((xmin + xmax) / 2) / width
    y_center = ((ymin + ymax) / 2) / height
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height

    return tuple(min(max(value, 0.0), 1.0)
                 for value in (x_center, y_center, box_width, box_height))


def convert_xml_to_yolo_txt(xml_file: Path) -> int:
    """Write one VOC .xml out as a YOLO .txt beside it. Returns boxes written.

    The .txt lands next to the .xml so find_image_label_pairs() can match them
    by filename. It is a generated file inside the SOURCE folder -- see the
    note in main() about re-running.
    """
    root = ET.parse(xml_file).getroot()

    size = root.find("size")
    if size is None:
        return 0

    width = float(size.find("width").text)
    height = float(size.find("height").text)

    lines = []
    for obj in root.findall("object"):
        name = obj.find("name")
        box = obj.find("bndbox")
        if name is None or name.text is None or box is None:
            continue
        if name.text.strip().lower() not in DETECTION_SOURCE_NAMES:
            continue

        x_center, y_center, box_width, box_height = convert_box_to_yolo(
            float(box.find("xmin").text),
            float(box.find("ymin").text),
            float(box.find("xmax").text),
            float(box.find("ymax").text),
            width,
            height,
        )
        lines.append(
            f"{DETECTION_CLASS_ID} {x_center:.6f} {y_center:.6f} "
            f"{box_width:.6f} {box_height:.6f}"
        )

    xml_file.with_suffix(".txt").write_text("\n".join(lines), encoding="utf-8")
    return len(lines)


def find_image_label_pairs() -> list[tuple[Path, Path]]:
    """Every image that has a .txt label next to it, as (image, label) pairs."""
    return [
        (image, image.with_suffix(".txt"))
        for image in find_source_files()
        if image.suffix.lower() in DETECTION_IMAGE_EXTENSIONS
        and image.with_suffix(".txt").exists()
    ]


def reset_target_dirs() -> None:
    """Delete and recreate the YOLO dataset folder.

    Deleted rather than overwritten: leftovers from a previous run with
    different settings would otherwise be silently trained on.
    """
    if YOLO_DATA_DIR.exists():
        shutil.rmtree(YOLO_DATA_DIR)

    for split in SPLITS:
        (YOLO_DATA_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DATA_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def split_pairs(pairs: list) -> dict[str, list]:
    """Shuffle once, then cut into train / val / test.

    Uses the same TEST_SIZE and VAL_SIZE as the classifier (params.py), so both
    objectives hold out the same proportions -- 70 / 15 / 15 -- and the same
    RANDOM_STATE, so the split is identical on every machine.

    Copies the list before shuffling: random.shuffle works in place, and a
    function that silently reorders its caller's data is a nasty surprise.
    """
    pairs = list(pairs)
    random.Random(RANDOM_STATE).shuffle(pairs)

    n_test = int(TEST_SIZE * len(pairs))
    n_val = int(VAL_SIZE * len(pairs))

    return {
        "test": pairs[:n_test],
        "val": pairs[n_test:n_test + n_val],
        "train": pairs[n_test + n_val:],
    }


def link_split(split_name: str, pairs: list) -> None:
    """Point one split's images and labels at the source files, without copying.

    YOLO only ever reads these files, and it finds them through data.yaml -- it
    never cares whether what it opens is a real file or a symlink. Copying would
    write a second full copy of the dataset (~6 GB) to disk for no benefit, which
    on the VM is the difference between a step that finishes in seconds and one
    that fills the disk.

    Symlinks are not available everywhere (Windows without developer mode, some
    network drives), so a failure falls back to a real copy rather than stopping
    the run.
    """
    for image_file, label_file in pairs:
        for source, kind in ((image_file, "images"), (label_file, "labels")):
            destination = YOLO_DATA_DIR / kind / split_name / source.name
            try:
                destination.symlink_to(source.resolve())
            except OSError:
                shutil.copy2(source, destination)


def write_dataset_yaml() -> None:
    """The one file YOLO is pointed at; everything else it finds from here."""
    YOLO_DATASET_YAML.write_text(
        "\n".join([
            f"path: {YOLO_DATA_DIR}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "",
            "names:",
            f"  {DETECTION_CLASS_ID}: {DETECTION_CLASS_NAME}",
        ]),
        encoding="utf-8",
    )


def prepare_data() -> dict[str, list]:
    """VOC XML in, YOLO dataset out. Returns the splits."""
    xml_files = find_source_files("*.xml")
    if not xml_files:
        raise FileNotFoundError(f"No XML annotations found in {DETECTION_DATA_DIR}")
    print(f"✅ Found {len(xml_files):,} XML annotation file(s)")

    # NOTE: this writes a .txt next to every .xml, inside the source dataset.
    # Re-running is safe (each file is overwritten), but a .txt whose .xml was
    # deleted would linger and still be picked up as a pair.
    total_boxes = sum(convert_xml_to_yolo_txt(xml_file) for xml_file in xml_files)
    print(f"✅ Converted to YOLO format: {total_boxes:,} bounding box(es)")

    pairs = find_image_label_pairs()
    if not pairs:
        raise FileNotFoundError("No image/label pairs found after the XML -> TXT conversion")
    print(f"✅ Matched {len(pairs):,} image/label pair(s)")

    reset_target_dirs()
    splits = split_pairs(pairs)
    for split_name in SPLITS:
        link_split(split_name, splits[split_name])
    print(
        f"✅ Split: {len(splits['train']):,} train / "
        f"{len(splits['val']):,} val / {len(splits['test']):,} test"
    )

    write_dataset_yaml()
    print(f"✅ Dataset ready: {YOLO_DATASET_YAML}")
    return splits


if __name__ == "__main__":
    prepare_data()
