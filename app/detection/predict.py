"""
Objective 2, Step 4 -- single-image detection.

Owns: running the trained detector on one photo and returning the bounding
boxes as plain data, plus cutting those boxes out as crops for Objective 3.

The detection twin of app/classification/predict.py, function for function:

    predict_boxes(image)        <- predict_image(image)
    load_image_for_detection()  <- load_image_for_prediction()

"Plain data" is the whole point of this file existing separately from
main.py's detect(), which prints. A printed box cannot be turned into JSON,
cropped, or counted -- and all three are things something else in this
project needs to do with it: FastAPI serialises the boxes into a response,
and Objective 3's pipeline crops each one and hands it to the classifier.

Run directly:
    python -m app.detection.predict path/to/photo.jpg
"""

import sys
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps
from pillow_heif import register_heif_opener

from app.detection.registry import load_detector
from app.params import DETECTION_CONFIDENCE, DETECTION_CROP_PADDING, ANNOTATION_COLOR

# Teach Pillow to open iPhone HEIC photos. One call, at import, and every
# Image.open() in this process -- here, in the pipeline, in the Streamlit
# app -- handles .heic like any other format. Without it, an AirDropped
# demo photo dies with "cannot identify image file". Registering twice is
# harmless, so nobody has to wonder whether it already happened.
register_heif_opener()


def load_image_for_detection(image) -> Image.Image:
    """Normalise any supported input into one RGB PIL image.

    `image` is a path (str or Path), raw image bytes, or an already-open PIL
    image. Bytes are what a FastAPI upload hands you -- `await file.read()` --
    and writing them to a temp file just to have a path to read back would be
    pure ceremony, plus a file to clean up in a container.

    ultralytics accepts paths and PIL images directly, so routing everything
    through PIL here rather than passing paths straight through is a choice,
    and it buys two things that matter for photos taken on a phone:

    convert("RGB") -- a PNG upload arrives with an alpha channel, i.e. 4
    channels where the model wants 3.

    exif_transpose() -- a phone writes the image sideways and adds an EXIF
    tag saying "rotate me". ultralytics' path loader goes through cv2, which
    ignores that tag, so a portrait photo would be detected in landscape and
    every box would come back at the wrong coordinates. Doing it here means a
    path and an upload of the SAME photo give the same boxes, which is what
    makes a local test meaningful for the deployed API.
    """
    if isinstance(image, (bytes, bytearray)):
        image = Image.open(BytesIO(image))
    elif isinstance(image, (str, Path)):
        image = Image.open(image)

    return ImageOps.exif_transpose(image).convert("RGB")


def predict_boxes(
    image, confidence: float = DETECTION_CONFIDENCE, detector=None
) -> list[dict]:
    """Find every brick in one image. Returns a list of plain dicts:

        [{"box": [x_min, y_min, x_max, y_max], "confidence": 0.94,
          "class_name": "lego"}, ...]

    Coordinates are pixels in the image's own frame, top-left origin, ready
    to hand to PIL's crop() as-is.

    Every value is a Python int / float / str, deliberately: ultralytics
    hands back torch tensors and numpy floats, and FastAPI raises at
    serialisation time on both. Converting here rather than in the endpoint
    means the notebook and the API get the same objects.

    `detector` is optional, and the API should pass one. Loading a detector
    takes seconds, so a Cloud Run endpoint loads it ONCE at module import and
    passes it in on every request; left as None this falls back to
    load_detector()'s usual local-then-bucket lookup, which is what a
    notebook wants.

    Boxes come back sorted by confidence, highest first, so a caller showing
    only the best few does not have to sort them itself.
    """
    if detector is None:
        detector = load_detector()

    result = detector.predict(
        load_image_for_detection(image), conf=confidence, verbose=False
    )[0]

    boxes = [
        {
            "box": [round(float(v)) for v in box.xyxy[0]],
            "confidence": float(box.conf[0]),
            # From the run's own data.yaml, not a hardcoded "lego": the day
            # this becomes a multi-class detector, this line already works.
            "class_name": result.names[int(box.cls[0])],
        }
        for box in result.boxes
    ]

    return sorted(boxes, key=lambda found: found["confidence"], reverse=True)


def crop_boxes(
    image, boxes: list[dict], padding: float = DETECTION_CROP_PADDING
) -> list[Image.Image]:
    """Cut each detected box out of the image. The Objective 3 handoff.

    One crop per box, in the same order, so crops[i] belongs to boxes[i] --
    which is what lets the pipeline attach a part ID back to the box it came
    from.

    Kept here rather than in the pipeline because it has to use the same
    normalised image predict_boxes() measured the coordinates against;
    re-opening the file in the pipeline is how you get crops that are subtly
    offset on EXIF-rotated photos.

    `padding` grows each box by that fraction before cutting. The detector
    draws tight boxes, and the classifier was trained on images where the
    brick sits inside a margin -- so an edge-to-edge crop is a small
    train/serve mismatch, and this puts the margin back. Clamped to the
    image, so a brick against the edge is padded as far as the photo allows
    and no further. Pass 0 for the raw detected box.
    """
    picture = load_image_for_detection(image)
    width, height = picture.size

    crops = []
    for found in boxes:
        x_min, y_min, x_max, y_max = found["box"]
        margin_x = round((x_max - x_min) * padding)
        margin_y = round((y_max - y_min) * padding)
        crops.append(
            picture.crop((
                max(0, x_min - margin_x),
                max(0, y_min - margin_y),
                min(width, x_max + margin_x),
                min(height, y_max + margin_y),
            ))
        )
    return crops


def draw_boxes(
    image, boxes: list[dict], labels: list[str] | None = None
) -> Image.Image:
    """Draw every box in `boxes` on a COPY of `image`, return that copy.

    The source is never mutated: draw_boxes() only ever reads what
    predict_boxes() found, so calling it more than once on the same photo
    (once to look at, once to save) never draws the same box twice on top
    of itself.

    `labels`, when given, must be the same length and IN THE SAME ORDER as
    `boxes` -- e.g. Objective 3's classifier output ("3001 94%") per box,
    so the picture shows what each box was classified as, not just that
    something was detected there. Left as None, each box is labelled with
    its own detection confidence instead, which is enough to sanity-check
    the detector alone (README Objective 3 Step 2).

    Uses PIL's default font on purpose: no font file to ship or look up
    across machines. load_default(size=...) (Pillow >= 10.1) renders it at
    any size, so labels and line widths SCALE WITH THE PHOTO -- a 4000px
    iPhone shot gets ~100px labels a room can read, a 500px render gets
    small ones, and both stay proportionate.
    """
    picture = load_image_for_detection(image).copy()
    draw = ImageDraw.Draw(picture)

    shortest = min(picture.size)
    font_size = max(14, shortest // 30)
    line_width = max(3, shortest // 400)
    pad = max(2, font_size // 5)
    font = ImageFont.load_default(size=font_size)

    for i, found in enumerate(boxes):
        x_min, y_min, x_max, y_max = found["box"]
        label = labels[i] if labels is not None else f"{found['confidence']:.0%}"

        draw.rectangle((x_min, y_min, x_max, y_max), outline=ANNOTATION_COLOR, width=line_width)

        # Label sits just above the box (or just below it, if the box is
        # flush with the top edge) on a filled background, so it stays
        # readable over a busy photo instead of blending into it.
        text_width = draw.textlength(label, font=font)
        text_y = y_min - font_size - 2 * pad - line_width
        if text_y < 0:
            text_y = y_max + line_width
        draw.rectangle(
            (x_min, text_y, x_min + text_width + 2 * pad, text_y + font_size + 2 * pad),
            fill=ANNOTATION_COLOR,
        )
        draw.text((x_min + pad, text_y + pad), label, fill="black", font=font)

    return picture


if __name__ == "__main__":
    # python -m app.detection.predict path/to/photo.jpg
    if len(sys.argv) != 2:
        sys.exit("Usage: python -m app.detection.predict <image_path>")

    found = predict_boxes(sys.argv[1])
    print(f"\n{len(found)} brick(s) detected in {sys.argv[1]}:")
    for rank, brick in enumerate(found, start=1):
        x_min, y_min, x_max, y_max = brick["box"]
        print(
            f"  {rank:>3}. {brick['class_name']:<10} {brick['confidence']:.2%}   "
            f"({x_min}, {y_min}) -> ({x_max}, {y_max})"
        )
