import os
from io import BytesIO
from pathlib import Path

import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from pillow_heif import register_heif_opener


# iPhone photos AirDropped to the Mac arrive as .heic, which Pillow (and so
# st.image) cannot open on its own. One call at import fixes that for the
# whole process. Registering twice is harmless.
register_heif_opener()


API_URL = os.getenv("API_URL")

# Must be the FIRST Streamlit command executed -- page title (browser tab)
# and favicon, the icon cut from the logo's brick mark.
st.set_page_config(
    page_title="LegoFitter",
    page_icon=str(Path(__file__).parent / "assets" / "favicon.png"),
    layout="centered",
)


@st.cache_data(show_spinner=False)
def load_part_image(url: str) -> bytes | None:
    """Download a Rebrickable part photo and make its white background
    transparent, so the brick floats on the page instead of sitting in a
    stark white tile.

    Flood fill from the four corners, not a global threshold: only white
    that is CONNECTED to the border becomes transparent, so a white brick
    keeps its own white pixels. thresh=40 also catches the off-white
    shadow gradient around Rebrickable's studio shots.

    Cached by URL (Streamlit reruns this whole script on every click);
    returns None on any failure so a broken image never breaks the page.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        picture = Image.open(BytesIO(response.content)).convert("RGBA")

        corners = [
            (0, 0),
            (picture.width - 1, 0),
            (0, picture.height - 1),
            (picture.width - 1, picture.height - 1),
        ]
        for corner in corners:
            ImageDraw.floodfill(picture, corner, (0, 0, 0, 0), thresh=40)

        output = BytesIO()
        picture.save(output, format="PNG")
        return output.getvalue()

    except Exception:
        return None


# ============================================================
# HEADER
# ============================================================

# The logo has a transparent background with black lettering, which is
# why .streamlit/config.toml pins the light theme -- on Streamlit's dark
# theme "FITTER" would vanish. Resolved from this file, not the CWD, so
# it works from the repo root, the container and anywhere else.
st.image(str(Path(__file__).parent / "assets" / "logo.png"), width=520)

st.write("Test de connexion à l'API LegoFitter")


if st.button("Tester l'API"):
    try:
        # 30s, not 10: a cold-started Cloud Run container imports
        # TensorFlow + PyTorch before it can answer, and the whole point of
        # this button is to tell "API down" apart from "API waking up".
        response = requests.get(
            f"{API_URL}/ping",
            timeout=30,
        )

        response.raise_for_status()

        st.success("API connectée")
        st.json(response.json())

    except requests.RequestException as e:
        st.error(f"Erreur de connexion à l'API : {e}")


# ============================================================
# ANALYSE LEGO
# ============================================================

st.divider()

st.subheader("Identifier des pièces LEGO")

uploaded_file = st.file_uploader(
    "Choisissez une image",
    type=["jpg", "jpeg", "png", "heic", "heif"],
)


if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Image à analyser",
        width=400,
    )

    if st.button("Analyser l'image"):

        try:

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type,
                )
            }

            with st.spinner(
                "Détection des briques et recherche d'un set LEGO..."
            ):

                response = requests.post(
                    f"{API_URL}/predict",
                    files=files,
                    timeout=120,
                )

            # ====================================================
            # SUCCÈS
            # ====================================================

            if response.status_code == 200:

                result = response.json()

                st.success(
                    f"{result['total_bricks']} pièce(s) LEGO détectée(s)"
                )

                # ------------------------------------------------
                # BOUNDING BOXES
                # ------------------------------------------------

                annotated_image = Image.open(
                    BytesIO(uploaded_file.getvalue())
                ).convert("RGB")

                draw = ImageDraw.Draw(annotated_image)

                # Same style as app/detection/predict.py's draw_boxes():
                # labels and line widths SCALE WITH THE PHOTO (a 4000px
                # iPhone shot gets labels a room can read, a 500px render
                # gets small ones), each label sits on a filled background
                # so it never blends into a busy photo, and the colour is
                # a vivid green -- red vanished against warm-coloured
                # bricks and its bare text was unreadable.
                # Keep in sync with params.ANNOTATION_COLOR (not imported:
                # the slim Streamlit container has no dotenv, and one
                # colour is not worth widening that container for).
                annotation_color = "#00E676"

                shortest = min(annotated_image.size)
                font_size = max(16, shortest // 30)
                line_width = max(4, shortest // 300)
                pad = max(2, font_size // 5)
                font = ImageFont.load_default(size=font_size)

                for detection in result.get("detections", []):

                    x1, y1, x2, y2 = detection["bbox"]

                    part_id = detection["part_id"]
                    confidence = detection[
                        "classification_confidence"
                    ]

                    label = f"{part_id} {confidence:.0%}"

                    draw.rectangle(
                        [x1, y1, x2, y2],
                        outline=annotation_color,
                        width=line_width,
                    )

                    # Label just above the box; just below it when the box
                    # touches the top edge of the photo.
                    text_width = draw.textlength(label, font=font)
                    text_y = y1 - font_size - 2 * pad - line_width
                    if text_y < 0:
                        text_y = y2 + line_width

                    draw.rectangle(
                        [
                            x1,
                            text_y,
                            x1 + text_width + 2 * pad,
                            text_y + font_size + 2 * pad,
                        ],
                        fill=annotation_color,
                    )

                    draw.text(
                        (x1 + pad, text_y + pad),
                        label,
                        fill="black",
                        font=font,
                    )

                st.subheader("Pièces détectées")

                st.image(
                    annotated_image,
                    caption="Détection YOLO + classification CNN",
                    width=600,
                )

                # ------------------------------------------------
                # INVENTAIRE
                # ------------------------------------------------

                st.subheader("Briques identifiées")

                details = result.get("inventory_details")

                if details:

                    columns = st.columns(4)

                    for i, part in enumerate(details):

                        with columns[i % 4]:

                            image_bytes = (
                                load_part_image(part["img_url"])
                                if part.get("img_url")
                                else None
                            )

                            if image_bytes:
                                st.image(image_bytes, width=150)

                            st.markdown(
                                f"**{part['count']} ×** "
                                f"{part.get('name') or part['part_id']}"
                            )

                            st.caption(f"Réf. {part['part_id']}")

                else:
                    # API without part details (no Rebrickable key, or an
                    # older deployment): raw counts beat nothing.
                    st.json(result["inventory"])

                # ------------------------------------------------
                # SET RECOMMANDÉ
                # ------------------------------------------------

                recommended_set = result.get(
                    "recommended_set"
                )

                if recommended_set:

                    st.divider()

                    st.subheader("Set LEGO recommandé")

                    st.write(
                        f"### {recommended_set['name']}"
                    )

                    st.write(
                        f"Set : {recommended_set['set_num']}"
                    )

                    if recommended_set.get("set_img_url"):

                        st.image(
                            recommended_set["set_img_url"],
                            width=400,
                        )

                    st.write(
                        "Pièces photographiées utilisées : "
                        f"{recommended_set['inventory_coverage']:.1%}"
                    )

                    st.write(
                        "Complétion du set : "
                        f"{recommended_set['compatibility']:.1%}"
                    )

                    if recommended_set["buildable"]:

                        st.success(
                            "Ce set est entièrement constructible "
                            "avec les pièces détectées."
                        )

                    else:

                        st.info(
                            "C'est le set le plus compatible "
                            "avec les pièces détectées."
                        )

                else:

                    st.warning(
                        "Aucun set LEGO compatible trouvé."
                    )

            # ====================================================
            # FORMAT NON SUPPORTÉ
            # ====================================================

            elif response.status_code == 415:

                st.error(
                    response.json().get(
                        "detail",
                        "Format d'image non supporté.",
                    )
                )

            # ====================================================
            # AUTRE ERREUR
            # ====================================================

            else:

                st.error(
                    f"Erreur API ({response.status_code}) : "
                    f"{response.text}"
                )

        except requests.RequestException as e:

            st.error(
                f"Impossible de contacter l'API : {e}"
            )

