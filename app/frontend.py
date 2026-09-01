import os
from io import BytesIO

import requests
import streamlit as st
from PIL import Image, ImageDraw
from pillow_heif import register_heif_opener


# iPhone photos AirDropped to the Mac arrive as .heic, which Pillow (and so
# st.image) cannot open on its own. One call at import fixes that for the
# whole process. Registering twice is harmless.
register_heif_opener()


API_URL = os.getenv("API_URL")


# ============================================================
# HEADER
# ============================================================

st.title("LegoFitter")

st.write("Test de connexion à l'API LegoFitter")


if st.button("Tester l'API"):
    try:
        response = requests.get(
            f"{API_URL}/ping",
            timeout=10,
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

                for detection in result.get("detections", []):

                    x1, y1, x2, y2 = detection["bbox"]

                    part_id = detection["part_id"]
                    confidence = detection[
                        "classification_confidence"
                    ]

                    draw.rectangle(
                        [x1, y1, x2, y2],
                        outline="red",
                        width=4,
                    )

                    draw.text(
                        (x1, max(0, y1 - 20)),
                        f"{part_id} - {confidence:.0%}",
                        fill="red",
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

                st.subheader("Inventaire détecté")

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


# ============================================================
# TEST MANUEL REBRICKABLE
# ============================================================

st.divider()

st.subheader("Rechercher une pièce LEGO")

part_id = st.text_input(
    "ID de la pièce",
    placeholder="Exemple : 3001",
)


if st.button("Rechercher la pièce"):

    if not part_id:

        st.warning("Entrez un ID de pièce.")

    else:

        try:

            response = requests.get(
                f"{API_URL}/parts/{part_id}",
                timeout=10,
            )

            if response.status_code == 200:

                part = response.json()

                st.success(
                    f"{part['name']} — {part['part_num']}"
                )

                if part.get("part_img_url"):

                    st.image(
                        part["part_img_url"],
                        width=300,
                    )

                st.write(
                    f"Année : "
                    f"{part['year_from']} → "
                    f"{part['year_to']}"
                )

                if part.get("part_url"):

                    st.link_button(
                        "Voir sur Rebrickable",
                        part["part_url"],
                    )

            elif response.status_code == 404:

                st.error(
                    "Pièce introuvable sur Rebrickable."
                )

            else:

                st.error(
                    f"Erreur API ({response.status_code}) : "
                    f"{response.text}"
                )

        except requests.RequestException as e:

            st.error(
                f"Impossible de contacter l'API : {e}"
            )
