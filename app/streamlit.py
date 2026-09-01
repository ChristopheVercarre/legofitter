import os

import requests
import streamlit as st
from pillow_heif import register_heif_opener

# iPhone photos AirDropped to the Mac arrive as .heic, which Pillow (and so
# st.image) cannot open on its own. One call at import fixes that for the
# whole process. Registering twice is harmless.
register_heif_opener()


API_URL = os.getenv("API_URL")


st.title("LegoFitter")

st.write("Test de connexion à l'API LegoFitter")


if st.button("Tester l'API"):
    try:
        response = requests.get(f"{API_URL}/ping", timeout=10)
        response.raise_for_status()

        data = response.json()

        st.success("API connectée")
        st.json(data)

    except requests.RequestException as e:
        st.error(f"Erreur de connexion à l'API : {e}")

st.divider()

st.subheader("Identifier une pièce LEGO")

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

            response = requests.post(
                f"{API_URL}/predict",
                files=files,
                timeout=30,
            )

            if response.status_code == 200:
                st.success("Image envoyée avec succès à l'API")
                st.json(response.json())

            elif response.status_code == 415:
                st.error(response.json()["detail"])

            else:
                st.error(
                    f"Erreur API ({response.status_code}) : "
                    f"{response.text}"
                )

        except requests.RequestException as e:
            st.error(f"Impossible de contacter l'API : {e}")

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

                st.success(f"{part['name']} — {part['part_num']}")

                if part.get("part_img_url"):
                    st.image(
                        part["part_img_url"],
                        width=300,
                    )

                st.write(f"Année : {part['year_from']} → {part['year_to']}")

                if part.get("part_url"):
                    st.link_button(
                        "Voir sur Rebrickable",
                        part["part_url"],
                    )

            elif response.status_code == 404:
                st.error("Pièce introuvable sur Rebrickable.")

            else:
                st.error(
                    f"Erreur API ({response.status_code}) : {response.text}"
                )

        except requests.RequestException as e:
            st.error(f"Impossible de contacter l'API : {e}")
