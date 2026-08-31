import os

import requests
import streamlit as st


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
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(
        uploaded_file,
        caption="Image à analyser",
        width=400
    )
