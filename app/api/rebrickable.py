import os

import httpx


REBRICKABLE_API_URL = "https://rebrickable.com/api/v3"


def get_part(part_id: str):

    api_key = os.getenv("REBRICKABLE_API_KEY")

    if not api_key:
        raise RuntimeError("REBRICKABLE_API_KEY is missing")

    url = f"{REBRICKABLE_API_URL}/lego/parts/{part_id}/"

    headers = {
        "Authorization": f"key {api_key}"
    }

    response = httpx.get(
        url,
        headers=headers,
        timeout=10,
    )

    response.raise_for_status()

    return response.json()

def get_set_inventory(set_num: str) -> dict[str, int]:
    """
    Récupère toutes les pièces nécessaires pour construire un set Rebrickable.

    Pour l'instant :
    - on ignore les couleurs ;
    - on ignore les pièces de rechange ;
    - on agrège les quantités par part_id.
    """

    api_key = os.getenv("REBRICKABLE_API_KEY")

    if not api_key:
        raise RuntimeError("REBRICKABLE_API_KEY is missing")

    url = f"{REBRICKABLE_API_URL}/lego/sets/{set_num}/parts/"

    headers = {
        "Authorization": f"key {api_key}"
    }

    inventory = {}

    while url:
        response = httpx.get(
            url,
            headers=headers,
            params={"page_size": 1000},
            timeout=20,
        )

        response.raise_for_status()

        data = response.json()

        for item in data["results"]:

            # On ignore les pièces de rechange
            if item.get("is_spare"):
                continue

            part_id = item["part"]["part_num"]
            quantity = item["quantity"]

            inventory[part_id] = (
                inventory.get(part_id, 0) + quantity
            )

        # Pagination Rebrickable
        url = data.get("next")

    return inventory

def get_candidate_sets_for_part(part_id: str, max_sets: int = 50) -> list[dict]:
    """
    Retourne des sets candidats contenant cette pièce,
    toutes couleurs confondues.
    """

    api_key = os.getenv("REBRICKABLE_API_KEY")

    if not api_key:
        raise RuntimeError("REBRICKABLE_API_KEY is missing")

    headers = {
        "Authorization": f"key {api_key}"
    }

    # 1. Récupère les couleurs connues pour cette pièce
    colors_url = (
        f"{REBRICKABLE_API_URL}/lego/parts/{part_id}/colors/"
    )

    response = httpx.get(
        colors_url,
        headers=headers,
        timeout=20,
    )
    response.raise_for_status()

    colors = response.json()["results"]

    candidate_sets = {}

    # 2. Cherche les sets pour chaque couleur
    for color in colors:

        color_id = color["color_id"]

        sets_url = (
            f"{REBRICKABLE_API_URL}"
            f"/lego/parts/{part_id}/colors/{color_id}/sets/"
        )

        response = httpx.get(
            sets_url,
            headers=headers,
            params={"page_size": max_sets},
            timeout=20,
        )

        if response.status_code != 200:
            continue

        data = response.json()

        for lego_set in data["results"]:

            candidate_sets[lego_set["set_num"]] = lego_set

            if len(candidate_sets) >= max_sets:
                break

        if len(candidate_sets) >= max_sets:
            break

    return list(candidate_sets.values())
