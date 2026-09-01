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
