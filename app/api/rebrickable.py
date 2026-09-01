"""Thin Rebrickable v3 client for the bonus objective.

HTTP only -- no scoring logic here (that lives in
app.recommendation.recommender). Every call needs REBRICKABLE_API_KEY,
read through params.py so .env works locally and a plain env var works
on Cloud Run.

Rebrickable throttles free API keys to roughly one request per second, so
the two natural call amplifiers are capped and cached:
  - a part's colour list is cut to the REBRICKABLE_MAX_COLORS_PER_PART
    colours that appear in the most sets (a common brick exists in dozens);
  - candidate-set and set-inventory lookups are cached for the life of the
    process, so repeated /predict calls stop re-asking about part 3001.
"""

import httpx

from app.params import (
    REBRICKABLE_API_KEY,
    REBRICKABLE_BASE_URL,
    REBRICKABLE_MAX_COLORS_PER_PART,
)

# Process-lifetime caches, keyed by part/set ID. Small, and safe to keep:
# Rebrickable's catalogue does not change mid-demo.
_candidate_sets_cache: dict[str, list] = {}
_set_inventory_cache: dict[str, dict] = {}
_part_cache: dict[str, dict] = {}


def _headers() -> dict:
    if not REBRICKABLE_API_KEY:
        raise RuntimeError("REBRICKABLE_API_KEY is missing")
    return {"Authorization": f"key {REBRICKABLE_API_KEY}"}


def get_part(part_id: str):
    if part_id in _part_cache:
        return _part_cache[part_id]

    response = httpx.get(
        f"{REBRICKABLE_BASE_URL}/lego/parts/{part_id}/",
        headers=_headers(),
        timeout=10,
    )
    response.raise_for_status()

    part = response.json()
    _part_cache[part_id] = part
    return part


def get_set_inventory(set_num: str) -> dict[str, int]:
    """Every part needed to build a set: {part_id: quantity}.

    Colours are merged per part_id and spare parts skipped. Paginated --
    big sets span several pages. Cached per set.
    """
    if set_num in _set_inventory_cache:
        return _set_inventory_cache[set_num]

    url = f"{REBRICKABLE_BASE_URL}/lego/sets/{set_num}/parts/"
    inventory: dict[str, int] = {}

    while url:
        response = httpx.get(
            url,
            headers=_headers(),
            params={"page_size": 1000},
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()

        for item in data["results"]:
            # Spare parts are not part of the build
            if item.get("is_spare"):
                continue

            part_id = item["part"]["part_num"]
            inventory[part_id] = inventory.get(part_id, 0) + item["quantity"]

        # Rebrickable pagination
        url = data.get("next")

    _set_inventory_cache[set_num] = inventory
    return inventory


def get_candidate_sets_for_part(part_id: str, max_sets: int = 50) -> list[dict]:
    """Sets that contain this part, any colour. Cached per part.

    Rebrickable only lists sets per (part, colour), so we fetch the part's
    colours first -- but keep only the few that appear in the most sets,
    which stops a 30-colour brick from costing 30 extra requests.
    """
    if part_id in _candidate_sets_cache:
        return _candidate_sets_cache[part_id]

    response = httpx.get(
        f"{REBRICKABLE_BASE_URL}/lego/parts/{part_id}/colors/",
        headers=_headers(),
        timeout=20,
    )

    if response.status_code != 200:
        # Unknown part or a rate-limit blip: this part simply contributes
        # no candidates; the recommender carries on with the others.
        return []

    colors = response.json()["results"]
    colors.sort(key=lambda color: color.get("num_sets") or 0, reverse=True)
    colors = colors[:REBRICKABLE_MAX_COLORS_PER_PART]

    candidate_sets: dict[str, dict] = {}

    for color in colors:
        response = httpx.get(
            f"{REBRICKABLE_BASE_URL}/lego/parts/{part_id}"
            f"/colors/{color['color_id']}/sets/",
            headers=_headers(),
            params={"page_size": max_sets},
            timeout=20,
        )

        if response.status_code != 200:
            continue

        for lego_set in response.json()["results"]:
            candidate_sets[lego_set["set_num"]] = lego_set

            if len(candidate_sets) >= max_sets:
                break

        if len(candidate_sets) >= max_sets:
            break

    result = list(candidate_sets.values())
    _candidate_sets_cache[part_id] = result
    return result
