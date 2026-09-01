"""Turn a detected inventory into "you could build this" set suggestions.

Strategy (every knob lives in params.py):
  1. for each detected part, ask Rebrickable which sets contain it;
  2. prefer sets of MIN_SET_SIZE..MAX_SET_SIZE pieces -- small enough that
     a photographed handful of bricks means something against them;
  3. fetch each surviving candidate's full inventory and score it;
  4. sort: buildable sets first, then by how much of YOUR pile they use.

A part or set whose HTTP lookup fails (rate limit, unknown ID) is skipped,
never fatal: losing one candidate is fine, losing the whole recommendation
because one request 429'd is not. A missing API key still raises, so the
API layer can log it once.
"""

from collections import Counter

import httpx

from app.api.rebrickable import (
    get_candidate_sets_for_part,
    get_set_inventory,
)
from app.params import (
    CANDIDATE_SETS_PER_PART,
    MAX_CANDIDATE_SETS,
    MAX_SET_SIZE,
    MIN_INVENTORY_COVERAGE,
    MIN_SET_SIZE,
    TARGET_SET_SIZE,
)


def score_inventory_match(
    owned_inventory: dict[str, int],
    required_inventory: dict[str, int],
) -> dict:
    """Compare what we photographed against what a set needs.

    Two ratios, two questions:
      - compatibility: how much of the SET is covered (1.0 = buildable);
      - inventory_coverage: how much of OUR pile the set uses (the demo
        metric -- "this set uses 80% of your bricks" lands better than
        "you own 2% of this set").
    """
    total_required = sum(required_inventory.values())

    if total_required == 0:
        return {
            "compatibility": 0.0,
            "inventory_coverage": 0.0,
            "buildable": False,
            "matched_parts": 0,
            "required_parts": 0,
            "missing_parts": {},
        }

    matched_parts = 0
    missing_parts = {}

    for part_id, required_quantity in required_inventory.items():
        owned_quantity = owned_inventory.get(part_id, 0)

        matched_parts += min(owned_quantity, required_quantity)

        if owned_quantity < required_quantity:
            missing_parts[part_id] = required_quantity - owned_quantity

    compatibility = matched_parts / total_required

    total_owned = sum(owned_inventory.values())

    inventory_coverage = (
        matched_parts / total_owned if total_owned > 0 else 0
    )

    return {
        "compatibility": round(compatibility, 4),
        "inventory_coverage": round(inventory_coverage, 4),
        "buildable": len(missing_parts) == 0,
        "matched_parts": matched_parts,
        "required_parts": total_required,
        "missing_parts": missing_parts,
    }


def recommend_sets(
    owned_inventory: dict[str, int],
    candidate_sets_per_part: int = CANDIDATE_SETS_PER_PART,
    max_candidates: int = MAX_CANDIDATE_SETS,
) -> list[dict]:
    """From the inventory the chaining detected, suggest LEGO sets."""

    candidate_counts = Counter()
    candidate_data = {}

    # 1. Collect candidate sets from every detected part
    for part_id in owned_inventory:
        try:
            sets = get_candidate_sets_for_part(
                part_id,
                max_sets=candidate_sets_per_part,
            )
        except httpx.HTTPError:
            # One part's lookup failing must not sink the others
            continue

        for lego_set in sets:
            set_num = lego_set["set_num"]

            candidate_counts[set_num] += 1
            candidate_data[set_num] = lego_set

    # 2. Prefer small sets (MIN_SET_SIZE..MAX_SET_SIZE pieces) already at
    #    candidate selection, so the expensive inventory downloads below
    #    are spent on sets we would actually show.
    size_candidate_ids = [
        set_num
        for set_num, count in candidate_counts.most_common()
        if (
            candidate_data[set_num].get("num_parts") is not None
            and MIN_SET_SIZE
            <= candidate_data[set_num]["num_parts"]
            <= MAX_SET_SIZE
        )
    ]

    # If no set falls in the preferred range, fall back to the most-cited
    # candidates so we always propose SOMETHING.
    if size_candidate_ids:
        best_candidate_ids = size_candidate_ids[:max_candidates]
    else:
        best_candidate_ids = [
            set_num
            for set_num, _ in candidate_counts.most_common(max_candidates)
        ]

    recommendations = []

    # 3. Real score against each candidate's full inventory
    for set_num in best_candidate_ids:
        try:
            required_inventory = get_set_inventory(set_num)
        except httpx.HTTPError:
            # Skip this candidate rather than losing the whole list
            continue

        score = score_inventory_match(
            owned_inventory=owned_inventory,
            required_inventory=required_inventory,
        )

        lego_set = candidate_data[set_num]

        recommendations.append(
            {
                "set_num": set_num,
                "name": lego_set.get("name"),
                "set_img_url": lego_set.get("set_img_url"),
                "year": lego_set.get("year"),
                "num_parts": lego_set.get("num_parts"),
                "candidate_matches": candidate_counts[set_num],
                **score,
            }
        )

    # 4a. Keep only sets in the preferred size range -- if any survive
    size_filtered = [
        recommendation
        for recommendation in recommendations
        if (
            recommendation["num_parts"] is not None
            and MIN_SET_SIZE
            <= recommendation["num_parts"]
            <= MAX_SET_SIZE
        )
    ]

    if size_filtered:
        recommendations = size_filtered

    # 4b. Prefer sets that use at least MIN_INVENTORY_COVERAGE of the
    #     photographed bricks -- applied only if at least one qualifies
    good_coverage = [
        recommendation
        for recommendation in recommendations
        if recommendation["inventory_coverage"] >= MIN_INVENTORY_COVERAGE
    ]

    if good_coverage:
        recommendations = good_coverage

    recommendations.sort(
        key=lambda x: (
            x["buildable"],
            x["inventory_coverage"] >= MIN_INVENTORY_COVERAGE,
            x["inventory_coverage"],
            x["compatibility"],
            -abs(
                (x["num_parts"] or TARGET_SET_SIZE)
                - TARGET_SET_SIZE
            ),
            x["candidate_matches"],
        ),
        reverse=True,
    )

    return recommendations
