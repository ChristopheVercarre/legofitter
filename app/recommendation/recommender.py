from collections import Counter

from app.api.rebrickable import (
    get_candidate_sets_for_part,
    get_set_inventory,
)


def score_inventory_match(
    owned_inventory: dict[str, int],
    required_inventory: dict[str, int],
) -> dict:

    total_required = sum(required_inventory.values())

    if total_required == 0:
        return {
            "compatibility": 0.0,
            "buildable": False,
            "matched_parts": 0,
            "required_parts": 0,
            "missing_parts": {},
        }

    matched_parts = 0
    missing_parts = {}

    for part_id, required_quantity in required_inventory.items():

        owned_quantity = owned_inventory.get(part_id, 0)

        matched_parts += min(
            owned_quantity,
            required_quantity,
        )

        if owned_quantity < required_quantity:
            missing_parts[part_id] = (
                required_quantity - owned_quantity
            )

    compatibility = matched_parts / total_required

    total_owned = sum(owned_inventory.values())

    inventory_coverage = (
        matched_parts / total_owned
        if total_owned > 0
        else 0
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
    candidate_sets_per_part: int = 20,
    max_candidates: int = 10,
) -> list[dict]:
    """
    À partir de l'inventaire détecté par le chaining :

    1. cherche des sets contenant les pièces détectées ;
    2. compte dans combien de pièces chaque set apparaît ;
    3. garde les meilleurs candidats ;
    4. récupère leur inventaire complet ;
    5. calcule leur compatibilité.
    """

    candidate_counts = Counter()
    candidate_data = {}

    # Recherche des sets candidats
    for part_id in owned_inventory:

        sets = get_candidate_sets_for_part(
            part_id,
            max_sets=candidate_sets_per_part,
        )

        for lego_set in sets:

            set_num = lego_set["set_num"]

            candidate_counts[set_num] += 1
            candidate_data[set_num] = lego_set

    # On privilégie dès la sélection les petits sets entre 50 et 250 pièces
    # Paramètres de recommandation
    TARGET_SET_SIZE = 100
    MIN_SET_SIZE = 50
    MAX_SET_SIZE = 250
    MIN_COMPATIBILITY = 0.10

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

    # Si on trouve des sets dans la bonne taille, on ne considère qu'eux.
    # Sinon on garde un fallback pour toujours proposer quelque chose.
    if size_candidate_ids:
        best_candidate_ids = size_candidate_ids[:max_candidates]
    else:
        best_candidate_ids = [
            set_num
            for set_num, _ in candidate_counts.most_common(max_candidates)
        ]

    recommendations = []

    # Calcul du vrai score sur l'inventaire complet du set
    for set_num in best_candidate_ids:

        required_inventory = get_set_inventory(set_num)

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


    # 1. Garder en priorité les sets entre 250 et 750 pièces
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


    # 2. Privilégier les sets avec au moins 10 % de compatibilité
    good_compatibility = [
        recommendation
        for recommendation in recommendations
        if recommendation["inventory_coverage"] >= MIN_COMPATIBILITY
    ]

    # On applique le seuil uniquement s'il reste au moins un candidat
    if good_compatibility:
        recommendations = good_compatibility


    recommendations.sort(
        key=lambda x: (
            x["buildable"],
            x["inventory_coverage"] >= MIN_COMPATIBILITY,
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
