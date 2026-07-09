from microbiome_src.fiber_response.matcher import detect_direction_conflicts, match_all_features
from microbiome_src.fiber_response.models import FeatureMatch, FiberResponseFeature, MatchSummary


def compute_match_summary(
    user_taxa: list[str],
    features: list[FiberResponseFeature],
    user_abundance=None,
) -> MatchSummary:
    matches = match_all_features(user_taxa, features, user_abundance)

    matched_feature_ids = set(m.paper_feature_id for m in matches)
    matched_count = len(matched_feature_ids)
    raw_coverage = matched_count / len(features) if features else 0.0

    total_importance = sum(f.importance for f in features)

    weighted_sum = 0.0
    seen_ids: set[str] = set()
    for m in matches:
        if m.paper_feature_id in seen_ids:
            continue
        seen_ids.add(m.paper_feature_id)
        if m.match_level == "family":
            continue
        weighted_sum += m.importance * m.confidence
    weighted_coverage = weighted_sum / total_importance if total_importance > 0 else 0.0

    high_weight = 0.0
    low_weight = 0.0
    for m in matches:
        if m.match_level == "family":
            continue
        norm_imp = m.importance / total_importance if total_importance > 0 else 0
        weighted_contrib = norm_imp * m.confidence
        if m.published_direction == "high_response_enriched":
            high_weight += weighted_contrib
        elif m.published_direction == "low_response_enriched":
            low_weight += weighted_contrib

    species_ids = set(m.paper_feature_id for m in matches if m.match_level == "exact_species")
    genus_ids = set(m.paper_feature_id for m in matches if m.match_level == "exact_genus")
    group_ids = set(m.paper_feature_id for m in matches if m.match_level == "named_group")
    family_ids = set(m.paper_feature_id for m in matches if m.match_level == "family")

    conflicts = detect_direction_conflicts(matches)

    return MatchSummary(
        total_paper_features=len(features),
        matched_features=matched_count,
        raw_coverage=round(raw_coverage, 4),
        importance_weighted_coverage=round(weighted_coverage, 4),
        high_response_weight=round(high_weight, 4),
        low_response_weight=round(low_weight, 4),
        exact_species_count=len(species_ids),
        exact_genus_count=len(genus_ids),
        named_group_count=len(group_ids),
        family_count=len(family_ids),
        conflict_count=len(conflicts),
        all_matches=matches,
    )
