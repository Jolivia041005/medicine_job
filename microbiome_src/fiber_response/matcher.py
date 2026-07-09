import pandas as pd

from microbiome_src.fiber_response.models import FeatureMatch, FiberResponseFeature
from microbiome_src.fiber_response.taxonomy_parser import (
    clean_taxon,
    extract_family_name,
    extract_genus_name,
    extract_species_name,
    is_named_group,
)

MATCH_CONFIDENCE = {
    "exact_species": 0.75,
    "exact_genus": 0.50,
    "named_group": 0.40,
    "family": 0.20,
    "no_match": 0.0,
}


def _build_paper_index(features: list[FiberResponseFeature]) -> dict[str, list[FiberResponseFeature]]:
    index: dict[str, list[FiberResponseFeature]] = {}
    for f in features:
        paper_tax_clean = clean_taxon(f.taxonomy)
        index.setdefault(paper_tax_clean, []).append(f)
        for alias in f.aliases:
            alias_clean = clean_taxon(alias)
            if alias_clean:
                index.setdefault(alias_clean, []).append(f)
    return index


def match_feature(
    user_taxon: str,
    features: list[FiberResponseFeature],
    paper_index: dict[str, list[FiberResponseFeature]],
    user_mean_abundance: float | None = None,
) -> list[FeatureMatch]:
    user_clean = clean_taxon(user_taxon)
    found: list[FeatureMatch] = []
    seen_ids: set[str] = set()

    def _add_matches(pfs, level, confidence):
        for pf in pfs:
            if pf.feature_id not in seen_ids:
                seen_ids.add(pf.feature_id)
                found.append(FeatureMatch(
                    user_feature=user_taxon,
                    paper_feature_id=pf.feature_id,
                    paper_taxonomy=pf.taxonomy,
                    match_level=level,
                    confidence=confidence,
                    importance=pf.importance,
                    published_direction=pf.published_direction,
                    user_mean_abundance=user_mean_abundance,
                    limitations=pf.limitations,
                ))

    user_species = extract_species_name(user_taxon)
    user_genus = extract_genus_name(user_taxon)
    user_family = extract_family_name(user_taxon)

    species_clean = clean_taxon(user_species) if user_species else None
    genus_clean = clean_taxon(user_genus) if user_genus else None
    family_clean = clean_taxon(user_family) if user_family else None

    for candidate in [user_clean, species_clean]:
        if not candidate:
            continue
        if candidate in paper_index:
            _add_matches(
                [pf for pf in paper_index[candidate] if pf.taxonomic_level == "species"],
                "exact_species", MATCH_CONFIDENCE["exact_species"]
            )

    if seen_ids:
        return found

    if genus_clean and genus_clean in paper_index:
        is_family_name = genus_clean.endswith("aceae") or genus_clean.endswith("aceae group")
        has_family_match = family_clean and family_clean in paper_index
        if not is_family_name or not has_family_match:
            _add_matches(paper_index[genus_clean], "exact_genus", MATCH_CONFIDENCE["exact_genus"])

    if seen_ids:
        return found

    if genus_clean and is_named_group(genus_clean):
        for name, pf_list in paper_index.items():
            for pf in pf_list:
                if clean_taxon(pf.taxonomy) == genus_clean:
                    _add_matches([pf], "named_group", MATCH_CONFIDENCE["named_group"])
    if seen_ids:
        return found

    if family_clean and family_clean in paper_index:
        _add_matches(paper_index[family_clean], "family", MATCH_CONFIDENCE["family"])

    return found


def match_all_features(
    user_taxa: list[str],
    features: list[FiberResponseFeature],
    user_abundance: pd.Series | None = None,
) -> list[FeatureMatch]:
    paper_index = _build_paper_index(features)
    all_matches: list[FeatureMatch] = []
    seen_pairs: set[tuple[str, str]] = set()

    for taxon in user_taxa:
        taxon_str = str(taxon)
        mean_abund = float(user_abundance[taxon_str]) if user_abundance is not None and taxon_str in user_abundance.index else None
        matches = match_feature(taxon_str, features, paper_index, mean_abund)
        for m in matches:
            pair = (taxon_str.lower().strip(), m.paper_feature_id)
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                all_matches.append(m)

    return all_matches


def detect_direction_conflicts(matches: list[FeatureMatch]) -> list[dict]:
    user_taxon_groups: dict[str, list[FeatureMatch]] = {}
    for m in matches:
        key = m.user_feature.lower().strip()
        user_taxon_groups.setdefault(key, []).append(m)

    conflicts = []
    for taxon, match_list in user_taxon_groups.items():
        if len(match_list) < 2:
            continue
        directions = {m.published_direction for m in match_list}
        if "high_response_enriched" in directions and "low_response_enriched" in directions:
            conflicts.append({
                "user_taxon": taxon,
                "match_count": len(match_list),
                "directions": sorted(directions),
                "match_ids": [m.paper_feature_id for m in match_list],
            })

    return conflicts
