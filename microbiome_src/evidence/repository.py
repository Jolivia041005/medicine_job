from pathlib import Path

import pandas as pd

from microbiome_src.evidence.normalize_taxon import (
    clean_taxon_string,
    extract_genus,
    extract_species,
    normalize_taxon_name,
)


class EvidenceRepository:
    def __init__(self, evidence_path: Path | str, guilds_path: Path | str):
        self.evidence = pd.read_csv(evidence_path)
        self.guilds = pd.read_csv(guilds_path)
        self._build_alias_index()

    def _build_alias_index(self):
        self._alias_map: dict[str, list[int]] = {}
        for idx, row in self.evidence.iterrows():
            canonical = clean_taxon_string(row["canonical_taxon"])
            self._alias_map.setdefault(canonical, []).append(idx)
            aliases_str = str(row.get("aliases", ""))
            if aliases_str and aliases_str != "nan":
                for alias in aliases_str.split(";"):
                    alias_clean = clean_taxon_string(alias.strip())
                    if alias_clean and alias_clean != canonical:
                        self._alias_map.setdefault(alias_clean, []).append(idx)

    def search(self, taxon_name: str) -> list[dict]:
        cleaned = clean_taxon_string(normalize_taxon_name(taxon_name))
        matched_indices: set[int] = set()
        confidence = "exact"

        if cleaned in self._alias_map:
            matched_indices.update(self._alias_map[cleaned])
            confidence = "exact"

        if not matched_indices:
            genus = extract_genus(taxon_name)
            if genus:
                genus_clean = clean_taxon_string(genus)
                if genus_clean in self._alias_map:
                    matched_indices.update(self._alias_map[genus_clean])
                    confidence = "genus_match"

        if not matched_indices:
            species = extract_species(taxon_name)
            if species:
                species_clean = clean_taxon_string(species)
                if species_clean in self._alias_map:
                    matched_indices.update(self._alias_map[species_clean])
                    confidence = "species_match"

        if not matched_indices:
            for word in cleaned.replace(";", " ").split():
                word_clean = word.strip()
                if word_clean and len(word_clean) > 2:
                    if word_clean in self._alias_map:
                        matched_indices.update(self._alias_map[word_clean])
                        confidence = "partial"

        results = []
        for idx in matched_indices:
            row = self.evidence.iloc[idx]
            results.append({
                "evidence_id": row["evidence_id"],
                "canonical_taxon": row["canonical_taxon"],
                "taxonomic_level": row["taxonomic_level"],
                "reported_direction": row["reported_direction"],
                "study": row["study"],
                "year": int(row["year"]) if not pd.isna(row["year"]) else None,
                "population": row["population"],
                "sequencing_type": row["sequencing_type"],
                "metformin_adjusted": row["metformin_adjusted"],
                "evidence_type": row["evidence_type"],
                "evidence_summary": row["evidence_summary"],
                "pmid": row.get("pmid", ""),
                "doi": row.get("doi", ""),
                "url": row.get("url", ""),
                "limitations": row.get("limitations", ""),
                "match_confidence": confidence,
            })

        return results

    def get_functional_guilds(self, taxon_name: str) -> list[dict]:
        cleaned = clean_taxon_string(normalize_taxon_name(taxon_name))
        results = []
        for _, row in self.guilds.iterrows():
            guild_taxon = clean_taxon_string(row["canonical_taxon"])
            if guild_taxon == cleaned or guild_taxon in cleaned or cleaned in guild_taxon:
                results.append({
                    "functional_guild": row["functional_guild"],
                    "confidence": row["confidence"],
                    "mechanism_summary": row["mechanism_summary"],
                    "strain_specific": row["strain_specific"],
                    "limitations": row["limitations"],
                })
        return results


def load_repository() -> EvidenceRepository:
    return EvidenceRepository(
        evidence_path=Path("data/seed/taxon_evidence.csv"),
        guilds_path=Path("data/seed/functional_guilds.csv"),
    )
