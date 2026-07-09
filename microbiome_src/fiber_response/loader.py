from pathlib import Path

import pandas as pd

from microbiome_src.fiber_response.models import FiberResponseFeature

VALID_DIRECTIONS = {"high_response_enriched", "low_response_enriched"}


def load_fiber_features(path: str | Path = "data/seed/fiber_response_features.csv") -> list[FiberResponseFeature]:
    df = pd.read_csv(path)

    assert len(df) == 44, f"Expected 44 features, got {len(df)}"
    assert df["feature_id"].is_unique, "feature_id not unique"
    assert df["lightgbm_importance"].dtype.kind in "if", "importance must be numeric"
    invalid = set(df["published_abundance_direction"]) - VALID_DIRECTIONS
    assert not invalid, f"Invalid direction values: {invalid}"

    features = []
    for _, row in df.iterrows():
        aliases_str = str(row.get("aliases", ""))
        aliases = [a.strip() for a in aliases_str.split(";") if a.strip()]

        features.append(FiberResponseFeature(
            feature_id=row["feature_id"],
            taxonomy=row["taxonomy"],
            taxonomic_level=row["taxonomic_level"],
            aliases=aliases,
            mean_low=float(row["mean_abundance_low_responders"]),
            mean_high=float(row["mean_abundance_high_responders"]),
            importance=float(row["lightgbm_importance"]),
            published_direction=row["published_abundance_direction"],
            source_study=row.get("source_study", ""),
            pmid=str(row.get("pmid", "")),
            doi=str(row.get("doi", "")),
            source_url=str(row.get("source_url", "")),
            limitations=str(row.get("limitations", "")),
        ))

    return features


def load_fiber_sources(path: str | Path = "data/seed/fiber_response_sources.csv") -> list[dict]:
    df = pd.read_csv(path)
    return df.to_dict(orient="records")
