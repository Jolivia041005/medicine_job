from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class FiberResponseFeature:
    feature_id: str
    taxonomy: str
    taxonomic_level: str
    aliases: list[str]
    mean_low: float
    mean_high: float
    importance: float
    published_direction: str
    source_study: str
    pmid: str
    doi: str
    source_url: str
    limitations: str


@dataclass(frozen=True)
class FeatureMatch:
    user_feature: str
    paper_feature_id: str
    paper_taxonomy: str
    match_level: str
    confidence: float
    importance: float
    published_direction: str
    user_mean_abundance: Optional[float] = None
    limitations: str = ""
    warning: Optional[str] = None


@dataclass
class MatchSummary:
    total_paper_features: int = 44
    matched_features: int = 0
    raw_coverage: float = 0.0
    importance_weighted_coverage: float = 0.0
    high_response_weight: float = 0.0
    low_response_weight: float = 0.0
    exact_species_count: int = 0
    exact_genus_count: int = 0
    named_group_count: int = 0
    family_count: int = 0
    conflict_count: int = 0
    all_matches: list[FeatureMatch] = field(default_factory=list)
