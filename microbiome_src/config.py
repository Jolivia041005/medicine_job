from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    MAX_FILE_SIZE_MB: int = 50
    MAX_SAMPLES: int = 2000
    MAX_RAW_TAXA: int = 5000
    MAX_DIFF_TAXA: int = 1000
    TOP_TAXA_DISPLAY: int = 30
    MAX_PCOA_SAMPLES: int = 800
    DUCKDB_MEMORY_LIMIT: str = "2GB"
    DUCKDB_THREADS: int = 2
    PREVALENCE_THRESHOLD: float = 0.10
    MIN_REL_ABUNDANCE: float = 0.0001
    DEFAULT_PSEUDOCOUNT: float = 1e-6
    FDR_THRESHOLD: float = 0.10
    MIN_EFFECT_SIZE: float = 0.2
