from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class ValidationIssue:
    level: str  # "error", "warning", "info"
    message: str
    field: Optional[str] = None


@dataclass(frozen=True)
class ValidationReport:
    is_valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    summary: str = ""


@dataclass(frozen=True)
class MicrobiomeDataset:
    abundance: pd.DataFrame   # rows=samples, columns=taxa
    metadata: pd.DataFrame    # indexed by SampleID
    group_column: str
    source_name: str
    abundance_scale: str = "unknown"  # "count", "relative", "unknown"


@dataclass(frozen=True)
class ConfounderFinding:
    variable: str
    issue: str
    severity: str  # "high", "medium", "low"
    detail: str = ""


@dataclass(frozen=True)
class ConfounderAuditReport:
    risk_level: str  # "low", "medium", "high"
    findings: list[ConfounderFinding] = field(default_factory=list)
