from pathlib import Path
from typing import Optional

import pandas as pd

from microbiome_src.models import MicrobiomeDataset


def load_demo_dataset() -> MicrobiomeDataset:
    processed_dir = Path("data/processed")
    abundance = pd.read_parquet(processed_dir / "qin_abundance.parquet")
    metadata = pd.read_parquet(processed_dir / "qin_metadata.parquet")
    return MicrobiomeDataset(
        abundance=abundance,
        metadata=metadata,
        group_column="Group",
        source_name="Qin2012 (Demo)",
        abundance_scale="count",
    )
