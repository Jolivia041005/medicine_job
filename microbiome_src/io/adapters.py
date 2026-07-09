from typing import Optional

import pandas as pd

from microbiome_src.io.loaders import (
    detect_orientation,
    detect_separator,
    infer_abundance_scale,
    read_table,
    standardize_to_sample_x_taxon,
)
from microbiome_src.io.validators import check_resource_limits, validate_abundance, validate_metadata
from microbiome_src.models import MicrobiomeDataset, ValidationIssue, ValidationReport


def build_dataset(
    abundance_file,
    metadata_file,
    orientation: Optional[str] = None,
    abundance_id_column: Optional[str] = None,
    metadata_id_column: Optional[str] = None,
    group_column: str = "Group",
    source_name: str = "upload",
) -> tuple[Optional[MicrobiomeDataset], ValidationReport, ValidationReport]:
    abundance_df = read_table(abundance_file)
    metadata_sep = detect_separator(metadata_file)
    metadata_df = pd.read_csv(metadata_file, sep=metadata_sep)

    if metadata_id_column is None:
        metadata_id_column = metadata_df.columns[0]
    if group_column not in metadata_df.columns:
        group_column = metadata_df.columns[1] if len(metadata_df.columns) > 1 else metadata_df.columns[0]

    if str(metadata_df.columns[0]).startswith("#"):
        metadata_df.rename(columns={metadata_df.columns[0]: "SampleID"}, inplace=True)
        metadata_id_column = "SampleID"
    elif metadata_id_column in metadata_df.columns:
        metadata_df.rename(columns={metadata_id_column: "SampleID"}, inplace=True)
    else:
        metadata_df.index = metadata_df.iloc[:, 0].astype(str).str.strip()
        metadata_df.index.name = "SampleID"

    if metadata_df.index.name != "SampleID" and "SampleID" not in metadata_df.columns:
        metadata_df.rename(columns={metadata_id_column: "SampleID"}, inplace=True)

    if "SampleID" not in metadata_df.index.names and "SampleID" in metadata_df.columns:
        metadata_df = metadata_df.set_index("SampleID")
    metadata_df.index = metadata_df.index.astype(str).str.strip()

    if orientation is None:
        orientation = detect_orientation(abundance_df, abundance_id_column)

    abundance = standardize_to_sample_x_taxon(abundance_df, id_column=abundance_id_column, orientation=orientation)

    resource_issues = check_resource_limits(
        file_size_mb=0,
        n_samples=abundance.shape[0],
        n_taxa=abundance.shape[1],
    )
    scale = infer_abundance_scale(abundance)
    abund_report = validate_abundance(abundance)
    if resource_issues:
        abund_issues = list(abund_report.issues) + resource_issues
        abund_report = ValidationReport(
            is_valid=not any(i.level == "error" for i in abund_issues),
            issues=abund_issues,
            summary=abund_report.summary,
        )
    meta_report = validate_metadata(metadata_df, abundance.index, group_column)

    if not abund_report.is_valid or not meta_report.is_valid:
        return None, abund_report, meta_report

    common_ids = abundance.index.intersection(metadata_df.index)
    abundance = abundance.loc[common_ids]
    metadata = metadata_df.loc[common_ids]

    dataset = MicrobiomeDataset(
        abundance=abundance,
        metadata=metadata,
        group_column=group_column,
        source_name=source_name,
        abundance_scale=scale,
    )
    return dataset, abund_report, meta_report
