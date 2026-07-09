from typing import Optional

import pandas as pd

from microbiome_src.models import ValidationIssue, ValidationReport


MAX_SAMPLES = 2000
MAX_RAW_TAXA = 5000
MAX_FILE_SIZE_MB = 50


def check_resource_limits(file_size_mb: float, n_samples: int, n_taxa: int) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if file_size_mb > MAX_FILE_SIZE_MB:
        issues.append(ValidationIssue(
            "error",
            f"文件大小 ({file_size_mb:.1f} MB) 超过限制 ({MAX_FILE_SIZE_MB} MB)"
        ))
    if n_samples > MAX_SAMPLES:
        issues.append(ValidationIssue(
            "error",
            f"样本数 ({n_samples}) 超过限制 ({MAX_SAMPLES})"
        ))
    if n_taxa > MAX_RAW_TAXA:
        issues.append(ValidationIssue(
            "error",
            f"分类单元数 ({n_taxa}) 超过限制 ({MAX_RAW_TAXA})"
        ))
    return issues


def validate_abundance(df: pd.DataFrame) -> ValidationReport:
    issues: list[ValidationIssue] = []

    if df.empty:
        issues.append(ValidationIssue("error", "丰度表为空"))
        return ValidationReport(is_valid=False, issues=issues,
                                summary="丰度表为空")

    if df.index.duplicated().any():
        dup_ids = df.index[df.index.duplicated()].tolist()
        issues.append(ValidationIssue(
            "error", f"存在重复的 SampleID: {dup_ids[:10]}...",
            field="SampleID"
        ))

    non_numeric_cols = df.select_dtypes(exclude="number").columns.tolist()
    if non_numeric_cols:
        issues.append(ValidationIssue(
            "error",
            f"以下列包含非数值数据: {non_numeric_cols[:10]}...",
            field="abundance"
        ))

    numeric_df = df.select_dtypes(include="number")
    if not numeric_df.empty and not non_numeric_cols:
        neg_cols = (numeric_df < 0).any()
        neg_cols = neg_cols[neg_cols].index.tolist()
        if neg_cols:
            issues.append(ValidationIssue(
                "error",
                f"以下 taxa 包含负值: {neg_cols[:10]}...",
                field="abundance"
            ))

    zero_rows = (numeric_df.sum(axis=1) == 0)
    if zero_rows.any():
        zero_ids = df.index[zero_rows].tolist()
        issues.append(ValidationIssue(
            "warning",
            f"以下样本全为零: {zero_ids[:10]}...",
            field="SampleID"
        ))

    is_valid = not any(i.level == "error" for i in issues)
    return ValidationReport(
        is_valid=is_valid,
        issues=issues,
        summary=f"样本次数 {df.shape[0]}, taxa 数 {df.shape[1]}, "
                f"错误 {sum(1 for i in issues if i.level == 'error')}, "
                f"警告 {sum(1 for i in issues if i.level == 'warning')}"
    )


def validate_metadata(
    metadata: pd.DataFrame,
    abundance_sample_ids: pd.Index,
    group_column: str,
) -> ValidationReport:
    issues: list[ValidationIssue] = []

    if metadata.empty:
        return ValidationReport(is_valid=False, issues=[
            ValidationIssue("error", "Metadata 表为空")
        ], summary="Metadata 表为空")

    if metadata.index.duplicated().any():
        issues.append(ValidationIssue("error", "Metadata 包含重复 SampleID"))
        return ValidationReport(is_valid=False, issues=issues,
                                summary="Metadata 包含重复 SampleID")

    if group_column not in metadata.columns:
        issues.append(ValidationIssue(
            "error", f"Metadata 中未找到分组列 '{group_column}'"
        ))
        is_valid_cols = False
    else:
        is_valid_cols = True

    meta_ids = set(metadata.index)
    abund_ids = set(abundance_sample_ids)
    unmatched_abund = abund_ids - meta_ids
    unmatched_meta = meta_ids - abund_ids

    if unmatched_abund:
        issues.append(ValidationIssue(
            "warning",
            f"丰度表中 {len(unmatched_abund)} 个样本不在 metadata 中: "
            f"{sorted(list(unmatched_abund))[:10]}..."
        ))
    if unmatched_meta:
        issues.append(ValidationIssue(
            "warning",
            f"metadata 中 {len(unmatched_meta)} 个样本不在丰度表中: "
            f"{sorted(list(unmatched_meta))[:10]}..."
        ))

    if is_valid_cols:
        group_counts = metadata[group_column].value_counts()
        if len(group_counts) < 2:
            issues.append(ValidationIssue(
                "error",
                f"分组列 '{group_column}' 需要至少 2 个组, 当前: "
                f"{dict(group_counts)}"
            ))

    is_valid = not any(i.level == "error" for i in issues)
    return ValidationReport(
        is_valid=is_valid,
        issues=issues,
        summary=f"metadata 样本数 {len(metadata)}, "
                f"丰度表样本数 {len(abundance_sample_ids)}, "
                f"交集 {len(abund_ids & meta_ids)}"
    )
