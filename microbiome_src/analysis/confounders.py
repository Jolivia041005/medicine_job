import numpy as np
import pandas as pd

from microbiome_src.models import ConfounderAuditReport, ConfounderFinding

KEY_FIELD_ALIASES: dict[str, list[str]] = {
    "metformin": ["metformin", "Metformin", "二甲双胍", "metformin_use", "MetforminUse"],
    "bmi": ["bmi", "BMI", "体重指数", "BodyMassIndex", "body_mass_index"],
    "age": ["age", "Age", "年龄", "Age_years", "age_years"],
    "sex": ["sex", "Sex", "性别", "gender", "Gender"],
    "antibiotics": ["antibiotics", "Antibiotics", "抗生素", "antibiotic_use", "AntibioticUse"],
    "batch": ["batch", "Batch", "批次", "batch_id", "BatchID", "sequencing_run"],
    "region": ["region", "Region", "地区", "country", "Country", "site", "Site"],
    "sequencing_method": ["sequencing_method", "SequencingMethod", "测序方法", "seq_method", "platform"],
}

METFORMIN_ALIASES = KEY_FIELD_ALIASES["metformin"]


def _match_column(columns: pd.Index, aliases: list[str]) -> str | None:
    for alias in aliases:
        for col in columns:
            if col.strip().lower() == alias.strip().lower():
                return col
    return None


def _standardized_difference(group_a: np.ndarray, group_b: np.ndarray) -> float:
    mean_a, mean_b = np.mean(group_a), np.mean(group_b)
    var_a, var_b = np.var(group_a, ddof=1), np.var(group_b, ddof=1)
    pooled_var = (var_a + var_b) / 2
    if pooled_var < 1e-15:
        if abs(mean_a - mean_b) < 1e-15:
            return 0.0
        return 10.0  # complete separation
    return float(abs(mean_a - mean_b) / np.sqrt(pooled_var))


def run_confounder_audit(
    metadata: pd.DataFrame,
    group_column: str,
    case_label: str = "T2D",
    control_label: str = "Control",
) -> ConfounderAuditReport:
    findings: list[ConfounderFinding] = []

    case_mask = metadata[group_column] == case_label
    control_mask = metadata[group_column] == control_label
    case_df = metadata[case_mask]
    control_df = metadata[control_mask]

    for field_name, aliases in KEY_FIELD_ALIASES.items():
        col = _match_column(metadata.columns, aliases)
        if col is None:
            findings.append(ConfounderFinding(
                variable=field_name,
                issue=f"未提供 {field_name} 信息",
                severity="medium" if field_name in ("metformin", "bmi", "batch") else "low",
                detail=f"metadata 中未找到 {field_name} 相关字段 (尝试匹配: {', '.join(aliases)})",
            ))
            continue

        series = metadata[col]

        if pd.api.types.is_numeric_dtype(series):
            case_vals = case_df[col].dropna().values
            control_vals = control_df[col].dropna().values
            if len(case_vals) < 3 or len(control_vals) < 3:
                findings.append(ConfounderFinding(
                    variable=field_name,
                    issue=f"{field_name} 样本数不足",
                    severity="low",
                    detail=f"case={len(case_vals)}, control={len(control_vals)}",
                ))
                continue

            std_diff = _standardized_difference(case_vals, control_vals)
            if std_diff > 0.5:
                findings.append(ConfounderFinding(
                    variable=field_name,
                    issue=f"{field_name} 组间差异较大",
                    severity="high" if field_name == "metformin" else "medium",
                    detail=(f"标准化差异 = {std_diff:.3f}, "
                            f"case 中位数 = {np.median(case_vals):.2f}, "
                            f"control 中位数 = {np.median(control_vals):.2f}"),
                ))
            elif std_diff > 0.2:
                findings.append(ConfounderFinding(
                    variable=field_name,
                    issue=f"{field_name} 组间存在中等差异",
                    severity="low",
                    detail=(f"标准化差异 = {std_diff:.3f}, "
                            f"case 中位数 = {np.median(case_vals):.2f}, "
                            f"control 中位数 = {np.median(control_vals):.2f}"),
                ))
        else:
            case_counts = case_df[col].value_counts()
            control_counts = control_df[col].value_counts()
            all_cats = set(case_counts.index) | set(control_counts.index)
            total_case = len(case_df)
            total_control = len(control_df)

            for cat in all_cats:
                p_case = case_counts.get(cat, 0) / total_case
                p_control = control_counts.get(cat, 0) / total_control
                if abs(p_case - p_control) > 0.5:
                    findings.append(ConfounderFinding(
                        variable=field_name,
                        issue=f"{field_name}='{cat}' 在组间极度失衡",
                        severity="high",
                        detail=(f"T2D 组: {p_case:.0%}, "
                                f"Control 组: {p_control:.0%}"),
                    ))
                    break

            groups_per_batch = metadata.groupby([col, group_column]).size().unstack(fill_value=0)
            if groups_per_batch.shape[1] == len(groups_per_batch):
                has_overlap = (groups_per_batch > 0).all(axis=1)
                if not has_overlap.any():
                    findings.append(ConfounderFinding(
                        variable=field_name,
                        issue=f"{field_name} 与 {group_column} 完全共线",
                        severity="high",
                        detail=f"每个 {field_name} 仅出现在一个组中，无法区分批次与疾病效应。",
                    ))

    high_count = sum(1 for f in findings if f.severity == "high")
    medium_count = sum(1 for f in findings if f.severity == "medium")

    if high_count >= 2:
        risk_level = "high"
    elif high_count == 1 or medium_count >= 3:
        risk_level = "medium"
    else:
        risk_level = "low"

    metformin_missing = any(
        f.variable == "metformin" and "未提供" in f.issue for f in findings
    )

    return ConfounderAuditReport(risk_level=risk_level, findings=findings)
