import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


def run_mannwhitney_differential(
    abundance: pd.DataFrame,
    metadata: pd.DataFrame,
    group_column: str,
    case_label: str = "T2D",
    control_label: str = "Control",
    pseudocount: float = 1e-6,
) -> pd.DataFrame:
    case_mask = metadata[group_column] == case_label
    control_mask = metadata[group_column] == control_label

    common_idx = abundance.index.intersection(metadata.index)
    abundance = abundance.loc[common_idx]

    case_ids = metadata.index[metadata.index.isin(common_idx) & case_mask]
    control_ids = metadata.index[metadata.index.isin(common_idx) & control_mask]

    case_abund = abundance.loc[case_ids]
    control_abund = abundance.loc[control_ids]

    rows = []
    for taxon in abundance.columns:
        case_vals = case_abund[taxon].values.astype(float)
        control_vals = control_abund[taxon].values.astype(float)

        if np.std(case_vals) == 0 and np.std(control_vals) == 0:
            continue

        try:
            stat, pvalue = mannwhitneyu(case_vals, control_vals, alternative="two-sided")
        except ValueError:
            continue

        median_case = np.median(case_vals)
        median_control = np.median(control_vals)

        log2fc = np.log2(
            (median_case + pseudocount) / (median_control + pseudocount)
        )

        n_case = np.sum(case_vals > 0)
        n_control = np.sum(control_vals > 0)
        prevalence_case = n_case / len(case_vals)
        prevalence_control = n_control / len(control_vals)

        direction = "T2D > Control" if median_case > median_control else "Control > T2D"
        if abs(median_case - median_control) < 1e-15:
            direction = "no difference"

        rows.append({
            "taxon": taxon,
            "n_case": len(case_vals),
            "n_control": len(control_vals),
            "detected_case": int(n_case),
            "detected_control": int(n_control),
            "prevalence_case": round(prevalence_case, 4),
            "prevalence_control": round(prevalence_control, 4),
            "median_case": median_case,
            "median_control": median_control,
            "log2fc": round(log2fc, 4),
            "statistic": stat,
            "pvalue": pvalue,
            "direction": direction,
        })

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    _, fdr, _, _ = multipletests(result["pvalue"].values, method="fdr_bh")
    result["fdr"] = fdr

    result["effect_size"] = _rank_biserial_correlation(
        abundance, case_ids, control_ids, result["taxon"].tolist()
    )

    result = result.sort_values(["fdr", "effect_size"], ascending=[True, False])
    result = result.reset_index(drop=True)
    return result


def _rank_biserial_correlation(
    abundance: pd.DataFrame,
    case_ids: pd.Index,
    control_ids: pd.Index,
    taxa: list[str],
) -> list[float]:
    effect_sizes = []
    n1 = len(case_ids)
    n2 = len(control_ids)

    for taxon in taxa:
        case_vals = abundance.loc[case_ids, taxon].values
        control_vals = abundance.loc[control_ids, taxon].values
        combined = np.concatenate([case_vals, control_vals])
        ranks = np.argsort(np.argsort(combined)).astype(float) + 1
        case_ranks = ranks[:n1]
        r_mean = np.mean(case_ranks)
        rbc = 2 * (r_mean - (n1 + n2 + 1) / 2) / n2
        effect_sizes.append(round(float(rbc), 4))

    return effect_sizes


def filter_candidates(
    diff_results: pd.DataFrame,
    fdr_threshold: float = 0.10,
    min_effect_size: float = 0.2,
    min_prevalence: float = 0.10,
) -> pd.DataFrame:
    if diff_results.empty:
        return diff_results
    mask = (
        (diff_results["fdr"] < fdr_threshold)
        & (diff_results["effect_size"].abs() >= min_effect_size)
        & (diff_results[["prevalence_case", "prevalence_control"]].max(axis=1) >= min_prevalence)
    )
    return diff_results[mask].reset_index(drop=True)
