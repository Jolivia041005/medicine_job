import numpy as np
import pandas as pd


def to_relative_abundance(df: pd.DataFrame) -> pd.DataFrame:
    row_sums = df.sum(axis=1)
    zero_rows = row_sums == 0
    if zero_rows.any():
        zero_ids = df.index[zero_rows].tolist()
        raise ValueError(f"以下样本全为零: {zero_ids[:10]}")
    return df.div(row_sums, axis=0)


def calculate_prevalence(df: pd.DataFrame, detection_limit: float = 0.0) -> pd.Series:
    return (df > detection_limit).mean(axis=0)


def filter_by_prevalence(
    df: pd.DataFrame,
    prevalence: pd.Series,
    min_prevalence: float = 0.10,
) -> pd.DataFrame:
    keep = prevalence >= min_prevalence
    return df.loc[:, keep]


def filter_by_mean_abundance(
    df: pd.DataFrame,
    min_mean_abundance: float = 0.0001,
) -> pd.DataFrame:
    keep = df.mean(axis=0) >= min_mean_abundance
    return df.loc[:, keep]


def select_top_taxa(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    n = min(n, df.shape[1])
    top_cols = df.mean(axis=0).nlargest(n).index
    return df[top_cols]


def collapse_other_taxa(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    if df.shape[1] <= top_n:
        return df.copy()
    top_cols = df.mean(axis=0).nlargest(top_n).index
    other_cols = df.columns.difference(top_cols)
    result = df[top_cols].copy()
    result["Other"] = df[other_cols].sum(axis=1)
    return result
