import hashlib
import json
from pathlib import Path
from typing import Optional

import pandas as pd


def detect_separator(filepath: str | Path) -> str:
    with open(filepath, "r") as f:
        first_line = f.readline()
    if "\t" in first_line:
        return "\t"
    return ","


def read_table(filepath: str | Path, sep: Optional[str] = None) -> pd.DataFrame:
    if sep is None:
        sep = detect_separator(filepath)
    df = pd.read_csv(filepath, sep=sep, comment=None)
    header_val = str(df.columns[0])
    if header_val.startswith("#"):
        df.rename(columns={header_val: header_val.lstrip("#")}, inplace=True)
    return df


def detect_orientation(df: pd.DataFrame, id_column: Optional[str] = None) -> str:
    """
    Heuristic: if the first column is string-like and other columns are numeric,
    likely taxa-as-rows. Returns 'taxa_as_rows' or 'samples_as_rows'.
    """
    if id_column is None:
        id_column = df.columns[0]

    numeric_cols = df.drop(columns=[id_column]).select_dtypes(include="number").columns
    non_numeric = set(df.drop(columns=[id_column]).columns) - set(numeric_cols)
    if len(non_numeric) == 0 and len(numeric_cols) > 1:
        return "taxa_as_rows"
    if df[id_column].dtype.kind in "iuf":
        return "samples_as_rows"
    return "taxa_as_rows"


def standardize_to_sample_x_taxon(
    df: pd.DataFrame,
    id_column: Optional[str] = None,
    orientation: Optional[str] = None,
) -> pd.DataFrame:
    if id_column is None:
        id_column = df.columns[0]
    if orientation is None:
        orientation = detect_orientation(df, id_column)

    df = df.set_index(id_column)
    df.index = df.index.astype(str).str.strip()

    if orientation == "taxa_as_rows":
        df = df.T

    df.index.name = "SampleID"
    df.columns.name = None

    numeric_cols = df.select_dtypes(include="number").columns
    non_numeric = set(df.columns) - set(numeric_cols)
    if non_numeric:
        df = df.drop(columns=list(non_numeric))

    return df


def file_sha256(filepath: str | Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def infer_abundance_scale(df: pd.DataFrame, tol: float = 0.05) -> str:
    row_sums = df.sum(axis=1)
    if row_sums.min() < 0:
        return "invalid"
    all_close_to_one = ((row_sums - 1.0).abs() < tol).all()
    if all_close_to_one:
        return "relative"
    all_close_to_100 = ((row_sums - 100.0).abs() < tol * 100).all()
    if all_close_to_100:
        return "relative_100"
    if (row_sums > 1).all():
        return "count"
    if (row_sums >= 0).all() and (row_sums <= 1.1).all():
        return "relative"
    return "unknown"
