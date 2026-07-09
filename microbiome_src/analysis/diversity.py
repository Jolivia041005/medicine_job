import numpy as np
import pandas as pd


def calculate_shannon(df: pd.DataFrame, pseudocount: float = 1e-12) -> pd.Series:
    rel = df.div(df.sum(axis=1), axis=0)
    log_rel = np.log(rel.values + pseudocount)
    shannon = -np.sum(rel.values * log_rel, axis=1)
    return pd.Series(shannon, index=rel.index, name="Shannon")
