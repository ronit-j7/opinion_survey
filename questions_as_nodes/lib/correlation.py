"""Step 02: pairwise correlation matrix (Pearson on the 1-5 codes by default)."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from lib import config


@dataclass
class CorrelationResult:
    R: pd.DataFrame             # pairwise correlations
    n_pairwise: pd.DataFrame    # n_jk = respondents who answered both j and k
    min_eigenvalue: float       # > 0 means R is positive definite (only needed by glasso)
    n_undefined_pairs: int      # pairs with too few shared respondents or zero variance; set to 0


def pairwise_counts(X: pd.DataFrame) -> pd.DataFrame:
    obs = X.notna().to_numpy(dtype=float) # X  is (91 * 60 of survey answers). We convert that to have 1s where response is not Nan and 0 when it is Nan. 
    return pd.DataFrame((obs.T @ obs).astype(int), index=X.columns, columns=X.columns) # The operation X^T @ X gives us the total number of people who answered each 2 questions (60*60 matrix).


def pairwise_correlation(X: pd.DataFrame, method: str, min_periods: int) -> pd.DataFrame:
    """Correlation on pairwise-complete rows.

    Each pair uses only the respondents who answered both items.
    For "spearman", pandas re-ranks each pair on those rows with mid-ranks for ties.
    """
    if method not in ("pearson", "spearman"):
        raise ValueError(f"Unknown correlation method: {method}")
    return X.corr(method=method, min_periods=min_periods)


def estimate_correlation(X: pd.DataFrame,
                         method: str = config.CORR_METHOD,
                         min_periods: int = config.MIN_PAIRWISE_N) -> CorrelationResult:
    r = pairwise_correlation(X, method, min_periods)

    values = r.to_numpy(copy=True)
    iu = np.triu_indices(len(values), 1) # np.triu gives the indices associated with the upper triangular slice of the n*n matrix. (,1 ) means offset by one so the diagonal isn't included. 
    n_undefined = int(np.isnan(values[iu]).sum())
    values = np.nan_to_num(values, nan=0.0)
    np.fill_diagonal(values, 1.0)

    return CorrelationResult(
        R=pd.DataFrame(values, index=r.index, columns=r.columns),
        n_pairwise=pairwise_counts(X),
        min_eigenvalue=float(np.linalg.eigvalsh(values).min()),
        n_undefined_pairs=n_undefined,
    )
