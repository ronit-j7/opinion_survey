"""Step 02: pairwise Spearman -> latent Pearson scale -> valid correlation matrix."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from bn import config


@dataclass
class CorrelationResult:
    spearman: pd.DataFrame      # raw pairwise Spearman r^S
    R: pd.DataFrame             # 2 sin(pi r^S / 6), repaired if needed; input to glasso
    n_pairwise: pd.DataFrame    # n_jk = respondents who answered both j and k
    min_eig_before: float
    min_eig_after: float
    repaired: bool
    n_undefined_pairs: int      # pairs with too few shared respondents or zero variance; set to 0 in R


def pairwise_counts(X: pd.DataFrame) -> pd.DataFrame:
    obs = X.notna().to_numpy(dtype=float)
    return pd.DataFrame((obs.T @ obs).astype(int), index=X.columns, columns=X.columns)


def spearman_pairwise(X: pd.DataFrame, min_periods: int) -> pd.DataFrame:
    """Spearman on pairwise-complete rows.

    pandas re-ranks each pair on the rows where both items are observed,
    using average (mid) ranks for ties.
    """
    return X.corr(method="spearman", min_periods=min_periods)


def spearman_to_pearson(rs: np.ndarray) -> np.ndarray:
    """Gaussian-copula conversion r = 2 sin(pi r^S / 6) (Liu et al., 2012)."""
    R = 2.0 * np.sin(np.pi * rs / 6.0)
    np.fill_diagonal(R, 1.0)
    return R


def repair_min_eigenvalue(R: np.ndarray, floor: float) -> tuple[np.ndarray, bool]:
    """Clip eigenvalues below `floor`, then rescale back to a unit diagonal."""
    R = (R + R.T) / 2.0
    w, Q = np.linalg.eigh(R)
    if w.min() >= floor:
        return R, False
    A = (Q * np.clip(w, floor, None)) @ Q.T
    d = np.sqrt(np.diag(A))
    A = A / np.outer(d, d)
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 1.0)
    return A, True


def estimate_correlation(X: pd.DataFrame,
                         min_periods: int = config.MIN_PAIRWISE_N,
                         eig_floor: float = config.EIG_FLOOR) -> CorrelationResult:
    cols = X.columns
    rs = spearman_pairwise(X, min_periods)

    rs_values = rs.to_numpy(copy=True)
    iu = np.triu_indices(len(cols), 1)
    n_undefined = int(np.isnan(rs_values[iu]).sum())
    rs_values = np.nan_to_num(rs_values, nan=0.0)

    R = spearman_to_pearson(rs_values)
    min_before = float(np.linalg.eigvalsh((R + R.T) / 2.0).min())
    R, repaired = repair_min_eigenvalue(R, eig_floor)
    min_after = float(np.linalg.eigvalsh(R).min())

    return CorrelationResult(
        spearman=rs,
        R=pd.DataFrame(R, index=cols, columns=cols),
        n_pairwise=pairwise_counts(X),
        min_eig_before=min_before,
        min_eig_after=min_after,
        repaired=repaired,
        n_undefined_pairs=n_undefined,
    )
