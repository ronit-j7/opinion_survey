"""Step 02 logic: respondent measures.

Two representations per respondent:
- the 4x5 ordinal histogram stack (domain x answer-category, rows sum to 1),
- the ipsative domain-mean profile (4 values, rows sum to 0)
plus the scalar extremity (mean |x - 3|).
"""
import numpy as np
import pandas as pd

from rn import config
from rn.data import domain_columns


def histograms(X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (hist, answered_per_domain).

    hist: (n, 4, 5) with hist[i, d, c] = share of answered items in domain d
    scored c+1; each (i, d) row sums to 1. Missing answers are excluded from
    the denominator of their own domain.
    """
    doms = domain_columns(X)
    n = len(X)
    hist = np.zeros((n, len(config.DOMAINS), config.HIST_BINS))
    answered = np.zeros((n, len(config.DOMAINS)))
    codes = X.to_numpy()
    for d, dom in enumerate(config.DOMAINS):
        idx = [X.columns.get_loc(c) for c in doms[dom]]
        block = codes[:, idx]
        counts = np.stack([
            np.bincount(row[~np.isnan(row)].astype(int), minlength=6)[1:6]
            for row in block
        ])
        ans = counts.sum(axis=1, keepdims=True)
        hist[:, d, :] = np.divide(counts, ans,
                                  out=np.zeros_like(counts, dtype=float),
                                  where=ans > 0)
        answered[:, d] = ans.ravel()
    return hist, answered


def domain_means(X: pd.DataFrame) -> pd.DataFrame:
    doms = domain_columns(X)
    return pd.DataFrame({d: X[cols].mean(axis=1) for d, cols in doms.items()})


def ipsative(X: pd.DataFrame) -> pd.DataFrame:
    """Row-centered domain means: each respondent's profile sums to 0."""
    D = domain_means(X)
    return D.sub(D.mean(axis=1), axis=0)


def extremity(X: pd.DataFrame) -> pd.Series:
    return X.sub(3).abs().mean(axis=1)


def dominant(X: pd.DataFrame) -> pd.Series:
    return domain_means(X).idxmax(axis=1)


def item_entropy(X: pd.DataFrame) -> pd.Series:
    """Normalized Shannon entropy of each item's answer distribution."""
    out = {}
    for c in X.columns:
        p = X[c].value_counts(normalize=True).reindex(
            range(1, config.HIST_BINS + 1), fill_value=0
        ).to_numpy()
        p = p[p > 0]
        out[c] = float(-(p * np.log(p)).sum() / np.log(config.HIST_BINS))
    return pd.Series(out).sort_values(ascending=False)
