"""Statistics helpers: Mantel tests, permutation p/z, BH-FDR, Fisher z."""
import numpy as np
from scipy.stats import spearmanr


def upper_triangle(A: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(A.shape[0], 1)
    return A[iu]


def spearman_r(A: np.ndarray, B: np.ndarray) -> float:
    """Spearman correlation between two distance matrices (upper triangles)."""
    return float(spearmanr(upper_triangle(A), upper_triangle(B)).statistic)


def mantel_r(A: np.ndarray, B: np.ndarray) -> float:
    r = spearmanr(upper_triangle(A), upper_triangle(B)).statistic
    return float(r)


def mantel_permutation(A: np.ndarray, B: np.ndarray, n_perm: int, seed: int) -> dict:
    """Spearman Mantel test: permute the labels of one matrix."""
    rng = np.random.default_rng(seed)
    r_obs = mantel_r(A, B)
    n = A.shape[0]
    null = np.empty(n_perm)
    for p in range(n_perm):
        perm = rng.permutation(n)
        null[p] = mantel_r(A[np.ix_(perm, perm)], B)
    pval = (1 + np.sum(null >= r_obs)) / (n_perm + 1)
    return {"r": r_obs, "null_mean": float(null.mean()),
            "null_sd": float(null.std(ddof=1)), "p": float(pval)}


def perm_p_value(obs: float, null: np.ndarray, greater: bool = True) -> float:
    null = np.asarray(null)
    if greater:
        return float((1 + np.sum(null >= obs)) / (len(null) + 1))
    return float((1 + np.sum(null <= obs)) / (len(null) + 1))


def z_vs_null(obs: float, null: np.ndarray) -> float:
    null = np.asarray(null)
    sd = null.std(ddof=1)
    return float((obs - null.mean()) / sd) if sd > 0 else float("inf")


def bh_fdr(pvals: np.ndarray, q: float = 0.05):
    """Benjamini-Hochberg: returns (rejected mask, adjusted p-values)."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    ranked = p[order]
    adj = np.minimum.accumulate((m / np.arange(1, m + 1)) * ranked[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    adj_sorted = np.empty(m)
    adj_sorted[order] = adj
    return adj_sorted <= q, adj_sorted


def fisher_z(r: float) -> float:
    return float(np.arctanh(np.clip(r, -0.999999, 0.999999)))
