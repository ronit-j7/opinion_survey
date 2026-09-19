"""Step 03/04 logic: the W1 metric, transport plans, and barycenters.

W1(i, j) = aggregate over domains of EMD(hist_i[d], hist_j[d]; M) where
M[c, c'] = |c - c'| (primary) or (c - c')^2 (robustness axis).

For the ordinal |c - c'| cost the per-domain EMD equals the L1 distance
between CDFs exactly, so the n x n matrix is computed with the closed form
and cross-checked against ot.emd2.
"""
import numpy as np
import ot as pot

from rn import config


def ground_cost(kind: str = None) -> np.ndarray:
    kind = kind or config.COST_KIND
    c = np.arange(config.HIST_BINS, dtype=float)
    diff = np.abs(c[:, None] - c[None, :])
    return diff if kind == "abs" else diff ** 2


def _emd_line_convex(H: np.ndarray, power: int) -> np.ndarray:
    """Exact pairwise EMD for 1-D convex costs f(|i-j|), f(t)=t^power.

    On the line with convex cost the monotone (comonotone) coupling is
    optimal, so EMD(i,j) = sum_m moved_mass * |level_i - level_j|^2 over the
    north-west-corner sweep. Vectorized across all pairs at once.
    """
    n = H.shape[0]
    bins = H.shape[1]
    I, J = np.triu_indices(n, 1)
    IA = np.zeros(len(I), dtype=int)
    IB = np.zeros(len(I), dtype=int)
    RA = H[I, 0].copy()
    RB = H[J, 0].copy()
    COST = np.zeros(len(I))
    for _ in range(2 * bins):
        m = np.minimum(RA, RB)
        COST += m * np.abs(IA - IB) ** power
        RA -= m
        RB -= m
        adv_a = (RA <= 1e-12) & (IA < bins - 1)
        IA[adv_a] += 1
        RA[adv_a] = H[I[adv_a], IA[adv_a]]
        adv_b = (RB <= 1e-12) & (IB < bins - 1)
        IB[adv_b] += 1
        RB[adv_b] = H[J[adv_b], IB[adv_b]]
        if not (adv_a | adv_b).any() and (RA <= 1e-12).all() and (RB <= 1e-12).all():
            break
    D = np.zeros((n, n))
    D[I, J] = COST
    D[J, I] = COST
    return D


def w1_matrix(hist: np.ndarray, kind: str = None, agg: str = None,
              answered: np.ndarray = None) -> np.ndarray:
    """Pairwise W1 matrix from an (n, 4, 5) histogram stack.

    agg: 'sum' | 'max' | 'weighted' (weighted by answered items per domain).
    """
    kind = kind or config.COST_KIND
    agg = agg or config.AGG
    if kind == "abs":
        cdf = np.cumsum(hist, axis=2)
        per_dom = np.abs(cdf[:, None, :, :] - cdf[None, :, :, :]).sum(axis=3)
    elif kind == "sq":
        per_dom = np.stack([_emd_line_convex(hist[:, d, :], 2)
                            for d in range(hist.shape[1])], axis=2)
    else:
        raise ValueError(f"unknown cost kind {kind}")
    return _aggregate(per_dom, agg, answered)


def _aggregate(per_dom: np.ndarray, agg: str, answered: np.ndarray) -> np.ndarray:
    if agg == "sum":
        return per_dom.sum(axis=2)
    if agg == "max":
        return per_dom.max(axis=2)
    if agg == "weighted":
        if answered is None:
            raise ValueError("weighted aggregation needs answered counts")
        w = answered / answered.sum(axis=1, keepdims=True)
        return (per_dom * w[:, None, :]).sum(axis=2)
    raise ValueError(f"unknown aggregation {agg}")


def emd2_crosscheck(hist: np.ndarray, n_pairs: int = 50, seed: int = 0) -> float:
    """Max |ot.emd2 - closed form| over a sample of pairs (abs cost only)."""
    rng = np.random.default_rng(seed)
    M = ground_cost("abs")
    n = hist.shape[0]
    worst = 0.0
    cdf = np.cumsum(hist, axis=2)
    for _ in range(n_pairs):
        i, j = rng.integers(0, n, 2)
        for d in range(hist.shape[1]):
            v = pot.emd2(hist[i, d], hist[j, d], M)
            closed = np.abs(cdf[i, d] - cdf[j, d]).sum()
            worst = max(worst, abs(v - closed))
    return float(worst)


def mean_flow(hist: np.ndarray, kind: str = None) -> np.ndarray:
    """Mean transport plan per domain over all respondent pairs: (4, 5, 5)."""
    M = ground_cost(kind)
    n, nd = hist.shape[0], hist.shape[1]
    flow = np.zeros((nd, config.HIST_BINS, config.HIST_BINS))
    for d in range(nd):
        acc = np.zeros_like(flow[0])
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                acc += pot.emd(hist[i, d], hist[j, d], M)
        flow[d] = acc / (n * (n - 1))
    return flow


def flow_summary(flow: np.ndarray) -> dict:
    """Band vs midpoint-crossing structure of a 5x5 flow matrix."""
    band = 0.0
    cross = 0.0
    for c1 in range(5):
        for c2 in range(5):
            if {c1, c2} <= {2, 3, 4}:
                band += flow[c1, c2]
            if (c1 <= 1 and c2 >= 3) or (c2 <= 1 and c1 >= 3):
                cross += flow[c1, c2]
    return {"neutral_agree_sa_band": float(band), "midpoint_crossing": float(cross)}


def _barycenter_1d(A_hist: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Fixed-support W1 barycenter of (n, bins) histograms (uniform weights)
    as an LP over per-histogram transport plans plus the barycenter vector."""
    from scipy.optimize import linprog
    from scipy.sparse import lil_matrix

    n, bins = A_hist.shape
    nvar = n * bins * bins + bins
    c = np.concatenate([M.ravel()] * n + [np.zeros(bins)])

    Aeq = lil_matrix((2 * n * bins, nvar))
    beq = np.zeros(2 * n * bins)
    for k in range(n):
        off = k * bins * bins
        for i in range(bins):
            for j in range(bins):
                Aeq[k * bins + i, off + i * bins + j] = 1.0
                Aeq[n * bins + k * bins + j, off + i * bins + j] = 1.0
            Aeq[k * bins + i, n * bins * bins + i] = -1.0
            beq[n * bins + k * bins + i] = A_hist[k, i]
    res = linprog(c, A_eq=Aeq.tocsr(), b_eq=beq, method="highs")
    if not res.success:
        raise RuntimeError(f"barycenter LP failed: {res.message}")
    out = np.clip(res.x[n * bins * bins:], 0, None)
    return out / out.sum()


def barycenter_domain(hist: np.ndarray, kind: str = None) -> np.ndarray:
    """Wasserstein barycenter per domain from an (n, 4, 5) histogram stack."""
    M = ground_cost(kind)
    return np.stack([_barycenter_1d(hist[:, d, :], M)
                     for d in range(hist.shape[1])])
