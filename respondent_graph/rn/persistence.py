"""Step 05 logic: persistent homology on the W1 (or residual) metric.

Primary complex: Vietoris-Rips. Alternative: witness (gudhi), used in the
robustness grid. Null ensemble: independent column permutation (preserves
item marginals, destroys alignment). Bootstrap: respondent resampling.
"""
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from ripser import ripser

from rn import config
from rn.measures import histograms
from rn.w1 import w1_matrix


def rips_dgms(D: np.ndarray, maxdim: int = None) -> list[np.ndarray]:
    res = ripser(D, distance_matrix=True, maxdim=maxdim or config.MAXDIM)
    return res["dgms"]


def h1_max_persistence(dgms: list[np.ndarray]) -> float:
    if len(dgms) < 2 or len(dgms[1]) == 0:
        return 0.0
    d = dgms[1]
    return float(np.max(d[:, 1] - d[:, 0]))


def single_linkage(D: np.ndarray) -> np.ndarray:
    return linkage(squareform(D, checks=False), method="single")


def heights(Z: np.ndarray) -> np.ndarray:
    return Z[:, 2]


def max_gap_info(D: np.ndarray) -> dict:
    """Largest jump between consecutive single-linkage merge heights."""
    Z = single_linkage(D)
    h = heights(Z)
    if len(h) < 2:
        return {"max_gap": float(h[-1]) if len(h) else 0.0,
                "cut": float("inf"), "sizes": [len(D)], "gap_index": 0}
    gaps = np.diff(h)
    gi = int(np.argmax(gaps))
    cut = 0.5 * (h[gi] + h[gi + 1])
    labels = fcluster(Z, t=cut, criterion="distance")
    _, counts = np.unique(labels, return_counts=True)
    return {
        "max_gap": float(gaps[gi]),
        "gap_index": gi,
        "cut": float(cut),
        "sizes": [int(c) for c in np.sort(counts)[::-1]],
    }


def ph_summary(D: np.ndarray) -> dict:
    dgms = rips_dgms(D)
    return {
        "max_gap": max_gap_info(D)["max_gap"],
        "max_h0_death": float(np.nanmax(rips_dgms(D)[0][:, 1][np.isfinite(rips_dgms(D)[0][:, 1])])) if len(rips_dgms(D)[0]) else 0.0,
        "h1_max": h1_max_persistence(dgms),
    }


def w1_from_frame(X, kind=None, agg=None) -> np.ndarray:
    hist, answered = histograms(X)
    return w1_matrix(hist, kind=kind, agg=agg, answered=answered)


def null_ensemble(X, kind=None, agg=None, n_perm=None, seed=None):
    """Column-permutation null: permute every item independently, rebuild W1,
    return arrays of (max_gap, h1_max) per permutation."""
    rng = np.random.default_rng(config.SEED if seed is None else seed)
    n_perm = n_perm or config.N_PERM
    gaps, h1s, sizes = [], [], []
    Xp = X.copy()
    for _ in range(n_perm):
        for c in Xp.columns:
            Xp[c] = rng.permutation(Xp[c].to_numpy())
        D = w1_from_frame(Xp, kind=kind, agg=agg)
        info = max_gap_info(D)
        gaps.append(info["max_gap"])
        sizes.append(info["sizes"])
        h1s.append(h1_max_persistence(rips_dgms(D)))
    return {
        "max_gap": np.asarray(gaps),
        "h1_max": np.asarray(h1s),
        "sizes": sizes,
    }


def bootstrap(D: np.ndarray, n_boot=None, seed=None):
    rng = np.random.default_rng(config.SEED if seed is None else seed)
    n_boot = n_boot or config.N_BOOT
    gaps, h1s = [], []
    n = D.shape[0]
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        Db = D[np.ix_(idx, idx)]
        gaps.append(max_gap_info(Db)["max_gap"])
        h1s.append(h1_max_persistence(rips_dgms(Db)))
    return {"max_gap": np.asarray(gaps), "h1_max": np.asarray(h1s)}


def witness_dgms(D: np.ndarray, seed=None):
    """Witness complex via gudhi with maxmin landmark selection."""
    try:
        import gudhi
    except ImportError:
        return None
    rng = np.random.default_rng(config.SEED if seed is None else seed)
    n = D.shape[0]
    k = max(10, int(config.WITNESS_LANDMARK_FRAC * n))
    first = int(rng.integers(n))
    landmarks = [first]
    dmin = D[:, first].copy()
    for _ in range(k - 1):
        far = int(np.argmax(dmin))
        landmarks.append(far)
        dmin = np.minimum(dmin, D[:, far])
    witnesses = []
    for i in range(n):
        order = np.argsort(D[i, landmarks])
        witnesses.append([(int(order[j]), float(D[i, landmarks[order[j]]]))
                          for j in range(min(4, k))])
    st = gudhi.WitnessComplex(witnesses).create_simplex_tree(max_alpha_square=float("inf"))
    st.persistence(homology_coeff_field=2, min_persistence=0)
    return [np.asarray(st.persistence_intervals_in_dimension(dim), dtype=float)
            for dim in (0, 1)]


def witness_summary(D: np.ndarray, seed=None) -> dict:
    dgms = witness_dgms(D, seed=seed)
    if dgms is None:
        return {"available": False}
    h0 = dgms[0][:, 1]
    h0 = h0[np.isfinite(h0)]
    return {
        "available": True,
        "max_h0_death": float(np.max(h0)) if len(h0) else 0.0,
        "h1_max": h1_max_persistence(dgms),
    }
