"""Step 03 - the W1 metric and its validation gate.

In:  outputs/clean.parquet, outputs/measures.npy
Out: outputs/W1.npy                n x n W1 matrix
     outputs/metric_validation.json
     figures/03_validation.png

GATE: Spearman rho of W1 against the mean-priority (ipsative Euclidean)
distance must stay below GATE_RHO, otherwise OT has degenerated and steps
04-08 need rethinking. The negative control documents the discarded
60-dim formulation collapsing into cosine under a random ground cost.
"""
import numpy as np
import ot as pot
import pandas as pd

from rn import config
from rn.io import load_npy, save_json, save_npy
from rn.measures import ipsative
from rn.stats import mantel_permutation, perm_p_value, spearman_r
from rn.w1 import emd2_crosscheck, ground_cost, w1_matrix


def split_half_reliability(X: pd.DataFrame, n_splits=None, n_null=None, seed=None):
    """Repeated stratified item half-splits: rebuild W1 per half, Mantel-
    Spearman between the halves; null rebuilds both halves after permuting
    each item's answers across respondents."""
    rng = np.random.default_rng(config.SEED if seed is None else seed)
    n_splits = n_splits or config.N_SPLIT
    n_null = n_null or config.N_SPLIT_NULL
    doms = {d: [c for c in X.columns if c[0] == d] for d in config.DOMAINS}
    from rn.persistence import w1_from_frame

    def halves(Xf):
        A_cols, B_cols = [], []
        for d in config.DOMAINS:
            cols = list(rng.permutation(doms[d]))
            A_cols += cols[: len(cols) // 2]
            B_cols += cols[len(cols) // 2:]
        return w1_from_frame(Xf[A_cols]), w1_from_frame(Xf[B_cols])

    from rn.stats import mantel_r
    rs = np.array([mantel_r(*halves(X)) for _ in range(n_splits)])

    Xp = X.copy()
    null_rs = []
    for _ in range(n_null):
        for c in Xp.columns:
            Xp[c] = rng.permutation(Xp[c].to_numpy())
        null_rs.append(mantel_r(*halves(Xp)))
    null_rs = np.asarray(null_rs)
    return rs, null_rs


def negative_control(X: pd.DataFrame, seed=None) -> dict:
    """The discarded 60-dim formulation: respondent measures over statements,
    random ground cost - collapses toward cosine distance."""
    rng = np.random.default_rng(config.NEGCTRL_SEED if seed is None else seed)
    V = X.to_numpy()
    W = np.ascontiguousarray(V / V.sum(axis=1, keepdims=True))
    C = np.abs(rng.normal(size=(V.shape[1], V.shape[1])))
    C = np.ascontiguousarray((C + C.T) / 2)
    np.fill_diagonal(C, 0.0)
    n = len(X)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = pot.emd2(W[i], W[j], C)
    norm = np.linalg.norm(W, axis=1, keepdims=True)
    Un = W / norm
    cos = 1.0 - Un @ Un.T
    from rn.stats import spearman_r
    return {
        "rho_w1_random_cost_vs_cosine": float(spearman_r(D, cos)),
        "note": "60-dim statement measures, random ground cost: W1 is "
                "redundant with cosine, hence discarded.",
    }


def main() -> None:
    config.ensure_dirs()
    X = pd.read_parquet(config.OUTPUTS / "clean.parquet").drop(
        columns=["row_position"], errors="ignore")
    hist = load_npy(config.OUTPUTS / "measures.npy")

    W1 = w1_matrix(hist)
    save_npy(W1, config.OUTPUTS / "W1.npy")

    P = ipsative(X).to_numpy()
    profile_dist = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    rho = spearman_r(W1, profile_dist)
    mant = mantel_permutation(W1, profile_dist, 1000, config.SEED)

    check = emd2_crosscheck(hist, n_pairs=50, seed=config.SEED)

    rs, null_rs = split_half_reliability(X)
    from scipy.stats import spearmanr
    rel_r = float(np.mean(rs))
    rel_null_mean = float(null_rs.mean())
    rel_null_sd = float(null_rs.std(ddof=1))
    rel_z = (rel_r - rel_null_mean) / rel_null_sd
    rel_p = perm_p_value(rel_r, null_rs)

    nctrl = negative_control(X)

    val = {
        "n": len(X),
        "cost": config.COST_KIND,
        "aggregation": config.AGG,
        "emd2_vs_cdf_l1_max_abs_diff": check,
        "emd2_cdf_identity_holds": bool(check <= config.EMD_TOL),
        "rho_w1_vs_mean_priority": rho,
        "mantel_vs_mean_priority": mant,
        "gate_rho_max": config.GATE_RHO,
        "gate_passed": bool(rho < config.GATE_RHO),
        "split_half": {
            "n_splits": int(len(rs)),
            "mean_r": rel_r,
            "sd_r": float(rs.std(ddof=1)),
            "null_mean": rel_null_mean,
            "null_sd": rel_null_sd,
            "z": float(rel_z),
            "p": rel_p,
        },
        "negative_control": nctrl,
    }
    save_json(val, config.OUTPUTS / "metric_validation.json")

    _plot(W1, profile_dist, rs, null_rs, rho)

    print(f"n = {val['n']}")
    print(f"emd2 vs CDF-L1 max abs diff: {check:.2e} (identity "
          f"{'HOLDS' if val['emd2_cdf_identity_holds'] else 'VIOLATED'})")
    print(f"rho(W1, mean-priority) = {rho:.3f}  "
          f"[gate < {config.GATE_RHO}: {'PASS' if val['gate_passed'] else 'FAIL'}]")
    print(f"split-half: mean r {rel_r:.3f} vs null {rel_null_mean:.3f} "
          f"(sd {rel_null_sd:.3f})  z = {rel_z:+.1f}  p = {rel_p:.4f}")
    print("negative control rho:", round(
        nctrl["rho_w1_random_cost_vs_cosine"], 3))


def _plot(W1, profile_dist, rs, null_rs, rho):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from rn import viz
    from rn.stats import upper_triangle

    viz.set_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))

    axes[0].scatter(upper_triangle(profile_dist), upper_triangle(W1),
                    s=11, alpha=0.35, c=viz.magma(0.42), edgecolors="none")
    axes[0].set(xlabel="mean-priority (ipsative Euclidean) distance",
                ylabel="W1")
    axes[0].set_title(f"W1 is not the mean profile  (ρ = {rho:.2f})")

    sns.histplot(rs, ax=axes[1], color=viz.magma(0.42), fill=True,
                 alpha=viz.FILL_ALPHA, stat="density", label="observed")
    sns.histplot(null_rs, ax=axes[1], color=viz.NEUTRAL, fill=True,
                 alpha=0.30, stat="density", label="item-permutation null")
    axes[1].set_title(f"Split-half reliability (mean r = {rs.mean():.3f})")
    axes[1].set(xlabel="Mantel r between item half-samples")
    axes[1].legend()
    viz.savefig(fig, "03_validation.png")


if __name__ == "__main__":
    main()
