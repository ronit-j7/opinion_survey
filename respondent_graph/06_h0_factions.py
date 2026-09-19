"""Step 06 - H0 read as a single-linkage dendrogram: significant gaps,
faction verdict, and the Louvain comparison.

In:  outputs/W1.npy, outputs/clean.parquet, outputs/null_ensemble.npy,
     outputs/persistence_stats.json
Out: outputs/dendrogram.json    merge heights, gaps, cuts, component sizes
     outputs/faction_verdict.json
     figures/06_dendrogram.png, figures/06_louvain.png
"""
import json

import networkx as nx
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import adjusted_rand_score

from rn import config
from rn.graphs import build_graph, louvain_partition, modularity
from rn.io import load_json, load_npy, save_json
from rn.persistence import max_gap_info, single_linkage, heights, w1_from_frame
from rn.stats import z_vs_null


def significant_gaps(Z: np.ndarray, null_max_gap: np.ndarray) -> list[dict]:
    """All merge-height gaps whose size exceeds the null 95th percentile."""
    h = heights(Z)
    null95 = float(np.percentile(null_max_gap, 95))
    out = []
    for i in range(len(h) - 1):
        gap = h[i + 1] - h[i]
        if gap >= null95:
            cut = 0.5 * (h[i] + h[i + 1])
            labels = fcluster(Z, t=cut, criterion="distance")
            _, counts = np.unique(labels, return_counts=True)
            out.append({
                "gap": float(gap),
                "cut": float(cut),
                "sizes": [int(c) for c in np.sort(counts)[::-1]],
                "z": z_vs_null(gap, null_max_gap),
            })
    return out


def louvain_comparison(X: pd.DataFrame, W1: np.ndarray) -> dict:
    ids = list(X.index)
    G = build_graph(W1, ids, k=config.K_MUTUAL)
    part = louvain_partition(G)
    q_obs = modularity(G, part)

    rng = np.random.default_rng(config.SEED)
    Xp = X.copy()
    q_null = []
    for _ in range(config.N_NULL_GRAPH):
        for c in Xp.columns:
            Xp[c] = rng.permutation(Xp[c].to_numpy())
        Dp = w1_from_frame(Xp)
        Gp = build_graph(Dp, ids, k=config.K_MUTUAL)
        q_null.append(modularity(Gp, louvain_partition(Gp, seed=int(rng.integers(1e6)))))
    q_null = np.asarray(q_null)

    full_labels = {}
    for ci, comm in enumerate(part):
        for node in comm:
            full_labels[node] = ci

    doms = {d: [c for c in X.columns if c[0] == d] for d in config.DOMAINS}
    aris, coassign = [], []
    pairs = list(zip(*np.triu_indices(len(ids), 1)))
    co_mat = np.zeros((len(ids), len(ids)))
    for _ in range(config.N_SUBSAMPLE):
        cols_a = []
        for d in config.DOMAINS:
            dc = list(rng.permutation(doms[d]))
            cols_a += dc[: len(dc) // 2]
        Dh = w1_from_frame(X[cols_a])
        Gh = build_graph(Dh, ids, k=config.K_MUTUAL)
        ph = louvain_partition(Gh, seed=int(rng.integers(1e6)))
        lh = {}
        for ci, comm in enumerate(ph):
            for node in comm:
                lh[node] = ci
        aris.append(adjusted_rand_score(
            [full_labels[i] for i in ids], [lh[i] for i in ids]))
        for a, b in pairs:
            co_mat[a, b] += float(lh[ids[a]] == lh[ids[b]])
    coassign_share = co_mat / config.N_SUBSAMPLE
    frac_pairs_above_80 = float(np.mean(coassign_share[np.triu_indices(len(ids), 1)] >= 0.8))

    k_sweep = {}
    for k in config.K_SWEEP:
        Gk = build_graph(W1, ids, k=k)
        pk = louvain_partition(Gk)
        k_sweep[int(k)] = {
            "n_edges": Gk.number_of_edges(),
            "n_communities": len(pk),
            "modularity": modularity(Gk, pk),
        }

    return {
        "k": config.K_MUTUAL,
        "n_edges": G.number_of_edges(),
        "n_communities": len(part),
        "sizes": sorted((len(c) for c in part), reverse=True),
        "modularity_observed": q_obs,
        "modularity_null_mean": float(q_null.mean()),
        "modularity_null_sd": float(q_null.std(ddof=1)),
        "modularity_z": z_vs_null(q_obs, q_null),
        "ari_item_subsample_mean": float(np.mean(aris)),
        "ari_item_subsample_sd": float(np.std(aris, ddof=1)),
        "pairs_coassigned_above_80pct": frac_pairs_above_80,
        "k_stability_sweep": k_sweep,
        "partition": {str(i): int(full_labels[i]) for i in ids},
    }


def main() -> None:
    config.ensure_dirs()
    X = pd.read_parquet(config.OUTPUTS / "clean.parquet").drop(
        columns=["row_position"], errors="ignore")
    W1 = load_npy(config.OUTPUTS / "W1.npy")
    null = np.load(config.OUTPUTS / "null_max_gap.npy")

    Z = single_linkage(W1)
    h = heights(Z)
    gaps = np.diff(h)
    dendro = {
        "merge_heights": [float(v) for v in h],
        "gaps": [float(v) for v in gaps],
        "max_gap": max_gap_info(W1),
        "significant_gaps": significant_gaps(Z, null),
    }
    save_json(dendro, config.OUTPUTS / "dendrogram.json")

    min_size = max(2, int(config.FACTION_MIN_SIZE_FRAC * len(X)))
    sig = dendro["significant_gaps"]
    faction_splits = [
        g for g in sig
        if sum(1 for s in g["sizes"] if s >= min_size) >= 2
    ]
    verdict = {
        "n": len(X),
        "faction_min_size": min_size,
        "n_significant_gaps": len(sig),
        "significant_splits": sig,
        "faction_splits": faction_splits,
        "verdict": "factions" if faction_splits else "no_factions",
        "n_h1_significant": int(load_json(config.OUTPUTS / "persistence_stats.json")["w1"]["h1_p"] < config.ALPHA),
    }

    louv = louvain_comparison(X, W1)
    verdict["louvain_comparison"] = louv
    save_json(verdict, config.OUTPUTS / "faction_verdict.json")

    _plot(Z, sig, louv)

    print("significant gaps:", json.dumps(sig, default=str))
    print("verdict:", verdict["verdict"])
    print(f"louvain: k={louv['k']} edges={louv['n_edges']} "
          f"communities={louv['n_communities']} sizes={louv['sizes']}")
    print(f"Q obs {louv['modularity_observed']:.3f} vs null "
          f"{louv['modularity_null_mean']:.3f} (z {louv['modularity_z']:+.1f})")
    print(f"item-subsample ARI {louv['ari_item_subsample_mean']:.2f} | "
          f"pairs co-assigned >80%: {louv['pairs_coassigned_above_80pct']:.1%}")


def _plot(Z, sig, louv):
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram as scipy_dendro
    from rn import viz

    viz.set_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2),
                             gridspec_kw={"width_ratios": [1.6, 1]})

    scipy_dendro(Z, ax=axes[0], no_labels=True,
                 link_color_func=lambda _: "#777777", above_threshold_color="#777777")
    if sig:
        axes[0].axhline(sig[-1]["cut"], color="#C44530", ls="--", lw=1.2,
                        alpha=0.8, label=f"significant gap cut ({sig[-1]['cut']:.2f})")
        axes[0].legend()
    axes[0].set_title("Single-linkage dendrogram on W1")
    axes[0].set_ylabel("W1 merge height")

    sweep = louv["k_stability_sweep"]
    ks = sorted(int(k) for k in sweep)
    q = [(sweep[str(k)] if str(k) in sweep else sweep[k])["modularity"]
         for k in ks]
    axes[1].plot(ks, q, color=viz.magma(0.42), lw=1.6, alpha=0.85, zorder=1)
    axes[1].scatter(ks, q, s=22, color=[viz.magma(0.42)] * len(ks),
                    edgecolors="white", lw=0.6, zorder=2)
    axes[1].axvline(louv["k"], color="#C44530", ls="--", lw=1.2, alpha=0.8)
    axes[1].annotate(f"k = {louv['k']}", xy=(louv["k"], max(q)),
                     xytext=(4, 0), textcoords="offset points",
                     color="#C44530", va="top")
    axes[1].set(xlabel="mutual-kNN k", ylabel="modularity")
    axes[1].set_title("k stability sweep")
    viz.savefig(fig, "06_dendrogram.png")


if __name__ == "__main__":
    main()
