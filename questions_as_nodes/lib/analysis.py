"""Steps 05-08: item statistics, centrality, communities, bootstrap stability."""
import numpy as np
import networkx as nx
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from lib import config
from lib.correlation import estimate_correlation
from lib.graph import benjamini_hochberg, correlation_pvalues


# --- Step 05: item statistics -------------------------------------------------

def item_statistics(X: pd.DataFrame, items: pd.DataFrame,
                    consensus_share: float = config.CONSENSUS_SHARE,
                    split_min_share: float = config.SPLIT_MIN_SHARE) -> pd.DataFrame:
    """Mean, spread and agreement shares per statement, with a consensus / split / mixed label."""
    agree = (X >= 4).sum() / X.notna().sum()
    disagree = (X <= 2).sum() / X.notna().sum()

    label = np.where((agree >= consensus_share) | (disagree >= consensus_share), "consensus",
             np.where((agree >= split_min_share) & (disagree >= split_min_share), "split", "mixed"))

    stats = pd.DataFrame({
        "code": X.columns,
        "n_answers": X.notna().sum().to_numpy(),
        "mean": X.mean().to_numpy(),
        "sd": X.std().to_numpy(),
        "share_agree": agree.to_numpy(),
        "share_neutral": ((X == 3).sum() / X.notna().sum()).to_numpy(),
        "share_disagree": disagree.to_numpy(),
        "label": label,
    })
    return items[["code", "theme", "text"]].merge(stats, on="code")


def answer_shares(X: pd.DataFrame) -> pd.DataFrame:
    """Share of each answer 1-5 per item (rows = items, columns = 1..5)."""
    counts = X.apply(lambda c: c.value_counts().reindex([1, 2, 3, 4, 5], fill_value=0))
    return (counts / counts.sum()).T


# --- Step 06: centrality ------------------------------------------------------

def centrality_table(G: nx.Graph) -> pd.DataFrame:
    """Degree, strength (sum |r|), betweenness, and the share of strength going to other themes."""
    strength = {n: sum(d["abs_weight"] for _, _, d in G.edges(n, data=True)) for n in G}
    cross = {n: sum(d["abs_weight"] for _, v, d in G.edges(n, data=True)
                    if G.nodes[v]["theme"] != G.nodes[n]["theme"]) for n in G}
    betweenness = nx.betweenness_centrality(G, weight="distance", normalized=True)

    table = pd.DataFrame({
        "code": list(G.nodes),
        "theme": [G.nodes[n]["theme"] for n in G],
        "degree": [G.degree(n) for n in G],
        "strength": [strength[n] for n in G],
        "betweenness": [betweenness[n] for n in G],
        "cross_theme_share": [cross[n] / strength[n] if strength[n] else 0.0 for n in G],
    })
    return table.sort_values("strength", ascending=False).reset_index(drop=True)


# --- Step 07: communities -----------------------------------------------------

def positive_graph(G: nx.Graph) -> nx.Graph:
    """Modularity needs non-negative weights, so drop negative edges for clustering only."""
    H = G.copy()
    H.remove_edges_from([(u, v) for u, v, d in G.edges(data=True) if d["weight"] <= 0])
    return H


def best_louvain(H: nx.Graph, seeds: int = config.LOUVAIN_SEEDS,
                 resolution: float = 1.0) -> tuple[dict, float]:
    """Louvain is randomised: run it `seeds` times and keep the partition with the highest modularity."""
    best = None
    for seed in range(seeds):
        parts = nx.community.louvain_communities(H, weight="abs_weight",
                                                 resolution=resolution, seed=seed)
        q = nx.community.modularity(H, parts, weight="abs_weight", resolution=resolution)
        if best is None or q > best[0]:
            best = (q, parts)
    q, parts = best
    return {n: i for i, part in enumerate(parts) for n in part}, q


def partition_modularity(H: nx.Graph, labels: dict) -> float:
    groups = {}
    for node, lab in labels.items():
        groups.setdefault(lab, set()).add(node)
    return nx.community.modularity(H, list(groups.values()), weight="abs_weight")


def theme_modularity_test(H: nx.Graph, themes: dict, n_permutations: int = config.PERMUTATIONS,
                          seed: int = config.SEED) -> dict:
    """Is the theme split more modular than a random split of the same group sizes?"""
    observed = partition_modularity(H, themes)
    nodes = list(themes)
    labels = np.array([themes[n] for n in nodes])

    rng = np.random.default_rng(seed)
    at_least = 0
    for _ in range(n_permutations):
        shuffled = dict(zip(nodes, rng.permutation(labels)))
        if partition_modularity(H, shuffled) >= observed:
            at_least += 1

    return {
        "theme_modularity": observed,
        "n_permutations": n_permutations,
        "p_value": (1 + at_least) / (1 + n_permutations),
    }


def compare_partitions(a: dict, b: dict) -> dict:
    """NMI and ARI between two labellings of the same nodes (1 = identical, 0 = unrelated/chance)."""
    nodes = sorted(a)
    x, y = [a[n] for n in nodes], [b[n] for n in nodes]
    return {"nmi": float(normalized_mutual_info_score(x, y)),
            "ari": float(adjusted_rand_score(x, y))}


def theme_block_matrix(R: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    """Mean |r| within and between themes, over all item pairs (not only the kept edges)."""
    theme = items.set_index("code")["theme"]
    names = sorted(theme.unique())

    values = R.abs().to_numpy()
    np.fill_diagonal(values, np.nan)  # an item's correlation with itself is not a pair
    A = pd.DataFrame(values, index=R.index, columns=R.columns)

    block = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            cells = A.loc[theme[theme == a].index, theme[theme == b].index].to_numpy()
            block.loc[a, b] = np.nanmean(cells)
    return block


# --- Step 08: bootstrap stability ---------------------------------------------

def _keep_mask(res, rule: str, q: float, threshold: float) -> np.ndarray:
    iu = np.triu_indices(len(res.R), 1)
    if rule == "fdr":
        return benjamini_hochberg(correlation_pvalues(res.R, res.n_pairwise)[iu], q)
    return np.abs(res.R.to_numpy()[iu]) >= threshold


def bootstrap_stability(X: pd.DataFrame, observed_strength: pd.Series,
                        rule: str = config.EDGE_RULE,
                        q: float = config.FDR_Q,
                        threshold: float = config.R_THRESHOLD,
                        method: str = config.CORR_METHOD,
                        B: int = config.BOOTSTRAP_B,
                        seed: int = config.SEED) -> dict:
    """Resample respondents with replacement, rebuild the graph, and record what stays.

    Returns per-pair edge inclusion rates, per-node strength distributions, and how well
    each resample's strength ranking matches the observed one.
    """
    codes = list(X.columns)
    p = len(codes)
    iu = np.triu_indices(p, 1)
    rng = np.random.default_rng(seed)

    inclusion = np.zeros(len(iu[0]))
    strengths = np.zeros((B, p))
    rank_corr = np.zeros(B)
    observed_rank = observed_strength.reindex(codes).rank().to_numpy()
    n_failed = 0

    for b in range(B):
        rows = rng.integers(0, len(X), len(X))
        res = estimate_correlation(X.iloc[rows], method=method)
        keep = _keep_mask(res, rule, q, threshold)
        inclusion += keep

        W = np.zeros((p, p))
        W[iu[0][keep], iu[1][keep]] = np.abs(res.R.to_numpy()[iu][keep])
        W = W + W.T
        strengths[b] = W.sum(axis=1)

        if strengths[b].std() == 0:
            n_failed += 1
            rank_corr[b] = np.nan
        else:
            rank_corr[b] = np.corrcoef(pd.Series(strengths[b]).rank(), observed_rank)[0, 1]

    pairs = pd.DataFrame({
        "source": [codes[j] for j in iu[0]],
        "target": [codes[k] for k in iu[1]],
        "inclusion_rate": inclusion / B,
    })
    node_stats = pd.DataFrame({
        "code": codes,
        "strength_observed": observed_strength.reindex(codes).to_numpy(),
        "strength_mean": strengths.mean(axis=0),
        "strength_lo": np.percentile(strengths, 2.5, axis=0),
        "strength_hi": np.percentile(strengths, 97.5, axis=0),
    })
    return {"pairs": pairs, "nodes": node_stats, "rank_corr": rank_corr,
            "B": B, "n_failed": n_failed}
