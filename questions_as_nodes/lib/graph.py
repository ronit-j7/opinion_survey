"""Step 03: turn a correlation (or partial-correlation) matrix into a networkx graph."""
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

from lib import config


def _add_item_nodes(G: nx.Graph, items: pd.DataFrame) -> None:
    for row in items.itertuples(index=False):
        G.add_node(row.code, theme=row.theme, theme_name=row.theme_name,
                   text=row.text, n_responses=int(row.n_responses))


def graph_from_weights(W: pd.DataFrame, items: pd.DataFrame,
                       zero_tol: float = config.ZERO_TOL) -> nx.Graph:
    """Undirected, weighted, signed graph with an edge wherever |W_jk| > zero_tol.

    Edge attributes: weight (signed), abs_weight, sign (+1/-1),
    distance = 1/|weight| for shortest-path measures later.
    """
    codes = list(W.columns)
    if list(items["code"]) != codes:
        raise ValueError("Item order in weight matrix and items table differ.")

    G = nx.Graph()
    _add_item_nodes(G, items)
    values = W.to_numpy()
    for j, k in zip(*np.triu_indices(len(codes), 1)):
        w = float(values[j, k])
        if abs(w) > zero_tol:
            G.add_edge(codes[j], codes[k], weight=w, abs_weight=abs(w),
                       sign=int(np.sign(w)), distance=1.0 / abs(w))
    return G


def correlation_pvalues(R: pd.DataFrame, n_pairwise: pd.DataFrame) -> np.ndarray:
    """Two-sided p-values for H0: rho_jk = 0, t = r sqrt((n-2)/(1-r^2)). Returns p x p, NaN diagonal."""
    r = np.clip(R.to_numpy(), -0.999999, 0.999999)
    n = n_pairwise.to_numpy().astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = r * np.sqrt((n - 2.0) / (1.0 - r ** 2))
        p = 2.0 * stats.t.sf(np.abs(t), df=n - 2.0)
    p[n < 3] = 1.0
    np.fill_diagonal(p, np.nan)
    return p


def benjamini_hochberg(pvals: np.ndarray, q: float) -> np.ndarray:
    """Boolean mask of rejected hypotheses at FDR level q."""
    m = len(pvals)
    order = np.argsort(pvals)
    passed = pvals[order] <= q * np.arange(1, m + 1) / m
    reject = np.zeros(m, dtype=bool)
    if passed.any():
        reject[order[: np.nonzero(passed)[0].max() + 1]] = True
    return reject


def correlation_graph(R: pd.DataFrame, n_pairwise: pd.DataFrame, items: pd.DataFrame,
                      rule: str = config.EDGE_RULE,
                      q: float = config.FDR_Q,
                      threshold: float = config.R_THRESHOLD) -> nx.Graph:
    """Correlation graph: nodes = items, edge weight = r_jk.

    rule="fdr":       keep r_jk whose t-test is significant after Benjamini-Hochberg at level q.
    rule="threshold": keep |r_jk| >= threshold.
    """
    p = len(R)
    iu = np.triu_indices(p, 1)
    r = R.to_numpy()[iu]

    if rule == "fdr":
        keep = benjamini_hochberg(correlation_pvalues(R, n_pairwise)[iu], q)
    elif rule == "threshold":
        keep = np.abs(r) >= threshold
    else:
        raise ValueError(f"Unknown edge rule: {rule}")

    W = np.zeros((p, p))
    W[iu[0][keep], iu[1][keep]] = r[keep]
    W = W + W.T
    return graph_from_weights(pd.DataFrame(W, index=R.index, columns=R.columns), items)


def edges_table(G: nx.Graph) -> pd.DataFrame:
    rows = [{"source": u, "target": v,
             "weight": d["weight"], "abs_weight": d["abs_weight"], "sign": d["sign"],
             "theme_source": G.nodes[u]["theme"], "theme_target": G.nodes[v]["theme"],
             "cross_theme": G.nodes[u]["theme"] != G.nodes[v]["theme"]}
            for u, v, d in G.edges(data=True)]
    cols = ["source", "target", "weight", "abs_weight", "sign",
            "theme_source", "theme_target", "cross_theme"]
    return (pd.DataFrame(rows, columns=cols)
            .sort_values("abs_weight", ascending=False)
            .reset_index(drop=True))


def summarize(G: nx.Graph, n_strongest: int = 10) -> dict:
    edges = edges_table(G)
    p = G.number_of_nodes()
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    pair_counts = (edges.apply(lambda r: "-".join(sorted((r.theme_source, r.theme_target))), axis=1)
                   .value_counts().sort_index().to_dict()) if len(edges) else {}

    return {
        "n_nodes": p,
        "n_edges": len(edges),
        "density": len(edges) / (p * (p - 1) / 2),
        "n_positive": int((edges["sign"] > 0).sum()),
        "n_negative": int((edges["sign"] < 0).sum()),
        "share_negative": float((edges["sign"] < 0).mean()) if len(edges) else 0.0,
        "mean_abs_weight": float(edges["abs_weight"].mean()) if len(edges) else 0.0,
        "within_theme_edges": int((~edges["cross_theme"]).sum()),
        "cross_theme_edges": int(edges["cross_theme"].sum()),
        "edges_by_theme_pair": pair_counts,
        "n_components": len(components),
        "largest_component_size": len(components[0]) if components else 0,
        "isolated_nodes": sorted(nx.isolates(G)),
        "strongest_edges": edges.head(n_strongest)[["source", "target", "weight"]]
                                .round(4).values.tolist(),
    }
