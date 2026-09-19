"""Graph builders: MST + mutual-kNN from a W1 matrix, and the Louvain
comparison utilities shared by steps 06, 08, and 10."""
import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from rn import config


def similarity(w1: float) -> float:
    return 1.0 / (1.0 + w1)


def mutual_knn_edges(W1: np.ndarray, k: int) -> set[tuple[int, int]]:
    """Positional index pairs (a < b) that are mutual k-nearest neighbours."""
    n = W1.shape[0]
    order = np.argsort(W1, axis=1)
    sets = [set(order[a, 1:k + 1]) for a in range(n)]
    edges = set()
    for a in range(n):
        for b in sets[a]:
            if a < b < n and a in sets[b]:
                edges.add((a, b))
    return edges


def build_graph(W1: np.ndarray, ids, k: int = None) -> nx.Graph:
    """Undirected weighted graph on response ids: MST union mutual-kNN."""
    k = k or config.K_MUTUAL
    ids = list(ids)
    G = nx.Graph()
    G.add_nodes_from(int(i) for i in ids)

    mst = minimum_spanning_tree(csr_matrix(W1)).tocoo()
    mst_pairs = {(min(a, b), max(a, b)) for a, b in zip(mst.row, mst.col)}
    pairs = mst_pairs | mutual_knn_edges(W1, k)
    for a, b in sorted(pairs):
        w = float(W1[a, b])
        G.add_edge(int(ids[a]), int(ids[b]), w1=w, weight=similarity(w))
    return G


def louvain_partition(G: nx.Graph, seed: int = None):
    return list(nx.community.louvain_communities(
        G, weight="weight", seed=config.SEED if seed is None else seed
    ))


def modularity(G: nx.Graph, communities) -> float:
    return float(nx.community.modularity(G, communities, weight="weight"))
