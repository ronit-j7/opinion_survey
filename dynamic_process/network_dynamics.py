"""
network_dynamics.py
--------------------
0. Cleans the RAW survey CSV (drops incomplete respondents, encodes
   Likert answers to -2..+2, imputes "No Comments" as Neutral).

1. Builds a SINGLE people-people network from the cleaned survey data.
   Nodes: the 85 respondents.
   Edge weight: cosine similarity between two respondents' FULL
   60-answer vectors (i.e. how closely two people agree across every
   question overall). Edges are chosen via k-nearest-neighbors: each
   person is connected to their k most similar peers, regardless of
   the absolute similarity value. This guarantees no isolated nodes
   by construction (every node has degree >= k) while keeping each
   person's neighborhood small and genuinely "who's most like me"
   rather than "who clears an arbitrary similarity bar."
   This one network is built ONCE and reused for every simulation below.

2. Runs DeGroot consensus 60 separate times on that SAME fixed
   network -- once per survey item. Each run starts every person at
   their own actual answer to that one question, and lets opinions
   evolve based on how similar people are OVERALL (not just on that
   question). This is the standard DeGroot use case: real people,
   weighted by how much they resemble each other, converging (or not)
   as they interact.

3. Averages each person's 60 converged (per-item) values into a
   single "network-adjusted opinion", comparable to their original
   raw average -- showing how much the network pulled each person's
   overall stance toward their similar-minded peers.

4. Saves all plots AND videos into a "figures/" directory.

Usage:
    python network_dynamics.py
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import TwoSlopeNorm

FIGURES_DIR = "figures"
FPS = 2


# ---------------------------------------------------------------------
# 0. Clean the raw survey data
# ---------------------------------------------------------------------

LIKERT_MAP = {
    "Strongly Disagree": -2,
    "Disagree": -1,
    "Neutral": 0,
    "Agree": 1,
    "Strongly Agree": 2,
}

NO_COMMENTS_VALUE = "No Comments"
ID_COL = "id. Response ID"


def load_and_clean(raw_path: str, id_col: str = ID_COL) -> pd.DataFrame:
    """
    Loads the raw survey CSV and returns a cleaned, fully numeric
    DataFrame ready for build_people_network().

    Steps:
      1. Drop any respondent who left at least one question blank
         (partial/non-completers).
      2. Treat "No Comments" as missing and impute as Neutral (0).
      3. Map all Likert text responses to numeric values in [-2, 2].
    """
    df = pd.read_csv(raw_path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    item_cols = [c for c in df.columns if c != id_col]

    before = len(df)
    df = df.dropna(subset=item_cols, how="any").reset_index(drop=True)
    print(f"Dropped {before - len(df)} incomplete respondents "
          f"({before} -> {len(df)} remaining).")

    n_no_comments = (df[item_cols] == NO_COMMENTS_VALUE).sum().sum()
    print(f"Imputing {n_no_comments} 'No Comments' cells as Neutral (0).")
    df[item_cols] = df[item_cols].replace(NO_COMMENTS_VALUE, "Neutral")

    unmapped = set(pd.unique(df[item_cols].values.ravel())) - set(LIKERT_MAP.keys())
    if unmapped:
        raise ValueError(f"Found unexpected response values, not in Likert map: {unmapped}")

    df[item_cols] = df[item_cols].apply(lambda col: col.map(LIKERT_MAP))
    return df


# ---------------------------------------------------------------------
# 1. Build the SINGLE people-people network
# ---------------------------------------------------------------------

def cosine_similarity_matrix(M: np.ndarray) -> np.ndarray:
    """
    Generic row-wise cosine similarity: each ROW of M is treated as a
    vector. Returns an (n_rows, n_rows) similarity matrix.
    """
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    normalized = M / norms
    return normalized @ normalized.T


def _similarity_matrix(X: np.ndarray, similarity: str) -> np.ndarray:
    if similarity == "cosine":
        return cosine_similarity_matrix(X)
    elif similarity == "pearson":
        return np.corrcoef(X)
    else:
        raise ValueError("similarity must be 'cosine' or 'pearson'")


def build_people_network(
    df: pd.DataFrame,
    id_col: str = ID_COL,
    similarity: str = "cosine",       # "cosine" or "pearson"
    k: int = 8,                       # each node connects to its k most similar peers
) -> nx.Graph:
    """
    Builds ONE weighted people-people network, used for every
    per-question DeGroot run.

    Nodes: the respondents (their id_col value).
    Edge weight: the similarity value itself (can be negative).

    Edges are chosen via k-nearest-neighbors: each respondent is
    connected to their k most similar other respondents (by
    |similarity|), regardless of the absolute similarity value.
    Because this is applied to every node and the resulting edges are
    unioned into an undirected graph, every node ends up with
    degree >= k -- no isolated nodes by construction, without having
    to hand-tune a similarity cutoff. `k` is the number of nearest
    peers per person, and is fully tunable.

    `similarity` and `k` are both tunable:
      - similarity="cosine"  -> cosine_similarity_matrix() on raw rows
      - similarity="pearson" -> standard correlation matrix (np.corrcoef)
      - k                    -> raise for a denser graph with larger
                                 peer groups, lower for a sparser graph
                                 with tighter peer groups
    """
    item_cols = [c for c in df.columns if c != id_col]
    ids = df[id_col].tolist()
    X = df[item_cols].to_numpy(dtype=float)
    S = _similarity_matrix(X, similarity)

    G = nx.Graph()
    G.add_nodes_from(ids)

    n = len(ids)
    if k < 1:
        raise ValueError("k must be >= 1")
    if k >= n:
        raise ValueError(f"k ({k}) must be less than the number of respondents ({n})")

    for i in range(n):
        # rank every OTHER respondent by similarity strength (sign-agnostic),
        # take the top k, and add an (undirected) edge to each. S is
        # symmetric, so re-adding an edge from the other endpoint's own
        # top-k pass is a no-op with the same weight, not a conflict.
        candidates = [j for j in range(n) if j != i]
        candidates.sort(key=lambda j: abs(S[i, j]), reverse=True)
        for j in candidates[:k]:
            G.add_edge(ids[i], ids[j], weight=S[i, j])

    return G


# ---------------------------------------------------------------------
# 2. DeGroot consensus, run once per question on the SAME fixed network
# ---------------------------------------------------------------------

def simulate_dynamics(G: nx.Graph, initial_opinions: dict, steps: int = 50):
    """
    Runs the DeGroot consensus process on G, starting from
    `initial_opinions` (a dict: node -> starting value).
    Edges are always static: the graph structure never changes during
    the simulation, only node opinions update each step.

    Each node's opinion is updated to the weighted average of its own
    and its neighbors' opinions each step, using the actual edge
    weights (similarity values) as the influence weights.
    Formally: x_i(t+1) = sum_j w_ij * x_j(t), with weights per row
    normalized to sum to 1 (including a self-weight for node i).

    Returns
    -------
    history : list of dicts, one per time step.
    G : the (unchanged) graph, returned for API symmetry.
    """
    nodes = list(G.nodes())
    opinions = {n: initial_opinions[n] for n in nodes}

    # build row-normalized influence weights once (edges are static,
    # and this SAME network/weighting is reused for every item)
    W = {}
    for n in nodes:
        neighbors = list(G.neighbors(n))
        raw_weights = {n: 1.0}  # self-influence
        for nb in neighbors:
            raw_weights[nb] = abs(G[n][nb]["weight"])
        total = sum(raw_weights.values())
        W[n] = {k: v / total for k, v in raw_weights.items()}

    history = [dict(opinions)]
    for _ in range(steps):
        new_opinions = {}
        for n in nodes:
            new_opinions[n] = sum(w * opinions[m] for m, w in W[n].items())
        opinions = new_opinions
        history.append(dict(opinions))

    return history, G


def run_all_items(df: pd.DataFrame, G: nx.Graph, id_col: str = ID_COL, steps: int = 50):
    """
    Runs DeGroot 60 separate times on the SAME fixed people-network `G`
    -- once per survey item -- starting each person at their own raw
    answer to that item.

    Returns
    -------
    converged : DataFrame, index=respondent id, columns=item, values
                = that person's converged (final) opinion on that item.
    histories : dict item -> full history list, for ALL 60 items
                (cheap to keep in memory: 60 items x steps x n_people).
    """
    item_cols = [c for c in df.columns if c != id_col]
    ids = df[id_col].tolist()

    converged = {}
    histories = {}
    for item in item_cols:
        initial = dict(zip(ids, df[item]))
        history, _ = simulate_dynamics(G, initial, steps=steps)
        converged[item] = history[-1]
        histories[item] = history

    converged_df = pd.DataFrame(converged)  # index = respondent id, columns = items
    return converged_df, histories


# ---------------------------------------------------------------------
# 3. Plotting helpers
# ---------------------------------------------------------------------

def plot_example_trajectories(history, item_name, save_as=None):
    nodes = list(history[0].keys())
    steps = len(history)
    fig, ax = plt.subplots(figsize=(8, 5))
    for n in nodes:
        y = [history[t][n] for t in range(steps)]
        ax.plot(range(steps), y, alpha=0.4, linewidth=1)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Opinion (-2 to +2 scale)")
    ax.set_title(f"DeGroot consensus on: {item_name}")
    plt.tight_layout()
    if save_as:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        path = os.path.join(FIGURES_DIR, save_as)
        fig.savefig(path, dpi=150)
        print(f"Saved {path}")
    return fig


def plot_convergence_curve(history, item_name, save_as=None):
    nodes = list(history[0].keys())
    steps = len(history)
    disagreement = [np.var([history[t][n] for n in nodes]) for t in range(steps)]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(steps), disagreement, color="crimson")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Opinion variance across respondents")
    ax.set_title(f"Convergence on: {item_name}")
    plt.tight_layout()
    if save_as:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        path = os.path.join(FIGURES_DIR, save_as)
        fig.savefig(path, dpi=150)
        print(f"Saved {path}")
    return fig


def get_category(item_name: str) -> str:
    """First letter of the item code, e.g. 'T01. ...' -> 'T'."""
    return item_name[0]


def plot_category_trajectories_grid(histories: dict, item_cols: list, category: str,
                                     save_as=None):
    """
    One figure per category (T/E/S/V), laid out as a 5x3 grid (15
    items per category). Each subplot is that item's full DeGroot
    trajectory plot (all respondents' opinion lines over iterations).
    """
    cat_items = [c for c in item_cols if get_category(c) == category]
    if len(cat_items) > 15:
        raise ValueError(
            f"Category '{category}' has {len(cat_items)} items, but the grid "
            f"only has 15 slots (5x3) -- extra items would be silently dropped. "
            f"Enlarge the grid or split the category."
        )
    fig, axes = plt.subplots(5, 3, figsize=(15, 18))
    axes = axes.flatten()

    for ax, item in zip(axes, cat_items):
        history = histories[item]
        nodes = list(history[0].keys())
        steps = len(history)
        for n in nodes:
            y = [history[t][n] for t in range(steps)]
            ax.plot(range(steps), y, alpha=0.4, linewidth=0.8)
        code = item.split(".")[0]  # e.g. "T01"
        ax.set_title(code, fontsize=10)
        ax.set_xlabel("Iteration", fontsize=8)
        ax.set_ylabel("Opinion", fontsize=8)
        ax.tick_params(labelsize=7)

    # hide any unused subplots (shouldn't happen with exactly 15 items)
    for ax in axes[len(cat_items):]:
        ax.axis("off")

    fig.suptitle(f"DeGroot trajectories -- category {category}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if save_as:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        path = os.path.join(FIGURES_DIR, save_as)
        fig.savefig(path, dpi=150)
        print(f"Saved {path}")
    return fig


def plot_category_convergence_grid(histories: dict, item_cols: list, category: str,
                                    save_as=None):
    """
    One figure per category (T/E/S/V), laid out as a 5x3 grid (15
    items per category). Each subplot is that item's convergence
    curve (opinion variance across respondents, over iterations).
    """
    cat_items = [c for c in item_cols if get_category(c) == category]
    if len(cat_items) > 15:
        raise ValueError(
            f"Category '{category}' has {len(cat_items)} items, but the grid "
            f"only has 15 slots (5x3) -- extra items would be silently dropped. "
            f"Enlarge the grid or split the category."
        )
    fig, axes = plt.subplots(5, 3, figsize=(15, 18))
    axes = axes.flatten()

    for ax, item in zip(axes, cat_items):
        history = histories[item]
        nodes = list(history[0].keys())
        steps = len(history)
        disagreement = [np.var([history[t][n] for n in nodes]) for t in range(steps)]
        ax.plot(range(steps), disagreement, color="crimson", linewidth=1)
        code = item.split(".")[0]
        ax.set_title(code, fontsize=10)
        ax.set_xlabel("Iteration", fontsize=8)
        ax.set_ylabel("Variance", fontsize=8)
        ax.tick_params(labelsize=7)

    for ax in axes[len(cat_items):]:
        ax.axis("off")

    fig.suptitle(f"DeGroot convergence -- category {category}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if save_as:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        path = os.path.join(FIGURES_DIR, save_as)
        fig.savefig(path, dpi=150)
        print(f"Saved {path}")
    return fig


def save_item_summary_csv(df, converged_df, id_col=ID_COL, save_as="item_summary.csv"):
    """
    Saves a per-question CSV with one row per survey item:
        question, initial_average, final_average
    "initial_average" is the mean of everyone's raw (pre-DeGroot)
    answer to that item; "final_average" is the mean of everyone's
    converged (post-DeGroot) opinion on that item. Saved into
    FIGURES_DIR by default.
    """
    item_cols = [c for c in df.columns if c != id_col]
    initial_avg = df[item_cols].mean(axis=0)
    final_avg = converged_df[item_cols].mean(axis=0)

    summary = pd.DataFrame({
        "question": item_cols,
        "initial_average": initial_avg.values,
        "final_average": final_avg.values,
    })
    summary["change"] = summary["final_average"] - summary["initial_average"]

    path = save_as
    if save_as and not os.path.isabs(save_as) and os.sep not in save_as:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        path = os.path.join(FIGURES_DIR, save_as)
    summary.to_csv(path, index=False)
    print(f"Saved {path}")
    return summary


def plot_original_vs_adjusted(df, converged_df, id_col=ID_COL, save_as="original_vs_adjusted.png"):
    """
    Compares each person's ORIGINAL average opinion (mean of their raw
    60 answers) against their NETWORK-ADJUSTED average opinion (mean
    of their 60 DeGroot-converged values) -- showing how much the
    network pulled each person toward their similar-minded peers.
    """
    item_cols = [c for c in df.columns if c != id_col]
    original_avg = df.set_index(id_col)[item_cols].mean(axis=1)
    adjusted_avg = converged_df.mean(axis=1)
    adjusted_avg = adjusted_avg.reindex(original_avg.index)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(original_avg, adjusted_avg, alpha=0.6)
    lims = [min(original_avg.min(), adjusted_avg.min()), max(original_avg.max(), adjusted_avg.max())]
    ax.plot(lims, lims, "--", color="gray", label="no change")
    ax.set_xlabel("Original average opinion")
    ax.set_ylabel("Network-adjusted average opinion")
    ax.set_title("Effect of DeGroot consensus on each respondent's overall stance")
    ax.legend()
    plt.tight_layout()
    if save_as:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        path = os.path.join(FIGURES_DIR, save_as)
        fig.savefig(path, dpi=150)
        print(f"Saved {path}")
    return fig


# ---------------------------------------------------------------------
# 4. Video helpers
# ---------------------------------------------------------------------

def circular_layout_ordered(G):
    """
    Arranges nodes evenly around a circle (no clumping is possible --
    every node gets a fixed, evenly-spaced slot). Isolated nodes are
    grouped together on one arc; connected nodes are ordered by
    degree, reducing unnecessary edge crossings.
    """
    isolated = [n for n in G.nodes() if G.degree(n) == 0]
    connected = [n for n in G.nodes() if G.degree(n) > 0]
    connected.sort(key=lambda n: G.degree(n), reverse=True)
    ordered_nodes = connected + isolated

    n = len(ordered_nodes)
    pos = {}
    for i, node in enumerate(ordered_nodes):
        angle = 2 * np.pi * i / n
        pos[node] = (np.cos(angle), np.sin(angle))
    return pos


def make_degroot_video(G, item_name, initial_opinions, out_name=None, steps=15, subframes=6):
    """
    Animates ONE item's DeGroot run on the people-network. Color =
    each respondent's distance from their own final resting value for
    THIS item (fades to white as they converge). Interpolates between
    real iterations so fast convergence renders as a smooth fade.
    """
    history, _ = simulate_dynamics(G, initial_opinions, steps=steps)
    nodes = list(G.nodes())
    pos = circular_layout_ordered(G)

    final_vals = np.array([history[-1][n] for n in nodes])
    raw_devs = np.array([[history[t][n] - final_vals[i] for i, n in enumerate(nodes)]
                          for t in range(len(history))])

    interp_devs, interp_labels = [], []
    for t in range(len(raw_devs) - 1):
        max_delta = np.max(np.abs(raw_devs[t + 1] - raw_devs[t]))
        for s in range(subframes):
            frac = s / subframes
            interp_devs.append(raw_devs[t] * (1 - frac) + raw_devs[t + 1] * frac)
            interp_labels.append((t + frac, max_delta))
    interp_devs.append(raw_devs[-1])
    interp_labels.append((len(raw_devs) - 1, None))
    for _ in range(subframes * 2):
        interp_devs.append(raw_devs[-1])
        interp_labels.append((len(raw_devs) - 1, None))

    vmax = max(np.max(np.abs(raw_devs)), 1e-6)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = plt.cm.RdBu_r

    fig, ax = plt.subplots(figsize=(9, 9))

    def draw_frame(frame_idx):
        ax.clear()
        devs = interp_devs[frame_idx]
        colors = [cmap(norm(d)) for d in devs]
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4, width=0.8, edge_color="gray")
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors,
                                node_size=120, edgecolors="black", linewidths=0.3)
        real_t, max_delta = interp_labels[frame_idx]
        subtitle = f"max change this step: {max_delta:.4f}" if max_delta is not None else "converged / initial"
        ax.set_title(f"DeGroot on '{item_name}' -- iteration ~{real_t:.1f}/{steps}\n"
                     f"color = distance from each respondent's final value\n{subtitle}", fontsize=11)
        ax.axis("off")

    anim = animation.FuncAnimation(fig, draw_frame, frames=len(interp_devs), interval=1000 / FPS)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_name = out_name or f"degroot_{item_name[:20].replace(' ', '_')}.mp4"
    out_path = os.path.join(FIGURES_DIR, out_name)
    anim.save(out_path, writer="ffmpeg", fps=FPS * subframes, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def make_knn_sweep_video(df, id_col=ID_COL, out_name="knn_sweep.mp4",
                          k_values=None, similarity="cosine", final_k=8):
    """
    k-NN equivalent of make_threshold_sweep_video: sweeps k (number of
    nearest peers per person) on the PEOPLE network, showing edges
    appear as k grows -- visual justification for the chosen k.
    `final_k` should match build_people_network()'s k when
    method="knn".
    """
    item_cols = [c for c in df.columns if c != id_col]
    ids = df[id_col].tolist()
    X = df[item_cols].to_numpy(dtype=float)
    S = cosine_similarity_matrix(X) if similarity == "cosine" else np.corrcoef(X)
    n = len(ids)

    if k_values is None:
        max_k = min(final_k, n - 1)
        k_values = list(range(1, max_k + 1))

    def knn_graph(k):
        G = nx.Graph()
        G.add_nodes_from(ids)
        for i in range(n):
            candidates = [j for j in range(n) if j != i]
            candidates.sort(key=lambda j: abs(S[i, j]), reverse=True)
            for j in candidates[:k]:
                G.add_edge(ids[i], ids[j])
        return G

    G_final = knn_graph(final_k)
    pos = circular_layout_ordered(G_final)

    fig, ax = plt.subplots(figsize=(9, 9))

    def draw_frame(idx):
        k = k_values[idx]
        G = knn_graph(k)
        ax.clear()
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4, width=0.7, edge_color="gray")
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color="steelblue",
                                node_size=120, edgecolors="black", linewidths=0.3)
        density = nx.density(G)
        min_deg = min(d for _, d in G.degree())
        ax.set_title(f"k = {k}  ({G.number_of_edges()} edges, density={density:.2f}, "
                     f"min degree={min_deg})", fontsize=13)
        ax.axis("off")

    anim = animation.FuncAnimation(fig, draw_frame, frames=len(k_values), interval=1000 / FPS)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(FIGURES_DIR, out_name)
    anim.save(out_path, writer="ffmpeg", fps=FPS, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------
# Example run
# ---------------------------------------------------------------------

if __name__ == "__main__":
    df = load_and_clean("../Survey_Results_UC.csv")

    # --- tune these freely ---
    SIMILARITY_METRIC = "cosine"   # "cosine" or "pearson"
    K_NEIGHBORS = 4                # peers per person in the k-NN network
    # -------------------------

    G = build_people_network(df, similarity=SIMILARITY_METRIC, k=K_NEIGHBORS)
    print(f"People network (k-NN, k={K_NEIGHBORS}): "
          f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    isolated = sum(1 for n in G.nodes() if G.degree(n) == 0)
    print(f"Isolated nodes: {isolated} ({isolated / G.number_of_nodes():.0%})")

    # Run DeGroot once per item (all 60), on the SAME fixed network
    item_cols = [c for c in df.columns if c != ID_COL]
    converged_df, histories = run_all_items(df, G, steps=50)

    # Per-question summary CSV: question, initial_average, final_average
    save_item_summary_csv(df, converged_df)

    # Plots
    plot_original_vs_adjusted(df, converged_df)

    categories = sorted(set(get_category(c) for c in item_cols))  # e.g. ['E','S','T','V']
    for cat in categories:
        plot_category_trajectories_grid(histories, item_cols, cat,
                                         save_as=f"trajectories_{cat}.png")
        plot_category_convergence_grid(histories, item_cols, cat,
                                        save_as=f"convergence_{cat}.png")

    # Videos
    first_item = item_cols[0]
    initial_first = dict(zip(df[ID_COL], df[first_item]))
    make_degroot_video(G, first_item, initial_first, steps=15)
    make_knn_sweep_video(df, final_k=K_NEIGHBORS)