"""Network drawing shared by the main pipeline and the optional glasso pipeline."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from lib import config


def network_layout(G: nx.Graph, per_row: int = 15) -> dict:
    """Largest component: spring layout in [-1, 1]^2.
    Smaller components: a row of small spring layouts underneath.
    Isolated nodes: rows below that.
    """
    components = sorted((c for c in nx.connected_components(G) if len(c) > 1), key=len, reverse=True)
    isolates = sorted(n for n in G if G.degree(n) == 0)
    pos, y = {}, -1.3

    if components:
        main = G.subgraph(components[0])
        pos.update(nx.spring_layout(main, weight="abs_weight", seed=config.SEED, iterations=500,
                                    k=2.0 / np.sqrt(len(main))))

    small = components[1:]
    if small:
        slot = 2.0 / max(len(small), 1)
        for i, comp in enumerate(small):
            sub = nx.spring_layout(G.subgraph(comp), seed=config.SEED, scale=min(0.12, slot / 3))
            cx = -1.0 + slot * (i + 0.5)
            pos.update({n: xy + np.array([cx, y]) for n, xy in sub.items()})
        y -= 0.35

    for i, n in enumerate(isolates):
        row, col = divmod(i, per_row)
        pos[n] = np.array([-1.0 + 2.0 * col / (per_row - 1), y - 0.18 * row])
    return pos


def draw_network(G: nx.Graph, title: str, out: Path, edge_label: str = "correlation") -> None:
    """Spring layout on |weight|; nodes coloured by theme; blue/red = positive/negative edges.

    Isolated nodes are laid out separately in rows underneath, so they don't squash the
    connected part of the graph into the centre.
    """
    pos = network_layout(G)

    edges = sorted(G.edges(data=True), key=lambda e: e[2]["abs_weight"])  # strongest drawn last
    max_w = max((d["abs_weight"] for *_, d in edges), default=1.0)

    fig, ax = plt.subplots(figsize=(12, 10))
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edgelist=[(u, v) for u, v, _ in edges],
        width=[0.4 + 5.0 * d["abs_weight"] / max_w for *_, d in edges],
        edge_color=[config.POSITIVE_EDGE_COLOR if d["weight"] > 0 else config.NEGATIVE_EDGE_COLOR
                    for *_, d in edges],
        alpha=0.6,
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_size=420, linewidths=0.8, edgecolors="white",
        node_color=[config.THEME_COLORS[G.nodes[n]["theme"]] for n in G.nodes],
    )
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_color="white", font_weight="bold")

    handles = [Patch(color=c, label=f"{k}  {config.THEMES[k]}") for k, c in config.THEME_COLORS.items()]
    handles += [Line2D([], [], color=config.POSITIVE_EDGE_COLOR, lw=2, label=f"positive {edge_label}"),
                Line2D([], [], color=config.NEGATIVE_EDGE_COLOR, lw=2, label=f"negative {edge_label}")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=9)
    isolates = [n for n in G if G.degree(n) == 0]
    if isolates:
        top = max(pos[n][1] for n in isolates)
        ax.text(-1.05, top + 0.1, f"No edges ({len(isolates)} statements)",
                fontsize=9, color="#555555", ha="left")
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
