"""Network drawing shared by the main pipeline and the optional glasso pipeline."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
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


def draw_network(G: nx.Graph, title: str, out: Path, edge_label: str = "correlation",
                 node_colors: dict | None = None, legend_entries: list | None = None) -> None:
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
        node_color=[(node_colors or {}).get(n) or config.THEME_COLORS[G.nodes[n]["theme"]]
                    for n in G.nodes],
    )
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_color="white", font_weight="bold")

    entries = legend_entries or [(f"{k}  {config.THEMES[k]}", c) for k, c in config.THEME_COLORS.items()]
    handles = [Patch(color=c, label=label) for label, c in entries]
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


def draw_answer_shares(shares: pd.DataFrame, items: pd.DataFrame, title: str, out: Path) -> None:
    """Stacked bars: the share of each answer 1-5 per statement, grouped by theme."""
    order = items.sort_values(["theme", "code"])["code"].tolist()[::-1]  # barh draws bottom-up
    S = shares.loc[order]
    labels = ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"]

    fig, ax = plt.subplots(figsize=(10, 13))
    left = np.zeros(len(S))
    for value, colour, label in zip([1, 2, 3, 4, 5], config.LIKERT_COLORS, labels):
        ax.barh(S.index, S[value], left=left, color=colour, label=label, height=0.78)
        left += S[value].to_numpy()

    theme_of = items.set_index("code")["theme"]
    for tick in ax.get_yticklabels():
        tick.set_color(config.THEME_COLORS[theme_of[tick.get_text()]])
        tick.set_fontsize(8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of answers")
    ax.set_title(title, fontsize=11)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.04), frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def draw_ranking(table: pd.DataFrame, value: str, title: str, out: Path, top: int = 15) -> None:
    """Horizontal bar chart of the top items by `value`, coloured by theme."""
    top_items = table.nlargest(top, value).iloc[::-1]

    fig, ax = plt.subplots(figsize=(8, 0.42 * len(top_items) + 1.6))
    ax.barh(top_items["code"], top_items[value],
            color=[config.THEME_COLORS[t] for t in top_items["theme"]], height=0.72)
    ax.set_xlabel(value.replace("_", " "))
    ax.set_title(title, fontsize=11)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def draw_block_matrix(block: pd.DataFrame, title: str, out: Path) -> None:
    """Heatmap of mean |r| within and between themes."""
    fig, ax = plt.subplots(figsize=(5.6, 5))
    im = ax.imshow(block.to_numpy(), cmap="Blues", vmin=0)
    names = [f"{t}\n{config.THEMES[t]}" for t in block.index]
    ax.set_xticks(range(len(block)), names, fontsize=8)
    ax.set_yticks(range(len(block)), names, fontsize=8)
    for i in range(len(block)):
        for j in range(len(block)):
            v = block.to_numpy()[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9,
                    color="white" if v > block.to_numpy().max() * 0.6 else "#222222")
    fig.colorbar(im, ax=ax, shrink=0.8, label="mean |r|")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def draw_strength_intervals(nodes: pd.DataFrame, title: str, out: Path, top: int = 20) -> None:
    """Observed strength with its bootstrap 95% interval, for the strongest items."""
    d = nodes.nlargest(top, "strength_observed").iloc[::-1]
    y = np.arange(len(d))

    fig, ax = plt.subplots(figsize=(8, 0.4 * len(d) + 1.6))
    ax.hlines(y, d["strength_lo"], d["strength_hi"], color="#9AA4B2", lw=2)
    ax.plot(d["strength_observed"], y, "o", color="#2F6DB5", ms=5, label="observed")
    ax.set_yticks(y, d["code"], fontsize=8)
    ax.set_xlabel("strength (sum of |r| over a statement's edges)")
    ax.set_title(title, fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def draw_inclusion(pairs: pd.DataFrame, title: str, out: Path) -> None:
    """How often each observed edge survives a resample, sorted."""
    rates = pairs["inclusion_rate"].sort_values(ascending=False).to_numpy()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(np.arange(1, len(rates) + 1), rates, color="#2F6DB5", lw=1.6)
    for level, colour in [(0.9, "#55A868"), (0.5, "#C44E52")]:
        ax.axhline(level, color=colour, ls="--", lw=1)
        ax.text(len(rates), level, f" {level:.0%}", va="center", fontsize=8, color=colour)
    ax.set_xlabel("edges, strongest inclusion first")
    ax.set_ylabel("share of resamples containing the edge")
    ax.set_ylim(0, 1.02)
    ax.set_title(title, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
