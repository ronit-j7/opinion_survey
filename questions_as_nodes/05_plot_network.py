"""Step 05 - first look at the network (sanity check, not the final report figure).

In:  outputs/04_graph.graphml, outputs/03_selection.json
Out: figures/05_network.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from bn import config
from bn.files import read_json


def main() -> None:
    config.ensure_dirs()
    G = nx.read_graphml(config.OUTPUTS / "04_graph.graphml")
    selection = read_json(config.OUTPUTS / "03_selection.json")

    pos = nx.spring_layout(G, weight="abs_weight", seed=config.SEED, iterations=300)

    edges = sorted(G.edges(data=True), key=lambda e: e[2]["abs_weight"])  # strongest drawn last
    max_w = max((d["abs_weight"] for *_, d in edges), default=1.0)

    fig, ax = plt.subplots(figsize=(11, 9))
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edgelist=[(u, v) for u, v, _ in edges],
        width=[0.4 + 5.0 * d["abs_weight"] / max_w for *_, d in edges],
        edge_color=[config.POSITIVE_EDGE_COLOR if d["weight"] > 0 else config.NEGATIVE_EDGE_COLOR
                    for *_, d in edges],
        alpha=0.75,
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_size=520, linewidths=0.8, edgecolors="white",
        node_color=[config.THEME_COLORS[G.nodes[n]["theme"]] for n in G.nodes],
    )
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_color="white", font_weight="bold")

    handles = [Patch(color=c, label=f"{k}  {config.THEMES[k]}") for k, c in config.THEME_COLORS.items()]
    handles += [Line2D([], [], color=config.POSITIVE_EDGE_COLOR, lw=2, label="positive partial corr."),
                Line2D([], [], color=config.NEGATIVE_EDGE_COLOR, lw=2, label="negative partial corr.")]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9)
    ax.set_title(f"Belief network: {G.number_of_nodes()} statements, {G.number_of_edges()} edges "
                 f"(glasso, EBIC γ={selection['gamma']}, λ={selection['lambda']:.4f})", fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(config.FIGURES / "05_network.png", dpi=200)
    plt.close(fig)
    print(f"Saved {config.FIGURES / '05_network.png'}")


if __name__ == "__main__":
    main()
