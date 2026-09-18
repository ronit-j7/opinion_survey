"""Optional G2 - build and draw the partial-correlation (glasso) graph.

Run from the questions_as_nodes folder after G1:  python glasso_optional/g2_build_graph.py

In:  outputs/g1_partial_corr.csv, outputs/g1_selection.json, outputs/01_items.csv
Out: outputs/g2_graph.graphml, outputs/g2_edges.csv, outputs/g2_summary.json, figures/g2_network.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make lib/ importable

import networkx as nx

from lib import config
from lib.files import read_items, read_json, read_matrix, save_json
from lib.graph import edges_table, graph_from_weights, summarize
from lib.plot import draw_network


def main() -> None:
    config.ensure_dirs()
    items = read_items(config.OUTPUTS / "01_items.csv")
    omega = read_matrix(config.OUTPUTS / "g1_partial_corr.csv")
    selection = read_json(config.OUTPUTS / "g1_selection.json")

    G = graph_from_weights(omega, items)
    nx.write_graphml(G, config.OUTPUTS / "g2_graph.graphml")
    edges_table(G).to_csv(config.OUTPUTS / "g2_edges.csv", index=False, float_format="%.6g")
    s = summarize(G)
    save_json(s, config.OUTPUTS / "g2_summary.json")

    title = (f"Partial correlation graph: {G.number_of_nodes()} statements, {G.number_of_edges()} edges "
             f"(glasso, EBIC γ={selection['gamma']}, λ={selection['lambda']:.4f})")
    draw_network(G, title, config.FIGURES / "g2_network.png", edge_label="partial corr.")
    print(f"Edges: {s['n_edges']}, isolated nodes: {len(s['isolated_nodes'])}")
    print(f"Saved {config.FIGURES / 'g2_network.png'}")


if __name__ == "__main__":
    main()
