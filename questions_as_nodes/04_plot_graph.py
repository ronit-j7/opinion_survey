"""Step 04 - draw the correlation graph.

In:  outputs/03_graph.graphml, outputs/03_summary.json
Out: figures/04_network.png
"""
import networkx as nx

from lib import config
from lib.files import read_json
from lib.plot import draw_network


def main() -> None:
    config.ensure_dirs()
    G = nx.read_graphml(config.OUTPUTS / "03_graph.graphml")
    s = read_json(config.OUTPUTS / "03_summary.json")

    rule = (f"significant at FDR {s['fdr_q']}" if s["edge_rule"] == "fdr"
            else f"|r| ≥ {s['r_threshold']}")
    title = (f"{s['corr_method'].capitalize()} correlation graph: {G.number_of_nodes()} statements, "
             f"{G.number_of_edges()} edges ({rule})")
    out = config.FIGURES / "04_network.png"
    draw_network(G, title, out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
