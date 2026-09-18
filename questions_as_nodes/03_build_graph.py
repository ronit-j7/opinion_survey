"""Step 03 - build the correlation graph.

Nodes = the 60 statements. Edge between j and k with weight r_jk, kept according to
EDGE_RULE in lib/config.py ("fdr": statistically significant, at most FDR_Q false edges;
"threshold": |r| >= R_THRESHOLD).

In:  outputs/02_R.csv, outputs/02_pairwise_n.csv, outputs/01_items.csv
Out: outputs/03_graph.graphml     the graph (open in Gephi / networkx)
     outputs/03_edges.csv         edge list, strongest first
     outputs/03_summary.json      edge rule, counts, density, signs, theme mix, isolated nodes
"""
import networkx as nx

from lib import config
from lib.files import read_items, read_matrix, save_json
from lib.graph import correlation_graph, edges_table, summarize


def main() -> None:
    config.ensure_dirs()
    items = read_items(config.OUTPUTS / "01_items.csv")
    R = read_matrix(config.OUTPUTS / "02_R.csv")
    n_pairwise = read_matrix(config.OUTPUTS / "02_pairwise_n.csv")

    G = correlation_graph(R, n_pairwise, items)

    nx.write_graphml(G, config.OUTPUTS / "03_graph.graphml")
    edges_table(G).to_csv(config.OUTPUTS / "03_edges.csv", index=False, float_format="%.6g")

    rule = {"corr_method": config.CORR_METHOD, "edge_rule": config.EDGE_RULE}
    rule.update({"fdr_q": config.FDR_Q} if config.EDGE_RULE == "fdr" else {"r_threshold": config.R_THRESHOLD})
    s = summarize(G)
    save_json({**rule, **s}, config.OUTPUTS / "03_summary.json")

    print(f"Rule: {rule}")
    print(f"Edges: {s['n_edges']}  (density {s['density']:.3f})")
    print(f"Positive / negative: {s['n_positive']} / {s['n_negative']}")
    print(f"Within-theme / cross-theme: {s['within_theme_edges']} / {s['cross_theme_edges']}")
    print(f"Components: {s['n_components']}, largest: {s['largest_component_size']}")
    print(f"Isolated nodes ({len(s['isolated_nodes'])}): {', '.join(s['isolated_nodes']) or '-'}")


if __name__ == "__main__":
    main()
