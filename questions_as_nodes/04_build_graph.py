"""Step 04 - build the partial-correlation graph and the baseline marginal-correlation graph.

In:  outputs/03_partial_corr.csv, outputs/02_R.csv, outputs/02_pairwise_n.csv, outputs/01_items.csv
Out: outputs/04_graph.graphml             main graph (open in Gephi / networkx)
     outputs/04_edges.csv                 main graph edge list, strongest first
     outputs/04_baseline_graph.graphml    baseline: significant plain correlations (BH-FDR)
     outputs/04_baseline_edges.csv
     outputs/04_summary.json              edge counts, density, signs, theme mix, isolated nodes
"""
import networkx as nx

from bn import config
from bn.files import read_items, read_matrix, save_json
from bn.graph import edges_table, graph_from_weights, marginal_graph, summarize


def print_summary(name: str, s: dict) -> None:
    print(f"--- {name} ---")
    print(f"Edges: {s['n_edges']}  (density {s['density']:.3f})")
    print(f"Positive / negative: {s['n_positive']} / {s['n_negative']}")
    print(f"Within-theme / cross-theme: {s['within_theme_edges']} / {s['cross_theme_edges']}")
    print(f"Components: {s['n_components']}, largest: {s['largest_component_size']}")
    print(f"Isolated nodes ({len(s['isolated_nodes'])}): {', '.join(s['isolated_nodes']) or '-'}")


def main() -> None:
    config.ensure_dirs()
    items = read_items(config.OUTPUTS / "01_items.csv")
    omega = read_matrix(config.OUTPUTS / "03_partial_corr.csv")
    R = read_matrix(config.OUTPUTS / "02_R.csv")
    n_pairwise = read_matrix(config.OUTPUTS / "02_pairwise_n.csv")

    G = graph_from_weights(omega, items)
    B = marginal_graph(R, n_pairwise, items)

    nx.write_graphml(G, config.OUTPUTS / "04_graph.graphml")
    nx.write_graphml(B, config.OUTPUTS / "04_baseline_graph.graphml")
    edges_table(G).to_csv(config.OUTPUTS / "04_edges.csv", index=False, float_format="%.6g")
    edges_table(B).to_csv(config.OUTPUTS / "04_baseline_edges.csv", index=False, float_format="%.6g")

    summary = {"partial_correlation_graph": summarize(G),
               "baseline_marginal_graph": summarize(B),
               "baseline_fdr_q": config.FDR_Q}
    save_json(summary, config.OUTPUTS / "04_summary.json")

    print_summary("Partial correlation graph", summary["partial_correlation_graph"])
    print_summary("Baseline marginal graph", summary["baseline_marginal_graph"])


if __name__ == "__main__":
    main()
