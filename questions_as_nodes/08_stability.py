"""Step 08 - how much of the graph survives resampling the respondents? (Test 9)

Draw 91 respondents with replacement, rebuild the correlations, apply the same edge rule,
and repeat 1,000 times. Then report how often each edge comes back, and how stable the
strength ranking is.

In:  outputs/01_responses.csv, outputs/03_graph.graphml
Out: outputs/08_edge_stability.csv    every pair, its inclusion rate, whether it is in our graph
     outputs/08_node_stability.csv    per statement: observed strength and its 95% interval
     outputs/08_summary.json
     figures/08_edge_inclusion.png, figures/08_strength_intervals.png
"""
import networkx as nx
import numpy as np
import pandas as pd

from lib import config
from lib.analysis import bootstrap_stability
from lib.files import read_responses, save_json
from lib.plot import draw_inclusion, draw_strength_intervals


def main() -> None:
    config.ensure_dirs()
    X = read_responses(config.OUTPUTS / "01_responses.csv")
    G = nx.read_graphml(config.OUTPUTS / "03_graph.graphml")

    observed_strength = pd.Series(
        {n: sum(d["abs_weight"] for _, _, d in G.edges(n, data=True)) for n in G})
    in_graph = {tuple(sorted(e)) for e in G.edges()}

    print(f"Resampling {config.BOOTSTRAP_B} times (rule: {config.EDGE_RULE}) ...")
    res = bootstrap_stability(X, observed_strength)

    pairs = res["pairs"]
    pairs["in_graph"] = [tuple(sorted((s, t))) in in_graph
                         for s, t in zip(pairs["source"], pairs["target"])]
    pairs.sort_values(["in_graph", "inclusion_rate"], ascending=False).to_csv(
        config.OUTPUTS / "08_edge_stability.csv", index=False, float_format="%.4g")
    res["nodes"].to_csv(config.OUTPUTS / "08_node_stability.csv", index=False, float_format="%.4g")

    ours = pairs[pairs["in_graph"]]["inclusion_rate"]
    missed = pairs[~pairs["in_graph"]]["inclusion_rate"]
    rank_corr = res["rank_corr"][~np.isnan(res["rank_corr"])]
    summary = {
        "n_resamples": res["B"],
        "edge_rule": config.EDGE_RULE,
        "n_edges_observed": int(len(ours)),
        "edges_in_over_90pct": int((ours >= 0.9).sum()),
        "edges_in_over_75pct": int((ours >= 0.75).sum()),
        "edges_in_under_50pct": int((ours < 0.5).sum()),
        "median_inclusion_of_our_edges": float(ours.median()),
        "pairs_outside_graph_over_50pct": int((missed >= 0.5).sum()),
        "strength_rank_corr_mean": float(rank_corr.mean()),
        "strength_rank_corr_p05": float(np.percentile(rank_corr, 5)),
        "n_failed_resamples": res["n_failed"],
    }
    save_json(summary, config.OUTPUTS / "08_summary.json")

    draw_inclusion(pairs[pairs["in_graph"]],
                   f"Edge stability over {res['B']} resamples",
                   config.FIGURES / "08_edge_inclusion.png")
    draw_strength_intervals(res["nodes"],
                            "Strength with bootstrap 95% interval",
                            config.FIGURES / "08_strength_intervals.png")

    print(f"Our {summary['n_edges_observed']} edges: "
          f"{summary['edges_in_over_90pct']} appear in >90% of resamples, "
          f"{summary['edges_in_over_75pct']} in >75%, "
          f"{summary['edges_in_under_50pct']} in <50%.")
    print(f"Median inclusion rate: {summary['median_inclusion_of_our_edges']:.2f}")
    print(f"Pairs outside our graph that appeared in >50% of resamples: "
          f"{summary['pairs_outside_graph_over_50pct']}")
    print(f"Strength ranking vs observed: mean correlation {summary['strength_rank_corr_mean']:.3f}, "
          f"5th percentile {summary['strength_rank_corr_p05']:.3f}")


if __name__ == "__main__":
    main()
