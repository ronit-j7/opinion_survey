"""Step 05 - what the class thinks of each statement (Test 1).

Consensus statements have almost no spread, so they cannot correlate with anything
and tend to end up with few or no edges. This step is what explains those nodes.

In:  outputs/01_responses.csv, outputs/01_items.csv, outputs/03_graph.graphml
Out: outputs/05_item_stats.csv        mean, sd, agree/neutral/disagree shares, label, degree
     figures/05_answer_shares.png     stacked answer breakdown per statement
"""
import networkx as nx

from lib import config
from lib.analysis import answer_shares, item_statistics
from lib.files import read_items, read_responses
from lib.plot import draw_answer_shares


def main() -> None:
    config.ensure_dirs()
    X = read_responses(config.OUTPUTS / "01_responses.csv")
    items = read_items(config.OUTPUTS / "01_items.csv")
    G = nx.read_graphml(config.OUTPUTS / "03_graph.graphml")

    stats = item_statistics(X, items)
    stats["degree"] = stats["code"].map(dict(G.degree()))
    stats.to_csv(config.OUTPUTS / "05_item_stats.csv", index=False, float_format="%.4g")

    draw_answer_shares(answer_shares(X), items,
                       "Answer breakdown per statement", config.FIGURES / "05_answer_shares.png")

    counts = stats["label"].value_counts()
    print(f"Labels: {counts.to_dict()}")
    print("\nHighest agreement:")
    print(stats.nlargest(5, "share_agree")[["code", "mean", "share_agree", "degree"]].to_string(index=False))
    print("\nHighest disagreement:")
    print(stats.nlargest(5, "share_disagree")[["code", "mean", "share_disagree", "degree"]].to_string(index=False))
    print("\nMost split (agree and disagree both large):")
    split = stats[stats["label"] == "split"].nlargest(5, "sd")
    print(split[["code", "mean", "sd", "share_agree", "share_disagree", "degree"]].to_string(index=False))
    print("\nMean degree by label:")
    print(stats.groupby("label")["degree"].mean().round(2).to_string())


if __name__ == "__main__":
    main()
