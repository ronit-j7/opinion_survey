"""Step 06 - which statements are central (Tests 3 and 4).

degree            number of edges
strength          sum of |r| over a statement's edges
betweenness       how often the statement sits on the shortest path between two others
cross_theme_share share of its strength that goes to statements in other themes

In:  outputs/03_graph.graphml
Out: outputs/06_centrality.csv
     figures/06_strength.png, figures/06_cross_theme.png
"""
import networkx as nx

from lib import config
from lib.analysis import centrality_table
from lib.files import read_items
from lib.plot import draw_ranking


def main() -> None:
    config.ensure_dirs()
    G = nx.read_graphml(config.OUTPUTS / "03_graph.graphml")
    items = read_items(config.OUTPUTS / "01_items.csv")

    table = centrality_table(G)
    table = table.merge(items[["code", "text"]], on="code")
    table.to_csv(config.OUTPUTS / "06_centrality.csv", index=False, float_format="%.4g")

    draw_ranking(table, "strength", "Most central statements (strength)",
                 config.FIGURES / "06_strength.png")
    connected = table[table["degree"] > 0]
    draw_ranking(connected, "cross_theme_share", "Statements that bridge themes",
                 config.FIGURES / "06_cross_theme.png")

    print("Top 10 by strength:")
    print(table.head(10)[["code", "theme", "degree", "strength", "betweenness"]]
          .to_string(index=False, float_format="%.3f"))
    print("\nTop 5 by betweenness:")
    print(table.nlargest(5, "betweenness")[["code", "theme", "betweenness"]]
          .to_string(index=False, float_format="%.3f"))
    print("\nTop 5 bridging themes (of statements with edges):")
    print(connected.nlargest(5, "cross_theme_share")[["code", "theme", "degree", "cross_theme_share"]]
          .to_string(index=False, float_format="%.3f"))
    print("\nMean strength by theme:")
    print(table.groupby("theme")["strength"].mean().round(3).to_string())


if __name__ == "__main__":
    main()
