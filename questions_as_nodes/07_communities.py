"""Step 07 - does the graph rediscover the four survey themes? (Tests 6 and 8)

1. Find communities with the Louvain method (negative edges dropped: modularity needs
   non-negative weights).
2. Compare them with the themes: a community x theme table, plus NMI and ARI.
3. Test the themes directly: is their modularity higher than random splits of the same sizes?
4. Theme block matrix: mean |r| within and between themes, over all item pairs.

In:  outputs/03_graph.graphml, outputs/02_R.csv, outputs/01_items.csv
Out: outputs/07_communities.csv, outputs/07_community_theme_table.csv,
     outputs/07_theme_blocks.csv, outputs/07_summary.json
     figures/07_network_communities.png, figures/07_theme_blocks.png
"""
import networkx as nx
import pandas as pd

from lib import config
from lib.analysis import (best_louvain, compare_partitions, partition_modularity,
                          positive_graph, theme_block_matrix, theme_modularity_test)
from lib.files import read_items, read_matrix, save_json
from lib.plot import draw_block_matrix, draw_network


def main() -> None:
    config.ensure_dirs()
    G = nx.read_graphml(config.OUTPUTS / "03_graph.graphml")
    items = read_items(config.OUTPUTS / "01_items.csv")
    R = read_matrix(config.OUTPUTS / "02_R.csv")

    H = positive_graph(G)
    communities, modularity = best_louvain(H)
    themes = {n: G.nodes[n]["theme"] for n in G}

    table = pd.DataFrame({"code": list(G.nodes),
                          "theme": [themes[n] for n in G],
                          "community": [communities[n] for n in G]})
    table = table.merge(items[["code", "text"]], on="code").sort_values(["community", "theme", "code"])
    table.to_csv(config.OUTPUTS / "07_communities.csv", index=False)

    crosstab = pd.crosstab(table["community"], table["theme"])
    crosstab.to_csv(config.OUTPUTS / "07_community_theme_table.csv")

    blocks = theme_block_matrix(R, items)
    blocks.to_csv(config.OUTPUTS / "07_theme_blocks.csv", float_format="%.4f")

    sizes = table["community"].value_counts()
    summary = {
        "n_communities": len(set(communities.values())),
        "n_communities_with_2plus": int((sizes >= 2).sum()),
        "n_singletons": int((sizes == 1).sum()),
        "modularity_communities": modularity,
        "modularity_themes": partition_modularity(H, themes),
        **compare_partitions(communities, themes),
        **theme_modularity_test(H, themes),
        "community_sizes": table["community"].value_counts().sort_index().to_dict(),
        "negative_edges_dropped": G.number_of_edges() - H.number_of_edges(),
    }
    save_json(summary, config.OUTPUTS / "07_summary.json")

    colours = {n: ("#BBBBBB" if sizes[communities[n]] == 1
                   else config.COMMUNITY_COLORS[communities[n] % len(config.COMMUNITY_COLORS)])
               for n in G}
    legend = [(f"community {i} ({sizes[i]} statements)",
               config.COMMUNITY_COLORS[i % len(config.COMMUNITY_COLORS)])
              for i in sorted(sizes[sizes >= 2].index)]
    if (sizes == 1).any():
        legend.append((f"{int((sizes == 1).sum())} single statements (no positive edges)", "#BBBBBB"))
    draw_network(G, f"Communities found by Louvain: {summary['n_communities_with_2plus']} groups "
                    f"+ {summary['n_singletons']} single statements (modularity {modularity:.3f})",
                 config.FIGURES / "07_network_communities.png",
                 node_colors=colours, legend_entries=legend)
    draw_block_matrix(blocks, "Mean |r| within and between themes",
                      config.FIGURES / "07_theme_blocks.png")

    print(f"Communities: {summary['n_communities_with_2plus']} with 2+ statements, "
          f"{summary['n_singletons']} singletons; sizes {summary['community_sizes']}")
    print(f"Modularity: communities {modularity:.3f}, themes {summary['modularity_themes']:.3f}")
    print(f"Themes vs communities: NMI {summary['nmi']:.3f}, ARI {summary['ari']:.3f}")
    print(f"Theme modularity test: p = {summary['p_value']:.4f} "
          f"({summary['n_permutations']} random relabellings)")
    print("\nCommunity x theme:")
    print(crosstab.to_string())
    print("\nMean |r| by theme block:")
    print(blocks.round(3).to_string())


if __name__ == "__main__":
    main()
