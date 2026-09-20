# Questions as nodes

A network built from the class survey:
- **Nodes:** the 60 statements.
- **Edges:** the correlation between two statements across respondents.
- **Edge rule:** by default, a correlation becomes an edge only if it is statistically significant, allowing at most 5% false edges (FDR).

`00_plan.md` describes the tests we plan to run on the graph.

## Run

From this folder:

```bash
pip install -r requirements.txt
python 01_prepare_data.py
python 02_correlation_matrix.py
python 03_build_graph.py
python 04_plot_graph.py
python 05_item_stats.py
python 06_centrality.py
python 07_communities.py
python 08_stability.py
```

Each step reads the previous step's files from `outputs/`. It writes its own files with the same number prefix.

## Layout

| File | Does | Writes |
|---|---|---|
| `01_prepare_data.py` | Codes answers 1–5; "No Comments" becomes missing; drops empty rows | `01_responses.csv`, `01_items.csv`, `01_info.json` |
| `02_correlation_matrix.py` | Pearson correlation for every pair of statements, using respondents who answered both | `02_R.csv`, `02_pairwise_n.csv`, `02_info.json` |
| `03_build_graph.py` | Keeps significant correlations as edges | `03_graph.graphml`, `03_edges.csv`, `03_summary.json` |
| `04_plot_graph.py` | Draws the graph | `figures/04_network.png` |
| `05_item_stats.py` | Per statement: mean, spread, agree/disagree shares, consensus/split label | `05_item_stats.csv`, `figures/05_answer_shares.png` |
| `06_centrality.py` | Degree, strength, betweenness, share of strength going to other themes | `06_centrality.csv`, `figures/06_strength.png`, `figures/06_cross_theme.png` |
| `07_communities.py` | Louvain communities, comparison with the four themes, theme block matrix | `07_communities.csv`, `07_community_theme_table.csv`, `07_theme_blocks.csv`, `07_summary.json`, 2 figures |
| `08_stability.py` | Rebuilds the graph on 1,000 resamples of the respondents and reports what survives | `08_edge_stability.csv`, `08_node_stability.csv`, `08_summary.json`, 2 figures |

Settings live in `lib/config.py`:
- `CORR_METHOD`: `"pearson"` or `"spearman"`.
- `EDGE_RULE`: `"fdr"` (significance, uses `FDR_Q`) or `"threshold"` (keeps edges with |r| ≥ `R_THRESHOLD`).

The shared code lives in `lib/`:
- `config.py`: settings
- `data.py`: loading and encoding
- `correlation.py`: correlations
- `graph.py`: graph building and summaries
- `analysis.py`: item statistics, centrality, communities, bootstrap stability
- `plot.py`: drawing
- `files.py`: reading and writing intermediate files
- `network.py`: used only by the optional glasso scripts

## Optional: partial-correlation graph (graphical lasso)

This version is parked in `glasso_optional/`. It keeps only direct links but is more complex, and `TEMP_steps_explained.md` explains it. Run it after steps 01–02:

```bash
python glasso_optional/g1_estimate_network.py
python glasso_optional/g2_build_graph.py
```
