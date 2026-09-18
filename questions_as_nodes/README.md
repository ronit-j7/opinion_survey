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
```

Each step reads the previous step's files from `outputs/`. It writes its own files with the same number prefix.

## Layout

| File | Does | Writes |
|---|---|---|
| `01_prepare_data.py` | Codes answers 1–5; "No Comments" becomes missing; drops empty rows | `01_responses.csv`, `01_items.csv`, `01_info.json` |
| `02_correlation_matrix.py` | Pearson correlation for every pair of statements, using respondents who answered both | `02_R.csv`, `02_pairwise_n.csv`, `02_info.json` |
| `03_build_graph.py` | Keeps significant correlations as edges | `03_graph.graphml`, `03_edges.csv`, `03_summary.json` |
| `04_plot_graph.py` | Draws the graph | `figures/04_network.png` |

Settings live in `lib/config.py`:
- `CORR_METHOD`: `"pearson"` or `"spearman"`.
- `EDGE_RULE`: `"fdr"` (significance, uses `FDR_Q`) or `"threshold"` (keeps edges with |r| ≥ `R_THRESHOLD`).

The shared code lives in `lib/`:
- `config.py`: settings
- `data.py`: loading and encoding
- `correlation.py`: correlations
- `graph.py`: graph building and summaries
- `plot.py`: drawing
- `files.py`: reading and writing intermediate files
- `network.py`: used only by the optional glasso scripts

## Optional: partial-correlation graph (graphical lasso)

This version is parked in `glasso_optional/`. It keeps only direct links but is more complex, and `TEMP_steps_explained.md` explains it. Run it after steps 01–02:

```bash
python glasso_optional/g1_estimate_network.py
python glasso_optional/g2_build_graph.py
```
