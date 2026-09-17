# Questions as nodes

A belief network built from the class survey. Each of the 60 statements is a node, and edges are regularized partial correlations (graphical lasso + EBIC).

`00_plan.md` explains the method and the tests. `00_spec_full_math.html` has the longer derivations.

## Run

From this folder:

```bash
pip install -r requirements.txt
python 01_prepare_data.py
python 02_correlation_matrix.py
python 03_estimate_network.py
python 04_build_graph.py
python 05_plot_network.py
```

Each step reads the previous step's files from `outputs/`. It writes its own files with the same number prefix.

## Layout

| File | Does | Writes |
|---|---|---|
| `01_prepare_data.py` | Codes answers 1–5; "No Comments" becomes missing; drops empty rows | `01_responses.csv`, `01_items.csv`, `01_info.json` |
| `02_correlation_matrix.py` | Pairwise Spearman → $2\sin(\pi r/6)$ → eigenvalue repair if needed | `02_spearman.csv`, `02_R.csv`, `02_pairwise_n.csv`, `02_info.json` |
| `03_estimate_network.py` | Graphical lasso over a grid of λ; picks λ by EBIC; computes partial correlations | `03_ebic_path.csv`, `03_precision.csv`, `03_partial_corr.csv`, `03_selection.json`, `figures/03_ebic_curve.png` |
| `04_build_graph.py` | Builds the main graph plus a baseline graph of significant plain correlations | `04_graph.graphml`, `04_edges.csv`, `04_baseline_*`, `04_summary.json` |
| `05_plot_network.py` | Quick network plot | `figures/05_network.png` |

The logic lives in `bn/`, and the numbered scripts are thin wrappers around it. That way later steps, such as the bootstrap, can rerun the whole pipeline by importing it:

- `bn/config.py`: all settings (γ, λ grid, tolerances, colours, paths)
- `bn/data.py`: loading and encoding
- `bn/correlation.py`: Spearman, sine conversion, eigenvalue repair
- `bn/network.py`: graphical lasso, EBIC, partial correlations
- `bn/graph.py`: networkx graphs, BH-FDR baseline, summaries
- `bn/files.py`: reading and writing intermediate files
