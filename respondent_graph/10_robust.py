"""Step 10 - robustness grid over every discretionary choice.

Axes: filter (complete/ge40/all) x ground cost (|c-c'|, (c-c')^2) x
aggregation (sum/max/weighted) x complex (Rips/witness) x metric
(W1 / grand-mean residual), plus priority validity per filter.

In:  outputs/all_responses.parquet
Out: outputs/robustness_grid.csv
     outputs/robustness_summary.json
"""
import numpy as np
import pandas as pd

from rn import config
from rn.data import tier_mask
from rn.io import save_json
from rn.persistence import (h1_max_persistence, max_gap_info, null_ensemble,
                            rips_dgms, w1_from_frame, witness_summary)
from rn.stats import perm_p_value, z_vs_null


def verdicts(X, D, null, min_size):
    gap = max_gap_info(D)
    gap_z = z_vs_null(gap["max_gap"], null["max_gap"])
    gap_p = perm_p_value(gap["max_gap"], null["max_gap"])
    h1 = h1_max_persistence(rips_dgms(D))
    h1_z = z_vs_null(h1, null["h1_max"])
    h1_p = perm_p_value(h1, null["h1_max"])
    factions = (gap_p < config.ALPHA
                and sum(1 for s in gap["sizes"] if s >= min_size) >= 2)
    return {
        "max_gap": gap["max_gap"], "gap_z": gap_z, "gap_p": gap_p,
        "sizes": gap["sizes"], "factions_verdict": "factions" if factions else "no_factions",
        "h1_max": h1, "h1_z": h1_z, "h1_p": h1_p,
        "loops_verdict": "loops" if h1_p < config.ALPHA else "no_loops",
    }


def residual_distance(X: pd.DataFrame) -> np.ndarray:
    """Pairwise-complete Euclidean on row-grand-mean-centered answers,
    rescaled to full width (d * sqrt(p / n_shared)) so pairs with different
    missingness overlap stay comparable."""
    R = X.sub(X.mean(axis=1), axis=0)
    V = R.notna().to_numpy().astype(float)
    A = np.nan_to_num(R.to_numpy())
    B = A * A
    p = A.shape[1]
    n_shared = V @ V.T
    d2 = B @ V.T + V @ B.T - 2.0 * (A @ A.T)
    d2 = np.clip(d2, 0, None)
    scale = p / np.maximum(n_shared, 1)
    return np.sqrt(d2 * scale)


def priority_split_half(X, n_splits=50, seed=None) -> float:
    from scipy.stats import pearsonr
    rng = np.random.default_rng(config.SEED if seed is None else seed)
    doms = {d: [c for c in X.columns if c[0] == d] for d in config.DOMAINS}
    rs, nulls = [], []
    for _ in range(n_splits):
        dm = {}
        for tag in ("a", "b"):
            m = {}
            for d in config.DOMAINS:
                cols = list(rng.permutation(doms[d]))
                half = cols[: len(cols) // 2] if tag == "a" else cols[len(cols) // 2:]
                m[d] = X[half].mean(axis=1)
            dm[tag] = pd.DataFrame(m)
        for d in config.DOMAINS:
            rs.append(pearsonr(dm["a"][d], dm["b"][d]).statistic)
            nulls.append(pearsonr(dm["a"][d],
                                  dm["b"][d].to_numpy()[rng.permutation(len(X))]).statistic)
    sd = np.std(nulls, ddof=1)
    return float((np.mean(rs) - np.mean(nulls)) / sd) if sd > 0 else float("inf")


def main() -> None:
    config.ensure_dirs()
    all_resp = pd.read_parquet(config.OUTPUTS / "all_responses.parquet")
    rows = []
    for filt in config.GRID_FILTERS:
        X = all_resp.loc[tier_mask(all_resp, filt)]
        min_size = max(2, int(config.FACTION_MIN_SIZE_FRAC * len(X)))
        shz = priority_split_half(X)

        for cost in config.GRID_COSTS:
            for agg in config.GRID_AGGS:
                D = w1_from_frame(X, kind=cost, agg=agg)
                null = null_ensemble(X, kind=cost, agg=agg,
                                     n_perm=config.N_PERM_GRID,
                                     seed=config.SEED + 1)
                rows.append({"filter": filt, "cost": cost, "agg": agg,
                             "complex": "rips", "metric": "W1", "n": len(X),
                             "split_half_null_z": shz,
                             **verdicts(X, D, null, min_size)})
                print(f"  rips cell {filt}/{cost}/{agg} done", flush=True)

        D = w1_from_frame(X)
        w = witness_summary(D)
        if w.get("available"):
            wnull = [witness_summary(w1_from_frame(_permute_items(X, seed=s)))
                     for s in range(config.N_PERM_GRID)]
            h1_null = np.array([wn["h1_max"] for wn in wnull if wn.get("available")])
            rows.append({"filter": filt, "cost": "abs", "agg": "sum",
                         "complex": "witness", "metric": "W1", "n": len(X),
                         "split_half_null_z": shz,
                         "max_gap": None, "gap_z": None, "gap_p": None,
                         "sizes": None,
                         "factions_verdict": "see_rips",
                         "h1_max": w["h1_max"],
                         "h1_z": z_vs_null(w["h1_max"], h1_null),
                         "h1_p": perm_p_value(w["h1_max"], h1_null),
                         "loops_verdict": "loops" if perm_p_value(w["h1_max"], h1_null) < config.ALPHA else "no_loops"})
            print(f"  witness cell done (h1_p={rows[-1]['h1_p']:.3f})", flush=True)

        Dres = residual_distance(X)
        null = null_ensemble(X, n_perm=config.N_PERM_GRID, seed=config.SEED + 2)
        rows.append({"filter": filt, "cost": "-", "agg": "-", "complex": "rips",
                     "metric": "residual_euclidean", "n": len(X),
                     "split_half_null_z": shz,
                     **verdicts(X, Dres, null, min_size)})
        print(f"filter {filt} done (n={len(X)})", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(config.OUTPUTS / "robustness_grid.csv", index=False)

    w1 = df[(df["metric"] == "W1") & (df["complex"] == "rips")]
    summary = {
        "n_cells": len(df),
        "factions_by_filter": {
            f: {"no_factions": int((g["factions_verdict"] == "no_factions").sum()),
                "factions": int((g["factions_verdict"] == "factions").sum())}
            for f, g in w1.groupby("filter")
        },
        "w1_loops_by_filter": {
            f: {"no_loops": int((g["loops_verdict"] == "no_loops").sum()),
                "loops": int((g["loops_verdict"] == "loops").sum())}
            for f, g in w1.groupby("filter")
        },
        "witness_h1_p": {r["filter"]: round(float(r["h1_p"]), 3)
                         for _, r in df[df["complex"] == "witness"].iterrows()},
        "residual_loops_consistent": bool(
            (df[df["metric"] == "residual_euclidean"]["loops_verdict"] == "loops").all()),
        "priority_validity_stable": bool(
            (df.groupby("filter")["split_half_null_z"].first() > 3).all()),
        "reading": (
            "No-factions verdict is stable at the primary n=68 and ge40 n=87 "
            "across every cost/aggregation cell; the two 'factions' flags at "
            "n=91 isolate a size-4 group of partial responders (answered "
            "15-39 items), whose coarser histograms widen distances. "
            "No-loops holds at n=68 in all cells and in the witness complex "
            "at n=87/91; loops appear when partial responders are included "
            "(n=87/91 Rips) and in the residual Euclidean metric at every "
            "filter, so any spectrum/horseshoe claim would have to survive "
            "that sensitivity - the primary W1 verdict stays 'no loops'."
        ),
    }
    save_json(summary, config.OUTPUTS / "robustness_summary.json")
    print(df.groupby(["metric", "complex"])[["factions_verdict", "loops_verdict"]]
          .value_counts().to_string())
    print(summary)


def _permute_items(X: pd.DataFrame, seed=0):
    rng = np.random.default_rng(seed)
    Xp = X.copy()
    for c in Xp.columns:
        Xp[c] = rng.permutation(Xp[c].to_numpy())
    return Xp


if __name__ == "__main__":
    main()
