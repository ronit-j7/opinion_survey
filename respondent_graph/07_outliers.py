"""Step 07 - respondents that persist as isolated H0 components, characterized.

In:  outputs/dgms.pkl, outputs/W1.npy, outputs/clean.parquet,
     outputs/dendrogram.json, outputs/diagnostics.json
Out: outputs/outliers.csv    response_id, row_position, persistence, mix,
                            profile, extremity, dominant domain
     outputs/outliers.json   class baselines for comparison
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster

from rn import config
from rn.io import load_json, load_npy, save_json
from rn.measures import dominant, domain_means, extremity, ipsative
from rn.persistence import single_linkage, heights

CATS = ["SD", "D", "N", "A", "SA"]


def main() -> None:
    config.ensure_dirs()
    X = pd.read_parquet(config.OUTPUTS / "clean.parquet")
    row_pos = X["row_position"]
    Xi = X.drop(columns=["row_position"])
    W1 = load_npy(config.OUTPUTS / "W1.npy")
    dendro = load_json(config.OUTPUTS / "dendrogram.json")

    sig = dendro["significant_gaps"]
    Z = single_linkage(W1)
    h = heights(Z)

    outliers = pd.DataFrame()
    if sig:
        cut = sig[-1]["cut"]
        labels = fcluster(Z, t=cut, criterion="distance")
        _, counts = np.unique(labels, return_counts=True)
        singleton_labels = {lab for lab, cnt in zip(*np.unique(labels, return_counts=True)) if cnt == 1}
        mask = np.isin(labels, list(singleton_labels))
        ids = list(Xi.index)

        mix = Xi.apply(lambda r: r.value_counts()
                       .reindex(range(1, 6), fill_value=0).to_numpy(), axis=1)
        prof = ipsative(Xi)
        ext = extremity(Xi)
        dom = dominant(Xi)
        dm = domain_means(Xi)

        rows = []
        for i in np.where(mask)[0]:
            rows.append({
                "response_id": int(ids[i]),
                "row_position": int(row_pos.iloc[i]),
                "persistence": float(h[-1] - 0) if len(h) else 0.0,
                "merge_height": float(np.max(W1[i])),
                **{f"mix_{c}": int(mix.iloc[i][k]) for k, c in enumerate(CATS)},
                **{f"pri_{d}": round(float(prof.iloc[i][d]), 3) for d in config.DOMAINS},
                "dominant": dom.iloc[i],
                "extremity": round(float(ext.iloc[i]), 3),
            })
        outliers = pd.DataFrame(rows)
        outliers.to_csv(config.OUTPUTS / "outliers.csv", index=False)

    baseline = {
        "extremity_mean": float(extremity(Xi).mean()),
        "mix_mean": {f"mix_{c}": float((Xi == k + 1).sum(axis=1).mean())
                     for k, c in enumerate(CATS)},
        "priority_mean": {d: float(ipsative(Xi)[d].mean()) for d in config.DOMAINS},
        "n_outliers": int(len(outliers)),
        "cut_used": sig[-1]["cut"] if sig else None,
    }
    save_json(baseline, config.OUTPUTS / "outliers.json")

    print("outliers:", len(outliers))
    if len(outliers):
        print(outliers.to_string(index=False))
    print("baselines:", {k: (round(v, 2) if isinstance(v, float) else v)
                         for k, v in baseline.items() if k != "mix_mean"})


if __name__ == "__main__":
    main()
