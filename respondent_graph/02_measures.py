"""Step 02 - build the respondent measures.

In:  outputs/clean.parquet
Out: outputs/measures.npy    (n, 4, 5) histogram stack, rows sum to 1
     outputs/answered.npy     (n, 4) answered items per domain
     outputs/profiles.csv     response_id, t/e/s/v ipsative profile
     outputs/extremity.csv    response_id, extremity
     outputs/measures_info.json
"""
import numpy as np
import pandas as pd

from rn import config
from rn.io import save_json, save_npy
from rn.measures import dominant, domain_means, extremity, histograms, ipsative


def main() -> None:
    config.ensure_dirs()
    X = pd.read_parquet(config.OUTPUTS / "clean.parquet")
    X = X.drop(columns=["row_position"], errors="ignore")

    hist, answered = histograms(X)
    prof = ipsative(X)
    ext = extremity(X)
    dm = domain_means(X)
    dom = dominant(X)

    save_npy(hist, config.OUTPUTS / "measures.npy")
    save_npy(answered, config.OUTPUTS / "answered.npy")
    prof.round(4).to_csv(config.OUTPUTS / "profiles.csv", index_label="response_id")
    ext.round(4).rename("extremity").to_csv(
        config.OUTPUTS / "extremity.csv", index_label="response_id")

    info = {
        "n": len(X),
        "hist_shape": list(hist.shape),
        "row_sums_ok": bool(np.allclose(hist.sum(axis=2), 1.0)),
        "profile_rows_sum_zero": bool(np.allclose(prof.sum(axis=1), 0.0)),
        "domain_means": {d: float(dm[d].mean()) for d in config.DOMAINS},
        "corr_S_V": float(dm["S"].corr(dm["V"])),
        "corr_T_E": float(dm["T"].corr(dm["E"])),
        "dominant_domain_counts": dom.value_counts().to_dict(),
        "extremity_mean": float(ext.mean()),
        "extremity_sd": float(ext.std(ddof=1)),
    }
    save_json(info, config.OUTPUTS / "measures_info.json")

    print(f"n = {info['n']}  hist {info['hist_shape']}  rows sum 1: {info['row_sums_ok']}")
    print("domain means:", {k: round(v, 2) for k, v in info["domain_means"].items()})
    print(f"corr S-V {info['corr_S_V']:.2f} | T-E {info['corr_T_E']:.2f}")
    print("dominant-domain counts:", info["dominant_domain_counts"])
    print(f"extremity: mean {info['extremity_mean']:.2f} sd {info['extremity_sd']:.2f}")


if __name__ == "__main__":
    main()
