"""Step 05 - persistent homology on W1 (and the residual metric).

In:  outputs/W1.npy, outputs/clean.parquet
Out: outputs/dgms.pkl            ripser diagrams for both metrics
     outputs/null_ensemble.npy   null (max_gap, h1_max) arrays
     outputs/persistence_stats.json
     figures/05_barcodes.png

Second metric: Euclidean on row-grand-mean-centered answers (removes each
respondent's overall enthusiasm level, keeps domain priorities).
"""
import numpy as np
import pandas as pd

from rn import config
from rn.io import load_npy, save_json, save_pickle
from rn.persistence import (bootstrap, max_gap_info, null_ensemble, rips_dgms,
                            single_linkage, heights, h1_max_persistence)
from rn.stats import perm_p_value, z_vs_null


def residual_distance(X: pd.DataFrame) -> np.ndarray:
    """User choice: row grand-mean centered (removes enthusiasm level,
    keeps domain priorities)."""
    R = X.sub(X.mean(axis=1), axis=0).to_numpy()
    return np.linalg.norm(R[:, None, :] - R[None, :, :], axis=-1)


def residual_wd_distance(X: pd.DataFrame) -> np.ndarray:
    """Spec-original variant: within-domain centered (removes priorities too,
    keeps only within-domain response shape)."""
    cols = {d: [c for c in X.columns if c[0] == d] for d in config.DOMAINS}
    parts = [X[cols[d]].sub(X[cols[d]].mean(axis=1), axis=0) for d in config.DOMAINS]
    R = pd.concat(parts, axis=1)[X.columns].to_numpy()
    return np.linalg.norm(R[:, None, :] - R[None, :, :], axis=-1)


def main() -> None:
    config.ensure_dirs()
    X = pd.read_parquet(config.OUTPUTS / "clean.parquet").drop(
        columns=["row_position"], errors="ignore")
    W1 = load_npy(config.OUTPUTS / "W1.npy")
    Dres = residual_distance(X)
    Dwd = residual_wd_distance(X)

    dgms = {"w1": rips_dgms(W1), "residual": rips_dgms(Dres),
            "residual_wd": rips_dgms(Dwd)}
    save_pickle(dgms, config.OUTPUTS / "dgms.pkl")

    null = null_ensemble(X)
    np.save(config.OUTPUTS / "null_max_gap.npy", null["max_gap"])
    np.save(config.OUTPUTS / "null_h1_max.npy", null["h1_max"])

    boot = bootstrap(W1)

    stats = {}
    for name, D in (("w1", W1), ("residual", Dres), ("residual_wd", Dwd)):
        obs_gap = max_gap_info(D)
        h1 = h1_max_persistence(dgms[name])
        if name == "w1":
            null_m = null
        else:
            null_m = null_ensemble(X, n_perm=max(100, config.N_PERM // 2))
        gap_z = z_vs_null(obs_gap["max_gap"], null_m["max_gap"])
        gap_p = perm_p_value(obs_gap["max_gap"], null_m["max_gap"])
        h1_z = z_vs_null(h1, null_m["h1_max"])
        h1_p = perm_p_value(h1, null_m["h1_max"])
        stats[name] = {
            "max_gap": obs_gap["max_gap"],
            "gap_cut_sizes": obs_gap["sizes"],
            "gap_z": gap_z,
            "gap_p": gap_p,
            "h1_max": h1,
            "h1_z": h1_z,
            "h1_p": h1_p,
        }

    stats["bootstrap_w1"] = {
        "max_gap_ci95": [float(np.percentile(boot["max_gap"], 2.5)),
                         float(np.percentile(boot["max_gap"], 97.5))],
        "h1_max_ci95": [float(np.percentile(boot["h1_max"], 2.5)),
                        float(np.percentile(boot["h1_max"], 97.5))],
        "n_boot": config.N_BOOT,
    }
    stats["n_perm"] = config.N_PERM
    save_json(stats, config.OUTPUTS / "persistence_stats.json")

    _plot(dgms, null, stats)

    for name in ("w1", "residual", "residual_wd"):
        s = stats[name]
        print(f"[{name:11s}] H0 max gap {s['max_gap']:.4f} (z {s['gap_z']:+.1f}, "
              f"p {s['gap_p']:.3f}) sizes {s['gap_cut_sizes']}  | "
              f"H1 max {s['h1_max']:.4f} (z {s['h1_z']:+.1f}, p {s['h1_p']:.3f})")


def _plot(dgms, null, stats):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from rn import viz

    viz.set_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8),
                             gridspec_kw={"width_ratios": [1, 1, 1.2]})
    for k, (name, frac) in enumerate((("w1", 0.42), ("residual_wd", 0.64))):
        d0 = dgms[name][0]
        d0f = d0[np.isfinite(d0[:, 1])]
        d0f = d0f[np.argsort(d0f[:, 1])]
        lengths = d0f[:, 1] - d0f[:, 0]
        colors = viz.magma_ramp(lengths)
        axes[k].hlines(np.arange(len(d0f)), d0f[:, 0], d0f[:, 1],
                       color=colors, lw=2.2, alpha=0.9)
        axes[k].set_title(f"{name}: H0 barcode")
        axes[k].set_xlabel("merge height")
    sns.histplot(null["max_gap"], ax=axes[2], color=viz.NEUTRAL, fill=True,
                 alpha=0.30, stat="density", label="null max gap")
    axes[2].axvline(stats["w1"]["max_gap"], color=viz.magma(0.42), lw=1.6,
                    ls="--", label="observed W1")
    axes[2].set_title(f"H0 max-gap null  (p = {stats['w1']['gap_p']:.3f})")
    axes[2].set(xlabel="max gap")
    axes[2].legend()
    viz.savefig(fig, "05_barcodes.png")


if __name__ == "__main__":
    main()
