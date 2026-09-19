"""Step 09 - validity of the domain-priority measure.

Split-half reliability of the priorities, held-out domain prediction, and
priority -> contested-item regressions with Benjamini-Hochberg control.

Contestedness is defined on the n=87 (ge40) subset; regressions are fit on
the primary n=68. Sets reported:
- core8      both the minority-share and entropy definitions agree
- minority10 full minority-share set at ge40 (headline)
- entropy10  top-10 by normalized entropy at ge40
- minority7  the same rule at complete cases (mandatory sensitivity row)

In:  outputs/clean.parquet, outputs/all_responses.parquet
Out: outputs/validity.csv, outputs/prediction.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from rn import config
from rn.io import save_json
from rn.measures import item_entropy
from rn.stats import bh_fdr, fisher_z, perm_p_value

NAMES = {
    "core8": None,
    "minority10": "minority share >= 0.10 (ge40)",
    "entropy10": "top-10 entropy (ge40)",
    "minority7": "minority share >= 0.10 (complete)",
}


def split_half_priorities(X: pd.DataFrame, n_splits=100, seed=None) -> pd.DataFrame:
    rng = np.random.default_rng(config.SEED if seed is None else seed)
    doms = {d: [c for c in X.columns if c[0] == d] for d in config.DOMAINS}
    rows = []
    for _ in range(n_splits):
        halves = {}
        for tag, take_first in (("a", True), ("b", False)):
            dm = {}
            for d in config.DOMAINS:
                dc = list(rng.permutation(doms[d]))
                cols = dc[: len(dc) // 2] if take_first else dc[len(dc) // 2:]
                dm[d] = X[cols].mean(axis=1)
            dm = pd.DataFrame(dm)
            halves[tag] = dm.sub(dm.mean(axis=1), axis=0)
        null_r = []
        for _ in range(20):
            perm = rng.permutation(len(X))
            for d in config.DOMAINS:
                null_r.append(pearsonr(halves["a"][d].to_numpy(),
                                       halves["b"][d].to_numpy()[perm]).statistic)
        for d in config.DOMAINS:
            r = pearsonr(halves["a"][d], halves["b"][d]).statistic
            rows.append({"domain": d, "r": r, "z": fisher_z(r),
                         "null_r_mean": float(np.mean(null_r)),
                         "null_r_sd": float(np.std(null_r, ddof=1))})
    return pd.DataFrame(rows)


def held_out_prediction(X: pd.DataFrame, n_perm=None, seed=None) -> list[dict]:
    rng = np.random.default_rng(config.SEED if seed is None else seed)
    n_perm = n_perm or config.N_PERM_REG
    dm = pd.DataFrame({d: X[[c for c in X.columns if c[0] == d]].mean(axis=1)
                       for d in config.DOMAINS})
    out = []
    for held in config.DOMAINS:
        preds = [d for d in config.DOMAINS if d != held]
        A = np.column_stack([dm[p] for p in preds] + [np.ones(len(X))])
        y = dm[held].to_numpy()
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        yhat = A @ beta
        r_obs = float(pearsonr(y, yhat).statistic)
        null = np.empty(n_perm)
        for p in range(n_perm):
            yp = rng.permutation(y)
            null[p] = pearsonr(yp, yhat).statistic
        out.append({"held_out": held, "r": r_obs,
                    "p": perm_p_value(r_obs, null, greater=True)})
    return out


def contested_sets(all_resp: pd.DataFrame) -> dict[str, list[str]]:
    X87 = all_resp[all_resp.notna().sum(axis=1) >= config.MIN_ANSWERED]
    X68 = all_resp.dropna()
    a = (X87 >= 4).mean()
    d = (X87 <= 2).mean()
    minority10 = [c for c in X87.columns
                  if min(a[c], d[c]) >= config.MINORITY_SHARE]
    a68 = (X68 >= 4).mean()
    d68 = (X68 <= 2).mean()
    minority7 = [c for c in X68.columns
                 if min(a68[c], d68[c]) >= config.MINORITY_SHARE]
    ent = item_entropy(X87)
    entropy10 = list(ent.head(config.ENTROPY_TOPK).index)
    core8 = [c for c in minority10 if c in entropy10]
    return {"core8": core8, "minority10": minority10,
            "entropy10": entropy10, "minority7": minority7}


def priority_regressions(X: pd.DataFrame, prof: pd.DataFrame,
                         sets: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for set_name, items in sets.items():
        for d in config.DOMAINS:
            for item in items:
                r, p = pearsonr(prof[d], X[item])
                rows.append({"set": set_name, "predictor": f"{d}-priority",
                             "item": item, "r": float(r), "p": float(p)})
    df = pd.DataFrame(rows)
    for name, g in df.groupby("set", sort=False):
        rej, padj = bh_fdr(g["p"].to_numpy(), q=config.BH_Q)
        df.loc[g.index, "p_bh"] = padj
        df.loc[g.index, "rejected"] = rej
    return df


def main() -> None:
    config.ensure_dirs()
    X = pd.read_parquet(config.OUTPUTS / "clean.parquet").drop(
        columns=["row_position"], errors="ignore")
    all_resp = pd.read_parquet(config.OUTPUTS / "all_responses.parquet")

    sh = split_half_priorities(X)
    null_sd = sh["null_r_sd"].mean()
    z_overall = float((sh["r"].mean() - sh["null_r_mean"].mean()) / null_sd)
    sh_summary = (sh.groupby("domain")
                  .agg(mean_r=("r", "mean"), mean_z=("z", "mean"))
                  .reset_index())
    sh_summary.to_csv(config.OUTPUTS / "validity.csv", index=False)

    held = held_out_prediction(X)

    sets = contested_sets(all_resp)
    prof = pd.read_csv(config.OUTPUTS / "profiles.csv", index_col="response_id")
    reg = priority_regressions(X, prof, sets)
    reg = reg[["set", "predictor", "item", "r", "p", "p_bh", "rejected"]]
    reg.to_csv(config.OUTPUTS / "prediction.csv", index=False)

    save_json({
        "contested_sets": sets,
        "set_definitions": NAMES,
        "overall_split_half_z": float(sh["z"].mean()),
        "split_half_null_z": z_overall,
        "held_out": held,
        "n_rejected_per_set": {
            s: int(g["rejected"].sum()) for s, g in reg.groupby("set")},
    }, config.OUTPUTS / "validity_info.json")

    print("contested sets:", {k: v for k, v in sets.items()})
    print(sh_summary.round(3).to_string(index=False))
    print(f"overall split-half Fisher z: {sh['z'].mean():+.2f}  "
          f"null z: {z_overall:+.2f}")
    print("held-out r:", {h['held_out']: round(h['r'], 3) for h in held})
    sig = reg[reg["rejected"]]
    print(f"significant after BH: {len(sig)} / {len(reg)}")
    if len(sig):
        print(sig.sort_values("r", key=abs, ascending=False)
              .head(12).round(3).to_string(index=False))

    _plot(reg, sets)


def _plot(reg, sets):
    import matplotlib.pyplot as plt
    from rn import viz

    viz.set_style()
    sig = (reg[reg["rejected"]]
           .groupby(["predictor", "item"], as_index=False)
           .agg(r=("r", "first"), p_bh=("p_bh", "min"),
                n_sets=("set", "count")))
    n_sets_total = len(sets)
    sig = sig.sort_values("r")
    labels = [f"{it} ← {pr[0]}" for pr, it in zip(sig["predictor"], sig["item"])]

    fig, ax = plt.subplots(figsize=(7, 0.42 * len(sig) + 1.6))
    colors = viz.magma_ramp(1.0 - sig["p_bh"] / sig["p_bh"].max())
    ax.hlines(range(len(sig)), 0, sig["r"], color="#C9C9C9", lw=1.4, zorder=1)
    ax.scatter(sig["r"], range(len(sig)), s=52, c=colors,
               edgecolors="#2B2B2B", lw=0.5, zorder=3)
    ax.axvline(0, color="#C9C9C9", lw=0.9)
    for y, (_, row) in enumerate(sig.iterrows()):
        ax.annotate(f"{row['n_sets']}/{n_sets_total}", xy=(row["r"], y),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=7.5, color="#999999")
    ax.set_yticks(range(len(sig)), labels)
    ax.set_xlabel("Pearson r  (BH q = 0.05 survivors; annotation = sets containing item)")
    ax.set_title("What domain priorities predict on contested items")
    viz.savefig(fig, "09_prediction.png")


if __name__ == "__main__":
    main()
