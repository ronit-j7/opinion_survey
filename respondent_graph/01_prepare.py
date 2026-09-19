"""Step 01 - encode answers, apply the filter cascade, emit diagnostics.

In:  ../Survey_Results_UC.csv
Out: outputs/clean.parquet        primary matrix (complete cases) + row_position
     outputs/all_responses.parquet full non-blank matrix for re-filtering
     outputs/items.csv            item table
     outputs/diagnostics.json     per-subset diagnostics (all / ge40 / complete)
"""
from rn import config
from rn.data import load_encoded, subset_stats, tier_mask
from rn.io import save_json

TIERS = ("all", "ge40", "complete")


def main() -> None:
    config.ensure_dirs()
    responses, items, info = load_encoded(config.DATA_CSV)

    diag = {"cascade": {t: int(tier_mask(responses, t).sum()) for t in TIERS}}
    diag["cascade"]["rows_in_file"] = info["n_rows_in_file"]
    diag["row_position"] = info["row_position"]
    for t in TIERS:
        X = responses.loc[tier_mask(responses, t)]
        diag[t] = subset_stats(X)
    diag["primary_filter"] = config.PRIMARY_FILTER
    diag["primary_n"] = diag["cascade"][config.PRIMARY_FILTER]

    primary = responses.loc[tier_mask(responses, config.PRIMARY_FILTER)]
    primary["row_position"] = [info["row_position"][str(i)] for i in primary.index]
    primary.to_parquet(config.OUTPUTS / "clean.parquet")
    responses.to_parquet(config.OUTPUTS / "all_responses.parquet")
    items.to_csv(config.OUTPUTS / "items.csv", index=False)
    save_json(diag, config.OUTPUTS / "diagnostics.json")

    for t in TIERS:
        s = diag[t]
        print(f"[{t:8s}] n={s['n']:3d}  agree-side {s['agree_side_share']:.3f}  "
              f"<5% dis. items {s['items_under_5pct_disagreement']:2d}  "
              f"E03 dis. {s['E03_disagree_share']:.2f}  E02 dis. {s['E02_disagree_share']:.2f}")
    print("primary filter:", config.PRIMARY_FILTER, "-> n =", diag["primary_n"])


if __name__ == "__main__":
    main()
