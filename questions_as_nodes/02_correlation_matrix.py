"""Step 02 - pairwise Spearman correlations, converted to the latent Pearson scale.

In:  outputs/01_responses.csv
Out: outputs/02_spearman.csv      raw pairwise Spearman r^S
     outputs/02_R.csv             2 sin(pi r^S / 6), repaired if not positive definite (glasso input)
     outputs/02_pairwise_n.csv    n_jk = respondents who answered both items
     outputs/02_info.json         eigenvalue check, repair flag, pairwise n range
"""
import numpy as np

from bn import config
from bn.correlation import estimate_correlation
from bn.files import read_responses, save_json, save_matrix


def main() -> None:
    config.ensure_dirs()
    X = read_responses(config.OUTPUTS / "01_responses.csv")
    res = estimate_correlation(X)

    save_matrix(res.spearman, config.OUTPUTS / "02_spearman.csv")
    save_matrix(res.R, config.OUTPUTS / "02_R.csv")
    save_matrix(res.n_pairwise, config.OUTPUTS / "02_pairwise_n.csv")

    iu = np.triu_indices(X.shape[1], 1)
    n_pairs = res.n_pairwise.to_numpy()[iu]
    r_offdiag = res.R.to_numpy()[iu]
    info = {
        "n_items": X.shape[1],
        "pairwise_n_min": n_pairs.min(),
        "pairwise_n_median": float(np.median(n_pairs)),
        "pairwise_n_max": n_pairs.max(),
        "n_undefined_pairs": res.n_undefined_pairs,
        "min_eigenvalue_before_repair": res.min_eig_before,
        "min_eigenvalue_after_repair": res.min_eig_after,
        "repaired": res.repaired,
        "eig_floor": config.EIG_FLOOR,
        "mean_abs_r": float(np.abs(r_offdiag).mean()),
        "share_negative_r": float((r_offdiag < 0).mean()),
    }
    save_json(info, config.OUTPUTS / "02_info.json")

    print(f"Pairwise n (min / median / max): {info['pairwise_n_min']} / "
          f"{info['pairwise_n_median']:g} / {info['pairwise_n_max']}")
    print(f"Undefined pairs (set to 0):      {info['n_undefined_pairs']}")
    print(f"Min eigenvalue before repair:    {res.min_eig_before:.3e}")
    print(f"Repaired:                        {res.repaired}"
          + (f" (min eigenvalue now {res.min_eig_after:.3e})" if res.repaired else ""))
    print(f"Mean |r|, share negative:        {info['mean_abs_r']:.3f}, {info['share_negative_r']:.3f}")


if __name__ == "__main__":
    main()
