"""Step 02 - pairwise correlation matrix (method set by CORR_METHOD in lib/config.py).

In:  outputs/01_responses.csv
Out: outputs/02_R.csv             pairwise correlations r_jk
     outputs/02_pairwise_n.csv    n_jk = respondents who answered both items
     outputs/02_info.json         method, pairwise n range, summary of r
"""
import numpy as np

from lib import config
from lib.correlation import estimate_correlation
from lib.files import read_responses, save_json, save_matrix


def main() -> None:
    config.ensure_dirs()
    X = read_responses(config.OUTPUTS / "01_responses.csv")
    res = estimate_correlation(X)

    save_matrix(res.R, config.OUTPUTS / "02_R.csv")
    save_matrix(res.n_pairwise, config.OUTPUTS / "02_pairwise_n.csv")

    # Summarise over the upper triangle only (cells above the diagonal): the 1,770 item pairs,
    # each once. The full matrix would count every pair twice ((j,k) and (k,j)) and include the
    # diagonal (r = 1, n = item's own response count), which would distort the averages.
    iu = np.triu_indices(X.shape[1], 1)  #Get the upper triangular indices for 60*60 matrix to use on the R matrix. 
    n_pairs = res.n_pairwise.to_numpy()[iu]
    r_offdiag = res.R.to_numpy()[iu]
    info = {
        "method": config.CORR_METHOD,
        "n_items": X.shape[1],
        "pairwise_n_min": n_pairs.min(),
        "pairwise_n_median": float(np.median(n_pairs)),
        "pairwise_n_max": n_pairs.max(),
        "n_undefined_pairs": res.n_undefined_pairs,
        "min_eigenvalue": res.min_eigenvalue,
        "mean_abs_r": float(np.abs(r_offdiag).mean()),
        "share_negative_r": float((r_offdiag < 0).mean()),
    }
    save_json(info, config.OUTPUTS / "02_info.json")

    print(f"Method:                          {config.CORR_METHOD}")
    print(f"Pairwise n (min / median / max): {info['pairwise_n_min']} / "
          f"{info['pairwise_n_median']:g} / {info['pairwise_n_max']}")
    print(f"Undefined pairs (set to 0):      {info['n_undefined_pairs']}")
    print(f"Mean |r|, share negative:        {info['mean_abs_r']:.3f}, {info['share_negative_r']:.3f}")


if __name__ == "__main__":
    main()
