"""Step 03 - graphical lasso with EBIC model selection -> partial correlation matrix.

In:  outputs/02_R.csv, outputs/02_pairwise_n.csv
Out: outputs/03_ebic_path.csv       lambda, n_edges, loglik, ebic, converged, failed
     outputs/03_precision.csv       selected Theta
     outputs/03_partial_corr.csv    Omega (edge weights)
     outputs/03_selection.json      chosen lambda, n used in EBIC, gamma, warnings
     figures/03_ebic_curve.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bn import config
from bn.files import read_matrix, save_json, save_matrix
from bn.network import partial_correlations, select_by_ebic


def plot_ebic_path(path: pd.DataFrame, lam: float, out) -> None:
    ok = path[~path["failed"]]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)
    ax1.plot(ok["lambda"], ok["ebic"], color="#333333", lw=1.5)
    ax1.set_ylabel("EBIC")
    ax2.plot(ok["lambda"], ok["n_edges"], color="#333333", lw=1.5)
    ax2.set_ylabel("Number of edges")
    ax2.set_xlabel(r"$\lambda$ (log scale)")
    for ax in (ax1, ax2):
        ax.axvline(lam, color=config.NEGATIVE_EDGE_COLOR, ls="--", lw=1)
        ax.set_xscale("log")
        ax.grid(alpha=0.3)
    ax1.set_title(rf"EBIC path ($\gamma$ = {config.EBIC_GAMMA}); selected $\lambda$ = {lam:.4f}")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main() -> None:
    config.ensure_dirs()
    R = read_matrix(config.OUTPUTS / "02_R.csv")
    n_pairwise = read_matrix(config.OUTPUTS / "02_pairwise_n.csv")

    iu = np.triu_indices(len(R), 1)
    n_ebic = config.N_EBIC or float(np.median(n_pairwise.to_numpy()[iu]))

    fit = select_by_ebic(R.to_numpy(), n=n_ebic)
    omega = partial_correlations(fit.theta)

    fit.path.to_csv(config.OUTPUTS / "03_ebic_path.csv", index=False, float_format="%.10g")
    save_matrix(pd.DataFrame(fit.theta, index=R.index, columns=R.columns),
                config.OUTPUTS / "03_precision.csv")
    save_matrix(pd.DataFrame(omega, index=R.index, columns=R.columns),
                config.OUTPUTS / "03_partial_corr.csv")

    selected = fit.path.iloc[fit.index]
    at_boundary = fit.index in (0, len(fit.path) - 1)
    selection = {
        "lambda": fit.lam,
        "lambda_index": fit.index,
        "n_lambda": len(fit.path),
        "at_grid_boundary": at_boundary,
        "n_edges": int(selected["n_edges"]),
        "ebic": float(selected["ebic"]),
        "converged": bool(selected["converged"]),
        "n_used_in_ebic": n_ebic,
        "gamma": fit.gamma,
        "n_failed_fits": int(fit.path["failed"].sum()),
        "n_unconverged_fits": int((~fit.path["converged"] & ~fit.path["failed"]).sum()),
    }
    save_json(selection, config.OUTPUTS / "03_selection.json")
    plot_ebic_path(fit.path, fit.lam, config.FIGURES / "03_ebic_curve.png")

    print(f"n used in EBIC: {n_ebic:g}, gamma: {fit.gamma}")
    print(f"Selected lambda: {fit.lam:.5f} (grid index {fit.index} of {len(fit.path)})")
    print(f"Edges: {selection['n_edges']}, EBIC: {selection['ebic']:.2f}")
    print(f"Failed fits: {selection['n_failed_fits']}, unconverged fits: {selection['n_unconverged_fits']}")
    if at_boundary:
        print("WARNING: selected lambda is at the edge of the grid; widen LAMBDA_MIN_RATIO.")
    if not selection["converged"]:
        print("WARNING: the selected fit did not converge; raise GLASSO_MAX_ITER.")


if __name__ == "__main__":
    main()
