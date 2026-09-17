"""Step 03: graphical lasso over a lambda grid, select lambda by EBIC, get partial correlations."""
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import graphical_lasso
from sklearn.exceptions import ConvergenceWarning

from bn import config


@dataclass
class GlassoFit:
    lam: float
    theta: np.ndarray       # selected precision matrix
    path: pd.DataFrame      # one row per lambda: lambda, n_edges, loglik, ebic, converged, failed
    n: float                # sample size used in the likelihood / EBIC
    gamma: float
    index: int              # position of the selected lambda in the grid (0 = largest)


def lambda_grid(R: np.ndarray, n_lambda: int, min_ratio: float) -> np.ndarray:
    """Log-spaced, descending from max |r_jk| (empty graph) to min_ratio * max."""
    p = R.shape[0]
    lam_max = np.abs(R[np.triu_indices(p, 1)]).max()
    return np.logspace(np.log10(lam_max), np.log10(lam_max * min_ratio), n_lambda)


def log_likelihood(R: np.ndarray, theta: np.ndarray, n: float) -> float:
    """(n/2) [log det Theta - tr(R Theta)], constants dropped."""
    sign, logdet = np.linalg.slogdet(theta)
    if sign <= 0:
        return -np.inf
    return 0.5 * n * (logdet - np.sum(R * theta))


def edge_count(theta: np.ndarray, tol: float) -> int:
    iu = np.triu_indices(theta.shape[0], 1)
    return int(np.count_nonzero(np.abs(theta[iu]) > tol))


def ebic(loglik: float, n_edges: int, n: float, p: int, gamma: float) -> float:
    return -2.0 * loglik + n_edges * np.log(n) + 4.0 * gamma * n_edges * np.log(p)


def fit_glasso(R: np.ndarray, lam: float, max_iter: int, tol: float) -> tuple[np.ndarray, bool]:
    """Solve max log det Theta - tr(R Theta) - lam * sum_{j!=k} |theta_jk|.

    scikit-learn penalises only the off-diagonal entries, matching the plan.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        _, theta = graphical_lasso(R, alpha=lam, max_iter=max_iter, tol=tol)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return (theta + theta.T) / 2.0, converged


def select_by_ebic(R: np.ndarray,
                   n: float,
                   gamma: float = config.EBIC_GAMMA,
                   n_lambda: int = config.N_LAMBDA,
                   min_ratio: float = config.LAMBDA_MIN_RATIO,
                   max_iter: int = config.GLASSO_MAX_ITER,
                   tol: float = config.GLASSO_TOL,
                   zero_tol: float = config.ZERO_TOL) -> GlassoFit:
    p = R.shape[0]
    rows, best = [], None

    for i, lam in enumerate(lambda_grid(R, n_lambda, min_ratio)):
        try:
            theta, converged = fit_glasso(R, lam, max_iter, tol)
        except (FloatingPointError, np.linalg.LinAlgError):
            rows.append({"lambda": lam, "n_edges": np.nan, "loglik": np.nan,
                         "ebic": np.nan, "converged": False, "failed": True})
            continue

        k = edge_count(theta, zero_tol)
        ll = log_likelihood(R, theta, n)
        score = ebic(ll, k, n, p, gamma)
        rows.append({"lambda": lam, "n_edges": k, "loglik": ll,
                     "ebic": score, "converged": converged, "failed": False})

        # strict '<' keeps the sparser (larger-lambda) solution on ties
        if np.isfinite(score) and (best is None or score < best[0]):
            best = (score, i, lam, theta)

    if best is None:
        raise RuntimeError("Graphical lasso failed for every lambda in the grid.")

    _, idx, lam, theta = best
    return GlassoFit(lam=lam, theta=theta, path=pd.DataFrame(rows), n=n, gamma=gamma, index=idx)


def partial_correlations(theta: np.ndarray, zero_tol: float = config.ZERO_TOL) -> np.ndarray:
    """omega_jk = -theta_jk / sqrt(theta_jj theta_kk); zero diagonal."""
    d = np.sqrt(np.diag(theta))
    omega = -theta / np.outer(d, d)
    omega[np.abs(omega) <= zero_tol] = 0.0
    np.fill_diagonal(omega, 0.0)
    return omega
