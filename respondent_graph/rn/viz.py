"""Shared figure style: seaborn white theme, magma palette, translucency.

Every plotting function calls set_style() first and savefig() last, so the
whole figure family stays in one palette: domain identity = 4 evenly spaced
magma samples, continuous quantities = full magma ramp, translucency
everywhere.
"""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import seaborn as sns  # noqa: E402

from rn import config  # noqa: E402

DOMAIN_FRACS = {"T": 0.20, "E": 0.42, "S": 0.64, "V": 0.86}

FILL_ALPHA = 0.45
LINE_ALPHA = 0.75
LINK_ALPHA = 0.35
NEUTRAL = "#8A8A8A"

_RC = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#C9C9C9",
    "axes.linewidth": 0.8,
    "axes.grid": False,
    "font.size": 9.5,
    "axes.titlesize": 10,
    "axes.titleweight": "regular",
    "axes.labelsize": 9.5,
    "axes.labelcolor": "#444444",
    "xtick.color": "#777777",
    "ytick.color": "#777777",
    "text.color": "#3A3A3A",
    "legend.frameon": False,
}


def set_style() -> None:
    sns.set_theme(style="white", context="notebook", rc=_RC)


def magma(frac: float) -> str:
    """Magma sample as a hex color."""
    return matplotlib.colors.to_hex(plt.cm.magma(frac))


def magma_ramp(values, frac_lo: float = 0.15, frac_hi: float = 0.92):
    """Map values to magma hex colors on a normalized ramp."""
    values = np.asarray(values, dtype=float)
    lo, hi = values.min(), values.max()
    if hi <= lo:
        return [magma(0.5 * (frac_lo + frac_hi))] * len(values)
    t = (values - lo) / (hi - lo)
    return [magma(frac_lo + (frac_hi - frac_lo) * v) for v in t]


def domain_color(domain: str) -> str:
    return config.THEME_COLORS[domain]


def savefig(fig, name: str) -> None:
    config.ensure_dirs()
    fig.savefig(config.FIGURES / name, dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
