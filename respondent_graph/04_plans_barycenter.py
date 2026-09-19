"""Step 04 - transport plans and Wasserstein barycenters.

In:  outputs/measures.npy, outputs/W1.npy
Out: outputs/flows.npy       (4, 5, 5) mean transport plan per domain
     outputs/barycenters.csv  (4, 5) barycenter mass per domain
     outputs/flows_info.json  band vs midpoint-crossing summary
     figures/04_flows.png
"""
import numpy as np
import pandas as pd

from rn import config
from rn.io import load_npy, save_json, save_npy
from rn.w1 import barycenter_domain, flow_summary, mean_flow

CATS = ["SD", "D", "N", "A", "SA"]


def main() -> None:
    config.ensure_dirs()
    hist = load_npy(config.OUTPUTS / "measures.npy")

    flows = mean_flow(hist)
    bary = barycenter_domain(hist)

    save_npy(flows, config.OUTPUTS / "flows.npy")
    pd.DataFrame(bary, index=list(config.DOMAINS), columns=CATS).round(4).to_csv(
        config.OUTPUTS / "barycenters.csv", index_label="domain")

    summary = {
        d: flow_summary(flows[i]) for i, d in enumerate(config.DOMAINS)
    }
    summary["barycenters"] = {
        d: {CATS[c]: float(bary[i, c]) for c in range(5)}
        for i, d in enumerate(config.DOMAINS)
    }
    save_json(summary, config.OUTPUTS / "flows_info.json")

    _plot(flows, bary, hist)

    for i, d in enumerate(config.DOMAINS):
        s = summary[d]
        print(f"[{d}] band(N-A-SA) {s['neutral_agree_sa_band']:.3f}  "
              f"midpoint-crossing {s['midpoint_crossing']:.3f}  "
              f"barycenter mode {CATS[int(np.argmax(bary[i]))]}")


def _plot(flows, bary, hist):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import PowerNorm
    from rn import viz

    viz.set_style()
    n_cat = len(CATS)
    fig, axes = plt.subplots(
        3, 4, figsize=(14, 12.5),
        gridspec_kw={"height_ratios": [0.85, 1.15, 1.30],
                     "hspace": 0.30, "wspace": 0.24})

    for i, d in enumerate(config.DOMAINS):
        ax_top, ax_mid, ax_hm = axes[0, i], axes[1, i], axes[2, i]
        f = flows[i]
        band = flow_summary(f)
        mean_share = hist[:, i, :].mean(axis=0)

        stem_colors = viz.magma_ramp(bary[i])
        ax_top.bar(range(n_cat), mean_share, width=0.62, color="#BDBDBD",
                   alpha=0.35, zorder=1)
        ax_top.vlines(range(n_cat), 0, bary[i], color=stem_colors, lw=1.8,
                      alpha=0.9)
        ax_top.scatter(range(n_cat), bary[i], s=42, c=stem_colors,
                       edgecolors="#2B2B2B", lw=0.5, zorder=3)
        ax_top.set_ylim(0, max(bary[i].max() * 1.3, mean_share.max() * 1.25))
        ax_top.set_title(f"{d} · {config.THEMES[d]}",
                         color=config.THEME_COLORS[d])
        if i == 0:
            ax_top.set_ylabel("barycenter mass\n(grey: class mean)")
        ax_top.set_xticks(range(n_cat), [""] * n_cat)

        offdiag = [(c1, c2, f[c1, c2]) for c1 in range(n_cat)
                   for c2 in range(n_cat) if c1 < c2 and f[c1, c2] > 5e-4]
        diag = [(c, f[c, c]) for c in range(n_cat) if f[c, c] > 5e-4]
        if offdiag:
            vals = np.array([v for *_, v in offdiag])
            cols = viz.magma_ramp(vals)
            for (c1, c2, v), col in zip(offdiag, cols):
                t = np.linspace(0, 1, 40)
                h = 0.25 + 0.55 * (abs(c2 - c1) / (n_cat - 1))
                x = c1 + (c2 - c1) * t
                y = h * np.sin(np.pi * t)
                ax_mid.plot(x, y, color=col,
                            alpha=0.45 + 0.45 * (v / vals.max()),
                            lw=1.2 + 7.8 * (v / vals.max()),
                            solid_capstyle="round")
        if diag:
            dmax = max(v for _, v in diag)
            for c, v in diag:
                ax_mid.plot([c, c], [0, 0.10 + 0.5 * v / dmax],
                            color=viz.magma(0.75),
                            alpha=0.50 + 0.35 * (v / dmax),
                            lw=1.0 + 6.0 * (v / dmax),
                            solid_capstyle="round")
        ax_mid.scatter(range(n_cat), [0] * n_cat, s=10, c="#2B2B2B", zorder=3)
        ax_mid.axvline(2.0, color="#C9C9C9", ls=":", lw=0.9)
        ax_mid.set_xticks(range(n_cat), CATS)
        ax_mid.set_ylim(-0.06, 1.0)
        ax_mid.set_yticks([])
        if i == 0:
            ax_mid.set_ylabel("mean transport plan\n(off-diagonal scaled)")

        im = ax_hm.imshow(f, cmap="magma",
                          norm=PowerNorm(gamma=0.45, vmin=0, vmax=f.max()))
        for c1 in range(n_cat):
            for c2 in range(n_cat):
                v = f[c1, c2]
                if v < 5e-4:
                    continue
                ax_hm.text(c2, c1, f"{v:.3f}", ha="center", va="center",
                           fontsize=7.5,
                           color="white" if v > 0.35 * f.max() else "#555555")
        ax_hm.set_xticks(range(n_cat), CATS)
        ax_hm.set_yticks(range(n_cat), CATS)
        ax_hm.set_title(f"plan matrix · band {band['neutral_agree_sa_band']:.2f}"
                        f" · cross {band['midpoint_crossing']:.2f}",
                        fontsize=9)
        if i == 0:
            ax_hm.set_ylabel("category -> category")

    fig.suptitle("Where the class varies: barycenter stance (top), transport "
                 "flows (middle), plan matrices (bottom; colour gamma-scaled)",
                 y=0.995)
    viz.savefig(fig, "04_flows.png")


if __name__ == "__main__":
    main()
