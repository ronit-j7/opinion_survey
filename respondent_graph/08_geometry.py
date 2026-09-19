"""Step 08 - the geometry data contract and the static ternary renderer.

The ipsative profile has 3 free dimensions (rows sum to 0), so the simplex
view is lossless. Node positions come from the simplex; edges come from W1
(MST + mutual-kNN), NOT from the positions - visible long edges are the
proof that W1 is not the mean profile.

In:  outputs/profiles.csv, outputs/W1.npy, outputs/extremity.csv,
     outputs/clean.parquet, outputs/outliers.csv
Out: outputs/geometry.json
     figures/08_ternary_panel.png, figures/08_graph_fr.png
"""
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.manifold import MDS

from rn import config
from rn.graphs import build_graph
from rn.io import load_npy, save_json
from rn.stats import spearman_r

TETRA = np.array([
    [1.0, 1.0, 1.0],
    [1.0, -1.0, -1.0],
    [-1.0, 1.0, -1.0],
    [-1.0, -1.0, 1.0],
]) / np.sqrt(3.0)


def main() -> None:
    config.ensure_dirs()
    X = pd.read_parquet(config.OUTPUTS / "clean.parquet")
    row_pos = X["row_position"]
    Xi = X.drop(columns=["row_position"])
    W1 = load_npy(config.OUTPUTS / "W1.npy")
    ids = list(Xi.index)

    prof = pd.read_csv(config.OUTPUTS / "profiles.csv", index_col="response_id")
    prof = prof.loc[ids]
    ext = pd.read_csv(config.OUTPUTS / "extremity.csv", index_col="response_id")
    ext = ext.loc[ids, "extremity"]
    try:
        outl = set(pd.read_csv(config.OUTPUTS / "outliers.csv")["response_id"])
    except FileNotFoundError:
        outl = set()

    W = prof[list(config.DOMAINS)].to_numpy() + 0.25
    xyz = W @ TETRA
    dom = prof[list(config.DOMAINS)].idxmax(axis=1)

    G = build_graph(W1, ids, k=config.K_MUTUAL)
    edges = [{"source": int(u), "target": int(v),
              "w1": float(d["w1"]), "weight": float(d["weight"])}
             for u, v, d in G.edges(data=True)]

    pos = nx.spring_layout(G, weight="weight", seed=config.SEED)

    mds = MDS(n_components=3, dissimilarity="precomputed", n_init=4,
              random_state=config.SEED)
    mds.fit_transform(W1)

    simplex_dist = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    geo = {
        "vertices": [{"domain": d, "xyz": [float(c) for c in TETRA[i]]}
                     for i, d in enumerate(config.DOMAINS)],
        "nodes": [{
            "id": int(i),
            "row_position": int(row_pos.loc[i]),
            "xyz": [float(c) for c in xyz[j]],
            "domain": str(dom.iloc[j]),
            "extremity": float(ext.iloc[j]),
            "is_outlier": bool(int(i) in outl),
            "profile": [float(prof.iloc[j][d]) for d in config.DOMAINS],
        } for j, i in enumerate(ids)],
        "edges": edges,
        "layout_fr": [{"id": int(i), "xy": [float(pos[i][0]), float(pos[i][1])]}
                      for i in ids],
        "meta": {
            "n": len(ids),
            "k_mutual": config.K_MUTUAL,
            "edge_rule": "MST union mutual-kNN on W1",
            "mds_stress_3d": float(mds.stress_),
            "rho_w1_vs_simplex": spearman_r(W1, simplex_dist),
        },
    }
    save_json(geo, config.OUTPUTS / "geometry.json")

    _ternary_panel(geo, prof, ext, dom, outl)
    _fr_graph(G, ids, dom, outl, ext)

    m = geo["meta"]
    print(f"nodes {m['n']}  edges {len(edges)}  mds_stress_3d "
          f"{m['mds_stress_3d']:.1f}  rho(W1, simplex) {m['rho_w1_vs_simplex']:.3f}")


FACES = [("T", "E", "S"), ("T", "E", "V"), ("T", "S", "V"), ("E", "S", "V")]


def _cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _inside_triangle(p, tri):
    v0, v1, v2 = tri
    d1 = _cross2(v1 - v0, p - v0)
    d2 = _cross2(v2 - v1, p - v1)
    d3 = _cross2(v0 - v2, p - v2)
    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return ~(has_neg & has_pos)


def _ternary_panel(geo, prof, ext, dom, outl):
    import matplotlib.pyplot as plt
    from rn import viz

    viz.set_style()
    ids = [n["id"] for n in geo["nodes"]]
    fig, axes = plt.subplots(2, 2, figsize=(11, 10.5))
    tri = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])
    centroid = tri.mean(axis=0)
    doms = list(config.DOMAINS)
    W4 = prof[doms].to_numpy() + 0.25
    label_offsets = [(-10, -14, "right"), (10, -14, "left"), (0, 9, "center")]

    for ax, face in zip(axes.ravel(), FACES):
        fidx = [doms.index(d) for d in face]
        didx = next(i for i in range(4) if i not in fidx)
        pos2 = W4[:, fidx] @ tri + W4[:, [didx]] * centroid
        inside = _inside_triangle(pos2, tri)

        ax.add_patch(plt.Polygon(tri, fill=False, color="#C9C9C9", lw=1.1,
                                 zorder=1))
        for e in geo["edges"]:
            u, v = e["source"], e["target"]
            iu, iv = ids.index(u), ids.index(v)
            ax.plot([pos2[iu, 0], pos2[iv, 0]], [pos2[iu, 1], pos2[iv, 1]],
                    color="#9A9A9A", lw=0.6, alpha=0.30, zorder=1)

        for j, i in enumerate(ids):
            color = config.THEME_COLORS[dom.iloc[j]]
            size = 16 + 30 * (ext.iloc[j] / ext.max())
            if i in outl:
                ax.scatter(*pos2[j], s=size + 16, c=color, marker="X",
                           edgecolors="#C44530", lw=1.8, zorder=4)
            elif inside[j]:
                ax.scatter(*pos2[j], s=size, c=color, alpha=0.85,
                           edgecolors="#2B2B2B", lw=0.4, zorder=3)
            else:
                ax.scatter(*pos2[j], s=size, facecolors="none",
                           edgecolors=color, lw=1.2, zorder=3)

        pad = 0.16
        ax.set_xlim(min(tri[:, 0].min(), pos2[:, 0].min()) - pad,
                    max(tri[:, 0].max(), pos2[:, 0].max()) + pad)
        ax.set_ylim(min(tri[:, 1].min(), pos2[:, 1].min()) - pad,
                    max(tri[:, 1].max(), pos2[:, 1].max()) + pad)
        ax.set_aspect("equal")
        ax.axis("off")
        for k, d in enumerate(face):
            dx, dy, ha = label_offsets[k]
            ax.annotate(d, xy=tri[k], xytext=(dx, dy),
                        textcoords="offset points", ha=ha, va="center",
                        fontsize=11, color="#777777")
        ax.set_title(f"{'-'.join(face)} · excludes {doms[didx]}", fontsize=10)

    fig.suptitle("Opinion simplex: positions from ipsative profile, "
                 "edges from W1", y=0.99, color="#3A3A3A")
    fig.text(0.5, 0.015,
             "filled = inside the simplex hull · hollow = one-sided profile "
             "outside the hull · X (red ring) = structural outlier",
             ha="center", fontsize=8.5, color="#999999")
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    viz.savefig(fig, "08_ternary_panel.png")


def _fr_graph(G, ids, dom, outl, ext):
    import matplotlib.pyplot as plt
    import networkx as nx
    from rn import viz

    viz.set_style()
    fig, ax = plt.subplots(figsize=(9, 8))
    pos = nx.spring_layout(G, weight="weight", seed=config.SEED)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#9A9A9A",
                           alpha=0.35, width=0.9)
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=[config.THEME_COLORS[dom.loc[i]] for i in G.nodes()],
        node_size=[46 + 130 * (ext.loc[i] / ext.max()) for i in G.nodes()],
        alpha=0.88, edgecolors=["#C44530" if i in outl else "#2B2B2B"
                                for i in G.nodes()],
        linewidths=[1.6 if i in outl else 0.4 for i in G.nodes()])
    ax.set_title("Respondent graph (MST + mutual-kNN on W1)")
    ax.axis("off")
    fig.tight_layout()
    viz.savefig(fig, "08_graph_fr.png")


if __name__ == "__main__":
    main()
