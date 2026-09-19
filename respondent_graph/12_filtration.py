"""Step 12 - filtration timelapse: the epsilon-sweep of the graph edges.

Left panel: edges materialize as epsilon crosses their W1 value; nodes
recolour by connected component. Right panel: the H0 barcode with an
epsilon cursor. Positions are FIXED across frames (comparability).

Outputs:
  figures/interactive/filtration.html   scrub epsilon with a slider (+ play)
  figures/12_filtration.mp4             video via ffmpeg
  figures/12_filtration_frames.png      static key-frame strip for the PDF

In:  outputs/W1.npy, outputs/geometry.json, outputs/dgms.pkl,
     outputs/dendrogram.json
"""
import json

import numpy as np
import pandas as pd

from rn import config, viz
from rn.graphs import build_graph
from rn.io import load_json, load_npy, load_pickle

N_FRAMES = 60
PALETTE = ["#3B0F6F", "#942B80", "#EA5560", "#FEBC82", "#5C156E",
           "#B93C7A", "#F97A57", "#FED3A1", "#251155", "#7A2080"]
GREY = "#9A9A9A"


def _frames(W1, ids, edges_list, deaths):
    ew = np.array([e[2] for e in edges_list])
    eps_grid = np.unique(np.quantile(ew, np.linspace(0, 1, N_FRAMES)))
    eps_grid = np.append(eps_grid, ew.max() * 1.05)
    frames = []
    for eps in eps_grid:
        parent = list(range(len(ids)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        active = []
        for k, (a, b, w) in enumerate(edges_list):
            if w <= eps:
                active.append(k)
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
        comps = {}
        comp_idx = []
        for a in range(len(ids)):
            r = find(a)
            if r not in comps:
                comps[r] = len(comps)
            comp_idx.append(comps[r])
        frames.append({"eps": float(eps), "active": active,
                       "comp": comp_idx, "nComp": len(set(comp_idx))})
    return frames, deaths


def _shared(W1, ids):
    geo = load_json(config.OUTPUTS / "geometry.json")
    layout = {d["id"]: d["xy"] for d in geo["layout_fr"]}
    xs = np.array([layout[i][0] for i in ids])
    ys = np.array([layout[i][1] for i in ids])
    xs = 0.08 + 0.84 * (xs - xs.min()) / (xs.max() - xs.min() + 1e-12)
    ys = 0.08 + 0.84 * (ys - ys.min()) / (ys.max() - ys.min() + 1e-12)
    G = build_graph(W1, ids)
    pos = {int(i): k for k, i in enumerate(ids)}
    edges_list = [(pos[u], pos[v], d["w1"]) for u, v, d in G.edges(data=True)]
    dgms = load_pickle(config.OUTPUTS / "dgms.pkl")
    d0 = dgms["w1"][0]
    deaths = np.sort(d0[np.isfinite(d0[:, 1])][:, 1])
    dendro = load_json(config.OUTPUTS / "dendrogram.json")
    cut = dendro["significant_gaps"][-1]["cut"] if dendro["significant_gaps"] else None
    return xs, ys, edges_list, deaths, cut


def build_html(frames, nodes_js, edges_js, deaths, cut, path):
    html = FILT_TEMPLATE.replace(
        "__DATA__",
        json.dumps({"frames": frames, "nodes": nodes_js, "edges": edges_js,
                    "deaths": [float(d) for d in deaths],
                    "cut": cut, "palette": PALETTE, "grey": GREY}))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)


def _mp4(frames, xs, ys, edges_list, ids, deaths, cut):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    viz.set_style()
    emax = deaths.max() * 1.05
    fig, (ax_n, ax_b) = plt.subplots(
        1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})

    def draw(fidx):
        f = frames[fidx]
        ax_n.clear()
        ax_b.clear()
        for k in f["active"]:
            a, b, _ = edges_list[k]
            ax_n.plot([xs[a], xs[b]], [ys[a], ys[b]], color=GREY, lw=0.8,
                      alpha=0.35, zorder=1)
        cols = [PALETTE[c % len(PALETTE)] for c in f["comp"]]
        ax_n.scatter(xs, ys, s=16, c=cols, edgecolors="#2B2B2B", lw=0.3,
                     zorder=2, alpha=0.9)
        ax_n.set_title(f"ε = {f['eps']:.2f} · {f['nComp']} components")
        ax_n.set_xlim(0, 1); ax_n.set_ylim(0, 1)
        ax_n.set_xticks([]); ax_n.set_yticks([])

        lengths = deaths
        bar_cols = viz.magma_ramp(lengths)
        ax_b.hlines(np.arange(len(deaths)), 0, deaths, color=bar_cols,
                    lw=2.0, alpha=0.9)
        ax_b.axvline(f["eps"], color="#C44530", lw=1.4)
        if cut:
            ax_b.axvline(cut, color="#9A9A9A", ls="--", lw=0.9)
        ax_b.set_xlim(0, emax)
        ax_b.set_title("H0 barcode (red: ε cursor)")
        ax_b.set_xlabel("merge height")

    ani = FuncAnimation(fig, draw, frames=len(frames), interval=180)
    writer = FFMpegWriter(fps=8, bitrate=2400)
    fig.tight_layout()
    ani.save(config.FIGURES / "12_filtration.mp4", writer=writer)
    plt.close(fig)


def _frames_strip(frames, xs, ys, edges_list, deaths):
    import matplotlib.pyplot as plt

    viz.set_style()
    qs = [0.25, 0.50, 0.75, 1.0]
    ew = np.array([e[2] for e in edges_list])
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.6))
    for ax, q in zip(axes, qs):
        eps = float(np.quantile(ew, q))
        f = min(frames, key=lambda fr: abs(fr["eps"] - eps))
        for k in f["active"]:
            a, b, _ = edges_list[k]
            ax.plot([xs[a], xs[b]], [ys[a], ys[b]], color=GREY, lw=0.8,
                    alpha=0.35, zorder=1)
        cols = [PALETTE[c % len(PALETTE)] for c in f["comp"]]
        ax.scatter(xs, ys, s=13, c=cols, edgecolors="#2B2B2B", lw=0.3,
                   zorder=2, alpha=0.9)
        ax.set_title(f"ε = {f['eps']:.2f} · {f['nComp']} components")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Filtration key frames (25/50/75/100th pct of edge W1)", y=1.02)
    viz.savefig(fig, "12_filtration_frames.png")


FILT_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Filtration timelapse</title>
<style>
  html,body{margin:0;height:100%;background:#FAFAFA;color:#3A3A3A;
    font:13px/1.45 -apple-system,Segoe UI,Helvetica,Arial,sans-serif}
  #bar{display:flex;align-items:center;gap:14px;padding:10px 16px;
    border-bottom:1px solid #E4E4E4;background:#fff}
  .spacer{flex:1}
  button{border:1px solid #D9D9D9;background:#fff;border-radius:6px;
    padding:5px 12px;font:inherit;cursor:pointer}
  button:hover{border-color:#942B80;color:#942B80}
  #wrap{display:flex;gap:10px;height:calc(100% - 88px);padding:10px}
  canvas{background:#fff;border:1px solid #EAEAEA;border-radius:8px}
  #ctrl{display:flex;align-items:center;gap:12px;padding:8px 16px}
  input[type=range]{flex:1;accent-color:#942B80}
  #lab{font-variant-numeric:tabular-nums;min-width:230px}
</style></head><body>
<div id="bar"><b>Filtration timelapse</b>
  <span style="color:#9A9A9A;font-size:12px">edges appear as ε crosses their W1 value</span>
  <span class="spacer"></span><button id="play">play</button></div>
<div id="wrap">
  <canvas id="net"></canvas><canvas id="barc"></canvas></div>
<div id="ctrl"><span>ε</span><input type="range" id="sl" min="0" value="0" step="1">
  <span id="lab"></span></div>
<script>
const D = __DATA__;
const net = document.getElementById('net'), nx = net.getContext('2d');
const barc = document.getElementById('barc'), bx = barc.getContext('2d');
function size(c){ const dpr = window.devicePixelRatio||1, w = c.clientWidth,
  h = c.clientHeight; c.width = w*dpr; c.height = h*dpr;
  c.getContext('2d').setTransform(dpr,0,0,dpr,0,0); return [w,h]; }
let NW, NH, BW, BH;
function resize(){ [NW,NH] = size(net); [BW,BH] = size(barc); render(); }
const emax = Math.max(...D.deaths) * 1.05;
function render(){
  const f = D.frames[+sl.value];
  nx.clearRect(0,0,NW,NH);
  const M = 24;
  const X = i => M + D.nodes[i][0]*(NW-2*M), Y = i => M + D.nodes[i][1]*(NH-2*M);
  nx.strokeStyle = D.grey;
  for (const k of f.active){
    const [a,b] = D.edges[k];
    nx.globalAlpha = 0.30; nx.lineWidth = 0.9;
    nx.beginPath(); nx.moveTo(X(a),Y(a)); nx.lineTo(X(b),Y(b)); nx.stroke();
  }
  nx.globalAlpha = 0.92;
  D.nodes.forEach((n,i) => {
    nx.beginPath(); nx.arc(X(i), Y(i), 5, 0, 6.2832);
    nx.fillStyle = D.palette[f.comp[i] % D.palette.length]; nx.fill();
    nx.lineWidth = 0.5; nx.strokeStyle = '#2B2B2B'; nx.stroke();
  });
  nx.globalAlpha = 1;

  bx.clearRect(0,0,BW,BH);
  const L = 14, R = BW-16, T = 10, Hh = BH-30;
  const sx = v => L + (R-L)*v/emax;
  const step = Math.max(1, Math.ceil(D.deaths.length/(Hh/6)));
  D.deaths.forEach((d,i) => {
    if (i % step) return;
    const y = T + (i/(D.deaths.length))*(Hh-8)+4;
    bx.strokeStyle = D.palette[0]; bx.globalAlpha = 0.25+0.65*(d/emax);
    bx.lineWidth = 2;
    bx.beginPath(); bx.moveTo(sx(0), y); bx.lineTo(sx(d), y); bx.stroke();
  });
  bx.globalAlpha = 1;
  if (D.cut != null){
    bx.strokeStyle = '#9A9A9A'; bx.setLineDash([4,3]); bx.lineWidth = 0.9;
    bx.beginPath(); bx.moveTo(sx(D.cut), T); bx.lineTo(sx(D.cut), T+Hh); bx.stroke();
    bx.setLineDash([]);
  }
  bx.strokeStyle = '#C44530'; bx.lineWidth = 1.4;
  bx.beginPath(); bx.moveTo(sx(f.eps), T); bx.lineTo(sx(f.eps), T+Hh); bx.stroke();
  bx.fillStyle = '#777'; bx.font = '11px sans-serif';
  bx.fillText('H0 barcode (merge height)', L, BH-8);
  lab.textContent = `ε = ${f.eps.toFixed(3)} · ${f.nComp} components`;
}
const sl = document.getElementById('sl'), lab = document.getElementById('lab');
sl.max = D.frames.length - 1;
sl.oninput = render;
let timer = null;
document.getElementById('play').onclick = function(){
  if (timer){ clearInterval(timer); timer = null; this.textContent = 'play'; return; }
  this.textContent = 'pause';
  timer = setInterval(() => {
    sl.value = (+sl.value + 1) % D.frames.length; render();
  }, 160);
};
window.addEventListener('resize', resize);
resize();
</script></body></html>
"""


def main() -> None:
    config.ensure_dirs()
    W1 = load_npy(config.OUTPUTS / "W1.npy")
    X = pd.read_parquet(config.OUTPUTS / "clean.parquet")
    ids = [int(i) for i in X.index]
    xs, ys, edges_list, deaths, cut = _shared(W1, ids)
    frames, _ = _frames(W1, ids, edges_list, deaths)

    nodes_js = [[float(a), float(b)] for a, b in zip(xs, ys)]
    edges_js = [[int(a), int(b)] for a, b, _ in edges_list]
    build_html(frames, nodes_js, edges_js, deaths, cut,
               config.FIGURES / "interactive" / "filtration.html")

    _mp4(frames, xs, ys, edges_list, ids, deaths, cut)
    _frames_strip(frames, xs, ys, edges_list, deaths)
    print(f"wrote figures/interactive/filtration.html ({len(frames)} frames), "
          "figures/12_filtration.mp4, figures/12_filtration_frames.png")


if __name__ == "__main__":
    main()
