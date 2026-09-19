"""Step 11 - interactive displays.

Outputs (figures/interactive/):
- graph_2d.html     self-contained offline canvas + vanilla-JS force graph.
                    Settle-on-load, drag-to-perturb, freeze toggle; hover
                    cards carry the full node/edge payloads.
- simplex_3d.html   plotly 3D orbit view of the lossless tetrahedron
                    embedding (positions fixed by design - no physics).

Both consume outputs/geometry.json, so the viewer contract is unchanged.

In:  outputs/geometry.json
"""
import json

import numpy as np

from rn import config
from rn.io import load_json

MAGMA = {0.20: "#3B0F6F", 0.42: "#942B80", 0.64: "#EA5560", 0.86: "#FEBC82"}


def _payload(geo: dict) -> dict:
    ids = [n["id"] for n in geo["nodes"]]
    exts = [n["extremity"] for n in geo["nodes"]]
    emax = max(exts)
    layout = {d["id"]: d["xy"] for d in geo["layout_fr"]}
    xs = np.array([layout[i][0] for i in ids])
    ys = np.array([layout[i][1] for i in ids])

    def norm(v, lo=0.10, hi=0.90):
        v = (v - v.min()) / (v.max() - v.min() + 1e-12)
        return lo + v * (hi - lo)

    nodes = []
    for k, n in enumerate(geo["nodes"]):
        p = n["profile"]
        card = (f"<b>respondent {n['id']}</b><br>file row {n['row_position']}"
                f" · dominant {n['domain']}<br>extremity {n['extremity']:.2f}"
                f"<br>T {p[0]:+.2f} · E {p[1]:+.2f} · S {p[2]:+.2f} · V {p[3]:+.2f}")
        if n["is_outlier"]:
            card += "<br><b>STRUCTURAL OUTLIER</b>"
        nodes.append({
            "id": n["id"], "x": float(norm(xs)[k]), "y": float(norm(ys)[k]),
            "r": 5 + 9 * (n["extremity"] / emax),
            "color": config.THEME_COLORS[n["domain"]],
            "outlier": bool(n["is_outlier"]), "card": card,
        })
    idx = {i: k for k, i in enumerate(ids)}
    ws = [e["weight"] for e in geo["edges"]]
    wmin, wmax = min(ws), max(ws)
    edges = []
    for e in geo["edges"]:
        t = (e["weight"] - wmin) / (wmax - wmin + 1e-12)
        edges.append({
            "s": idx[e["source"]], "t": idx[e["target"]],
            "w1": e["w1"], "weight": e["weight"],
            "alpha": 0.12 + 0.55 * t, "width": 0.7 + 1.9 * t,
            "card": (f"<b>{e['source']} ↔ {e['target']}</b><br>"
                     f"W1 {e['w1']:.3f} · weight {e['weight']:.3f}"),
        })
    return {
        "nodes": nodes, "edges": edges,
        "meta": {"k": geo["meta"]["k_mutual"], "n": geo["meta"]["n"],
                 "rule": geo["meta"]["edge_rule"],
                 "domains": config.THEMES},
        "colors": config.THEME_COLORS,
    }


GRAPH_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Respondent graph</title>
<style>
  html,body{margin:0;height:100%;background:#FAFAFA;color:#3A3A3A;
    font:13px/1.45 -apple-system,Segoe UI,Helvetica,Arial,sans-serif}
  #bar{display:flex;align-items:center;gap:14px;padding:10px 16px;
    border-bottom:1px solid #E4E4E4;background:#fff}
  #bar b{font-weight:600}
  #bar .spacer{flex:1}
  button{border:1px solid #D9D9D9;background:#fff;border-radius:6px;
    padding:5px 12px;font:inherit;color:#3A3A3A;cursor:pointer}
  button:hover{border-color:#942B80;color:#942B80}
  #wrap{position:relative;height:calc(100% - 42px)}
  canvas{display:block;width:100%;height:100%}
  #card{position:absolute;display:none;pointer-events:none;background:#fff;
    border:1px solid #E0E0E0;border-radius:8px;padding:8px 11px;
    box-shadow:0 4px 14px rgba(0,0,0,.08);font-size:12px;max-width:260px}
  #legend{position:absolute;left:12px;bottom:10px;background:rgba(255,255,255,.85);
    border:1px solid #EAEAEA;border-radius:8px;padding:7px 11px;font-size:11.5px}
  .chip{display:inline-block;width:9px;height:9px;border-radius:50%;
    margin:0 4px 0 10px;vertical-align:-1px;border:1px solid #2B2B2B}
  #foot{position:absolute;right:12px;bottom:10px;color:#A0A0A0;font-size:11px}
</style></head><body>
<div id="bar"><b>Respondent graph</b>
  <span style="color:#9A9A9A;font-size:12px" id="subtitle"></span>
  <span class="spacer"></span>
  <button id="freeze">freeze</button><button id="reheat">reheat</button>
  <button id="reset">reset layout</button></div>
<div id="wrap"><canvas id="cv"></canvas><div id="card"></div>
  <div id="legend"></div><div id="foot">drag nodes · hover for info</div></div>
<script>
const DATA = __DATA__;
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
const card = document.getElementById('card');
let W, H, M = 46;
const nodes = DATA.nodes.map(n => ({...n, vx:0, vy:0, ix:n.x, iy:n.y}));
const edges = DATA.edges;

function resize(){
  const dpr = window.devicePixelRatio || 1;
  const r = cv.getBoundingClientRect();
  W = r.width; H = r.height;
  cv.width = W*dpr; cv.height = H*dpr; ctx.setTransform(dpr,0,0,dpr,0,0);
  nodes.forEach(n => { n.x = M + n.ix*(W-2*M); n.y = M + n.iy*(H-2*M);
                       n.vx = n.vy = 0; });
  running = true; draw();
}

let running = true, physicsOn = true, dragging = null;
const REP = 1600, SPR = 0.015, REST0 = 40, REST1 = 150, CEN = 0.0025,
      DAMP = 0.88, MAXV = 7, SETTLE = 0.5;

function step(){
  let energy = 0;
  for (let i = 0; i < nodes.length; i++){
    const a = nodes[i];
    let fx = (W/2 - a.x)*CEN, fy = (H/2 - a.y)*CEN;
    for (let j = 0; j < nodes.length; j++){
      if (i === j) continue;
      const b = nodes[j];
      let dx = a.x-b.x, dy = a.y-b.y;
      let d2 = dx*dx + dy*dy;
      if (d2 < 100) d2 = 100;
      const f = REP/d2, d = Math.sqrt(d2);
      fx += f*dx/d; fy += f*dy/d;
    }
    a.fx = fx; a.fy = fy;
  }
  for (const e of edges){
    const a = nodes[e.s], b = nodes[e.t];
    const dx = b.x-a.x, dy = b.y-a.y, d = Math.max(Math.hypot(dx,dy), 1);
    const rest = REST0 + REST1*(1-e.weight);
    const f = SPR*(d-rest);
    a.fx += f*dx/d; a.fy += f*dy/d; b.fx -= f*dx/d; b.fy -= f*dy/d;
  }
  for (const n of nodes){
    if (n === dragging) { n.vx = n.vy = 0; continue; }
    n.vx = (n.vx + n.fx)*DAMP; n.vy = (n.vy + n.fy)*DAMP;
    const v = Math.hypot(n.vx, n.vy);
    if (v > MAXV){ n.vx *= MAXV/v; n.vy *= MAXV/v; }
    n.x += n.vx; n.y += n.vy;
    energy += n.vx*n.vx + n.vy*n.vy;
  }
  if (energy < SETTLE) running = false;
}

let mouse = {x:-999, y:-999}, hoverNode = null, hoverEdge = null;
function pickNode(){
  let best = null, bd = 1e9;
  for (const n of nodes){
    const d = Math.hypot(mouse.x-n.x, mouse.y-n.y);
    if (d < n.r+5 && d < bd){ bd = d; best = n; }
  }
  return best;
}
function pickEdge(){
  let best = null, bd = 7;
  for (const e of edges){
    const a = nodes[e.s], b = nodes[e.t];
    const dx = b.x-a.x, dy = b.y-a.y, l2 = dx*dx+dy*dy;
    if (!l2) continue;
    let t = ((mouse.x-a.x)*dx + (mouse.y-a.y)*dy)/l2;
    t = Math.max(0, Math.min(1, t));
    const d = Math.hypot(mouse.x-(a.x+t*dx), mouse.y-(a.y+t*dy));
    if (d < bd){ bd = d; best = e; }
  }
  return best;
}

function draw(){
  ctx.clearRect(0, 0, W, H);
  const inc = new Set();
  if (hoverNode) for (const e of edges)
    if (nodes[e.s] === hoverNode || nodes[e.t] === hoverNode) inc.add(e);
  for (const e of edges){
    const a = nodes[e.s], b = nodes[e.t];
    const hot = inc.has(e) || e === hoverEdge;
    ctx.strokeStyle = hot ? '#942B80' : '#9A9A9A';
    ctx.globalAlpha = hot ? 0.95 : e.alpha;
    ctx.lineWidth = hot ? e.width + 0.8 : e.width;
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
  }
  ctx.globalAlpha = 1;
  for (const n of nodes){
    ctx.beginPath(); ctx.arc(n.x, n.y, n.r, 0, 6.2832);
    ctx.fillStyle = n.color; ctx.globalAlpha = 0.9; ctx.fill();
    ctx.globalAlpha = 1; ctx.lineWidth = n.outlier ? 2.2 : 0.6;
    ctx.strokeStyle = n.outlier ? '#C44530' : '#2B2B2B'; ctx.stroke();
    if (n === hoverNode){
      ctx.beginPath(); ctx.arc(n.x, n.y, n.r+4, 0, 6.2832);
      ctx.strokeStyle = '#942B80'; ctx.lineWidth = 1.4; ctx.stroke();
    }
  }
}

function frame(){
  if (physicsOn && (running || dragging)) step();
  draw();
  requestAnimationFrame(frame);
}

cv.addEventListener('mousemove', ev => {
  const r = cv.getBoundingClientRect();
  mouse.x = ev.clientX-r.left; mouse.y = ev.clientY-r.top;
  if (dragging){ dragging.x = mouse.x; dragging.y = mouse.y; }
  hoverNode = pickNode();
  hoverEdge = hoverNode ? null : pickEdge();
  if (hoverNode || hoverEdge){
    card.style.display = 'block';
    card.innerHTML = hoverNode ? hoverNode.card : hoverEdge.card;
    card.style.left = Math.min(mouse.x+16, W-270)+'px';
    card.style.top  = Math.min(mouse.y+14, H-90)+'px';
    cv.style.cursor = dragging ? 'grabbing' : 'pointer';
  } else {
    card.style.display = 'none';
    cv.style.cursor = dragging ? 'grabbing' : 'default';
  }
});
cv.addEventListener('mouseleave', () => {
  hoverNode = hoverEdge = null; card.style.display = 'none';
});
cv.addEventListener('mousedown', () => {
  const n = pickNode();
  if (n){ dragging = n; running = true; }
});
window.addEventListener('mouseup', () => {
  if (dragging){ dragging = null; running = true; }
});
document.getElementById('freeze').onclick = function(){
  physicsOn = !physicsOn;
  this.textContent = physicsOn ? 'freeze' : 'resume';
};
document.getElementById('reheat').onclick = () => { running = true; };
document.getElementById('reset').onclick = () => {
  nodes.forEach(n => { n.x = M+n.ix*(W-2*M); n.y = M+n.iy*(H-2*M);
                       n.vx = n.vy = 0; });
  running = true;
};

const lg = document.getElementById('legend');
let html = '';
for (const [d, name] of Object.entries(DATA.meta.domains))
  html += `<span class="chip" style="background:${DATA.colors[d]}"></span>${name}`;
html += '<span class="chip" style="background:#fff;border-color:#C44530"></span>outlier';
html += `<br><span style="color:#9A9A9A">node size = extremity · ${DATA.meta.rule}</span>`;
lg.innerHTML = html;
document.getElementById('subtitle').textContent =
  `n = ${DATA.meta.n} · MST ∪ mutual-kNN (k = ${DATA.meta.k}) on W1`;

window.addEventListener('resize', resize);
resize(); frame();
</script></body></html>
"""


def build_graph_2d(payload: dict, path) -> None:
    html = GRAPH_TEMPLATE.replace("__DATA__", json.dumps(payload))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)


def build_simplex_3d(geo: dict, path) -> None:
    import plotly.graph_objects as go

    ids = [n["id"] for n in geo["nodes"]]
    xyz = np.array([n["xyz"] for n in geo["nodes"]])
    dom = [n["domain"] for n in geo["nodes"]]
    ext = [n["extremity"] for n in geo["nodes"]]
    emax = max(ext)
    idx = {i: k for k, i in enumerate(ids)}

    fig = go.Figure()
    V = np.array([v["xyz"] for v in geo["vertices"]])
    vdom = [v["domain"] for v in geo["vertices"]]
    for a in range(4):
        for b in range(a + 1, 4):
            fig.add_trace(go.Scatter3d(
                x=[V[a][0], V[b][0]], y=[V[a][1], V[b][1]], z=[V[a][2], V[b][2]],
                mode="lines", line=dict(color="#C9C9C9", width=2, dash="dot"),
                hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter3d(
        x=V[:, 0], y=V[:, 1], z=V[:, 2], mode="markers+text",
        text=vdom, textposition="top center",
        marker=dict(size=6, color=[config.THEME_COLORS[d] for d in vdom],
                    line=dict(color="#2B2B2B", width=1)),
        hoverinfo="skip", showlegend=False))

    ex, ey, ez = [], [], []
    for e in geo["edges"]:
        a, b = idx[e["source"]], idx[e["target"]]
        ex += [xyz[a][0], xyz[b][0], None]
        ey += [xyz[a][1], xyz[b][1], None]
        ez += [xyz[a][2], xyz[b][2], None]
    fig.add_trace(go.Scatter3d(
        x=ex, y=ey, z=ez, mode="lines",
        line=dict(color="#9A9A9A", width=2), opacity=0.30,
        hoverinfo="skip", showlegend=False))

    texts = []
    for n in geo["nodes"]:
        p = n["profile"]
        t = (f"respondent {n['id']}<br>file row {n['row_position']} · "
             f"dominant {n['domain']}<br>extremity {n['extremity']:.2f}<br>"
             f"T {p[0]:+.2f} · E {p[1]:+.2f} · S {p[2]:+.2f} · V {p[3]:+.2f}")
        if n["is_outlier"]:
            t += "<br><b>STRUCTURAL OUTLIER</b>"
        texts.append(t)
    fig.add_trace(go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers", text=texts,
        hovertemplate="%{text}<extra></extra>",
        marker=dict(size=4 + 8 * (np.array(ext) / emax),
                    color=[config.THEME_COLORS[d] for d in dom],
                    line=dict(color="#2B2B2B", width=1), opacity=0.92),
        showlegend=False))

    fig.update_layout(
        template="plotly_white", margin=dict(l=0, r=0, t=30, b=0),
        title="Opinion simplex (lossless): positions from profile, edges from W1",
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False),
                   zaxis=dict(visible=False),
                   aspectmode="data"),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs=True)


def main() -> None:
    config.ensure_dirs()
    geo = load_json(config.OUTPUTS / "geometry.json")
    out = config.FIGURES / "interactive"
    build_graph_2d(_payload(geo), out / "graph_2d.html")
    build_simplex_3d(geo, out / "simplex_3d.html")
    print(f"wrote {out/'graph_2d.html'} (physics, offline)")
    print(f"wrote {out/'simplex_3d.html'} (plotly 3D, self-contained)")


if __name__ == "__main__":
    main()
