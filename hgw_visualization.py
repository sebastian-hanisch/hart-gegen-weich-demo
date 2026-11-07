"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import hgw_algorithm as A
import hgw_constants as C

FAM_COLORS = {0: "#4c78a8", 1: "#d8a13a", 2: "#7b5ea7"}


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_cloud(a, ts: dict, r: dict):
    """Alle Pläne (Zeitsumme gegen Kosten); die zulässigen blau; die Schwelle als senkrechte Linie; Kostenoptimum, hartes Optimum (Zielplan) und der von der Strafe gewählte Plan als Marken."""
    p = a.prob
    x, y = np.asarray(p.sB, dtype=float), np.asarray(p.sA, dtype=float)
    ok = x <= ts["T"]
    fig = go.Figure(go.Scattergl(x=x[~ok], y=y[~ok], mode="markers", marker=dict(size=4, color=C.COLOR_ALL), name="alle Pläne", hovertemplate="B %{x:.1f} / A %{y:.1f}<extra></extra>"))
    if ok.any():
        fig.add_trace(go.Scattergl(x=x[ok], y=y[ok], mode="markers", marker=dict(size=5, color=C.COLOR_FEAS), name="zulässig (B ≤ T)", hovertemplate="B %{x:.1f} / A %{y:.1f}<extra></extra>"))
    fig.add_vline(x=ts["T"], line=dict(color=C.COLOR_BAD, dash="dash"), annotation_text="Schwelle T", annotation_position="top")
    marks = [("Kostenoptimum", p.iA, "star", "#111111", 16), ("hartes Optimum (Zielplan)" if ts["feasible"] else "bester Plan (kleinste Verletzung)", ts["target"], "diamond", C.COLOR_OK, 15),
             (f"gewählt: {C.PENALTY_SHORT[r['form']]}", r["idx"], "x", C.COLOR_MARK, 14)]
    for name, i, sym, col, size in marks:
        fig.add_trace(go.Scatter(x=[x[i]], y=[y[i]], mode="markers", marker=dict(symbol=sym, size=size, color=col, line=dict(width=1.5, color="white")), name=name,
                                 hovertemplate=name + ": B %{x:.1f} / A %{y:.1f}<extra></extra>"))
    fig.update_xaxes(title_text="Summe B der Zeit (muss höchstens T sein)")
    fig.update_yaxes(title_text="Summe A der Kosten (kleiner ist besser)")
    return _base(fig, height=440)


def build_curve(rows: list[dict], a_t: float, form: str):
    """Der gewählte Plan je Gewicht 2^e: oben die Kosten (gestrichelt: Kosten des Zielplans), unten die Verletzung; grün: Treffer."""
    xs = [f"2^{r['e']}" for r in rows]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("Kosten A des gewählten Plans", "Verletzung der Schwelle"))
    fig.add_trace(go.Scatter(x=xs, y=[r["a"] for r in rows], mode="lines+markers", line=dict(color="#4c78a8"), marker=dict(size=7, color=[C.COLOR_OK if r["hit"] else C.COLOR_BAD for r in rows]),
                             name="A des gewählten Plans", hovertemplate="λ = %{x}: A %{y:.1f}<extra></extra>"), row=1, col=1)
    fig.add_hline(y=a_t, line=dict(color=C.COLOR_OK, dash="dash"), row=1, col=1)
    fig.add_trace(go.Bar(x=xs, y=[r["v"] for r in rows], marker_color=[C.COLOR_OK if r["hit"] else C.COLOR_BAD for r in rows], name="Verletzung", hovertemplate="λ = %{x}: Verletzung %{y:.2f}<extra></extra>"), row=2, col=1)
    fig.update_layout(showlegend=False)
    fig.update_xaxes(type="category", title_text=f"Gewicht λ ({C.PENALTY_SHORT[form]})", row=2, col=1)
    return _base(fig, height=420)


def build_reach_bars(res: dict, total: int = 20):
    """Instanzen, in denen die Gewichtssumme das harte Optimum erreicht, je Familie und Schwelle (die Strafe mit Schwelle: ab λ* immer)."""
    taus = sorted({tau for _, tau in res})
    fams = sorted({f for f, _ in res})
    fig = go.Figure([go.Bar(x=[f"Schwelle {t} %" for t in taus], y=[res[(f, t)]["reach"] for t in taus], name=("K = " + str(C.GRANULARITIES[f])) if C.GRANULARITIES[f] else "reell",
                            marker_color=FAM_COLORS[f]) for f in fams])
    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[0, total], title_text=f"Gewichtssumme erreicht das harte Optimum, auch nur im Gleichstand (von {total})")
    return _base(fig, height=320)


def build_small_lam(res: dict, total: int = 20):
    """Instanzen mit verletzter Schwelle je Gewicht (lineare Strafe, Schwelle 50 %), je Familie."""
    fams = sorted({k[1] for k in res})
    lams = sorted({k[2] for k in res})
    fig = go.Figure([go.Scatter(x=[f"{lam:g}" for lam in lams], y=[res[("linear", f, lam)]["violated"] for lam in lams], mode="lines+markers", name=("K = " + str(C.GRANULARITIES[f])) if C.GRANULARITIES[f] else "reell",
                                line=dict(color=FAM_COLORS[f])) for f in fams])
    fig.update_xaxes(type="category", title_text="Gewicht λ (lineare Strafe)")
    fig.update_yaxes(range=[0, total], title_text=f"Instanzen mit verletzter Schwelle (von {total})")
    return _base(fig, height=320)


def build_big_lam(res: dict, total: int = 20):
    """Treffer der linearen Strafe in float64 je Gewicht 2^e bei unzulässiger Schwelle, je Familie."""
    fams = sorted(res)
    exps = list(next(iter(res.values()))["infeasible"])
    fig = go.Figure([go.Scatter(x=[f"2^{e}" for e in exps], y=[res[f]["infeasible"][e] for e in exps], mode="lines+markers", name=("K = " + str(C.GRANULARITIES[f])) if C.GRANULARITIES[f] else "reell",
                                line=dict(color=FAM_COLORS[f])) for f in fams])
    fig.update_xaxes(type="category", title_text="Gewicht λ = 2^e (unzulässige Schwelle)")
    fig.update_yaxes(range=[0, total + 1], title_text=f"Treffer (von {total})")
    return _base(fig, height=320)
