"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import lgc_constants as C
import lgc_dynamics as K


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.2), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_series(a: dict):
    """Bestand i_t und Bestellung u_t über die ersten 150 Perioden; Perioden mit Klemmung rot markiert."""
    n = len(a["head_u"])
    t = list(range(n + 1))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=("Bestand i_t (Abweichung vom Soll)", "Bestellung u_t (Abweichung vom mittleren Bedarf D)"))
    fig.add_trace(go.Scatter(x=t, y=np.round(a["head_i"], 4), mode="lines+markers", line=dict(color=C.COLOR_I), marker=dict(size=4), name="i_t",
                             hovertemplate="t = %{x}: i = %{y:.3f}<extra></extra>"), row=1, col=1)
    tu = list(range(n))
    fig.add_trace(go.Scatter(x=tu, y=np.round(a["head_u"], 4), mode="lines+markers", line=dict(color=C.COLOR_U), marker=dict(size=4), name="u_t",
                             hovertemplate="t = %{x}: u = %{y:.3f}<extra></extra>"), row=2, col=1)
    cl = np.nonzero(a["head_clipped"])[0]
    if cl.size:
        fig.add_trace(go.Scatter(x=cl.tolist(), y=[-a["D"]] * cl.size, mode="markers", marker=dict(color=C.COLOR_CHAOS, size=8, symbol="x"), name="Klemmung (Bestellung 0)",
                                 hovertemplate="t = %{x}: geklemmt<extra></extra>"), row=2, col=1)
    fig.add_hline(y=-a["D"], line=dict(color=C.COLOR_GREY, dash="dot"), row=2, col=1)
    fig.update_xaxes(title_text="Periode t", row=2, col=1)
    return _base(fig, 520)


def build_phase(a: dict):
    """Phasenbild (i_{t-1}, i_t): Anfangsstück als Linie, Ende der langen Bahn als Punkte."""
    h, tl = a["head_i"], a["tail_i"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.round(tl[:-1], 3), y=np.round(tl[1:], 3), mode="markers", marker=dict(size=3, color=C.COLOR_LINE, opacity=0.6), name="Ende der Bahn (1500 Perioden)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=np.round(h[:-1][:60], 3), y=np.round(h[1:][:60], 3), mode="lines+markers", line=dict(color=C.COLOR_I, width=1), marker=dict(size=4), name="Anfang (60 Perioden)", hoverinfo="skip"))
    fig.update_xaxes(title_text="i_{t−1}")
    fig.update_yaxes(title_text="i_t")
    return _base(fig, 400)


def build_grid(g: dict, alpha_now: float):
    """Oben: Spitzen des Bestands (Bifurkationsbild) in Vielfachen von D; unten: Exponent je Periode (Benettin Linie, Zweibahn Punkte)."""
    L, al, res, wolf = g["L"], g["alphas"], g["res"], g["wolf"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=[0.55, 0.45],
                        subplot_titles=("Spitzen des Bestands / D", "größter Exponent je Periode"))
    cols, vals = res.peaks
    fig.add_trace(go.Scattergl(x=np.round(al[cols], 5), y=np.round(vals / g["D"], 3), mode="markers", marker=dict(size=2, color=C.COLOR_LINE), name="Spitzen von i_t", hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=np.round(al, 5), y=np.round(res.lam, 5), mode="lines", line=dict(color=C.COLOR_LINE, width=1.5), name="Benettin",
                             hovertemplate="α = %{x:.3f}: λ = %{y:.4f}<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=np.round(al, 5), y=np.round(wolf, 5), mode="markers", marker=dict(size=3, color=C.COLOR_U), name="Zweibahn",
                             hovertemplate="α = %{x:.3f}: λ = %{y:.4f}<extra></extra>"), row=2, col=1)
    for lo, hi in g["ranges"]:
        for r in (1, 2):
            fig.add_vrect(x0=lo, x1=hi, fillcolor=C.COLOR_CHAOS_BAND, line_width=0, row=r, col=1)
    ak = K.alpha_crit(L)
    for r in (1, 2):
        fig.add_vline(x=ak, line=dict(color=C.COLOR_CHAOS, dash="dash"), row=r, col=1)
        fig.add_vline(x=alpha_now, line=dict(color=C.COLOR_I, dash="dot"), row=r, col=1)
    fig.add_hline(y=0, line=dict(color=C.COLOR_GREY), row=2, col=1)
    fig.add_hline(y=K.LAMBDA_SCHWELLE, line=dict(color=C.COLOR_GREY, dash="dot"), row=2, col=1)
    fig.update_xaxes(title_text="α (rot gestrichelt: α_krit, grün gepunktet: eingestelltes α; rot hinterlegt: chaotisch)", row=2, col=1)
    return _base(fig, 640)


def build_horizon(h: dict):
    """Abstand der zwei Lager über die Zeit (logarithmisch); Gerade der Steigung λ ab dem Anfangsunterschied und Schwelle."""
    dist = h["dist"]
    t = np.arange(len(dist))
    keep = np.unique(np.concatenate([np.arange(0, len(dist), max(1, len(dist) // 800)), [len(dist) - 1]]))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t[keep], y=np.maximum(dist[keep], 1e-16), mode="lines", line=dict(color=C.COLOR_LINE), name="Abstand der Zustände",
                             hovertemplate="t = %{x}: %{y:.2e}<extra></extra>"))
    if h["lam"] > K.LAMBDA_SCHWELLE:
        fig.add_trace(go.Scatter(x=t[keep], y=h["d0"] * np.exp(np.minimum(h["lam"] * t[keep], 700.0)), mode="lines", line=dict(color=C.COLOR_CHAOS, dash="dash"), name="d₀ · e^(λ t)",
                                 hovertemplate="t = %{x}: %{y:.2e}<extra></extra>"))
    fig.add_hline(y=h["thr"], line=dict(color=C.COLOR_GREY, dash="dot"), annotation_text="sichtbar (1 % von D)", annotation_position="bottom right")
    top = max(float(dist.max()), h["thr"]) * 3
    fig.update_yaxes(type="log", range=[math.log10(h["d0"]) - 1, math.log10(top)], title_text="Abstand")
    fig.update_xaxes(title_text="Periode t")
    return _base(fig, 380)


def build_noise(n: dict):
    al = n["alphas"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.round(al, 4), y=np.round(n["lam0"], 5), mode="lines", line=dict(color=C.COLOR_GREY, width=1.5), name="ohne Rauschen",
                             hovertemplate="α = %{x:.3f}: λ = %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=np.round(al, 4), y=np.round(n["lam_s"], 5), mode="lines", line=dict(color=C.COLOR_CHAOS, width=1.5), name=f"Rauschen {n['s']:.2f} D".replace(".", ","),
                             hovertemplate="α = %{x:.3f}: λ = %{y:.4f}<extra></extra>"))
    fig.add_vline(x=K.alpha_crit(n["L"]), line=dict(color=C.COLOR_CHAOS, dash="dash"))
    fig.add_hline(y=0, line=dict(color=C.COLOR_GREY))
    fig.add_hline(y=K.LAMBDA_SCHWELLE, line=dict(color=C.COLOR_GREY, dash="dot"))
    fig.update_xaxes(title_text="α (rot gestrichelt: α_krit)")
    fig.update_yaxes(title_text="größter Exponent je Periode")
    return _base(fig, 380)


def build_nf(a: dict):
    trs, lam, pos = a["trs"], a["lam"], a["pos"]
    ok = ~np.isnan(lam)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.round(trs[ok & pos], 4), y=np.round(lam[ok & pos], 4), mode="markers", marker=dict(size=4, color=C.COLOR_CHAOS), name="(19) erfüllt",
                             hovertemplate="τ_R = %{x:.3f}: λ = %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=np.round(trs[ok & ~pos], 4), y=np.round(lam[ok & ~pos], 4), mode="markers", marker=dict(size=4, color=C.COLOR_LINE), name="(19) nicht erfüllt",
                             hovertemplate="τ_R = %{x:.3f}: λ = %{y:.3f}<extra></extra>"))
    div = ~ok
    if div.any():
        fig.add_trace(go.Scatter(x=np.round(trs[div], 4), y=[0.0] * int(div.sum()), mode="markers", marker=dict(size=5, color=C.COLOR_GREY, symbol="x"), name="unbeschränkt",
                                 hovertemplate="τ_R = %{x:.3f}: unbeschränkt<extra></extra>"))
    if a["tr"] <= trs.max():
        fig.add_vline(x=a["tr"], line=dict(color=C.COLOR_I, dash="dot"))
    fig.add_hline(y=0, line=dict(color=C.COLOR_GREY))
    fig.update_xaxes(title_text="τ_R (nur Werte unterhalb −(1 + δ_R))")
    fig.update_yaxes(title_text="größter Exponent je Periode")
    return _base(fig, 380)
