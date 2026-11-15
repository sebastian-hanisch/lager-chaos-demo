"""Chaos im Lager – Verzögerung plus Klemmung - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Ein Lager mit linearer Bestellregel (Anpassung α, Lieferzeit L) schwingt hinter α_krit; darf die Bestellung nicht unter 0 fallen, ist die
Schwingung über weite Bereiche von α chaotisch. Zwei Verfahren für den Lyapunov-Exponenten, Rauschen, Vorhersagehorizont, die Normalform der
Randkollision und die Zweierpotenz-Falle der Gleitkommarechnung. Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import lgc_constants as C
import lgc_dynamics as K
import lgc_evaluation as E
from lgc_evaluation import Settings, de
from lgc_layout import dunkles_design, kopf_css
from lgc_presets import (
    SETTING_SPECS,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    power_of_two_alpha,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from lgc_visualization import build_grid, build_horizon, build_nf, build_noise, build_phase, build_series

st.set_page_config(page_title="Chaos im Lager – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _orbit(L, alpha, D, i0):
    return E.orbit_analysis(L, alpha, D, i0)


@st.cache_data(show_spinner="Rechne das Gitter über α (rund eine Sekunde) …")
def _grid(L, D):
    return E.grid_analysis(L, D)


@st.cache_data(show_spinner=False)
def _horizon(L, alpha, D):
    return E.horizon_analysis(L, alpha, D)


@st.cache_data(show_spinner="Rechne das Gitter mit und ohne Rauschen (rund eine Sekunde) …")
def _noise(L, alpha, D, s, seed):
    return E.noise_analysis(L, alpha, D, s, seed)


@st.cache_data(show_spinner=False)
def _lager_nf(alpha, D, i0):
    return E.lager_nf_analysis(alpha, D, i0)


@st.cache_data(show_spinner="Rechne die Normalform über τ_R (rund eine Sekunde) …")
def _nf(tl, dl, dr, tr):
    return E.nf_analysis(tl, dl, dr, tr)


@st.cache_data(show_spinner="Suche Zyklen …")
def _pow2(L, alpha, D, art, cap, seed):
    return E.pow2_analysis(L, alpha, D, art, cap, seed)


@st.cache_data(show_spinner=False)
def _frozen():
    return E.load_frozen()


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in lgc_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("📦 Chaos im Lager – Verzögerung plus Klemmung")
st.markdown(
    """
Ein **Lager** bestellt nach einer einfachen Regel nach: Bestellung = mittlerer Bedarf minus **α** mal Abweichung des Bestands vom Soll; die Ware kommt erst nach **L Perioden** an.
Ohne weitere Regel schwingt das Lager hinter der Grenze **α_krit = 2 sin(π/(4L − 2))** immer stärker auf. **Darf die Bestellung nicht unter 0 fallen (Klemmung)**, blieb die Schwingung in allen gerechneten Fällen beschränkt,
und über weite Bereiche von α (bis 2,4, nach dem Kriterium λ > 0,003) ist sie **chaotisch**: größter Lyapunov-Exponent positiv, ohne jedes Rauschen, bei konstantem Bedarf (in diesem Modell, bei den gemessenen Gittern).
Die Demo misst das mit zwei Verfahren und zeigt, was an den Zahlen Messung und was ein Rechner-Artefakt ist.

Weiter unten: **❓ Kernfragen**, die Reiter **🔬 Experimente** (Gitter über α, Vorhersagehorizont, Rauschen, Randkollision, Zweierpotenz-Falle), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert das Lagermodell", expanded=True):
    st.markdown(
        """
1. **Zustand:** In Periode t hat das Lager die Abweichung **i_t** vom Soll-Bestand; unterwegs sind die Bestellungen **u_{t−1}, …, u_{t−L+1}** (Abweichungen vom mittleren Bedarf **D**). Der Zustand hat die Dimension L.
2. **Regel mit Klemmung:** **u_t = max(−α · i_t, −D)**. Die Bestellung D + u_t darf nicht negativ werden; ist sie 0, hängt sie nicht mehr vom Bestand ab, die Rückkopplung ist dort „gelöscht“.
3. **Lieferung:** **i_{t+1} = i_t + u_{t+1−L}**: die Bestellung von vor L − 1 Perioden kommt an. Ohne Klemmung ist alles linear, die Grenze α_krit folgt aus den Eigenwerten; mit Klemmung ist die Regel stückweise linear.
4. **Messen:** Der größte Lyapunov-Exponent λ (je Periode) sagt, ob zwei fast gleiche Lager auseinanderlaufen (λ > 0) oder zusammen (λ < 0). Hier gilt **chaotisch** als λ > 0,003 bei 6000 Messschritten.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Lager**")
    L = st.slider("Lieferzeit L (Perioden)", *bounds("L_slider"), key="L_slider", help="Wie viele Perioden die Ware unterwegs ist; der Zustand hat die Dimension L.")
    alpha = st.slider("Anpassung α", *bounds("alpha_slider"), step=C.ALPHA_STEP, key="alpha_slider", help="Anteil der Bestandsabweichung, den die Bestellung je Periode ausgleicht.")
    ak = K.alpha_crit(int(L))
    st.caption(f"Grenze α_krit = {de(ak, 3)}; α ist das {de(alpha / ak)}fache davon.")
    D = st.slider("Mittlerer Bedarf D je Periode", *bounds("D_slider"), key="D_slider", help="Skala des Modells: Zustand und Spannweite wachsen mit D, der Exponent nicht.")
    i0 = st.slider("Startbestand i₀ (Vielfache von D)", *bounds("i0_slider"), step=C.I0_STEP, key="i0_slider", help="Startbestand der Bahn im Ergebnis; die Leitung ist am Anfang leer.")
    st.markdown("**Zufall**")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1, help="Gilt für das Bedarfsrauschen und die gestreuten Startwerte der Zweierpotenz-Falle.")
    st.button("🎲 Neuer Zufalls-Seed", width="stretch", on_click=randomize_seed)

L, D = int(L), float(D)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis: Bahn und Klemmung")
orb = _orbit(L, float(alpha), D, float(i0))
summ = orb["summary"]
m1, m2, m3, m4 = st.columns(4)
m1.metric("α / α_krit", de(alpha / ak), help="Unter 1 ist die lineare Regel stabil.")
m2.metric("Größter Exponent λ je Periode", de(summ["lam"], 4, True), help="Benettin-Verfahren, 6000 Einschwing- und 6000 Messschritte.")
m3.metric("Spannweite des Bestands", f"{de(summ['span'])} D", help="Höchster minus tiefster Bestand im Messfenster, in Vielfachen von D.")
m4.metric("Perioden mit Klemmung", de(100.0 * summ["clip_frac"], 1) + " %", help=f"Anteil im Messfenster ({summ['clip_count']} von {K.REC} Perioden nach 6000 Einschwingschritten).")
c1, c2 = st.columns([3, 2])
c1.plotly_chart(build_series(orb), width="stretch", key="series_chart")
c2.plotly_chart(build_phase(orb), width="stretch", key="phase_chart")
fc = orb["first_clip"]
st.caption(
    "Links: die ersten 150 Perioden ab dem Startbestand; rote Kreuze: Perioden, in denen die Klemmung greift (Bestellung genau 0, also u = −D). "
    + (f"Erster Klemmschritt bei t = {fc}. " if fc is not None else "In den ersten 5000 Perioden greift die Klemmung nie. ")
    + "Rechts: Phasenbild (i_{t−1}, i_t); Punkte: Ende der langen Bahn, Linie: Anfang."
)

# --- Kernfragen ----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfragen")
st.markdown("**Was macht das Lager bei den eingestellten Werten?**")
kind, text = E.verdict_bahn(orb)
getattr(st, kind)(text)
st.markdown("**Wie oft ist das Lager hinter der Grenze chaotisch?** (Gitter über α für die eingestellte Lieferzeit, jeder Punkt eine eigene Bahn)")
g = _grid(L, D)
kind, text = E.verdict_gitter(g)
getattr(st, kind)(text)

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experimente")
tab_grid, tab_hor, tab_noise, tab_rand, tab_pow = st.tabs(["📈 Chaos-Gitter über α", "🔭 Vorhersagehorizont", "🎲 Rauschen im Bedarf", "🧱 Randkollision", "💾 Zweierpotenz-Falle"])

with tab_grid:
    st.markdown(
        "Jeder Punkt ist eine eigene Bahn mit festem α (6000 Einschwing-, 6000 Messschritte, Startbestand 0,05 D, Leitung leer). Oben die Spitzen des Bestands (wie ein Bifurkationsbild), "
        "unten der größte Exponent mit **zwei Verfahren**: Benettin (Tangentialvektor, Linie) und Zweibahn (Abstand zweier Bahnen, Punkte). Rot hinterlegt: Bereiche, in denen jeder Gitterpunkt chaotisch ist."
    )
    st.plotly_chart(build_grid(g, float(alpha)), width="stretch", key="grid_chart")
    kind, text = E.verdict_gitter(g)
    getattr(st, kind)(text)
    rng_txt = ", ".join(de(a, 3) if a == b else f"{de(a, 3)} bis {de(b, 3)}" for a, b in g["ranges"][:8])
    st.caption(
        f"Zusammenhängende chaotische Bereiche oberhalb α_krit (ein einzelner Wert ist ein einzelner Gitterpunkt): {rng_txt if rng_txt else 'keine'}"
        + (" …" if len(g["ranges"]) > 8 else "")
        + f". Gitter: {len(g['alphas'])} Punkte von α = {de(g['alphas'][0], 2)} bis {de(g['alphas'][-1], 2)}, der Schritt ist nie eine Zweierpotenz. "
        "Zwischen den chaotischen Bereichen liegen periodische Fenster (Exponent unter −0,003); schmalere Fenster als der Gitterabstand sind nicht ausgeschlossen."
    )

with tab_hor:
    st.markdown(
        "Zwei Lager starten auf derselben Bahn, das zweite mit einem Bestand, der um 10⁻⁹ höher liegt (keine Zweierpotenz). Der Abstand ihrer Zustände wächst, wenn der Exponent positiv ist, "
        "etwa wie d₀ · e^(λ t); 1/λ ist der Vorhersagehorizont je Faktor e. Gilt für die Einstellungen in der Seitenleiste (L, α, D)."
    )
    hor = _horizon(L, float(alpha), D)
    st.plotly_chart(build_horizon(hor), width="stretch", key="horizon_chart")
    kind, text = E.verdict_horizon(hor)
    getattr(st, kind)(text)

with tab_noise:
    st.markdown(
        "Der Bedarf schwankt zufällig um den Mittelwert: e_t ~ Normalverteilung mit Standardabweichung s · D (SplitMix64 aus dem Zufalls-Seed der Seitenleiste). Gemessen wird der Exponent **entlang derselben Rauschfolge** "
        "(beide Nachbarbahnen sehen dasselbe Rauschen); das ist nicht der Exponent über alle Rauschfolgen. Gilt für die Lieferzeit L der Seitenleiste."
    )
    s = st.slider("Rauschen s (Standardabweichung in Vielfachen von D)", *bounds("s_slider"), step=C.S_STEP, key="s_slider", help="0: kein Rauschen. Höhere Werte: stärkere Bedarfsschwankung.")
    nz = _noise(L, float(alpha), D, float(s), int(seed))
    st.plotly_chart(build_noise(nz), width="stretch", key="noise_chart")
    kind, text = E.verdict_noise(nz)
    getattr(st, kind)(text)

with tab_rand:
    st.markdown(
        "**Randkollision** heißt: eine stückweise lineare Abbildung verändert ihre Dynamik, wenn ein Fixpunkt die Knickgrenze erreicht. Banerjee, Yorke und Grebogi (PRL 80, 1998, S. 3049) geben für eine Normalform "
        "Bedingungen an, unter denen **robustes Chaos** (Umgebung im Parameterraum ohne periodische Attraktoren und ohne koexistierende Attraktoren) auftritt. Das Lager mit L = 2 ist genau eine solche Normalform. "
        "*Die Bedingungen (17) bis (19) wurden gegen den LaTeX-Quelltext der ar5iv-Fassung (arXiv chao-dyn/9803001) gelesen, nicht gegen das PDF.*"
    )
    st.markdown("**(a) Das Lager mit L = 2 als Normalform** (α und D aus der Seitenleiste; die Lieferzeit ist hier fest 2)")
    ln = _lager_nf(float(alpha), D, float(i0))
    rows = ["| Lesart | τ_L | δ_L | τ_R | δ_R | (17) | (18) | (19) links | Gebiet des Aufsatzes |", "|---|---|---|---|---|---|---|---|---|"]
    for x in ln["lese"]:
        c19 = "komplexe Eigenwerte" if x["c19"] is None else de(x["c19"], 3)
        rows.append(f"| {x['name']} | {de(x['tl'], 2)} | {de(x['dl'], 2)} | {de(x['tr'], 2)} | {de(x['dr'], 2)} | {'ja' if x['c17'] else 'nein'} | {'ja' if x['c18'] else 'nein'} | {c19} | {'ja' if x['region'] else 'nein'} |")
    st.markdown("\n".join(rows))
    kind, text = E.verdict_lager_nf(ln)
    getattr(st, kind)(text)
    st.caption(
        "Koordinaten x = i − D/α, y = u_{t−1} + D; links der Knickgrenze (x ≤ 0) greift die Klemmung nicht, rechts schon. Der Aufsatz formuliert die Bedingungen für μ > 0 und 1 > δ > 0 auf beiden Seiten; "
        "das Lager hat μ = −D und auf einer Seite δ = 0. Deshalb sind beide Lesarten (so wie das Lager steht, und gespiegelt) angegeben."
    )
    st.markdown("**(b) Positivkontrolle: eine Normalform, die die Bedingungen erfüllt** (μ = 1; 300 Werte von τ_R unterhalb −(1 + δ_R))")
    r1, r2, r3, r4 = st.columns(4)
    tl = r1.slider("τ_L", *bounds("tl_slider"), step=0.05, key="tl_slider", help="Spur der linken Jacobi-Matrix.")
    dl = r2.slider("δ_L", *bounds("dl_slider"), step=0.05, key="dl_slider", help="Determinante der linken Jacobi-Matrix.")
    dr = r3.slider("δ_R", *bounds("dr_slider"), step=0.05, key="dr_slider", help="Determinante der rechten Jacobi-Matrix.")
    tr = r4.slider("τ_R", *bounds("tr_slider"), step=0.05, key="tr_slider", help="Spur der rechten Jacobi-Matrix; die Kurve zeigt τ_R von −3 bis −(1 + δ_R).")
    nfa = _nf(float(tl), float(dl), float(dr), float(tr))
    st.plotly_chart(build_nf(nfa), width="stretch", key="nf_chart")
    kind, text = E.verdict_nf(nfa)
    getattr(st, kind)(text)
    pt = nfa["point"]
    c19 = "komplexe Eigenwerte" if pt["c19"] is None else de(pt["c19"], 3)
    st.markdown(
        f"**Eingestellter Punkt** (τ_L, δ_L, τ_R, δ_R) = ({de(tl)}, {de(dl)}, {de(tr)}, {de(dr)}): (17) {'ja' if pt['c17'] else 'nein'}, (18) {'ja' if pt['c18'] else 'nein'}, "
        f"(19) links {c19} ({'> 0' if pt['c19'] is not None and pt['c19'] > 0 else 'nicht > 0'}), Gebiet für robustes Chaos {'ja' if pt['region'] else 'nein'}; Exponent dort "
        + (de(nfa["lam_pt"], 3, True) if nfa["lam_pt"] == nfa["lam_pt"] else "unbeschränkt")
        + " je Periode (60000 Schritte)."
    )
    st.markdown("**(c) Das Lager mit L = 3, 6 und 12 im feinen Gitter** (vorgerechnet: `tools/make_frozen.py`, 4000 Punkte je Bereich, 6000 + 6000 Schritte)")
    fz = _frozen()
    if fz:
        rows = ["| L | Bereich α | chaotisch | Punkte mit Exponent unter −0,003 | kleinster Exponent |", "|---|---|---|---|---|"]
        for key in ("3", "6", "12"):
            f = fz["fine_grid"][key]
            rows.append(f"| {key} | {de(f['lo'], 3)} bis {de(f['hi'], 3)} | {f['n_chaotic']} von {f['n']} | {f['n_window']} | {de(f['lam_min'], 4, True)} |")
        st.markdown("\n".join(rows))
    st.info(
        "**Verdikt:** L = 2: Normalform ja, Bedingungen nein, periodische Fenster ja. L ≥ 3: Der Zustand hat mindestens die Dimension 3, die Normalform ist nicht anwendbar; "
        "dass eine Randkollision das Diagramm erklärt, ist eine **Hypothese**. In den feinen Gittern der Chaos-Bereiche (oben) zeigt sich kein periodisches Fenster, zwischen den Bereichen liegen welche (Gitter über α)."
    )

with tab_pow:
    st.markdown(
        "Der Rechner kennt nur endlich viele Zahlen, jede Float-Bahn ist deshalb irgendwann periodisch. Meist ist die Periode so lang, dass man sie nie sieht. Ist **α eine Zweierpotenz** und hat der Startwert nur wenige "
        "Mantissenbits, rechnet der Rechner **exakt** in Dualbrüchen, und die Bahn endet nach wenigen Schritten in einem Zyklus. Verglichen werden α der Seitenleiste und α + 10⁻⁹·√2, mit Brent-Suche bis zur Schrittgrenze."
    )
    p1, p2, p3 = st.columns([2, 2, 2])
    art = p1.selectbox("Startwerte", C.START_OPTIONS, key="start_select", help="einfach: 0,5; 0,25; 0,125; 3; 7,5 und 0,1 (wenige Mantissenbits). gestreut: 5 Startwerte in [−50; 50] aus dem Zufalls-Seed der Seitenleiste.")
    cap_key = p2.selectbox("Schrittgrenze der Zyklussuche", C.CAP_OPTIONS, key="cap_select", help="So viele Schritte läuft die Suche je Startwert und je α, dann gibt sie auf. Die Suche bis 3 000 000 Schritte ist vorgerechnet (unten).")
    p3.button("α auf die Zweierpotenz 2 setzen", on_click=power_of_two_alpha, help="Setzt die Anpassung α der Seitenleiste auf 2, den benannten Testfall.")
    pw = _pow2(L, float(alpha), D, art, cap_key, int(seed))
    rows = ["| Startwert i₀ | bei α der Seitenleiste | bei α + 10⁻⁹·√2 |", "|---|---|---|"]
    for r in pw["rows"]:
        fa = "kein Zyklus gefunden" if r["a"] is None else f"Periode {r['a'][0]} (Vorperiode {r['a'][1]})"
        fb = "kein Zyklus gefunden" if r["b"] is None else f"Periode {r['b'][0]} (Vorperiode {r['b'][1]})"
        rows.append(f"| {de(r['start'], 3)} | {fa} | {fb} |")
    st.markdown("\n".join(rows))
    kind, text = E.verdict_pow2(pw)
    getattr(st, kind)(text)
    fz = _frozen()
    if fz:
        pz = fz["pow2"]
        st.markdown(
            f"**Vorgerechnet** (L = 3, D = 10, Schrittgrenze 3·10⁶, `tools/make_frozen.py`): α = 2, einfache Startwerte: {pz['einfach_alpha2']['n_cycle']} von {pz['einfach_alpha2']['n']} mit Zyklus; "
            f"gestreute Startwerte (SplitMix64, {pz['gestreut_alpha2']['n']} Starts in [−50; 50]): bei α = 2 {pz['gestreut_alpha2']['n_cycle']}, bei α + 10⁻⁹·√2 {pz['gestreut_alpha2b']['n_cycle']}, bei α = 1,1 {pz['gestreut_alpha11']['n_cycle']} mit Zyklus."
        )

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Gemessen |
|---|---|---|
| **Konstanter Bedarf** | Mit Rauschen bleibt das Chaos (Abschnitt Rauschen), aber der Exponent entlang der Rauschfolge ist nicht der über alle Folgen | Reiter Rauschen |
| **Kein Leitungsgewicht (β = 0)** | Wer die bestellte Ware mitzählt, verschiebt die Grenze α_krit; die Demo misst nur β = 0 | nicht gemessen |
| **Gitterpunkte statt aller α** | Fenster, die schmaler als der Gitterabstand sind, bleiben unsichtbar | Reiter Chaos-Gitter |
| **Startwert und Messdauer** | Wenige Punkte hängen am Startwert (Koexistenz); λ nahe 0,003 braucht längere Messdauer | README |
| **Mechanismus für L ≥ 3** | Die Randkollision ist nur für L = 2 als Normalform nachgewiesen, für L ≥ 3 eine Hypothese | Reiter Randkollision |
| **Praxis** | Chaos tritt je nach Lieferzeit bei Spannweiten von etwa 1 mal dem Bedarf (L = 1) bis über 200 mal dem Bedarf (L = 12) auf; bei α ≤ 1 für L = 3 nirgends, ob reale Lager dort arbeiten, ist offen | Gitter, README |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind die Lieferzeit $L$, die Anpassung $\alpha$ und der mittlere Bedarf $D$; gesucht ist, ob der Bestand periodisch schwingt oder chaotisch, gemessen am größten Lyapunov-Exponenten $\lambda$ je Periode.

| Symbol | Bedeutung |
|---|---|
| $i_t$ | Bestand minus Soll in Periode $t$ |
| $u_t$ | Bestellung minus mittlerer Bedarf $D$ |
| $L$ | Lieferzeit in Perioden; der Zustand hat die Dimension $L$ |
| $\alpha$ | Anpassung der Bestellregel |
| $x_t$ | Zustand $(i_t, u_{t-1}, \dots, u_{t-L+1})$ |
| $\lambda$ | größter Lyapunov-Exponent je Periode |

**Modell.**

$$u_t = \max(-\alpha\, i_t,\; -D)$$

$$i_{t+1} = i_t + u_{t+1-L}$$

**Grenze der linearen Regel** (ohne Klemmung, Eigenwerte der Zustandsmatrix vom Betrag 1):

$$\alpha_{\mathrm{krit}} = 2 \sin\!\left(\frac{\pi}{4L-2}\right), \qquad \text{kritische Periode } 4L-2$$

**Lyapunov-Exponent** (Benettin): der Tangentialvektor $v_t$ folgt der Jacobi-Matrix $J_t$ des Schritts; bei Klemmung ist die Ableitung der Bestellung nach dem Bestand 0.

$$v_{t+1} = J_t\, v_t, \qquad \lambda = \lim_{T \to \infty} \frac{1}{T} \sum_{t=1}^{T} \ln \frac{\lVert v_t \rVert}{\lVert v_{t-1} \rVert}$$

Zweibahn-Verfahren: zwei Bahnen im Abstand $d_0$, nach jedem Schritt auf $d_0$ zurückgesetzt; $\lambda$ ist das Mittel von $\ln(d_t/d_0)$. Gemessen werden 6000 Schritte nach 6000 Einschwingschritten; **chaotisch** heißt hier $\lambda > 0{,}003$.

**Normalform der Randkollision** (L = 2, mit $x = i - D/\alpha$, $y = u_{t-1} + D$):

$$G(x, y) = \begin{cases} (\tau_L x + y + \mu,\; -\delta_L x) & x \le 0 \\ (\tau_R x + y + \mu,\; -\delta_R x) & x > 0 \end{cases}$$

Für das Lager gilt $\tau_L = \tau_R = 1$, $\delta_L = \alpha$, $\delta_R = 0$, $\mu = -D$. Bedingungen aus Banerjee, Yorke, Grebogi (1998), mit den Eigenwerten $\lambda_1^L \ge \lambda_2^L$ der linken Matrix:

$$\text{(17)}\quad \tau_L > 1 + \delta_L \;\text{ und }\; \tau_R < 1 + \delta_R$$

$$\text{(18)}\quad \tau_L > 1 + \delta_L \;\text{ und }\; \tau_R < -(1 + \delta_R)$$

$$\text{(19)}\quad \delta_L \tau_R \lambda_1^L - \delta_R \lambda_1^L \lambda_2^L + \delta_R \lambda_2^L - \delta_L \tau_R + \tau_L \delta_L - \delta_L^2 - \lambda_2^L \delta_L > 0$$

Robustes Chaos nach dem Aufsatz: $1 > \delta_L > 0$, $1 > \delta_R > 0$, $\tau_R < -(1+\delta_R)$, $\tau_L > 2\sqrt{\delta_L}$ und (19). Die Gleichungen des Lagers selbst sind vollständig; für (17) bis (19) und das Gebiet gilt der Wortlaut der Quelle (nur die Gleichungen, nicht ihr Beweis).

**Literatur.** Banerjee, S., Yorke, J. A., Grebogi, C.: Robust Chaos. Physical Review Letters 80 (1998), 3049 (Formeln aus der ar5iv-Fassung, arXiv chao-dyn/9803001). Chaos in Lagermodellen ist in der Fachliteratur beschrieben (Mosekilde und Larsen 1988; Sterman 1988); diese Arbeiten wurden hier **nicht gelesen**, nur als Verzeichniseinträge gefunden; ob sie dieses Modell betreffen, ist offen.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)

sync_query_params({key: st.session_state[key] for key in SETTING_SPECS})
