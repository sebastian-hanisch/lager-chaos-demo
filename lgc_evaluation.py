"""Auswertungen der Abschnitte der App und ihre Verdikte (reine Funktionen: (Art, Text)).

Jede Auswertung ruft nur den Kern `lgc_dynamics` auf; die App legt sie hinter `st.cache_data`. Zahlen in den Verdikten kommen aus der Rechnung."""

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

import lgc_constants as C
import lgc_dynamics as K
from lgc_rng import SplitMix64


@dataclass(frozen=True)
class Settings:
    L: int = C.DEFAULT_L
    alpha: float = C.DEFAULT_ALPHA
    D: float = C.DEFAULT_D
    i0: float = C.DEFAULT_I0          # Startbestand in Vielfachen von D
    seed: int = C.DEFAULT_SEED
    s: float = C.DEFAULT_S
    tl: float = C.DEFAULT_TL
    dl: float = C.DEFAULT_DL
    dr: float = C.DEFAULT_DR
    tr: float = C.DEFAULT_TR
    start: str = C.START_EINFACH
    cap: str = C.DEFAULT_CAP


def de(x: float, nd: int = 2, sign: bool = False) -> str:
    """Zahl mit Dezimalkomma (Anzeige)."""
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace(".", ",")


_SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def pow10(x: float) -> str:
    """Zehnerpotenz als Text, z. B. 1e-9 -> 10⁻⁹."""
    return "10" + str(round(math.log10(x))).translate(_SUP)


def fmt_int(n: int) -> str:
    """Ganze Zahl mit geschütztem Leerzeichen als Tausendertrenner (300000 -> 300 000)."""
    return f"{n:,}".replace(",", " ")


def pct(x: float, nd: int = 1) -> str:
    return de(100.0 * x, nd) + " %"


# --- Abschnitt Bahn --------------------------------------------------------------------------------------------------------------------

def orbit_analysis(L: int, alpha: float, D: float, i0r: float) -> dict:
    """Bahn ab dem Startbestand i0r * D (Leitung leer): Anfangsstück für die Anzeige, Kennzahlen aus 6000 + 6000 Perioden, Zyklussuche."""
    i0 = i0r * D
    orb = K.simulate(L, alpha, D, i0, K.BURN + 5000)
    summ = K.orbit_summary(L, alpha, D, i0)
    period = K.detect_period(orb, L, max_p=4000)
    return {"head_i": orb.i[:151], "head_u": orb.u[:150], "head_clipped": orb.clipped[:150], "tail_i": orb.i[-1500:], "summary": summ, "period": period,
            "first_clip": (int(np.argmax(orb.clipped)) if orb.clipped.any() else None), "D": D, "L": L, "alpha": alpha, "i0": i0}


def verdict_bahn(a: dict) -> tuple[str, str]:
    L, alpha, D, i0 = a["L"], a["alpha"], a["D"], a["i0"]
    s, per = a["summary"], a["period"]
    ak = K.alpha_crit(L)
    q = alpha / ak
    kopf = f"α = {de(alpha)} ist das {de(q)}fache der Grenze α_krit = {de(ak, 3)}"
    lam = s["lam"]
    fenster = f"{s['clip_count']} von {K.REC} Perioden im Messfenster mit Klemmung"
    if alpha <= ak * 0.9995:
        if s["clip_count"] == 0:
            extra = " Nahe der Grenze klingt die Störung nur langsam ab." if abs(lam) <= K.LAMBDA_SCHWELLE else ""
            return "success", f"{kopf}: **stabil**. Die Störung klingt ab (größter Exponent {de(lam, 4, True)} je Periode), die Klemmung greift im Messfenster nie.{extra}"
        return "info", f"{kopf}: unterhalb der Grenze, aber im Messfenster greift die Klemmung noch ({fenster}); der Startbestand {de(i0, 1)} liegt weit außen oder die Störung klingt langsam ab (Exponent {de(lam, 4, True)})."
    if lam > K.LAMBDA_SCHWELLE and per == 1:
        return "info", (f"{kopf}: **Start im instabilen Gleichgewicht**, die Bahn bleibt stehen (Periode 1, Spannweite {de(s['span'])} D, {fenster}). "
                        f"Der Exponent {de(lam, 4, True)} je Periode gehört zu diesem Gleichgewicht: es ist instabil, eine kleine Störung würde wachsen. "
                        "Das ist weder ein Zyklus noch Chaos; ein anderer Startbestand zeigt das Verhalten der Bahn.")
    if lam > K.LAMBDA_SCHWELLE and per:
        return "info", (f"{kopf}: der Zustand kehrt nach **{per} Perioden** wieder, der Exponent {de(lam, 4, True)} ist aber positiv: ein **instabiler Zyklus, kein Chaos**. "
                        f"Der Rechner hält ihn nur, weil er bei wenigen Mantissenbits von α und Startwert exakt rechnet (Abschnitt Zweierpotenz-Falle); Spannweite {de(s['span'])} D, {fenster}.")
    if lam > K.LAMBDA_SCHWELLE:
        return "warning", (f"{kopf}: **chaotisch**. Größter Exponent {de(lam, 4, True)} je Periode (über {de(K.LAMBDA_SCHWELLE, 3)}, 6000 Messschritte), "
                           f"Vorhersagehorizont 1/λ etwa {de(1 / lam, 0)} Perioden, Spannweite des Bestands {de(s['span'])} D, {fenster}. Kein Rauschen, konstanter Bedarf.")
    if lam < -K.LAMBDA_SCHWELLE:
        zyk = f"Zyklus der Periode {per}" if per else "Zyklus (Periode über 4000 oder nicht wiedergefunden)"
        return "info", f"{kopf}: **{zyk}**. Größter Exponent {de(lam, 4, True)} je Periode, Spannweite des Bestands {de(s['span'])} D, {fenster}."
    if per:
        return "info", f"{kopf}: **Zyklus der Periode {per}**, Exponent {de(lam, 4, True)} (Betrag höchstens {de(K.LAMBDA_SCHWELLE, 3)}), Spannweite {de(s['span'])} D, {fenster}."
    return "info", (f"{kopf}: **keine Entscheidung**. Der Exponent {de(lam, 4, True)} liegt im Band ±{de(K.LAMBDA_SCHWELLE, 3)}, eine Wiederkehr des Zustands bis Periode 4000 wurde nicht gefunden "
                    f"(sehr langer Zyklus oder Chaos mit kleinem Exponenten); Spannweite {de(s['span'])} D.")


# --- Abschnitt Chaos-Gitter ------------------------------------------------------------------------------------------------------------

LO_GRID = {3: 0.15, 6: 0.08, 12: 0.04}   # untere Grenze des Gitters über alpha (sonst 0,25 alpha_krit); L = 1 und 2 laufen von 0,05 bis 3,0


def grid_range(L: int) -> tuple[float, float]:
    if L <= 2:
        return 0.05, 3.0
    return LO_GRID.get(L, round(0.25 * K.alpha_crit(L), 2)), C.ALPHA_GRID_MAX


def grid_points(L: int) -> int:
    return 3000 if L <= 2 else (450 if L <= 6 else 300)


def grid_analysis(L: int, D: float = K.D_STANDARD, n: int | None = None, burn: int = K.BURN, rec: int = K.REC) -> dict:
    """Gitter über alpha: Benettin (mit Spitzen des Bestands) und Zweibahn, Zählung oberhalb alpha_krit, chaotische Bereiche."""
    lo, hi = grid_range(L)
    n = n or grid_points(L)
    al = K.alpha_grid(lo, hi, n)
    res = K.lyapunov_grid(L, al, D=D, burn=burn, rec=rec, i0=0.05 * D, peaks=400, peaks_per=8 if L <= 2 else 20)
    wolf = K.two_orbit_grid(L, al, D=D, burn=burn, rec=rec, i0=0.05 * D)
    counts = K.chaos_counts(res, al, L)
    below = (al <= 1.0) & (al > K.alpha_crit(L) * K.OBERHALB)
    counts["n_below1"] = int(below.sum())
    counts["n_chaotic_below1"] = int(((res.lam > K.LAMBDA_SCHWELLE) & below).sum())
    above = al > K.alpha_crit(L) * K.OBERHALB
    counts["agree"] = int((((res.lam > K.LAMBDA_SCHWELLE) == (wolf > K.LAMBDA_SCHWELLE)) & above).sum())
    counts["max_dev"] = float(np.max(np.abs(res.lam - wolf)[above])) if above.any() else 0.0
    counts["first_chaotic"] = float(al[(res.lam > K.LAMBDA_SCHWELLE) & above][0]) if ((res.lam > K.LAMBDA_SCHWELLE) & above).any() else None
    return {"L": L, "D": D, "alphas": al, "res": res, "wolf": wolf, "counts": counts, "ranges": K.chaotic_ranges(al, res.lam, L)}


def verdict_gitter(g: dict) -> tuple[str, str]:
    L, c = g["L"], g["counts"]
    ak = K.alpha_crit(L)
    lo, hi = grid_range(L)
    kopf = (f"L = {L}: {c['n_chaotic']} von {c['n_above']} Gitterpunkten oberhalb α_krit = {de(ak, 3)} (bis α = {de(hi, 1)}) sind chaotisch "
            f"({pct(c['share_chaotic'])}; Exponent über {de(K.LAMBDA_SCHWELLE, 3)}, 6000 Messschritte)")
    if c["n_below1"]:
        unter = f"; im Bereich α ≤ 1,0: {c['n_chaotic_below1']} von {c['n_below1']} ({pct(c['n_chaotic_below1'] / c['n_below1'])})"
    else:
        unter = "; im Bereich α ≤ 1,0 liegt kein Gitterpunkt oberhalb der Grenze"
    zwei = f" Zweites Verfahren (Zweibahn): gleiches Urteil in {c['agree']} von {c['n_above']} Punkten, größte Abweichung {de(c['max_dev'], 4)}."
    kind = "warning" if c["n_chaotic"] else "info"
    return kind, kopf + unter + "." + zwei


# --- Abschnitt Vorhersagehorizont -----------------------------------------------------------------------------------------------------

def horizon_analysis(L: int, alpha: float, D: float) -> dict:
    d0 = K.D0_ZWEIBAHN
    dist = K.two_orbit_distance(L, alpha, D=D, n=C.HORIZON_SCHRITTE, d0=d0, i0=0.05 * D)
    lam = float(K.lyapunov_grid(L, [alpha], D=D, i0=0.05 * D).lam[0])
    thr = C.HORIZON_FAKTOR * D
    return {"L": L, "alpha": alpha, "D": D, "dist": dist, "lam": lam, "thr": thr, "d0": d0, "t_visible": K.first_visible(dist, thr)}


def verdict_horizon(h: dict) -> tuple[str, str]:
    lam, tv = h["lam"], h["t_visible"]
    kopf = f"Anfangsunterschied {pow10(h['d0'])} im Bestand"
    if lam > K.LAMBDA_SCHWELLE and tv is not None:
        return "warning", (f"{kopf}: sichtbar (Abstand über {de(h['thr'], 2)} = 1 % von D) nach **{tv} Perioden**. Der Exponent {de(lam, 4, True)} gibt 1/λ = {de(1 / lam, 1)} Perioden je Faktor e; "
                           f"ln({de(h['thr'])} / {pow10(h['d0'])}) = {de(math.log(h['thr'] / h['d0']), 1)} solche Faktoren ergeben {de(math.log(h['thr'] / h['d0']) / lam, 0)} Perioden.")
    if lam > K.LAMBDA_SCHWELLE:
        return "info", f"{kopf}: der Abstand bleibt in {C.HORIZON_SCHRITTE} Perioden unter {de(h['thr'], 2)}; der Exponent {de(lam, 4, True)} ist klein, der Horizont 1/λ = {de(1 / lam, 0)} Perioden länger als die Rechnung."
    return "success", f"{kopf}: der Unterschied wächst nicht (Exponent {de(lam, 4, True)} je Periode, höchstens {de(K.LAMBDA_SCHWELLE, 3)} gilt als kein Chaos): beide Lager laufen zusammen oder bleiben dicht beieinander."


# --- Abschnitt Rauschen ----------------------------------------------------------------------------------------------------------------

def noise_range(L: int) -> tuple[float, float]:
    return LO_GRID.get(L, round(0.25 * K.alpha_crit(L), 2)), C.ALPHA_GRID_MAX


def noise_analysis(L: int, alpha: float, D: float, s: float, seed: int) -> dict:
    lo, hi = noise_range(L)
    al = K.alpha_grid(lo, hi, C.NOISE_POINTS)
    T = K.BURN + K.REC
    r0 = K.lyapunov_grid(L, al, D=D, i0=0.05 * D)
    rs = K.lyapunov_grid(L, al, D=D, i0=0.05 * D, noise=K.noise_sequence(seed, T, s, D))
    # Streuung zwischen Rauschfolgen am eingestellten alpha: jede Spalte hat ihre eigene Folge
    seqs = np.stack([K.noise_sequence(seed + j, T, s, D) for j in range(C.NOISE_SEEDS)], axis=1)
    spread = K.lyapunov_grid(L, [alpha] * C.NOISE_SEEDS, D=D, i0=0.05 * D, noise=seqs).lam
    # stabiler Bereich: Klemmung bei 0,7 alpha_krit
    stable = K.lyapunov_grid(L, [0.7 * K.alpha_crit(L)] * C.NOISE_SEEDS, D=D, i0=0.0, noise=seqs)
    return {"L": L, "alphas": al, "lam0": r0.lam, "lam_s": rs.lam, "c0": K.chaos_counts(r0, al, L), "cs": K.chaos_counts(rs, al, L), "s": s, "alpha": alpha,
            "spread": spread, "stable_clip": float(stable.clip_frac.mean()), "stable_n": int(C.NOISE_SEEDS * K.REC)}


def verdict_noise(n: dict) -> tuple[str, str]:
    c0, cs, s = n["c0"], n["cs"], n["s"]
    if s == 0:
        return "info", "Rauschen 0: Gitter ohne und mit Rauschen sind gleich. Regler s im Reiter erhöhen."
    d0, ds = cs["n_window"] / max(cs["n_above"], 1), c0["n_window"] / max(c0["n_above"], 1)
    kopf = (f"Rauschen {de(s)} D: Anteil chaotischer Gitterpunkte oberhalb α_krit {pct(c0['share_chaotic'])} → {pct(cs['share_chaotic'])} "
            f"(Exponent entlang derselben Rauschfolge; {c0['n_above']} Punkte); Punkte mit Exponent unter −{de(K.LAMBDA_SCHWELLE, 3)} (Fenster) {pct(ds)} → {pct(d0)}.")
    sp = n["spread"]
    fuss = (f" Am eingestellten α streut der Exponent zwischen {C.NOISE_SEEDS} Rauschfolgen von {de(float(sp.min()), 3, True)} bis {de(float(sp.max()), 3, True)}."
            f" Bei α = 0,7 α_krit greift die Klemmung in {pct(n['stable_clip'], 3)} der Perioden ({C.NOISE_SEEDS} Folgen zu je {K.REC}).")
    return "info", kopf + fuss


# --- Reiter Randkollision --------------------------------------------------------------------------------------------------------------

def lager_nf_analysis(alpha: float, D: float, i0r: float = 0.5, steps: int = 200) -> dict:
    """Das Lager mit L = 2 gegen seine Normalform (x = i - D/alpha, y = w + D) und die Bedingungen des Aufsatzes in beiden Lesarten."""
    i0 = i0r * D
    orb = K.simulate(2, alpha, D, i0, steps)
    nf = K.normal_form_orbit(alpha, D, i0, 0.0, steps)
    x_lager = orb.i - D / alpha
    y_lager = np.concatenate([[0.0], orb.u]) + D
    dev = float(max(np.max(np.abs(x_lager - nf[:, 0])), np.max(np.abs(y_lager - nf[:, 1]))))
    nfp = K.lager_normal_form(alpha, D)
    lese = []
    for name, tl, dl, tr, dr in (
        ("Lager, μ = −D", nfp["tau_L"], nfp["delta_L"], nfp["tau_R"], nfp["delta_R"]),
        ("gespiegelt, μ > 0 (Lesart des Aufsatzes)", nfp["tau_R"], nfp["delta_R"], nfp["tau_L"], nfp["delta_L"]),
    ):
        lese.append({"name": name, "tl": tl, "dl": dl, "tr": tr, "dr": dr, "c17": K.condition_17(tl, dl, tr, dr), "c18": K.condition_18(tl, dl, tr, dr),
                     "c19": K.condition_19_value(tl, dl, tr, dr), "region": K.robust_chaos_region(tl, dl, tr, dr)})
    return {"alpha": alpha, "dev": dev, "steps": steps, "lese": lese, "lam": K.orbit_summary(2, alpha, D, i0)["lam"]}


def verdict_lager_nf(a: dict) -> tuple[str, str]:
    alle_falsch = not any(x["c17"] or x["c18"] or x["region"] for x in a["lese"])
    kopf = (f"L = 2, α = {de(a['alpha'])}: Normalform ja (größte Abweichung der Bahnen {a['dev']:.1e} über {a['steps']} Perioden"
            f"{'; bei positivem Exponenten verstärkt sich der Rundungsunterschied' if a.get('lam', 0.0) > K.LAMBDA_SCHWELLE else ''})")
    if alle_falsch:
        return "info", kopf + ", Bedingungen (17) und (18) und das Gebiet des Aufsatzes nein (in beiden Lesarten verletzt; auf der rechten Seite gilt δ = 0 statt δ > 0 und τ = +1 statt τ < −1)."
    return "success", kopf + ", Bedingungen des Aufsatzes erfüllt."


def nf_analysis(tl: float, dl: float, dr: float, tr: float, n: int = 60000) -> dict:
    """Normalform mit μ = 1: Exponent über τ_R von −3 bis −(1 + δ_R) − 0,01 (300 Punkte) und am eingestellten τ_R."""
    trs = np.linspace(-3.0, -(1 + dr) - 0.01, 300)
    lam_all = K.normal_form_lyapunov(tl, dl, np.append(trs, tr), dr, n=n)
    lam, lam_pt = lam_all[:-1], float(lam_all[-1])
    c19 = np.array([K.condition_19_value(tl, dl, float(t), dr) for t in trs], dtype=object)
    pos = np.array([v is not None and v > 0 for v in c19])
    ch, per, div = lam > K.LAMBDA_SCHWELLE, lam < -K.LAMBDA_SCHWELLE, np.isnan(lam)
    return {"trs": trs, "lam": lam, "pos": pos, "tl": tl, "dl": dl, "dr": dr, "tr": tr, "lam_pt": lam_pt,
            "n_chaotic": int(ch.sum()), "n_window": int(per.sum()), "n_diverged": int(div.sum()), "n_pos": int(pos.sum()),
            "n_pos_chaotic": int((pos & ch).sum()), "n_pos_window": int((pos & per).sum()), "n_nonpos_chaotic": int((~pos & ch).sum()),
            "n_nonpos_diverged": int((~pos & div).sum()), "point": {"c17": K.condition_17(tl, dl, tr, dr), "c18": K.condition_18(tl, dl, tr, dr),
                                                                       "c19": K.condition_19_value(tl, dl, tr, dr), "region": K.robust_chaos_region(tl, dl, tr, dr)}}


def verdict_nf(a: dict) -> tuple[str, str]:
    kopf = (f"τ_L = {de(a['tl'])}, δ_L = {de(a['dl'])}, δ_R = {de(a['dr'])}, 300 Werte von τ_R unterhalb −(1 + δ_R): {a['n_chaotic']} chaotisch, "
            f"{a['n_window']} mit Fenster (Exponent unter −{de(K.LAMBDA_SCHWELLE, 3)}), {a['n_diverged']} unbeschränkt; Bedingung (19) gilt in {a['n_pos']} Punkten, davon chaotisch {a['n_pos_chaotic']}, mit Fenster {a['n_pos_window']}.")
    return ("success" if a["n_pos"] and a["n_pos_chaotic"] == a["n_pos"] and a["n_pos_window"] == 0 else "info"), kopf


# --- Abschnitt Zweierpotenz-Falle -------------------------------------------------------------------------------------------------------

EINFACHE_STARTS = (0.5, 0.25, 0.125, 3.0, 7.5, 0.1)


def mantissa_bits(x: float) -> int:
    """Zahl der Binärstellen der Mantisse (2 → 1, 1,25 → 3, 1,1 → 52 oder 53)."""
    num = abs(Fraction(x).numerator)
    while num and num % 2 == 0:
        num //= 2
    return max(1, num.bit_length())


def starts_for(art: str, seed: int) -> list[float]:
    if art == C.START_EINFACH:
        return list(EINFACHE_STARTS)
    rng = SplitMix64(seed)
    return [round(rng.uniform(-50.0, 50.0), 6) for _ in range(5)]


def pow2_analysis(L: int, alpha: float, D: float, art: str, cap_key: str, seed: int) -> dict:
    cap = C.CAP_VALUES[cap_key]
    alpha2 = alpha + 1e-9 * math.sqrt(2)
    step_a, step_b = K.make_step(L, alpha, D), K.make_step(L, alpha2, D)
    rows = []
    for s0 in starts_for(art, seed):
        ra = K.brent_cycle(step_a, K.start_state(L, s0), cap)
        rb = K.brent_cycle(step_b, K.start_state(L, s0), cap)
        rows.append({"start": s0, "a": ra, "b": rb})
    first = rows[0]
    mult = None
    if first["a"] is not None and first["a"][0] <= 5000:
        per, pre = first["a"]
        x = K.start_state(L, first["start"])
        for _ in range(pre):
            x = step_a(x)
        states = []
        for _ in range(per):
            states.append(x)
            x = step_a(x)
        mult = float(K.cycle_multipliers(L, alpha, D, states)[0])
    exact = None
    if art == C.START_EINFACH and mantissa_bits(alpha) <= 12 and float(D).is_integer():
        e = K.exact_cycle(L, Fraction(alpha), Fraction(D), Fraction(first["start"]), cap=20000)
        if e is not None:
            exact = {"pre": e[0], "per": e[1]}
    return {"L": L, "alpha": alpha, "alpha2": alpha2, "D": D, "cap": cap, "cap_key": cap_key, "rows": rows, "mult": mult, "exact": exact,
            "bits": mantissa_bits(alpha), "art": art}


def verdict_pow2(a: dict) -> tuple[str, str]:
    rows = a["rows"]
    n = len(rows)
    ka = [r["a"] for r in rows if r["a"] is not None]
    kb = [r["b"] for r in rows if r["b"] is not None]
    kopf = f"α = {de(a['alpha'], 3)} ({a['bits']} Mantissenbit{'s' if a['bits'] != 1 else ''}): {len(ka)} von {n} Startwerten enden innerhalb von {fmt_int(a['cap'])} Schritten in einem Zyklus"
    if ka:
        kopf += f" (Perioden {min(k[0] for k in ka)} bis {max(k[0] for k in ka)})"
    kopf += f"; bei α + 10⁻⁹·√2: {len(kb)} von {n}."
    teil = ""
    if a["mult"] is not None and rows[0]["a"] is not None:
        per = rows[0]["a"][0]
        teil = (f" Zum ersten Start (Periode {per}): größter Multiplikator {de(a['mult'], 2)}, Exponent ln({de(a['mult'], 2)})/{per} = {de(math.log(a['mult']) / per, 4, True)} je Periode"
                f"{' (positiv: ein instabiler Zyklus, kein Chaos)' if a['mult'] > 1 + 1e-9 else ''}.")
    if a["exact"] is not None:
        teil += f" Exakt mit Brüchen: Vorperiode {a['exact']['pre']}, Periode {a['exact']['per']} (gleich der Float-Bahn: {'ja' if rows[0]['a'] == (a['exact']['per'], a['exact']['pre']) else 'nein'})."
    return ("warning" if ka and not kb else "info"), kopf + teil


def load_frozen() -> dict | None:
    """Vorgerechnete Messreihen (`data/lgc_frozen.json`, erzeugt von `tools/make_frozen.py`); None, wenn die Datei fehlt."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parent / "data" / "lgc_frozen.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
