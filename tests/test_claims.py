"""Claims-Tests: jede Zahl der README steht hier als Band (Zählungen: Rundungsunterschiede der Plattform können einzelne Punkte nahe der Schwelle 0,003 kippen).

Die Spalte „Test“ der README-Tabelle nennt die Funktionen dieser Datei. Messbedingungen überall: D = 10, 6000 Einschwing- und 6000 Messschritte, chaotisch heißt Exponent über 0,003,
Gitter nie auf Zweierpotenzen (`alpha_grid`: Schritt nicht dyadisch, kleiner irrationaler Versatz), Zufalls-Seed 35 wo Zufall gebraucht wird."""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

import lgc_constants as C
import lgc_dynamics as K
import lgc_evaluation as E

ROOT = Path(__file__).resolve().parent.parent


def _near(value, expected, tol):
    assert abs(value - expected) <= tol, (value, expected, tol)


# --- M1: Anteil chaotisch oberhalb der Grenze --------------------------------------------------------------------------------------------

GRID_EXPECT = {  # L: (Punkte oberhalb, chaotisch, Toleranz)
    3: (356, 181, 3), 6: (410, 221, 3), 12: (288, 150, 3), 2: (2033, 673, 5), 1: (1016, 1012, 3),
}


@pytest.fixture(scope="module")
def grids():
    return {L: E.grid_analysis(L) for L in GRID_EXPECT}


@pytest.mark.parametrize("L", list(GRID_EXPECT))
def test_share_chaotic_above_the_limit(grids, L):
    n_above, n_ch, tol = GRID_EXPECT[L]
    c = grids[L]["counts"]
    assert c["n_above"] == n_above
    _near(c["n_chaotic"], n_ch, tol)


def test_shares_in_percent_as_in_the_readme(grids):
    share = {L: grids[L]["counts"]["share_chaotic"] for L in grids}
    _near(share[3], 0.508, 0.01)
    _near(share[6], 0.539, 0.01)
    _near(share[12], 0.521, 0.01)
    _near(share[2], 0.331, 0.003)
    _near(share[1], 0.996, 0.004)


def test_no_chaos_for_L3_below_alpha_1_and_some_for_L6_L12(grids):
    assert (grids[3]["counts"]["n_below1"], grids[3]["counts"]["n_chaotic_below1"]) == (76, 0)
    c6, c12 = grids[6]["counts"], grids[12]["counts"]
    assert c6["n_below1"] == 139 and abs(c6["n_chaotic_below1"] - 9) <= 2        # 6,5 %
    assert c12["n_below1"] == 110 and abs(c12["n_chaotic_below1"] - 22) <= 3     # 20,0 %


def test_first_chaotic_point_in_multiples_of_alpha_crit(grids):
    for L, expected, tol in ((3, 1.63, 0.03), (6, 2.62, 0.1), (12, 4.29, 0.1)):
        _near(grids[L]["counts"]["first_chaotic"] / K.alpha_crit(L), expected, tol)


def test_chaotic_ranges_of_L3(grids):
    r = grids[3]["ranges"]
    assert len(r) == 3
    for (lo, hi), (elo, ehi) in zip(r, ((1.007, 1.157), (1.337, 1.357), (1.677, 2.397))):
        _near(lo, elo, 0.002)
        _near(hi, ehi, 0.002)


def test_L2_has_periodic_windows(grids):
    c = grids[2]["counts"]
    _near(c["n_window"] / c["n_above"], 0.575, 0.005)       # 57,5 % der Punkte mit Exponent unter -0,003


# --- M2: zwei Verfahren -----------------------------------------------------------------------------------------------------------------

def test_two_methods_agree_in_the_grids(grids):
    for L in (2, 3, 6, 12):
        c = grids[L]["counts"]
        assert c["agree"] >= c["n_above"] - 1, L
        assert c["max_dev"] < 0.004, L
    assert grids[3]["counts"]["agree"] == 356 and grids[3]["counts"]["max_dev"] < 0.0015


# --- M4: feines Gitter und Gegenprobe (vorgerechnet) ---------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def frozen():
    data = E.load_frozen()
    assert data is not None, "data/lgc_frozen.json fehlt (tools/make_frozen.py)"
    return data


@pytest.mark.parametrize("L,lo,hi,n_ch,tol", [(3, 1.677, 2.397, 4000, 0), (6, 1.520, 2.381, 3988, 6), (12, 1.632, 2.340, 3942, 20)])
def test_fine_grid_has_no_window_and_matches_the_frozen_numbers(frozen, L, lo, hi, n_ch, tol):
    al = K.alpha_grid(lo, hi, 4000)
    lam = K.lyapunov_grid(L, al).lam
    ch = int((lam > K.LAMBDA_SCHWELLE).sum())
    assert abs(ch - n_ch) <= tol
    assert int((lam < -K.LAMBDA_SCHWELLE).sum()) == 0                      # kein Punkt mit Exponent unter -0,003
    f = frozen["fine_grid"][str(L)]
    assert (f["lo"], f["hi"], f["n"], f["n_window"]) == (lo, hi, 4000, 0)
    assert abs(f["n_chaotic"] - n_ch) <= tol and abs(f["n_chaotic"] - ch) <= max(tol, 2)
    assert float(lam.min()) > 0 and float(lam.min()) < 0.005               # kleinster Exponent liegt knapp über der Schwelle


def test_counter_check_with_random_alphas_and_starts_is_frozen_as_6000_of_6000(frozen):
    g = frozen["gegenprobe"]
    assert (g["n_alpha"], g["n_start"], g["runs"], g["n_chaotic"], g["burn"], g["rec"]) == (1000, 6, 6000, 6000, 12000, 12000)
    assert 0.003 < g["lam_min"] < 0.006
    spec = importlib.util.spec_from_file_location("make_frozen", ROOT / "tools" / "make_frozen.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    again = mod.gegenprobe()
    assert again["n_chaotic"] == 6000 and again["runs"] == 6000


# --- M3: Rauschen -----------------------------------------------------------------------------------------------------------------------

def test_noise_keeps_chaos_and_windows():
    n0 = E.noise_analysis(3, 1.8, 10.0, 0.0, 35)["cs"]
    n25 = E.noise_analysis(3, 1.8, 10.0, 0.25, 35)["cs"]
    assert n0["n_above"] == n25["n_above"] == 79
    _near(n0["n_chaotic"], 40, 2)           # 50,6 %
    _near(n25["n_chaotic"], 49, 2)          # 62,0 %
    _near(n0["n_window"], 31, 2)            # 39,2 %
    _near(n25["n_window"], 30, 2)           # 38,0 %
    m0 = E.noise_analysis(6, 1.8, 10.0, 0.0, 35)["cs"]
    m25 = E.noise_analysis(6, 1.8, 10.0, 0.25, 35)["cs"]
    assert m0["n_above"] == 91
    _near(m0["n_chaotic"], 50, 2)           # 54,9 %
    _near(m25["n_chaotic"], 58, 2)          # 63,7 %
    assert n25["n_chaotic"] > n0["n_chaotic"] and m25["n_chaotic"] > m0["n_chaotic"]


def test_noise_clamp_in_the_stable_range_is_rare_for_s_025_and_visible_for_s_05():
    """alpha = 0,7 alpha_krit, je 5 Rauschfolgen zu 40000 Perioden (Zufalls-Seeds 100 bis 104): Zahl der Perioden mit Klemmung."""
    T = 6000 + 40000
    result = {}
    for L in (3, 6, 12):
        for s in (0.25, 0.5):
            seqs = np.stack([K.noise_sequence(100 + j, T, s) for j in range(5)], axis=1)
            r = K.lyapunov_grid(L, [0.7 * K.alpha_crit(L)] * 5, burn=6000, rec=40000, i0=0.0, noise=seqs)
            result[(L, s)] = float(r.clip_frac.sum() * 40000)
    assert all(result[(L, 0.25)] <= 5 for L in (3, 6, 12))                  # höchstens 5 von 200000 Perioden (0,0025 %)
    assert result[(12, 0.5)] == 0
    _near(result[(3, 0.5)] / 200000, 0.0177, 0.006)                         # 1,8 % der Perioden bei s = 0,5
    _near(result[(6, 0.5)] / 200000, 0.00095, 0.0007)


# --- M8: Vorhersagehorizont ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("L,alpha,lam,tol", [(3, 2.15, 0.0811, 0.003), (3, 1.0269, 0.0129, 0.002), (6, 1.8864, 0.0272, 0.002), (12, 2.0096, 0.0091, 0.0015), (12, 1.2229, 0.0078, 0.0015)])
def test_horizon_is_about_ln_ratio_over_lambda(L, alpha, lam, tol):
    h = E.horizon_analysis(L, alpha, 10.0)
    _near(h["lam"], lam, tol)
    assert h["t_visible"] is not None
    ratio = h["t_visible"] * h["lam"] / math.log(h["thr"] / h["d0"])        # ln(0,1 / 1e-9) = 18,4
    assert 0.7 < ratio < 1.3, ratio


def test_horizon_values_quoted_in_the_readme():
    h = E.horizon_analysis(3, 2.15, 10.0)
    _near(1 / h["lam"], 12.3, 0.5)
    _near(h["t_visible"], 215, 40)
    assert 100 < 1 / E.horizon_analysis(12, 2.0096, 10.0)["lam"] < 130
    assert 100 < 1 / E.horizon_analysis(12, 1.2229, 10.0)["lam"] < 160


# --- Bahn: Spannweiten und Presets -----------------------------------------------------------------------------------------------------

def test_span_behind_the_limit_for_L3():
    for q, expected in ((1.2, 4.300), (1.001, 3.249)):
        for D in (1.0, 10.0, 100.0):
            _near(K.orbit_summary(3, q * K.alpha_crit(3), D, 0.5 * D)["span"], expected, 0.003)


def test_spans_at_chaotic_points_quoted_in_the_readme():
    _near(K.orbit_summary(3, 2.15, 10.0, 5.0)["span"], 12.87, 0.1)           # Start 0,5 D wie im Preset: 12,87 D (Start 0,05 D: 12,85)
    _near(K.orbit_summary(6, 1.8864, 10.0, 0.5)["span"], 45.6, 0.3)
    _near(K.orbit_summary(12, 2.0096, 10.0, 0.5)["span"], 185.5, 0.6)       # "etwa 186 D"


def test_directly_behind_the_limit_L3_cycles_of_period_10_and_81_and_one_undecided_point():
    """L = 3: bei 1,001 alpha_krit Zyklus der Periode 10 (= 4L - 2), bei 1,2 alpha_krit Zyklus der Periode 81 mit Exponent im Band ±0,003; bei alpha = 0,74
    (1,198 alpha_krit) liegt der Exponent im Band und bis Periode 4000 kehrt der Zustand nicht wieder: kein Urteil (Befund gegenüber dem Plan, der dort „Grenzzyklus“ schrieb)."""
    near = E.orbit_analysis(3, 1.001 * K.alpha_crit(3), 10.0, 0.5)
    assert near["period"] == 10 and near["summary"]["lam"] < -K.LAMBDA_SCHWELLE
    a = E.orbit_analysis(3, 1.2 * K.alpha_crit(3), 10.0, 0.5)
    assert a["period"] == 81 and abs(a["summary"]["lam"]) <= K.LAMBDA_SCHWELLE
    b = E.orbit_analysis(3, 0.74, 10.0, 0.5)
    assert b["period"] is None and abs(b["summary"]["lam"]) <= K.LAMBDA_SCHWELLE
    assert E.verdict_bahn(b)[1].count("keine Entscheidung") == 1


PRESET_BAHN = {  # Preset: (Art, erwartete Textstücke)
    "Von Hand (L = 2)": ("info", ["Zyklus der Periode 6", "-0,1090", "2,40 D", "2000 von 6000"]),
    "Grenzzyklus": ("info", ["Zyklus der Periode 10", "-0,0420", "3,87 D", "1800 von 6000"]),
    "Chaos L = 3": ("warning", ["chaotisch", "+0,0803", "12,87 D", "etwa 12 Perioden"]),
    "Rauschen": ("warning", ["chaotisch", "+0,0413", "10,75 D"]),
    "Randkollision": ("warning", ["chaotisch", "+0,0472", "4,04 D"]),
    "Zweierpotenz-Falle": ("info", ["kehrt nach **16 Perioden** wieder", "+0,0687", "instabiler Zyklus, kein Chaos"]),
}


@pytest.mark.parametrize("name", list(PRESET_BAHN))
def test_each_preset_shows_the_expected_verdict_line(name):
    p = C.PRESETS[name]
    kind, text = E.verdict_bahn(E.orbit_analysis(p["L"], p["alpha"], float(p["D"]), p["i0"]))
    exp_kind, parts = PRESET_BAHN[name]
    assert kind == exp_kind
    for part in parts:
        assert part in text, (name, part, text)


# --- Handbeispiele ---------------------------------------------------------------------------------------------------------------------

def test_critical_values_and_periods_from_the_handbook():
    assert K.alpha_crit(2) == pytest.approx(1.0) and K.alpha_crit(3) == pytest.approx(0.6180340, abs=1e-7) and K.alpha_crit(6) == pytest.approx(0.2846297, abs=1e-7)
    assert [4 * L - 2 for L in (2, 3, 6)] == [6, 10, 22]


def test_hand_example_clamp_and_normal_form_numbers():
    orb = K.simulate(2, 1.2, 10.0, 5.0, 8)
    assert int(np.argmax(orb.clipped)) == 6
    ln = E.lager_nf_analysis(1.2, 10.0, 0.5, 2000)
    assert ln["dev"] < 1e-13                                               # 8,9e-15 über 2000 Perioden
    assert E.lager_nf_analysis(1.6, 10.0, 0.5, 200)["dev"] > 1e-12         # bei positivem Exponenten verstärkt sich der Rundungsunterschied


# --- M5: das Lager mit L = 2 und die Bedingungen des Aufsatzes -----------------------------------------------------------------------------

@pytest.mark.parametrize("alpha", [0.9, 1.2, 1.6, 2.4])
def test_L2_store_violates_conditions_17_18_and_the_region_in_both_readings(alpha):
    for x in E.lager_nf_analysis(alpha, 10.0, 0.5, 200)["lese"]:
        assert not x["c17"] and not x["c18"] and not x["region"], (alpha, x["name"])
    mirrored = E.lager_nf_analysis(alpha, 10.0, 0.5, 200)["lese"][1]
    assert (mirrored["tl"], mirrored["dl"], mirrored["tr"], mirrored["dr"]) == (1.0, 0.0, 1.0, alpha)
    assert not 1.0 > 1.0 + 0.0                                              # (17) scheitert gespiegelt an der Gleichheit 1 > 1 + 0


# --- M6: Positivkontrolle der Normalform ------------------------------------------------------------------------------------------------------

def test_normal_form_positive_control_367_of_367():
    total_pos = total_pos_ch = 0
    for (tl, dl, dr), (n_ch, n_div, n_pos, tol) in (((1.9, 0.5, 0.5), (23, 277, 7, 2)), ((1.7, 0.3, 0.05), (202, 98, 202, 3)), ((1.7, 0.3, 0.3), (158, 142, 158, 3))):
        a = E.nf_analysis(tl, dl, dr, -1.8)
        assert a["n_window"] == 0
        _near(a["n_chaotic"], n_ch, tol)
        _near(a["n_diverged"], n_div, tol)
        _near(a["n_pos"], n_pos, 1)
        assert a["n_pos_window"] == 0
        total_pos += a["n_pos"]
        total_pos_ch += a["n_pos_chaotic"]
        if (tl, dl, dr) != (1.9, 0.5, 0.5):
            assert a["n_nonpos_chaotic"] == 0 and a["n_nonpos_diverged"] == a["n_diverged"]          # (19) <= 0: alle unbeschränkt
    assert total_pos == 367 and total_pos_ch == 367
    lam = E.nf_analysis(1.7, 0.3, 0.05, -1.8)
    chaotic_pos = lam["lam"][lam["pos"] & (lam["lam"] > K.LAMBDA_SCHWELLE)]
    assert 0.22 < chaotic_pos.min() < 0.24 and 0.63 < chaotic_pos.max() < 0.66


def test_normal_form_contrast_with_condition_18_violated():
    trs = np.linspace(-1.0, 1.4, 49)
    lam = K.normal_form_lyapunov(1.7, 0.3, trs, 0.05)
    assert int((lam > K.LAMBDA_SCHWELLE).sum()) == 0
    _near(int((lam < -K.LAMBDA_SCHWELLE).sum()), 41, 2)
    _near(int(np.isnan(lam).sum()), 7, 2)


def test_source_conditions_are_evaluated_at_the_paper_example_sets():
    """(1,9; 0,5; 0,5) und (1,7; 0,3; 0,05): tau_L > 2 sqrt(delta_L) und (17), (18) für tau_R < -(1 + delta_R) gelten; (19) kippt beim ersten Satz bei tau_R = -1,54 von positiv zu negativ."""
    assert 1.9 > 2 * math.sqrt(0.5) and 1.7 > 2 * math.sqrt(0.3)
    assert K.robust_chaos_region(1.7, 0.3, -1.8, 0.05) and K.robust_chaos_region(1.9, 0.5, -1.52, 0.5)
    assert not K.robust_chaos_region(1.9, 0.5, -1.55, 0.5) and not K.robust_chaos_region(1.9, 0.5, -1.8, 0.5)     # dort (19) <= 0: unbeschränkt


# --- M7: Zweierpotenz-Falle --------------------------------------------------------------------------------------------------------------

def test_power_of_two_case_five_of_six_simple_starts_end_in_a_cycle():
    p = E.pow2_analysis(3, 2.0, 10.0, C.START_EINFACH, "300000", 35)
    got = {r["start"]: r["a"] for r in p["rows"]}
    assert got == {0.5: (62, 17), 0.25: (60, 17), 0.125: (63, 42), 3.0: (30, 9), 7.5: (15, 2), 0.1: None}
    assert all(r["b"] is None for r in p["rows"])                          # alpha + 1e-9 sqrt 2: kein Zyklus in 3·10^5 Schritten
    assert p["exact"] == {"pre": 17, "per": 62} and p["bits"] == 1
    _near(p["mult"], 81.0, 1e-6)
    _near(math.log(p["mult"]) / 62, 0.0709, 5e-5)                          # positiver Exponent auf einem instabilen Zyklus


def test_alpha_1_1_has_no_cycle_for_simple_or_scattered_starts_in_3e5_steps():
    for art in (C.START_EINFACH, C.START_GESTREUT):
        p = E.pow2_analysis(3, 1.1, 10.0, art, "300000", 35)
        assert all(r["a"] is None for r in p["rows"]), art
    assert E.pow2_analysis(3, 2.0, 10.0, C.START_GESTREUT, "300000", 35)["rows"][0]["a"] is None      # gestreute Starts: bei alpha = 2 kein Zyklus


def test_frozen_search_to_3e6_steps(frozen):
    p = frozen["pow2"]
    assert p["cap"] == 3_000_000
    assert p["einfach_alpha2"]["n_cycle"] == 5 and p["einfach_alpha2"]["periods"] == [15, 30, 60, 62, 63]
    assert p["einfach_alpha2b"]["n_cycle"] == 0 and p["einfach_alpha11"]["n_cycle"] == 0
    for key in ("gestreut_alpha2", "gestreut_alpha2b", "gestreut_alpha11"):
        assert p[key]["n"] == 100 and p[key]["n_cycle"] == 0
    # zwei der gestreuten Starts mit der vollen Grenze neu gerechnet (je etwa 0,6 s)
    for al in (2.0, 1.1):
        starts = E.starts_for(C.START_GESTREUT, C.DEFAULT_SEED)[:1]
        step = K.make_step(3, al, 10.0)
        assert K.brent_cycle(step, K.start_state(3, starts[0]), C.FROZEN_CAP) is None


def test_hand_cycle_with_alpha_2_in_L2():
    pre, per, cyc = K.exact_cycle(2, E.Fraction(2), E.Fraction(1), E.Fraction(1, 2))
    assert (pre, per) == (1, 8)
    _near(math.log(K.cycle_multipliers(2, E.Fraction(2), E.Fraction(1), cyc)[0]) / 8, 0.2012, 5e-5)
