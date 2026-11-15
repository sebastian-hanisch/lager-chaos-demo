"""Eigenschaftstests (ORACLE-PLAYBOOK §5.2): Aussagen, die für das Lager gelten müssen, ohne dass man das richtige Ergebnis kennt, und Randfälle.

Alle alpha, D und Startwerte sind keine Zweierpotenzen (Ausnahme: der benannte Testfall in test_claims.py)."""

import math
from fractions import Fraction as Fr

import numpy as np
import pytest

import lgc_dynamics as K


@pytest.mark.parametrize("L,alpha", [(2, 1.2), (3, 1.15), (3, 2.15), (6, 1.11), (6, 1.89)])
def test_exponent_does_not_depend_on_the_scale_D(L, alpha):
    """D -> c D und Startbestand -> c Startbestand: dieselbe Rechnung in anderer Skala (die Regel ist positiv homogen).

    Mit c als Zweierpotenz ist die Float-Bahn bitgleich skaliert, der Exponent gleich auf 1e-9 (alpha bleibt keine Zweierpotenz). Mit anderem c rechnet der Rechner
    eine andere gerundete Bahn: bei positivem Exponenten schwankt die Messung dann mit der endlichen Messdauer (6000 Schritte), gefunden bis 0,0045 (L = 3, alpha = 1,15);
    das Band im Test ist 0,006."""
    base = K.lyapunov_grid(L, [alpha], D=10.0, i0=0.5).lam[0]
    for c in (0.5, 8.0):
        assert K.lyapunov_grid(L, [alpha], D=10.0 * c, i0=0.5 * c).lam[0] == pytest.approx(base, abs=1e-9), c
    for c in (0.3, 7.0, 130.0):
        assert K.lyapunov_grid(L, [alpha], D=10.0 * c, i0=0.5 * c).lam[0] == pytest.approx(base, abs=0.006), c


def test_span_is_proportional_to_D():
    """Spannweite des Bestands hinter der Grenze: 1,2 alpha_krit (L = 3), D = 1, 10, 100 (Startbestand 0,5 D): dieselbe Zahl in Vielfachen von D."""
    a = 1.2 * K.alpha_crit(3)
    spans = [K.orbit_summary(3, a, D, 0.5 * D)["span"] for D in (1.0, 10.0, 100.0)]
    assert spans == pytest.approx([spans[0]] * 3, rel=1e-3)
    assert spans[0] == pytest.approx(4.300, abs=5e-3)


@pytest.mark.parametrize("L", [2, 3, 6, 12])
def test_below_the_limit_the_clamp_changes_nothing_for_small_disturbances(L):
    """Unterhalb von alpha_krit und bei Startbestand unter D/alpha greift die Klemmung nie: geklemmte und ungeklemmte Bahn sind gleich."""
    a = 0.7 * K.alpha_crit(L)
    i0 = 0.5 * 10.0 / a                       # halbe Randlinie D/alpha: Betrag |u| <= D/2 am Start
    clamped = K.simulate(L, a, 10.0, i0, 3000)
    linear = K.simulate(L, a, math.inf, i0, 3000)
    assert not clamped.clipped.any()
    assert np.array_equal(clamped.i, linear.i)


@pytest.mark.parametrize("start", [-200.0, -50.0, -10.0, 10.0, 50.0, 200.0])
def test_below_the_limit_large_disturbances_decay_even_with_the_clamp(start):
    """Störung bis 200 D: Unterhalb von alpha_krit (L = 3, 0,7 alpha_krit) klingt der Bestand trotz Klemmung ab."""
    a = 0.7 * K.alpha_crit(3)
    orb = K.simulate(3, a, 10.0, start * 10.0, 20000)
    assert np.max(np.abs(orb.i[-500:])) < 1e-6 * 10.0


@pytest.mark.parametrize("L", [1, 2, 3, 6, 12])
def test_unclamped_exponent_is_negative_below_and_growth_above_the_limit(L):
    ak = K.alpha_crit(L)
    assert K.lyapunov_grid(L, [0.9 * ak], D=math.inf, burn=300, rec=600).lam[0] < 0
    above = K.simulate(L, 1.1 * ak, math.inf, 1.0, 1500)          # ohne Klemmung: lineares Wachstum, kein Chaos
    assert abs(above.i[-1]) > 1e3 * abs(above.i[0])
    assert K.lyapunov_grid(L, [1.1 * ak], D=math.inf, burn=300, rec=600).lam[0] > 0


def test_same_seed_gives_the_same_noise_and_orbit_other_seed_differs():
    e1, e2, e3 = K.noise_sequence(35, 2000, 0.1), K.noise_sequence(35, 2000, 0.1), K.noise_sequence(36, 2000, 0.1)
    assert np.array_equal(e1, e2) and not np.array_equal(e1, e3)
    o1 = K.simulate(3, 1.8, 10.0, 5.0, 2000, noise=e1)
    o2 = K.simulate(3, 1.8, 10.0, 5.0, 2000, noise=e2)
    o3 = K.simulate(3, 1.8, 10.0, 5.0, 2000, noise=e3)
    assert np.array_equal(o1.i, o2.i) and not np.array_equal(o1.i, o3.i)


def test_the_two_methods_see_the_same_noise_sequence():
    """Beide Nachbarbahnen bekommen dieselbe Rauschfolge: Zweibahn und Benettin liefern entlang dieser Folge nahe beieinander liegende Exponenten."""
    al = [1.4, 1.8, 2.15]
    noise = K.noise_sequence(35, K.BURN + K.REC, 0.1)
    lam = K.lyapunov_grid(3, al, noise=noise).lam
    wolf = K.two_orbit_grid(3, al, noise=noise)
    assert np.max(np.abs(lam - wolf)) < 0.003


def test_noise_columns_with_their_own_sequence_equal_single_runs():
    seqs = np.stack([K.noise_sequence(40 + j, 3000, 0.1) for j in range(3)], axis=1)
    both = K.lyapunov_grid(3, [1.8] * 3, burn=1000, rec=2000, noise=seqs).lam
    for j in range(3):
        single = K.lyapunov_grid(3, [1.8], burn=1000, rec=2000, noise=seqs[:, j]).lam[0]
        assert both[j] == pytest.approx(single, abs=1e-12)


def test_grid_columns_are_independent_of_each_other():
    """Ein Gitterpunkt rechnet dasselbe, ob er allein oder im Gitter steht (kein Übersprechen zwischen den Spalten)."""
    al = np.array([0.8, 1.15, 1.7, 2.1])
    grid = K.lyapunov_grid(3, al, burn=1500, rec=2500).lam
    for k, a in enumerate(al):
        assert K.lyapunov_grid(3, [a], burn=1500, rec=2500).lam[0] == pytest.approx(grid[k], abs=1e-12)


def test_D_zero_orders_only_when_the_stock_is_negative():
    """D = 0: u = max(-alpha i, 0). Von Hand (L = 2, alpha = 1, Start i = -1): i = -1, -1, 0, 1, 1, 1 ...; die Bahn bleibt bei i = 1 mit Bestellung 0."""
    orb = K.simulate(2, 1.0, 0.0, -1.0, 8)
    assert list(orb.i[:6]) == [-1.0, -1.0, 0.0, 1.0, 1.0, 1.0]
    assert float(np.max(np.abs(orb.u[3:]))) == 0.0


def test_start_exactly_on_the_edge_line_is_continuous():
    """Auf der Randlinie i = D/alpha ist -alpha i = -D: geklemmt oder nicht, die Bestellung ist dieselbe (Fractions: exakt)."""
    step = K.make_step(2, Fr(6, 5), Fr(10))
    assert step((Fr(25, 3), Fr(0))) == (Fr(25, 3), Fr(-10))
    below, above = step((Fr(25, 3) - Fr(1, 10**9), Fr(0))), step((Fr(25, 3) + Fr(1, 10**9), Fr(0)))
    assert abs(below[1] - above[1]) <= Fr(6, 5) * Fr(2, 10**9)


def test_L_equal_1_has_no_pipeline_and_limit_2():
    """L = 1: i_{t+1} = i_t - alpha i_t (geklemmt: i_t - D), Grenze alpha = 2: bei alpha = 1,5 (unter 2) konvergiert der Bestand gegen 0 ohne Klemmung nach Start im Innern."""
    orb = K.simulate(1, 1.5, 10.0, 2.0, 5)
    assert list(orb.i) == pytest.approx([2.0, -1.0, 0.5, -0.25, 0.125, -0.0625])
    assert not orb.clipped.any()
    clipped = K.simulate(1, 1.5, 10.0, 20.0, 2)          # -1,5 * 20 = -30 < -10: geklemmt, i_1 = 20 - 10 = 10
    assert clipped.clipped[0] and clipped.i[1] == pytest.approx(10.0)


def test_alpha_at_the_limit_is_only_a_band_not_an_equality():
    """Am Rand alpha_krit selbst gilt weder Stabilität noch Instabilität eindeutig; die Auswertung vergleicht mit einem Band (0,9995 und 1,0005)."""
    ak = K.alpha_crit(3)
    assert K.lyapunov_grid(3, [ak * 0.9995], D=math.inf, burn=100, rec=300).lam[0] < 0 < K.lyapunov_grid(3, [ak * 1.0005], D=math.inf, burn=100, rec=300).lam[0]
