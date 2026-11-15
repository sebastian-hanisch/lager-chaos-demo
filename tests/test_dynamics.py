"""Einheiten des Kerns gegen Handrechnung (DEMO-PLAYBOOK §1b): jede Einheit an einer Mini-Instanz, deren Ergebnis von Hand gerechnet ist.

Handbeispiel (Quelle: dynamische-systeme-planung/handbeispiele_06.py, hier neu in Brüchen nachgerechnet): L = 2, D = 10, alpha = 6/5, Start (i, w) = (5, 0)."""

import math
from fractions import Fraction as Fr

import numpy as np
import pytest

import lgc_dynamics as K

HAND = [  # t: (i_t, w_t = u_{t-1}, u_t)
    (Fr(5), Fr(0), Fr(-6)),
    (Fr(5), Fr(-6), Fr(-6)),
    (Fr(-1), Fr(-6), Fr(6, 5)),
    (Fr(-7), Fr(6, 5), Fr(42, 5)),
    (Fr(-29, 5), Fr(42, 5), Fr(174, 25)),
    (Fr(13, 5), Fr(174, 25), Fr(-78, 25)),
    (Fr(239, 25), Fr(-78, 25), Fr(-10)),
    (Fr(161, 25), Fr(-10), Fr(-966, 125)),
]


def test_alpha_crit_by_hand():
    assert K.alpha_crit(2) == pytest.approx(1.0, abs=1e-15)                      # 2 sin(pi/6)
    assert K.alpha_crit(3) == pytest.approx((math.sqrt(5) - 1) / 2, abs=1e-15)   # 2 sin(pi/10): Goldener Schnitt
    assert K.alpha_crit(1) == pytest.approx(2.0)
    assert K.alpha_crit(6) == pytest.approx(0.2846297, abs=1e-7)


def test_hand_orbit_in_fractions_is_exact():
    step = K.make_step(2, Fr(6, 5), Fr(10))
    x = (Fr(5), Fr(0))
    for t, (i, w, u) in enumerate(HAND):
        assert x == (i, w), t
        x = step(x)
        assert x[1] == u, t
    # erster Klemmschritt bei t = 6: -6/5 * 239/25 = -11,472 < -10
    assert [t for t, (i, w, u) in enumerate(HAND) if u == -10][0] == 6
    assert Fr(-6, 5) * Fr(239, 25) < -10


def test_float_orbit_matches_hand_values_and_marks_the_first_clip():
    orb = K.simulate(2, 1.2, 10.0, 5.0, 8)
    assert list(orb.i[:8]) == pytest.approx([float(h[0]) for h in HAND], abs=1e-12)
    assert list(orb.u[:8]) == pytest.approx([float(h[2]) for h in HAND], abs=1e-12)
    assert int(np.argmax(orb.clipped)) == 6 and orb.clipped.sum() == 1


def test_normal_form_of_the_two_period_store_is_exact_in_fractions():
    """x = i - D/alpha, y = w + D: x' = x + y - D, y' = -alpha x (x <= 0) bzw. 0 (x > 0), von Hand hergeleitet."""
    al, D = Fr(6, 5), Fr(10)
    step = K.make_step(2, al, D)
    x = (Fr(5), Fr(0))
    nx, ny = x[0] - D / al, x[1] + D
    for _ in range(40):
        x = step(x)
        nx, ny = nx + ny - D, (-al * nx if nx <= 0 else Fr(0))
        assert (x[0] - D / al, x[1] + D) == (nx, ny)


def test_normal_form_parameters_of_the_store():
    nf = K.lager_normal_form(1.2, 10.0)
    assert nf == {"tau_L": 1.0, "delta_L": 1.2, "tau_R": 1.0, "delta_R": 0.0, "mu": -10.0}


def test_two_period_cycle_with_alpha_2_by_hand():
    """L = 2, alpha = 2, D = 1, Start 1/2: Vorperiode 1, Zyklus der Länge 8; Produkt der Matrizen hat die Beträge 5 und 0, Exponent ln 5 / 8."""
    pre, per, cyc = K.exact_cycle(2, Fr(2), Fr(1), Fr(1, 2))
    assert (pre, per) == (1, 8)
    assert cyc == [(Fr(1, 2), Fr(-1)), (Fr(-1, 2), Fr(-1)), (Fr(-3, 2), Fr(1)), (Fr(-1, 2), Fr(3)), (Fr(5, 2), Fr(1)), (Fr(7, 2), Fr(-1)), (Fr(5, 2), Fr(-1)), (Fr(3, 2), Fr(-1))]
    mult = K.cycle_multipliers(2, Fr(2), Fr(1), cyc)
    assert mult == pytest.approx([5.0, 0.0], abs=1e-12)
    assert math.log(mult[0]) / per == pytest.approx(0.2012, abs=5e-5)


def test_three_period_cycle_float_equals_exact_fractions():
    """L = 3, alpha = 2, D = 10, Start 1/2: Vorperiode 17, Periode 62; die Float-Bahn ist dieselbe (Dualbrüche sind in Float exakt)."""
    pre, per, _ = K.exact_cycle(3, Fr(2), Fr(10), Fr(1, 2))
    assert (pre, per) == (17, 62)
    assert K.brent_cycle(K.make_step(3, 2.0, 10.0), (0.5, 0.0, 0.0), 10000) == (62, 17)


def test_brent_on_a_hand_sequence():
    """x -> (x * x + 1) mod 7 ab 2: 2, 5, 5, ... (5*5+1 = 26 mod 7 = 5): Periode 1, Vorperiode 1; ab 0: 0, 1, 2, 5, 5: Periode 1, Vorperiode 3.
    Rückgabe (Periode, Vorperiode)."""
    step = lambda x: ((x[0] * x[0] + 1) % 7,)  # noqa: E731
    assert K.brent_cycle(step, (2,), 100) == (1, 1)
    assert K.brent_cycle(step, (0,), 100) == (1, 3)
    # eine Folge ohne Zyklus innerhalb der Grenze
    assert K.brent_cycle(lambda x: (x[0] + 1,), (0,), 50) is None


def test_detect_period_on_the_exact_cycle():
    orb = K.simulate(2, 2.0, 1.0, 0.5, 6000)
    assert K.detect_period(orb, 2, max_p=4000) == 8


def test_detect_period_returns_none_for_a_chaotic_orbit():
    orb = K.simulate(3, 2.15, 10.0, 0.5, 11000)
    assert K.detect_period(orb, 3, max_p=4000) is None


def test_classify_thresholds_are_exclusive_at_the_band_edge():
    assert K.classify(0.0031) == "chaotisch" and K.classify(0.003) == "unentschieden"
    assert K.classify(-0.003) == "unentschieden" and K.classify(-0.0031) == "periodisch"


def test_count_runs_and_chaotic_ranges_on_a_small_row():
    assert K.count_runs([1, 1, 0, 0, 1, 0, 1, 1]) == 3
    assert K.count_runs([0, 0]) == 0 and K.count_runs([]) == 0
    al = [0.1, 0.7, 0.8, 0.9, 1.0, 1.1]                       # alpha_krit(3) = 0,618: der erste Punkt liegt darunter
    lam = [0.5, 0.01, 0.02, -0.1, 0.01, 0.02]
    assert K.chaotic_ranges(al, lam, 3) == [(0.7, 0.8), (1.0, 1.1)]
    c = K.chaos_counts(lam, al, 3)
    assert (c["n_above"], c["n_chaotic"], c["n_window"], c["runs_non_chaotic"]) == (5, 4, 1, 1)


def test_conditions_by_hand():
    # (17), (18) an Zahlen: tau_L = 1,9 > 1 + 0,5; tau_R = -2 < 1 + 0,5 und < -(1 + 0,5)
    assert K.condition_17(1.9, 0.5, -2.0, 0.5) and K.condition_18(1.9, 0.5, -2.0, 0.5)
    assert K.condition_17(1.9, 0.5, 0.0, 0.5) and not K.condition_18(1.9, 0.5, 0.0, 0.5)   # tau_R = 0: (17) ja, (18) nein
    assert not K.condition_17(1.0, 0.0, 1.0, 0.5) and not K.condition_17(1.4, 0.5, -2.0, 0.5)  # 1 > 1 falsch; 1,4 > 1,5 falsch


def test_condition_19_by_hand_value():
    """tau_L = 2, delta_L = 1: Eigenwerte der linken Matrix 1 und 1 (Doppelwurzel). delta_R = 0,5, tau_R = -3:
    1*(-3)*1 - 0,5*1*1 + 0,5*1 - 1*(-3) + 2*1 - 1 - 1*1 = -3 - 0,5 + 0,5 + 3 + 2 - 1 - 1 = 0."""
    assert K.condition_19_value(2.0, 1.0, -3.0, 0.5) == pytest.approx(0.0, abs=1e-12)
    assert K.condition_19_value(1.0, 1.0, -3.0, 0.5) is None      # komplexe Eigenwerte (tau^2 < 4 delta)


def test_robust_region_requires_delta_inside_the_open_unit_interval():
    assert K.robust_chaos_region(1.7, 0.3, -1.8, 0.05)
    assert not K.robust_chaos_region(1.7, 0.3, -1.8, 0.0)         # delta_R = 0: außerhalb des Gebiets
    assert not K.robust_chaos_region(1.7, 0.3, -1.0, 0.05)        # tau_R nicht < -(1 + delta_R)


def test_peak_points_on_a_hand_series():
    hist = np.array([[0.0, 5.0], [2.0, 1.0], [1.0, 3.0], [4.0, 2.0], [3.0, 0.0]])     # zwei Spalten
    cols, vals = K.peak_points(hist)
    assert list(cols) == [0, 0, 1] and list(vals) == [2.0, 4.0, 3.0]


def test_jacobian_by_hand():
    J = K.jacobian(3, 0.5, False)
    assert J.tolist() == [[1.0, 0.0, 1.0], [-0.5, 0.0, 0.0], [0.0, 1.0, 0.0]]
    assert K.jacobian(3, 0.5, True)[1, 0] == 0.0
    assert K.jacobian(1, 0.5, False).tolist() == [[0.5]]
