"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text) und bekommt Zeilen statt Seeds."""

import pytest

import lgc_evaluation as ev


def _orbit(L=3, alpha=0.4, lam=-0.05, clip=0, per=None, span=0.0):
    return {"L": L, "alpha": alpha, "D": 10.0, "i0": 5.0, "period": per,
            "summary": {"lam": lam, "span": span, "clip_frac": clip / 6000, "clip_count": clip, "urteil": ""}}


def test_number_formatting_uses_decimal_commas():
    assert ev.de(1.5, 2) == "1,50" and ev.de(-0.05, 4, True) == "-0,0500" and ev.pct(0.508, 1) == "50,8 %"
    assert ev.mantissa_bits(2.0) == 1 and ev.mantissa_bits(1.25) == 3 and ev.mantissa_bits(3.0) == 2 and ev.mantissa_bits(7.5) == 4 and ev.mantissa_bits(1.1) > 40


def test_verdict_bahn_stable_without_clamp():
    kind, text = ev.verdict_bahn(_orbit(alpha=0.4, lam=-0.05, clip=0))          # 0,4 / 0,618 = 0,647
    assert kind == "success" and "stabil" in text and "das 0,65fache" in text and "-0,0500" in text and "nie" in text


def test_verdict_bahn_stable_near_the_limit_names_the_slow_decay():
    kind, text = ev.verdict_bahn(_orbit(alpha=0.6, lam=-0.001, clip=0))
    assert kind == "success" and "langsam" in text


def test_verdict_bahn_below_the_limit_with_clamp_active_is_info():
    kind, text = ev.verdict_bahn(_orbit(alpha=0.4, lam=-0.05, clip=17))
    assert kind == "info" and "17 von 6000" in text


def test_verdict_bahn_chaotic_names_exponent_horizon_and_span():
    kind, text = ev.verdict_bahn(_orbit(alpha=2.15, lam=0.1, clip=1800, span=12.34))
    assert kind == "warning" and "chaotisch" in text and "+0,1000" in text and "1/λ etwa 10 Perioden" in text and "12,34 D" in text and "1800 von 6000" in text


def test_verdict_bahn_cycle_with_negative_exponent():
    kind, text = ev.verdict_bahn(_orbit(alpha=0.74, lam=-0.1, per=12, span=4.3))
    assert kind == "info" and "Zyklus der Periode 12" in text and "4,30 D" in text
    assert "nicht wiedergefunden" in ev.verdict_bahn(_orbit(alpha=0.74, lam=-0.1, per=None))[1]


def test_verdict_bahn_band_with_and_without_period():
    kind, text = ev.verdict_bahn(_orbit(alpha=0.9, lam=0.002, per=8))
    assert kind == "info" and "Zyklus der Periode 8" in text
    kind, text = ev.verdict_bahn(_orbit(alpha=0.9, lam=0.002, per=None))
    assert kind == "info" and "keine Entscheidung" in text and "±0,003" in text


def _grid(n_chaotic=181, n_above=356, below=(76, 0), agree=356, dev=0.0009, L=3):
    return {"L": L, "counts": {"n_chaotic": n_chaotic, "n_above": n_above, "share_chaotic": n_chaotic / n_above, "n_below1": below[0], "n_chaotic_below1": below[1], "agree": agree, "max_dev": dev}}


def test_verdict_gitter_counts_every_number():
    kind, text = ev.verdict_gitter(_grid())
    assert kind == "warning"
    assert "181 von 356" in text and "50,8 %" in text and "α_krit = 0,618" in text and "0 von 76 (0,0 %)" in text and "356 von 356" in text and "0,0009" in text


def test_verdict_gitter_without_points_below_one_and_without_chaos():
    kind, text = ev.verdict_gitter(_grid(n_chaotic=0, below=(0, 0)))
    assert kind == "info" and "kein Gitterpunkt oberhalb der Grenze" in text


def _horizon(lam=0.08109, tv=215):
    return {"lam": lam, "t_visible": tv, "thr": 0.1, "d0": 1e-9}


def test_verdict_horizon_positive_exponent():
    kind, text = ev.verdict_horizon(_horizon())
    assert kind == "warning" and "nach **215 Perioden**" in text and "1/λ = 12,3" in text and "ln(0,10 / 10⁻⁹) = 18,4" in text and "227 Perioden" in text and "Anfangsunterschied 10⁻⁹" in text


def test_verdict_horizon_other_branches():
    kind, text = ev.verdict_horizon(_horizon(lam=0.004, tv=None))
    assert kind == "info" and "unter 0,10" in text and "250 Perioden" in text
    kind, text = ev.verdict_horizon(_horizon(lam=-0.01, tv=None))
    assert kind == "success" and "wächst nicht" in text and "-0,0100" in text


def _noise(s=0.25, share0=0.506, win0=40, share_s=0.62, win_s=30, n_above=79):
    return {"s": s, "c0": {"share_chaotic": share0, "n_window": win0, "n_above": n_above}, "cs": {"share_chaotic": share_s, "n_window": win_s, "n_above": n_above},
            "spread": __import__("numpy").array([0.043, 0.05]), "stable_clip": 0.00003, "stable_n": 30000}


def test_verdict_noise_names_both_shares_windows_and_spread():
    kind, text = ev.verdict_noise(_noise(win0=31, win_s=30))
    assert kind == "info"
    assert "50,6 % → 62,0 %" in text and "79 Punkte" in text and "39,2 % → 38,0 %" in text          # 31/79 und 30/79
    assert "von +0,043 bis +0,050" in text and "5 Rauschfolgen" in text and "0,003 %" in text


def test_verdict_noise_without_noise_asks_for_a_value():
    assert ev.verdict_noise(_noise(s=0.0))[0] == "info" and "Regler s" in ev.verdict_noise(_noise(s=0.0))[1]


def test_verdict_lager_nf_all_violated_and_one_satisfied():
    bad = {"alpha": 1.2, "dev": 2.5e-14, "steps": 200, "lese": [{"c17": False, "c18": False, "region": False}, {"c17": False, "c18": False, "region": False}]}
    kind, text = ev.verdict_lager_nf(bad)
    assert kind == "info" and "Normalform ja" in text and "2.5e-14" in text and "nein" in text and "δ = 0 statt δ > 0" in text
    good = {"alpha": 1.2, "dev": 1e-15, "steps": 200, "lese": [{"c17": True, "c18": False, "region": False}]}
    assert ev.verdict_lager_nf(good)[0] == "success"


def test_verdict_nf_success_only_if_every_condition_point_is_chaotic_without_window():
    a = {"tl": 1.7, "dl": 0.3, "dr": 0.05, "n_chaotic": 202, "n_window": 0, "n_diverged": 98, "n_pos": 202, "n_pos_chaotic": 202, "n_pos_window": 0}
    kind, text = ev.verdict_nf(a)
    assert kind == "success" and "202 chaotisch" in text and "0 mit Fenster" in text and "98 unbeschränkt" in text and "in 202 Punkten, davon chaotisch 202" in text
    assert ev.verdict_nf({**a, "n_pos_chaotic": 200})[0] == "info"
    assert ev.verdict_nf({**a, "n_pos": 0, "n_pos_chaotic": 0})[0] == "info"       # keine Punkte mit (19): nichts bestätigt


def _pow(ka, kb, mult=None, exact=None, n=6):
    rows = [{"start": 0.5, "a": ka[0] if ka else None, "b": kb[0] if kb else None}] + [{"start": 0.25, "a": a, "b": b} for a, b in zip(ka[1:], kb[1:])]
    return {"alpha": 2.0, "bits": 1, "cap": 300000, "rows": rows, "mult": mult, "exact": exact}


def test_verdict_pow2_cycle_with_positive_exponent_and_exact_orbit():
    p = _pow([(62, 17), (60, 17)], [None, None], mult=1.5, exact={"pre": 17, "per": 62})
    kind, text = ev.verdict_pow2(p)
    assert kind == "warning" and "2 von 2 Startwerten enden innerhalb von 300 000 Schritten" in text and "Perioden 60 bis 62" in text and "0 von 2" in text and "1 Mantissenbit)" in text
    assert "Periode 62" in text and "ln(1,50)/62 = +0,0065" in text and "instabiler Zyklus" in text and "Vorperiode 17, Periode 62 (gleich der Float-Bahn: ja)" in text


def test_verdict_pow2_no_cycle_is_info():
    kind, text = ev.verdict_pow2(_pow([None], [None]))
    assert kind == "info" and "0 von 1" in text


def test_verdict_bahn_positive_exponent_with_returning_state_is_an_unstable_cycle_not_chaos():
    kind, text = ev.verdict_bahn(_orbit(alpha=2.0, lam=0.0687, per=16, clip=4125, span=12.0))
    assert kind == "info" and "**16 Perioden**" in text and "+0,0687" in text and "kein Chaos" in text and "Zweierpotenz-Falle" in text and "4125 von 6000" in text


def test_verdict_bahn_start_in_the_unstable_equilibrium_is_not_called_a_cycle():
    """Fund der unabhängigen Prüfung: bei Startbestand 0 bleibt die Bahn im instabilen Gleichgewicht stehen (Periode 1, Spannweite 0, Exponent ln des Spektralradius > 0);
    das Verdikt nannte das „instabilen Zyklus“ und begründete ihn mit der Zweierpotenz-Falle."""
    kind, text = ev.verdict_bahn(_orbit(L=2, alpha=1.2, lam=0.0912, per=1, clip=0, span=0.0))
    assert kind == "info" and "Start im instabilen Gleichgewicht" in text and "Periode 1" in text and "+0,0912" in text and "0,00 D" in text
    assert "Zweierpotenz-Falle" not in text and "instabiler Zyklus" not in text and "Mantissenbits" not in text
    assert "Zweierpotenz-Falle" in ev.verdict_bahn(_orbit(alpha=2.0, lam=0.0687, per=2, clip=4125, span=12.0))[1]      # ab Periode 2 bleibt der Hinweis


@pytest.mark.parametrize("L, alpha, lam", [(2, 1.2, 0.0912), (3, 1.8, 0.3150)])
def test_orbit_analysis_from_start_inventory_zero_stays_put_and_gets_the_equilibrium_verdict(L, alpha, lam):
    """Mit dem echten Kern: Startbestand 0 (der Regler erlaubt ihn), D = 10: Periode 1, Spannweite 0, Exponent positiv (L = 2, α = 1,2: +0,0912; L = 3, α = 1,8: +0,3150)."""
    a = ev.orbit_analysis(L, alpha, 10.0, 0.0)
    assert a["period"] == 1 and a["summary"]["span"] == 0.0 and a["summary"]["clip_count"] == 0 and a["summary"]["lam"] == pytest.approx(lam, abs=5e-4)
    kind, text = ev.verdict_bahn(a)
    assert "Start im instabilen Gleichgewicht" in text and "Zweierpotenz-Falle" not in text
    kind, text = ev.verdict_bahn(ev.orbit_analysis(L, alpha, 10.0, 0.5))
    assert "Start im instabilen Gleichgewicht" not in text
