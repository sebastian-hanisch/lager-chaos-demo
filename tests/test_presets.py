"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, nie eine Zweierpotenz außer dem benannten Testfall."""

import math

import lgc_constants as C
import lgc_evaluation as ev
import lgc_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_widget_bounds_and_options():
    for name, preset in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            value = preset[key]
            if spec.options is not None:
                assert value in spec.options, (name, key)
            else:
                assert spec.lo <= value <= spec.hi, (name, key)


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Von Hand (L = 2)"]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def _is_power_of_two(x: float) -> bool:
    return x > 0 and math.frexp(x)[0] == 0.5


def test_no_preset_uses_a_power_of_two_except_the_named_test_case():
    for name, p in C.PRESETS.items():
        for key in ("alpha", "D", "tl", "dl", "dr"):
            if name == "Zweierpotenz-Falle" and key == "alpha":
                assert _is_power_of_two(p[key])
                continue
            assert not _is_power_of_two(float(p[key])), (name, key, p[key])
    assert not _is_power_of_two(C.DEFAULT_ALPHA) and not _is_power_of_two(C.DEFAULT_D)


def test_chaos_presets_sit_where_the_measurement_says():
    """Preset „Chaos L = 3“ (alpha = 2,15) liegt im chaotischen Bereich 1,677 bis 2,397; „Grenzzyklus“ liegt beim 1,13fachen der Grenze von L = 3."""
    import lgc_dynamics as K
    p = C.PRESETS["Chaos L = 3"]
    assert p["L"] == 3 and 1.677 < p["alpha"] < 2.397
    g = C.PRESETS["Grenzzyklus"]
    assert g["alpha"] / K.alpha_crit(g["L"]) == __import__("pytest").approx(1.13, abs=0.01)
