"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio)."""

import random
from dataclasses import dataclass

import streamlit as st

import lgc_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    options: tuple | None = None


SETTING_SPECS = {
    "L_slider": SettingSpec("L", int, C.DEFAULT_L, C.L_MIN, C.L_MAX),
    "alpha_slider": SettingSpec("a", float, C.DEFAULT_ALPHA, C.ALPHA_MIN, C.ALPHA_MAX),
    "D_slider": SettingSpec("D", int, C.DEFAULT_D, C.D_MIN, C.D_MAX),
    "i0_slider": SettingSpec("i0", float, C.DEFAULT_I0, C.I0_MIN, C.I0_MAX),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
    "s_slider": SettingSpec("s", float, C.DEFAULT_S, C.S_MIN, C.S_MAX),
    "tl_slider": SettingSpec("tl", float, C.DEFAULT_TL, C.TL_MIN, C.TL_MAX),
    "dl_slider": SettingSpec("dl", float, C.DEFAULT_DL, C.DL_MIN, C.DL_MAX),
    "dr_slider": SettingSpec("dr", float, C.DEFAULT_DR, C.DR_MIN, C.DR_MAX),
    "tr_slider": SettingSpec("tr", float, C.DEFAULT_TR, C.TR_MIN, C.TR_MAX),
    "start_select": SettingSpec("start", str, C.START_EINFACH, options=C.START_OPTIONS),
    "cap_select": SettingSpec("cap", str, C.DEFAULT_CAP, options=C.CAP_OPTIONS),
}
PRESET_KEYS = {
    "L": "L_slider", "alpha": "alpha_slider", "D": "D_slider", "i0": "i0_slider", "seed": "seed_input", "s": "s_slider",
    "tl": "tl_slider", "dl": "dl_slider", "dr": "dr_slider", "tr": "tr_slider", "start": "start_select", "cap": "cap_select",
}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt (Auswahlfelder nur mit gültigem Eintrag)."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            if spec.options is not None:
                if value not in spec.options:
                    continue
            else:
                if value != value:  # NaN
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
            st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            st.session_state[state_key] = C.PRESETS[name][key]


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen
    Standardwert) mit dem Session State. Ein später ergänzter Regler braucht nur seinen Eintrag in SETTING_SPECS,
    dann hebt auch seine Änderung die Markierung auf (tests/test_layout.py prüft das für jeden Regler)."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(
            st.session_state.get(state_key, spec.default) == preset.get(preset_key_of.get(state_key), spec.default)
            for state_key, spec in SETTING_SPECS.items()
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden (setzt auch die Regler der Reiter weiter unten):")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def power_of_two_alpha():
    """Knopf im Abschnitt Zweierpotenz-Falle: alpha auf 2 setzen."""
    st.session_state["alpha_slider"] = 2.0


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
