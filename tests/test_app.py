"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "\u25b6"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


# --- Zusätze für diese Demo -------------------------------------------------------------------------------------------------------------

import pytest  # noqa: E402

import lgc_constants as C  # noqa: E402
import lgc_presets as P  # noqa: E402


def _texts(at) -> str:
    return " ".join([m.value for m in at.markdown] + [c.value for c in at.caption] + [s.value for s in at.success] + [i.value for i in at.info] + [w.value for w in at.warning])


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_runs_and_sets_its_values(name):
    at = _fresh_app()
    at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.sidebar.slider(key="L_slider").value == C.PRESETS[name]["L"]
    assert at.sidebar.slider(key="alpha_slider").value == pytest.approx(C.PRESETS[name]["alpha"])


def test_all_minima_and_all_maxima_together_do_not_raise():
    for edge in ("min", "max"):
        at = _fresh_app()
        for key, spec in P.SETTING_SPECS.items():
            if spec.options is not None:
                continue
            for bereich in (at.sidebar, at.main):
                for art in ("slider", "number_input"):
                    try:
                        w = getattr(bereich, art)(key=key)
                    except KeyError:
                        continue
                    value = spec.lo if edge == "min" else spec.hi
                    w.set_value(int(value) if spec.caster is int else float(value))
        at.run(timeout=TIMEOUT)
        assert not at.exception, (edge, [str(e) for e in at.exception])


@pytest.mark.parametrize("key,options", [("start_select", C.START_OPTIONS), ("cap_select", C.CAP_OPTIONS)])
def test_every_option_of_the_selectboxes_runs(key, options):
    for option in options:
        at = _fresh_app()
        at.selectbox(key=key).set_value(option).run(timeout=TIMEOUT)
        assert not at.exception, (key, option, [str(e) for e in at.exception])


def test_power_of_two_button_sets_alpha_2_and_the_verdict_names_the_cycles():
    at = _fresh_app()
    at.sidebar.button(key="preset_Chaos L = 3").click().run(timeout=TIMEOUT)
    at.button[[b.label for b in at.button].index("α auf die Zweierpotenz 2 setzen")].click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.sidebar.slider(key="alpha_slider").value == 2.0
    assert "5 von 6 Startwerten enden innerhalb von 300 000 Schritten in einem Zyklus" in _texts(at)
    assert "bei α + 10⁻⁹·√2: 0 von 6" in _texts(at)


def test_hand_preset_shows_the_first_clamp_in_period_6_and_the_two_period_normal_form():
    at = _fresh_app()
    texts = _texts(at)
    assert "Erster Klemmschritt bei t = 6." in texts
    assert "Normalform ja" in texts and "Bedingungen (17) und (18) und das Gebiet des Aufsatzes nein" in texts
