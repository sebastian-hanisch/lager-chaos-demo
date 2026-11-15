"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets.

Nie alpha, D oder Startwerte als Zweierpotenz voreinstellen (Rechner-Artefakt, siehe Abschnitt Zweierpotenz-Falle); die einzige Ausnahme ist das
ausdrücklich benannte Preset „Zweierpotenz-Falle“ (alpha = 2)."""

L_MIN, L_MAX, DEFAULT_L = 1, 12, 2
ALPHA_MIN, ALPHA_MAX, DEFAULT_ALPHA, ALPHA_STEP = 0.05, 3.0, 1.2, 0.01
D_MIN, D_MAX, DEFAULT_D = 1, 100, 10
I0_MIN, I0_MAX, DEFAULT_I0, I0_STEP = -3.0, 3.0, 0.5, 0.05   # Startbestand in Vielfachen von D
SEED_MAX, DEFAULT_SEED = 999999, 35
S_MIN, S_MAX, DEFAULT_S, S_STEP = 0.0, 0.25, 0.1, 0.01          # Rauschen: Standardabweichung in Vielfachen von D
TL_MIN, TL_MAX, DEFAULT_TL = 1.0, 2.5, 1.7
DL_MIN, DL_MAX, DEFAULT_DL = 0.05, 0.9, 0.3
DR_MIN, DR_MAX, DEFAULT_DR = 0.0, 0.9, 0.05
TR_MIN, TR_MAX, DEFAULT_TR = -3.0, 1.5, -1.8
START_EINFACH = "einfach"           # 0,5; 0,25; 0,125; 3; 7,5; 0,1 (wenige Mantissenbits)
START_GESTREUT = "gestreut"         # 5 Starts aus dem Zufalls-Seed in [-50; 50]
START_OPTIONS = (START_EINFACH, START_GESTREUT)
CAP_OPTIONS = ("30000", "300000")                 # live; die Suche bis 3·10⁶ Schritte ist vorgerechnet (data/lgc_frozen.json)
CAP_VALUES = {"30000": 30_000, "300000": 300_000}
FROZEN_CAP = 3_000_000
DEFAULT_CAP = "300000"

# Messkonventionen (feste Annahmen der Demo, alle in der README genannt)
ALPHA_GRID_MAX = 2.4
NOISE_POINTS = 100
NOISE_SEEDS = 5
HORIZON_FAKTOR = 0.01                   # "sichtbar": Abstand der Zustände über 1 % von D
HORIZON_SCHRITTE = 4000

COLOR_I = "#2F6B65"
COLOR_U = "#c4782b"
COLOR_CHAOS = "#c0392b"
COLOR_LINE = "#4c78a8"
COLOR_GREY = "rgba(150,150,150,0.8)"
COLOR_CHAOS_BAND = "rgba(192,57,43,0.12)"

_BASE = {
    "L": DEFAULT_L, "alpha": DEFAULT_ALPHA, "D": DEFAULT_D, "i0": DEFAULT_I0, "seed": DEFAULT_SEED, "s": DEFAULT_S,
    "tl": DEFAULT_TL, "dl": DEFAULT_DL, "dr": DEFAULT_DR, "tr": DEFAULT_TR, "start": START_EINFACH, "cap": DEFAULT_CAP,
}
PRESETS = {
    "Von Hand (L = 2)": dict(_BASE),
    "Grenzzyklus": {**_BASE, "L": 3, "alpha": 0.7},
    "Chaos L = 3": {**_BASE, "L": 3, "alpha": 2.15},
    "Rauschen": {**_BASE, "L": 3, "alpha": 1.8, "s": 0.25},
    "Randkollision": {**_BASE, "L": 2, "alpha": 1.6, "tl": 1.7, "dl": 0.3, "dr": 0.05, "tr": -1.8},
    "Zweierpotenz-Falle": {**_BASE, "L": 3, "alpha": 2.0},
}
PRESET_HELP = {
    "Von Hand (L = 2)": "Das Handbeispiel: Lieferzeit 2, alpha = 1,2, mittlerer Bedarf 10, Start bei i = 5. Die Bahn lässt sich in Brüchen nachrechnen; erster Klemmschritt in Periode 6.",
    "Grenzzyklus": "Lieferzeit 3, alpha = 0,7 (das 1,13fache der Grenze 0,618): die Schwingung wächst nicht ins Unendliche, sondern geht in einen Zyklus der Periode 10 über.",
    "Chaos L = 3": "Lieferzeit 3, alpha = 2,15: positiver größter Exponent ohne jedes Rauschen. Unten: das Gitter über alpha und der Vorhersagehorizont.",
    "Rauschen": "Lieferzeit 3, alpha = 1,8 mit Bedarfsrauschen 0,25 D: das Chaos bleibt, Fenster mit negativem Exponenten bleiben, die Feinstruktur verschwindet (Abschnitt Rauschen).",
    "Randkollision": "Lieferzeit 2 als Normalform der Randkollision; die Regler des Reiters Randkollision stehen auf einem Punkt, an dem die Bedingungen des Aufsatzes gelten.",
    "Zweierpotenz-Falle": "Lieferzeit 3, alpha = 2 (eine Zweierpotenz): der Rechner rechnet in Dualbrüchen exakt, die Bahn endet in einem Zyklus, obwohl der Exponent positiv ist (Abschnitt Zweierpotenz-Falle).",
}
