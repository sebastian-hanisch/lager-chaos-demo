# Chaos im Lager – Verzögerung plus Klemmung – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-lager-chaos-demo.streamlit.app/)**

Stück der Linie **Dynamische Systeme** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Die Linie fragt, was passiert, wenn man eine Regel immer wieder auf ihr eigenes Ergebnis anwendet:
Ruhepunkt, Kreislauf oder Chaos, und was davon bleibt im Rechner. Dieses Stück hängt an zwei Vorgängern: [`lyapunov-chaos-demo`](https://github.com/sebastian-hanisch/lyapunov-chaos-demo) misst den Lyapunov-Exponenten an einer eindimensionalen Abbildung
mit Ableitung entlang der Bahn, [`verzoegerung-hopf-demo`](https://github.com/sebastian-hanisch/verzoegerung-hopf-demo) erzeugt Chaos in stetiger Zeit mit glatter Nichtlinearität. Offen blieb, ob eine einfache **stückweise lineare** Regel mit Gedächtnis
(Zustand der Dimension L) ebenfalls Chaos erzeugt und wie man es dort misst (Tangentialvektor im L-dimensionalen Zustand, zweite Methode). Das Lager ist das Beispiel, in dem eine Klemmung (Bestellung ≥ 0) genügt. Die Zweierpotenz-Falle am Ende hat ihre Theorie in
[`gleitkomma-chaos-demo`](https://github.com/sebastian-hanisch/gleitkomma-chaos-demo); hier ist sie ein Testfall.

## Warum dieses Problem

Ein Lager, das nach einer einfachen linearen Regel nachbestellt, schwingt hinter der Grenze **α_krit = 2 sin(π/(4L − 2))** immer stärker auf (L: Lieferzeit in Perioden). Praktisch kann man aber nicht negativ bestellen. Mit dieser Klemmung blieb die Schwingung in allen gerechneten Fällen beschränkt,
und über weite Bereiche von α (bis 2,4, nach dem Kriterium λ > 0,003) ist sie **chaotisch**, ohne jedes Rauschen und bei konstantem Bedarf. Die Demo fragt: **Wie oft, wo und wie sicher ist das so, und was an den Zahlen ist Messung, was ein Rechner-Artefakt?**
Chaos in Lagermodellen ist in der Fachliteratur beschrieben (Mosekilde und Larsen 1988; Sterman 1988); diese Arbeiten wurden **nicht gelesen** (nur Verzeichniseinträge), ob sie dieses Modell (ein Lager, lineare Regel, Klemmung) betreffen, ist offen. Der eigene Anteil sind die Lyapunov-Gitter mit zweitem Verfahren, das Rauschen, die Zweierpotenz-Falle und der Reiter Randkollision.

## Modell

Abweichungen vom Soll, mittlerer Bedarf D, Lieferzeit L, Anpassung α, **kein Leitungsgewicht** (β = 0): Zustand x_t = (i_t, u_{t−1}, …, u_{t−L+1}) mit Bestand i (minus Soll) und Bestellungen u (minus D).

- Regel mit Klemmung: **u_t = max(−α i_t, −D)**; die Bestellung D + u_t darf nicht unter 0 fallen.
- Lieferung: **i_{t+1} = i_t + u_{t+1−L} − e_t**, e_t die Bedarfsabweichung (0 ohne Rauschen, sonst normalverteilt mit Standardabweichung s · D).
- Alle Exponenten gelten je Periode. **Chaotisch** heißt in dieser Demo: größter Lyapunov-Exponent **über 0,003** bei 6000 Messschritten nach 6000 Einschwingschritten (Schwelle und Dauer sind gesetzt). Unter −0,003 gilt der Punkt als Fenster (periodisch); dazwischen steht „keine Entscheidung“.
- Die Formeln stehen im Expander 📐 der App; D, α und der Startbestand sind Regler (die Skala D ändert den Exponenten nicht).

## Methodik

| Verfahren | Wie es arbeitet | Grenze |
|---|---|---|
| **Benettin** (`lgc_dynamics.lyapunov_grid`) | Tangentialvektor folgt der Jacobi-Matrix (bei Klemmung ist die Ableitung der Bestellung 0), nach jedem Schritt auf Länge 1; Mittel von ln(Länge); über ein Gitter von α vektorisiert | endliche Messdauer: Punkte nahe 0,003 hängen an ihr |
| **Zweibahn** (`two_orbit_grid`) | Nachbarbahn im Abstand 10⁻⁹, nach jedem Schritt auf diesen Abstand zurückgesetzt; Mittel von ln(Abstand/10⁻⁹) | gleiche Messdauer; zweites Verfahren, kein zweiter Beweis |
| **Zyklussuche nach Brent** (`brent_cycle`) | Wiederkehr eines bitweise gleichen Float-Zustands, bis zu einer Schrittgrenze; dazu die exakte Bahn mit Brüchen (`fractions`) | ohne Wiederkehr innerhalb der Grenze steht „kein Zyklus gefunden“, nicht „kein Zyklus“ |
| **Normalform der Randkollision** (`normal_form_lyapunov`) | Benettin an der zweidimensionalen Abbildung G(x, y) = (τx + y + μ, −δx) mit zwei Seiten, über τ_R vektorisiert | 60000 Schritte, Zustände über 10⁸ gelten als unbeschränkt |

Alle Zufallsfolgen (Bedarfsrauschen, gestreute Startwerte) kommen aus einem eigenen **SplitMix64** (`lgc_rng.py`, vektorisiert gegen die Klasse geprüft); das Gitter über α hat einen nicht-dyadischen Schritt und einen kleinen irrationalen Versatz, **nie** liegt α auf einer Zweierpotenz.
Die Live-Rechnung der App (jeder Abschnitt) braucht je Eingabe etwa 0,2 bis 1,1 Sekunden (Messung auf dem Entwicklungsrechner, nicht festgeschrieben); die schweren Reihen sind **vorgerechnet** (`data/lgc_frozen.json`, erzeugt von `tools/make_frozen.py`):
das feine Gitter, die Gegenprobe mit 6000 Läufen und die Zyklussuche bis 3·10⁶ Schritte.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: D = 10, α-Gitter von 0,15 (L = 3), 0,08 (L = 6), 0,04 (L = 12) bis 2,4 mit 450 / 450 / 300 Punkten (L = 1 und 2: 0,05 bis 3,0 mit 3000 Punkten), Startbestand 0,05 D, 6000 + 6000 Schritte. Die Tests prüfen **Bänder** (einzelne Punkte nahe der Schwelle können auf einer anderen Plattform kippen);
die Zahlen unten sind die des Entwicklungsrechners.

| Frage | Befund | Test |
|---|---|---|
| Wie oft ist das Lager oberhalb von α_krit chaotisch? | **181 von 356 Gitterpunkten (50,8 %) bei L = 3, 221 von 410 (53,9 %) bei L = 6, 150 von 288 (52,1 %) bei L = 12**, jeweils bis α = 2,4; L = 2: 673 von 2033 (33,1 %); L = 1 (α_krit = 2, bis 3,0): 1012 von 1016 (99,6 %) | `test_share_chaotic_above_the_limit`, `test_shares_in_percent_as_in_the_readme` |
| Und bei α ≤ 1 (als übliche Einstellung ohne Quelle angenommen)? | L = 3: 0 von 76 Punkten; L = 6: 9 von 139 (6,5 %); L = 12: 22 von 110 (20,0 %). Der erste chaotische Gitterpunkt liegt bei 1,63 (L = 3), 2,62 (L = 6) und 4,29 (L = 12) mal α_krit | `test_no_chaos_for_L3_below_alpha_1_and_some_for_L6_L12`, `test_first_chaotic_point_in_multiples_of_alpha_crit` |
| Zusammenhängende Bereiche (L = 3) | α von 1,007 bis 1,157, 1,337 bis 1,357 und 1,677 bis 2,397; dazwischen Fenster (131 Punkte mit Exponent unter −0,003) | `test_chaotic_ranges_of_L3` |
| Stimmen die zwei Verfahren überein? | Gleiches Urteil (chaotisch ja oder nein) in 356 von 356 (L = 3), 410 von 410 (L = 6), 287 von 288 (L = 12), 2033 von 2033 (L = 2) Punkten; größte Zahlenabweichung 0,0009 (L = 3 und 6), 0,0011 (L = 12), 0,0020 (L = 2). Für L = 1 nicht behauptet (1013 von 1016, größte Abweichung 0,06) | `test_two_methods_agree_in_the_grids`, `test_benettin_agrees_with_the_qr_product_at_20_points`, `test_two_orbit_method_agrees_with_benettin_at_20_points` |
| Zeigt das feine Gitter in den Chaos-Bereichen ein Fenster? | L = 3, α in [1,677; 2,397]: **4000 von 4000 chaotisch**, kleinster Exponent +0,0044; L = 6, [1,520; 2,381]: 3988 von 4000, kleinster +0,0015; L = 12, [1,632; 2,340]: 3942 von 4000, kleinster +0,0016. **Kein Punkt mit Exponent unter −0,003** (4000 Punkte je Bereich, Abstand 1,8·10⁻⁴ bis 2,2·10⁻⁴). Gegenprobe L = 3 mit 1000 zufälligen α und je 6 zufälligen Anfangszuständen in [−50; 50], doppelter Messdauer: **6000 von 6000 Läufen** mit Exponent über 0,003 (kleinster +0,0042). Schmalere Fenster als der Gitterabstand sind nicht ausgeschlossen | `test_fine_grid_has_no_window_and_matches_the_frozen_numbers`, `test_counter_check_with_random_alphas_and_starts_is_frozen_as_6000_of_6000`, `test_counter_check_no_window_in_the_L3_chaos_range_with_random_alphas_and_starts` |
| Bleibt das Chaos mit Rauschen im Bedarf? | Entlang derselben Rauschfolge (Zufalls-Seed 35, 100 Gitterpunkte, 79 oberhalb α_krit bei L = 3, 91 bei L = 6): Anteil chaotisch bei s = 0 → 0,25 (Standardabweichung in Vielfachen von D): **L = 3: 50,6 → 62,0 %**, L = 6: 54,9 → 63,7 %. Fenster mit Exponent unter −0,003: L = 3 39,2 → 38,0 % bei Seed 35 (34 bis 38 % über andere Rausch-Seeds; 31 → 30 Punkte, der chaotische Anteil bleibt in allen Seeds 62,0 %); die Feinstruktur verschwindet bei L = 3 (Punkte im Band ±0,003: von 8 auf 0), bei L = 6 nur zum Teil (von 16 auf 7). Bei α = 0,7 α_krit greift die Klemmung selbst bei s = 0,25 in höchstens 5 von 200 000 Perioden (je 5 Folgen zu 40 000 Perioden, L = 3, 6, 12); bei s = 0,5: L = 3 in 1,8 % der Perioden, L = 6 in 0,09 %, L = 12 nie | `test_noise_keeps_chaos_and_windows`, `test_noise_clamp_in_the_stable_range_is_rare_for_s_025_and_visible_for_s_05` |
| Wie lange ist das Lager vorhersagbar? | Anfangsunterschied 10⁻⁹ im Bestand, „sichtbar“ heißt Abstand der Zustände über 1 % von D: L = 3, α = 2,15 (Exponent +0,0811): 1/λ = **12,3 Perioden**, sichtbar nach 215 Perioden; L = 3, α = 1,0269: 1/λ = 77; L = 6, α = 1,8864: 37; L = 12, α = 2,0096: 110 und α = 1,2229: 128. Die Zeit bis „sichtbar“ gleicht ln(10⁸)/λ = 18,4/λ auf 30 % (5 Punkte). Voraussetzung: Das Lager schwingt schon mit Spannweite 4 bis 200 D (Tabelle unten) | `test_horizon_is_about_ln_ratio_over_lambda`, `test_horizon_values_quoted_in_the_readme` |
| Wie groß ist die Spannweite hinter der Grenze? | L = 3: **3,24 D bei 1,001 α_krit** (Zyklus der Periode 10 = 4L − 2 nach langem Einschwingen; nach 6000 + 6000 Schritten noch 3,249 D, ein Zyklus ist dort nicht erkennbar) und **4,300 D bei 1,2 α_krit** (Zyklus der Periode 81, Exponent −0,0026), gleich für D = 1, 10 und 100; α = 2,15: 12,87 D; L = 6, α = 1,8864: 45,6 D; L = 12, α = 2,0096: etwa 186 D (`test_spans_at_chaotic_points_quoted_in_the_readme`) | `test_span_behind_the_limit_for_L3`, `test_spans_at_chaotic_points_quoted_in_the_readme`, `test_directly_behind_the_limit_L3_cycles_of_period_10_and_81_and_one_undecided_point` |
| Das Handbeispiel (L = 2, α = 1,2, D = 10, Start (5, 0)) | Erster Klemmschritt in Periode 6; Zyklus der Periode 6 (= 4L − 2 ist hier die kritische, nicht diese Periode), Exponent −0,1090, Spannweite 2,40 D, Klemmung in 2000 von 6000 Perioden; die Bahn in Brüchen stimmt mit der Hand-Tabelle der Periode 0 bis 7 exakt | `test_hand_orbit_in_fractions_is_exact`, `test_each_preset_shows_the_expected_verdict_line` |
| Das Lager mit L = 2 und der Aufsatz von Banerjee, Yorke und Grebogi | Das Lager ist **exakt** eine Normalform der Randkollision (größte Abweichung der Bahnen 8,9·10⁻¹⁵ über 2000 Perioden bei α = 1,2; bei α = 1,6, Exponent positiv, wächst der Rundungsunterschied auf 4·10⁻⁸ in 200 Perioden). Die Bedingungen (17) und (18) und das Gebiet für robustes Chaos sind **nicht erfüllt** (geprüft für α = 0,9; 1,2; 1,6; 2,4 in beiden Lesarten, siehe unten), und L = 2 hat periodische Fenster: 57,5 % der Gitterpunkte oberhalb α_krit (1169 von 2033) | `test_hand_example_clamp_and_normal_form_numbers`, `test_L2_store_violates_conditions_17_18_and_the_region_in_both_readings`, `test_L2_has_periodic_windows`, `test_two_period_store_equals_its_normal_form_over_2000_steps` |
| Positivkontrolle: eine Normalform, die den Bedingungen genügt | μ = 1, 300 Werte von τ_R in [−3; −(1 + δ_R) − 0,01], 60000 Schritte, drei Parametersätze (τ_L, δ_L, δ_R) = (1,9; 0,5; 0,5), (1,7; 0,3; 0,05), (1,7; 0,3; 0,3): chaotisch 23, 202, 158; unbeschränkt 277, 98, 142; **kein Fenster**. **Alle 367 Punkte mit (19) > 0 sind chaotisch** (Exponent +0,228 bis +0,645, keiner mit Fenster); in den zwei letzten Sätzen sind alle Punkte mit (19) ≤ 0 unbeschränkt, im ersten sind 16 chaotisch. Kontrast (1,7; 0,3; 0,05), τ_R von −1,0 bis 1,4 (49 Punkte, Bedingung (18) verletzt): 0 chaotisch, 41 mit Fenster, 7 unbeschränkt | `test_normal_form_positive_control_367_of_367`, `test_normal_form_contrast_with_condition_18_violated`, `test_source_conditions_are_evaluated_at_the_paper_example_sets` |
| Zweierpotenz-Falle (L = 3, D = 10, Float64) | α = 2: von 6 einfachen Startwerten (0,5; 0,25; 0,125; 3; 7,5; 0,1) enden **5 in einem Zyklus** (Perioden 62, 60, 63, 30, 15; Vorperioden 17, 17, 42, 9, 2), der Start 0,1 nicht (Suche bis 3·10⁵ und 3·10⁶ Schritte). Die exakte Bahn mit Brüchen (Start 1/2) hat Vorperiode 17 und Periode 62, **gleich der Float-Bahn**; Multiplikator 81, Exponent ln 81/62 = **+0,0709**: ein instabiler Zyklus, kein Chaos. Bei α + 10⁻⁹·√2 und α = 1,1: 0 von 6 einfachen Startwerten. 100 gestreute Startwerte (SplitMix64, Zufalls-Seed 35, in [−50; 50]): bei α = 2, α + 10⁻⁹·√2 und α = 1,1 **keiner** mit Zyklus in 3·10⁶ Schritten | `test_power_of_two_case_five_of_six_simple_starts_end_in_a_cycle`, `test_alpha_1_1_has_no_cycle_for_simple_or_scattered_starts_in_3e5_steps`, `test_frozen_search_to_3e6_steps`, `test_three_period_cycle_float_equals_exact_fractions` |
| Der Zyklus mit positivem Exponenten im Handbeispiel | L = 2, α = 2, D = 1, Start 1/2: Vorperiode 1, Zyklus der Länge 8, Multiplikatoren 5 und 0, Exponent ln 5/8 = +0,2012 | `test_two_period_cycle_with_alpha_2_by_hand`, `test_hand_cycle_with_alpha_2_in_L2` |

Ergebnis der Bahn je Preset (Verdikt der App, Start 0,5 D): Von Hand (L = 2, α = 1,2): Zyklus der Periode 6; Grenzzyklus (L = 3, α = 0,7, das 1,13fache der Grenze): Zyklus der Periode 10, Exponent −0,0420, Spannweite 3,87 D; Chaos L = 3 (α = 2,15): Exponent +0,0803, Spannweite 12,87 D;
Rauschen (α = 1,8, ohne Rauschen gerechnet): +0,0413; Randkollision (L = 2, α = 1,6): +0,0472, Spannweite 4,04 D; Zweierpotenz-Falle (α = 2, Start 5): Zyklus der Periode 16 bei Exponent +0,0687, also kein Chaos (`test_each_preset_shows_the_expected_verdict_line`).

### Befunde und Korrekturen gegenüber dem Plan

- **Kleine Abweichungen der Zählung:** Der Plan nannte 220 von 410 (L = 6) und 151 von 288 (L = 12) aus einer Messung mit 8000 + 4000 Schritten; mit 6000 + 6000 Schritten sind es 221 und 150. Bei α ≤ 1 für L = 12 nennt der Plan 19,1 %, hier sind es 22 von 110 (20,0 %). Die neue Messung gilt.
- **„Grenzzyklus“ hinter der Grenze:** Der Plan schrieb bei 1,2 α_krit „Grenzzyklus, Spannweite 4,3 D“. Die Spannweite stimmt (4,300 D); der Exponent liegt aber im Band ±0,003 (−0,0026), die Periode ist 81, und bei α = 0,74 (1,198 α_krit, ein Reglerwert der App) kehrt der Zustand bis Periode 4000 nicht wieder (Exponent +0,0001): dort steht „keine Entscheidung“. Das Preset „Grenzzyklus“ liegt deshalb bei α = 0,7 (Periode 10).
- **Zweierpotenz und der Exponent allein:** Das Preset „Zweierpotenz-Falle“ hat α = 2; Benettin meldet dort +0,0687 und „chaotisch“ wäre falsch. Die App meldet bei positivem Exponenten **und** wiederkehrendem Zustand „instabiler Zyklus, kein Chaos“. Das schränkt die frühere Aussage der Planung („jede Bahn landet auf einer instabilen periodischen Bahn“) ein: gemessen für 5 von 6 einfachen Startwerten, nicht für gestreute (0 von 100).
- **Lesart der Bedingungen:** Der Plan wertete (17) mit τ_L = τ_R = 1, δ_L = α, δ_R = 0 und μ = −D aus. Der Aufsatz formuliert für μ > 0; die gespiegelte Lesart (x → −x) vertauscht links und rechts. Die App gibt **beide** an; in beiden ist (17) verletzt (ungespiegelt mit Abstand α, gespiegelt an der Gleichheit 1 > 1 + 0).
- **Gebiet des Aufsatzes:** Es verlangt 1 > δ_L > 0 und 1 > δ_R > 0; im Plan stand nur „δ > 0“. Das Lager hat auf einer Seite δ = 0 und liegt auch deshalb außerhalb.
- **Skala:** Ein Test „gleicher Exponent bei anderem D“ gilt bitgenau nur für Skalierung mit einer Zweierpotenz (Exponent gleich auf 10⁻⁹); bei anderen Faktoren schwankt die Messung bis 0,0045 (L = 3, α = 1,15), weil der Rechner eine andere gerundete Bahn rechnet (`test_exponent_does_not_depend_on_the_scale_D`).
- **Mechanismus:** Für L ≥ 3 ist die Normalform nicht anwendbar (Zustand ab Dimension 3). Dass eine Randkollision das Diagramm erklärt, bleibt **Hypothese**; die Messung belegt nur, dass in den feinen Gittern der Chaos-Bereiche kein Fenster auftritt (unten die Grenzen).

## Quellen

| Quelle | Stufe | Was hier verwendet wird |
|---|---|---|
| Banerjee, Yorke, Grebogi: Robust Chaos, Physical Review Letters 80 (1998) 3049 (arXiv chao-dyn/9803001) | Formeln und Gebietsbeschreibung im LaTeX-Quelltext der ar5iv-Fassung gelesen (2026-11-14), **nicht gegen das PDF**, den ganzen Aufsatz nicht gelesen | Normalform, (17), (18), (19), das Gebiet für robustes Chaos (1 > δ > 0, τ_R < −(1 + δ_R), τ_L > 2√δ_L, (19)); „robustes Chaos“ als Umgebung im Parameterraum ohne periodische Attraktoren und ohne koexistierende Attraktoren |
| Mosekilde und Larsen 1988; Sterman 1988 (Chaos in Lagermodellen) | nur Verzeichniseinträge, **nicht gelesen** | Hinweis „in der Fachliteratur beschrieben“ |

## Ehrliche Grenzen

- **Ein Modell:** β = 0, additives normalverteiltes Rauschen, konstanter mittlerer Bedarf, ein Lager. Chaos hier heißt Exponent über 0,003 bei 6000 Messschritten; Punkte nahe der Schwelle hängen an der Messdauer (Test mit 12000 + 12000 Schritten: 6000 von 6000 Läufen im Bereich von L = 3 bleiben darüber).
- **Anfangszustände:** Das Gitter startet je Punkt bei 0,05 D; weitere Anfangszustände sind nur in der Gegenprobe (6 je α, L = 3) geprüft. In der Planung fand sich bei L = 12, α = 0,609 eine Koexistenz (Exponent von −0,0205 bis +0,0040); eine spätere Stichprobe mit 100 Startwerten (unabhängige Prüfung, nicht im Repo) fand bis +0,096 (99 Starts negativ, 1 über 0,003). Die 20,0 % bei L = 12, α ≤ 1 gelten deshalb nur für den Gitterstart.
- **Rauschen:** Der Exponent gilt **entlang derselben Rauschfolge** (beide Nachbarbahnen sehen sie); der Exponent über alle Rauschfolgen ist nicht gemessen. Bei s ≤ 0,25 ist das Rauschen für L = 3, 6 und 12 klein gegen die Schwingung (Spannweiten 6 bis 210 D), für L = 1 und 2 (Spannweiten 1 bis 5 D) nicht.
- **Fenster schmaler als der Gitterabstand** sind nicht ausgeschlossen; „kein Fenster in den Chaos-Bereichen“ gilt für die Gitter dieser README und die Gegenprobe, nicht für alle α.
- **Mechanismus für L ≥ 3 offen** (nur Hypothese Randkollision). Die Formeln (17) bis (19) wurden nur im Quelltext der ar5iv-Fassung gelesen; die Voraussetzung 1 > δ > 0 steht dort, das Lager erfüllt sie nicht.
- **Praxis:** Chaos tritt je nach Lieferzeit bei Spannweiten von etwa 1 mal dem Bedarf (L = 1) bis über 200 mal dem Bedarf (L = 12) auf; ob reale Lager in diesem Bereich arbeiten, ist offen. Gemessen: bei α ≤ 1 für L = 3 nirgends chaotisch, für L = 6 in 6,5 % und für L = 12 in 20,0 % der Gitterpunkte.
- **Float64:** Ein Gitterpunkt rechnet eine gerundete Bahn; die Exponenten sind Messungen an ihr, kein Beweis für die exakte Dynamik (bei Zweierpotenz-α ist sie sogar eine andere).
- **Nicht mit echtem Mausklick getestet:** Die App ist mit AppTest und im Browser über die Presets geprüft; Plotly-Interaktionen sind gesperrt (keine Auswahl).

## Verifikation

Die Suite hat **168 Tests** (`pytest --collect-only -q tests`), davon die Orakel-, Eigenschafts- und Claims-Tests unten; sie läuft lokal in etwa einer Minute mit `-n 2`.

- **Kern (`tests/test_dynamics.py`):** α_krit, die Bahn des Handbeispiels in Brüchen (Periode 0 bis 7, erster Klemmschritt in Periode 6), die Normalform exakt in Brüchen, der Zyklus von L = 2 und L = 3, Brent an Handfolgen, Bedingungen (17) bis (19) an Handzahlen, Jacobi-Matrix, Spitzen, Zählung der Läufe.
- **Orakel (`tests/test_oracle_lager_chaos.py`):** Benettin gegen das Produkt der Jacobi-Matrizen mit QR-Zerlegung an 20 Punkten (Zahl auf 0,002), Benettin gegen Zweibahn an 20 Punkten (gleiches Urteil, Zahl auf 0,002), ungeklemmt gleicht der Exponent ln des Spektralradius (auf 10⁻³), das Lager mit L = 2 gegen seine Normalform (10⁻¹² über 2000 Schritte), α_krit gegen `numpy.linalg.eigvals` für L = 1 bis 12 (auch die kritische Periode 4L − 2), Normalform gegen QR-Produkt,
  (19) im Sonderfall δ_L = δ_R gegen die verkürzte Form des Aufsatzes (200 Zufallspunkte), Brent gegen Wörterbuchzählung, SplitMix64 gegen Referenzwerte, Gegenprobe für „kein Fenster“ mit zufälligen α und Anfangszuständen.
- **Eigenschaften (`tests/test_eigenschaften_lager_chaos.py`):** Skala D, Spannweite proportional zu D, unterhalb der Grenze ändert die Klemmung nichts (Störungen bis 200 D klingen ab), ungeklemmt negativ unterhalb und Wachstum oberhalb von α_krit, gleicher Zufalls-Seed gleiche Bahn, dieselbe Rauschfolge für beide Verfahren, Gitterspalten unabhängig, Randfälle (D = 0, Start auf der Randlinie, L = 1, α am Rand nur als Band).
- **Claims (`tests/test_claims.py`):** jede Zahl der Tabelle oben als Band, je Preset die erwartete Verdikt-Zeile. **Verdikte (`tests/test_verdicts.py`):** jede Schwelle und jede Zahl im Text an handgesetzten Werten.
- **Presets und Oberfläche:** Presets vollständig, nie eine Zweierpotenz außer dem benannten Testfall; AppTest-Rauchtest (jeder Button, jeder Regler am Minimum und Maximum, alle Minima und alle Maxima zusammen, jedes Preset, jede Auswahl), Layout (Schnellstart in der Seitenleiste, aktives Preset, jeder Regler in `SETTING_SPECS`), Footer, Laufzeit-Importe.
- **Nicht im Test:** die Zahlen der „vorgerechneten“ Reihen sind nur als Stichprobe neu gerechnet (feines Gitter und Gegenprobe vollständig, Zyklussuche bis 3·10⁶ Schritte für einen Start je α); Zahlen zur Messdauer 12000 + 12000 stehen nur in der Gegenprobe.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: Bahn, Kernfragen, fünf Reiter (Chaos-Gitter, Vorhersagehorizont, Rauschen, Randkollision, Zweierpotenz-Falle) |
| `lgc_dynamics.py` | Kern: Lager, Benettin, Zweibahn, Normalform der Randkollision, Bedingungen (17) bis (19), Zyklussuche |
| `lgc_rng.py` | SplitMix64 (Kopie aus `restaurant-tischmix-demo`, erweitert um vektorisierte Felder) |
| `lgc_evaluation.py` | Auswertungen der Abschnitte und ihre Verdikte (reine Funktionen) |
| `lgc_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `lgc_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `lgc_layout.py` | Fester Kopf (Stylesheet) |
| `lgc_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `data/lgc_frozen.json` | Vorgerechnete Reihen (feines Gitter, Gegenprobe, Zyklussuche bis 3·10⁶ Schritte) |
| `tools/make_frozen.py` | Erzeugt `data/lgc_frozen.json` (einige Minuten) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `tests/` | Kern, Orakel, Eigenschaften, Claims, Verdikte, Presets, Layout, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Leitungsgewicht β:** Wer die bestellte Ware mitzählt, verschiebt die Grenze α_krit; die Demo misst nur β = 0 (dort steht das lineare Lager in der Reihe über Rückkopplung).
- **Mechanismus für L ≥ 3, höherdimensionale Bedingungen der Randkollision:** offen (Hypothese).
- **Exponent über alle Rauschfolgen:** nicht gemessen.
- **Float64-Zykluslängen und stochastisches Runden:** Theorie in [`gleitkomma-chaos-demo`](https://github.com/sebastian-hanisch/gleitkomma-chaos-demo); hier nur der Testfall.
- **PDF-Export:** nicht eingebaut.
- **Anwendungen und Optimum** (Kette mit Prognose, Kosten): `bullwhip-demo`, `lager-und-haendler-demo`; keine Überschneidung bei Lyapunov, Klemmung oder Randkollision.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
