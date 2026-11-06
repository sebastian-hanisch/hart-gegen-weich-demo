# Hart gegen weich – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-hart-gegen-weich-demo.streamlit.app/)** -->

Stück 6 der Linie **Zielfunktionsdesign** („Das Ziel ist das Modell“) in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“ (Arbeitsname; Name, Bereich und Themenfeld werden erst bei der Einbindung in die Website festgelegt). Es baut auf Stück 1 bis 5 auf (`zielvergleich-demo`, `preis-der-fairness-demo`, `zwei-kriterien-demo`, `lexikographisch-demo`, `gleichstand-demo`): dieselbe Zuordnung, jetzt mit einer **Schwelle** auf der Zeitsumme B. **Hart** heißt: unter allen Plänen mit B ≤ T die kleinsten Kosten A. **Weich** heißt: die Schwelle wird zum Strafterm im Ziel, `A + λ·v` mit der Verletzung `v = max(0, B − T)`. Alles ist **simuliert**; keine Aussage über echte Personen, Betriebe oder Lasten.

## Warum dieses Problem

Ein Strafterm ist ein **Gewicht**, und ein Gewicht hat einen Bereich: zu klein bricht er die Schwelle, ab einem endlichen Wert ist er exakt, zu groß kann er Zahlen verschlucken, und „hart“ ist im Löser auch nur eine Toleranz. Die Fragen: Was kostet die Schwelle? Ab welchem λ wählt die Strafe das harte Optimum? Erreicht eine einfache Gewichtssumme ohne Schwelle es überhaupt? Was liefert die weiche Formulierung, wenn die Schwelle unzulässig ist? Und wo wird ein Löser steif?

## Modell

Gegeben Kosten `a[i][j]` und Zeit `b[i][j]` (Zeile Person, Spalte Aufgabe), zwei unabhängige Matrizen. Ein Plan ist eine Permutation `π`; `A(π) = Σ a[i][π(i)]`, ebenso `B(π)`. Die Schwelle ist `T = B_min + τ·(B(π_A) − B_min)` mit dem Kostenoptimum `π_A` (bei ganzen Zahlen abgerundet); `τ = 100 %` ist genau `B(π_A)`, unter 0 % liegt `T` unter der kleinsten Zeitsumme. **Hart:** `min A(π)` unter `B(π) ≤ T`. **Weich:** linear `A + λ·v`, quadratisch `A + λ·v²`, und zum Vergleich die **Gewichtssumme** `A + λ·B` ohne Schwelle. Der **Zielplan** minimiert (Verletzung, A) lexikographisch, bei Gleichstand dann B und der erste Plan; bei zulässiger Schwelle ist er das harte Optimum. **λ\*** ist das kleinste Gewicht, ab dem die Strafe den Zielplan unter den Minimierern hat: das Maximum über alle Pläne mit kleinerem A und größerer Verletzung von `(A_Ziel − A) / (v^p − v_Ziel^p)`; es wird exakt aus der endlichen Planmenge gerechnet. Bei ganzen Zahlen ist `v ≥ 1` und daher `λ* ≤ A_Ziel − A_min` (lineare Strafe). Das **Fenster** der Gewichtssumme sind die λ, für die der Zielplan `A + λ·B` minimiert: Pläne mit `B > B_Ziel` geben die untere, Pläne mit `B < B_Ziel` die obere Schranke; ist das Fenster leer, liegt der Zielplan nicht auf der unteren Hülle der Punkte (A, B) (Stück 3). Formeln stehen im 📐-Expander der App.

## Methodik

Drei Instanzfamilien mit zwei unabhängigen Matrizen: **ganze Zahlen 1 bis 9**, **ganze Zahlen 1 bis 99** und **reell gleichverteilt 0 bis 100**. Bis n = 8 werden alle Pläne aufgezählt (bis 40 320, exakt in ganzen Zahlen); darüber rechnet nur HiGHS (`scipy.optimize.milp`, Voreinstellungen) bis n = 12, mit Schlupfvariable `s ≥ B − T`, `s ≥ 0` und Ziel `A + λ·s`, und mit harter Nebenbedingung `B ≤ T`. Die Ungarische Methode trägt die Strafe nicht, weil die Summe `Σ b` nicht in Paarkosten zerlegbar ist. Gewichte sind Potenzen von 2 (exakt in float64); der Minimierer bei Gleichstand der Strafe ist der Plan mit der kleinsten Verletzung, dann der erste der Aufzählung. Die App zeigt: die Schwelle und ihre Kosten, den gewählten Plan je Gewicht (Kurve von 2⁻³ bis 2⁶⁰), λ\* und das Fenster der Gewichtssumme, die unzulässige Schwelle, HiGHS gegen die Aufzählung und die Toleranz der harten Nebenbedingung. Zählungen über 20 feste Instanzen liegen hinter Schaltern. Verdikte entstehen live aus der Rechnung, jede Zahl im Text kommt aus dem Ergebnis.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stammen aus den Auswertungsfunktionen der App und sind in `tests/test_claims.py` nachgerechnet. „Je 20 feste Instanzen“ heißt `SWEEP_SEEDS` (Seeds 100000 bis 100019), nie ein einzelner Seed; die Zahlen sind **Zählungen und Spannen über Instanzen**, kein Satz und keine Aussage über andere Instanzen. Soweit nicht anders angegeben n = 8 (alle 40 320 Pläne aufgezählt) und die Schwelle bei 50 % des Wegs vom Zeitminimum zum Kostenoptimum. Familien: ganze Zahlen 1 bis 9 / 1 bis 99 / reell.

| Frage | Befund |
|---|---|
| Was kostet die Schwelle? Anstieg von A des harten Optimums gegen das Kostenoptimum (Median, in Klammern Maximum), Schwelle bei 25 / 50 / 75 % | 1 bis 9: **58,4 %** (145,5 %) / **29,9 %** (85,7 %) / **12,1 %** (45,5 %); 1 bis 99: **101,2 %** (240,3 %) / **28,0 %** (151,4 %) / **13,6 %** (64,6 %); reell: **94,9 %** (225,8 %) / **51,1 %** (139,4 %) / **18,2 %** (75,0 %). Die Schwelle ist ohne Wirkung nur bei 1 bis 9 und 75 %: in **3** von 20 |
| Zu kleines Gewicht: Instanzen mit verletzter Schwelle (lineare Strafe), λ = 0,25 / 1 / 4 / 8 / 16 | 1 bis 9: **18 / 6 / 2 / 0 / 0**; 1 bis 99: **19 / 9 / 4 / 2 / 0**; reell: **17 / 12 / 5 / 4 / 2**; bei λ = 32 in keiner Instanz |
| dabei bei λ = 1 (Median über die verletzten Instanzen) | Verletzung **1 / 6 / 10,8** Zeiteinheiten, Kosten **18,8 % / 15,5 % / 17,3 %** unter dem harten Optimum: die weiche Lösung sieht billiger aus, weil sie die Schwelle bricht |
| λ\* der linearen Strafe (Median, in Klammern Maximum), Schwelle 25 / 50 / 75 % | 1 bis 9: **2,00** (5,00) / **0,90** (6,00) / **0,58** (4,00); 1 bis 99: **3,00** (42,00) / **0,96** (13,25) / **0,44** (5,75); reell: **2,09** (10,56) / **1,52** (24,29) / **0,48** (109,84) |
| λ\* der quadratischen Strafe (Median, in Klammern Maximum) | 1 bis 9: **1,12** (5,00) / **0,25** (6,00) / **0,22** (4,00); 1 bis 99: **0,61** (42,00) / **0,09** (6,00) / **0,01** (2,75); reell: **0,24** (6,70) / **0,08** (14,34) / **0,01** (500,55): das Gewicht hat eine Einheit (bei `v < 1` wird es riesig) |
| Quadratische Strafe, Instanzen mit verletzter Schwelle bei λ = 1 / 4 / 16 | 1 bis 9: **6 / 2 / 0**; 1 bis 99: **4 / 1 / 0**; reell: **4 / 4 / 0** |
| Erreicht die Gewichtssumme `A + λ·B` das harte Optimum? Instanzen von 20, Schwelle 25 / 50 / 75 % | 1 bis 9: **13 / 11 / 14**; 1 bis 99: **10 / 12 / 14**; reell: **9 / 9 / 11**: in den übrigen liegt es für **kein** λ auf der unteren Hülle |
| Unzulässige Schwelle (eins unter der kleinsten Zeitsumme, reell 95 % davon): Treffer „kleinste Verletzung, dann kleinste Kosten“ bei λ = 1 / 4 / 16 / 64 / 2²⁰ (von 20) | 1 bis 9: **2 / 12 / 20 / 20 / 20**; 1 bis 99: **0 / 7 / 18 / 19 / 20**; reell: **0 / 7 / 15 / 20 / 20**: das harte Problem ist unlösbar, die Strafe liefert einen Plan |
| Zu großes Gewicht in float64 (dieselbe Schwelle): Treffer bei λ = 2⁵² / 2⁵³ / 2⁵⁴ / 2⁵⁶ / 2⁵⁸ / 2⁶⁰ (von 20) | 1 bis 9: **20 / 19 / 19 / 16 / 15 / 14**; 1 bis 99 und reell: **20** bei jeder Stufe. Instanzen mit mehr als einem Plan bei kleinster Verletzung: **12** (1 bis 9) und **0** (1 bis 99) |
| Zulässige Schwelle, Treffer des harten Optimums bei λ = 1 / 4 / 16 und von 2⁶ bis 2⁶⁰ (von 20) | 1 bis 9: **14 / 18 / 20** und **20**; 1 bis 99: **11 / 16 / 20** und **20**; reell: **8 / 15 / 18** und **20**: das Gewicht hängt nur an der Verletzung, das harte Optimum hat `v = 0`, deshalb ist „zu groß“ hier harmlos |
| HiGHS mit Schlupfvariable, Treffer (zulässig, A gleich dem harten Optimum; in Klammern verletzt) bei λ = 0,25 / 1 / 4 / 16 | 1 bis 9: **2** (18) / **12** (8) / **18** (2) / **20** (0); 1 bis 99: **1** (19) / **10** (10) / **16** (4) / **20** (0); reell: **3** (17) / **8** (12) / **15** (5) / **18** (2); von λ = 10³ bis 10¹¹ in allen Familien **20** |
| HiGHS bei λ = 10¹² und 10¹⁴ | 1 bis 9 und 1 bis 99: **20**; reell: nur **7** von 20, alle zulässig, aber mit bis zu **168,2 %** höheren Kosten als das harte Optimum |
| Ist „hart“ im Löser hart? Die Schwelle liegt δ unter der Zeitsumme des Kostenoptimums (reell, von 20): Instanzen, in denen HiGHS das nur um δ verletzende Kostenoptimum liefert; δ = 10⁻⁹ / 10⁻⁷ / 10⁻⁶ / 10⁻⁵ / 10⁻⁴ / 10⁻³ / 10⁻¹ | harte Nebenbedingung: **20 / 20 / 19 / 20 / 4 / 0 / 0**; Strafe mit λ = 10⁶: **20 / 20 / 20 / 20 / 4 / 0 / 0** |
| Preset „Zu kleines Gewicht bricht die Schwelle“ (n = 8, 1 bis 9, Seed 53, Schwelle 50 %, lineare Strafe, λ = 1) | `T` = **26**, hartes Optimum A = **27** statt **12** (**125,0 %** mehr), **394** von 40 320 Plänen zulässig; bei λ = 1 überschreitet die Strafe die Schwelle um **9** (B = 35) und kostet A = 12, also **55,6 %** weniger; λ\* = **2,25** (quadratisch **0,75**); die Gewichtssumme trifft es für kein λ |
| Preset „Kleinstes Beispiel“ (n = 3, Seed 4) | `T` = **14**, **2** von 6 Plänen zulässig, hartes Optimum A = **26** statt **14**; λ\* = **4** (quadratisch **1,33**); die Gewichtssumme trifft es nie |
| Preset „Unzulässige Schwelle“ (n = 8, Seed 4, Schwelle −25 %) | `T` = **6** unter der kleinsten Zeitsumme **16**: kleinste Verletzung **10**, A = **40**; die Strafe trifft das ab λ\* = **3**; bei λ = 2⁵⁶ wählt sie A = **48** (A geht in float64 verloren) |
| Preset „HiGHS wird bei λ = 10¹² steif“ (reell, Seed 0) | hartes Optimum A = **191,6** statt **105,1** (**82,3 %** mehr); die Aufzählung trifft es bei λ = 2⁴⁰, HiGHS liefert dort **86,9 %** höhere Kosten |

**Korrektur der Planungshypothese:** Vor der Messung stand „zu kleines Gewicht erlaubt die Verletzung, zu großes macht numerisch steif“. Das erste stimmt (bei λ = 1 in 6 bis 12 von 20 Instanzen). Das zweite stimmt **nur dort, wo die Verletzung unvermeidbar ist** (unzulässige Schwelle, float64 ab λ = 2⁵³) oder der Löser selbst steif wird (HiGHS bei reellen Werten ab λ = 10¹²); bei zulässiger Schwelle bleibt jedes λ bis 2⁶⁰ in der Aufzählung bei 20 von 20. Außerdem: in der endlichen Planmenge ist **jede** Strafe ab einem endlichen Gewicht exakt, auch die quadratische (im stetigen Fall braucht sie ein Gewicht gegen unendlich; siehe `straf-barriere-demo`). Für die ℓ1-Strafe genügt im stetigen Fall ein endliches Gewicht oberhalb des größten Lagrange-Multiplikators (Suchtreffer, Primärtext nicht gelesen). Die HiGHS-Dokumentation nennt `primal_feasibility_tolerance` (Voreinstellung 10⁻⁷) und sagt nicht, ob sie auf skalierte oder unskalierte Zeilen wirkt; die Toleranz oben ist eine Messung der benutzten Version.

## Ehrliche Grenzen

- **Drei Familien, n = 8 und bis 12:** Wie stark die Schwelle bindet und wie groß λ\* wird, hängt an Familie, Größe und Schwellenlage; echte Daten können sich anders verhalten.
- **Eine Schwelle auf der Summe:** Mehrere Schwellen, eine Schwelle auf dem Maximum (Engpass) und Nebenbedingungen auf den Plänen selbst sind nicht gemessen.
- **Die Einheit von B:** λ\* der linearen Strafe skaliert mit 1/c, das der quadratischen mit 1/c², wenn man B mit c multipliziert; jedes λ in dieser Demo gilt für B in den Einheiten der Instanz.
- **HiGHS in einer Version:** Die HiGHS-Zeilen sind Messwerte von `scipy.optimize.milp` mit Voreinstellungen in der benutzten Version; andere Löser und Versionen können andere Toleranzen und andere Grenzen der Steifheit haben. Die Ursache der Fehlwahl bei λ ≈ 10¹² ist nicht untersucht.
- **Gleichstand der Strafe:** Bei Gleichstand gewinnt die kleinere Verletzung, dann der erste Plan der Aufzählung; eine andere Regel wählt anders (Stück 5).
- **Gleitkomma:** Bei ganzen Zahlen exakt; bei reellen Werten gilt Gleichheit der berechneten Zahl, und die Schwelle bei 100 % ist genau B des Kostenoptimums.

## Verifikation

Insgesamt 603 Tests (`pytest tests/`), alle grün.

- **Kern:** Handbeispiel (n = 3, sechs Pläne) von Hand: Summen aller Pläne, Schwelle, Zielplan, Verletzungen, λ\* 3 und 1,5, leeres Fenster der Gewichtssumme, die Wahl der Strafe bei λ = 1 und λ = 3, unzulässige Schwelle mit λ\* = 7/3, HiGHS gegen die Aufzählung; Schwelle, Zielplan, Wahl je Strafform, λ\* (mit `fractions.Fraction` und gegen ein Gitter ober- und unterhalb), Fenster der Gewichtssumme gegen ein Gitter, HiGHS hart und weich gegen reines Python (itertools, Tupelvergleich).
- **Eigenschaften:** das harte Optimum ist zulässig und kostenminimal; engere Schwelle, nie billiger; Schwelle bei 100 % bindet nicht; das Strafoptimum ist ab λ\* konstant und davor nie teurer; die Verletzung des gewählten Plans wächst nicht mit dem Gewicht; die Schranke `λ* ≤ A_Ziel − A_min` bei ganzen Zahlen; Skalieren von B skaliert λ\* mit 1/c und 1/c²; Umnummerieren ändert den Zielwert und λ\* nicht; Determinismus.
- **Randfälle:** konstante Matrizen, Schwelle gleich der kleinsten Zeitsumme, Kostenoptimum gleich schnellstem Plan, negative Schwelle (unzulässig), n = 3 bis 12, Aufzählungsgrenze n = 8 / 9, unbekannte Strafform, der Mechanismus hinter „zu großes Gewicht“ (float64).
- **Verdikte:** reine Funktionen, jede Verzweigung und jede Zahl im Text an Hand- oder Presetwerten (`tests/test_verdicts.py`).
- **Presets:** Vollständigkeit, Werte innerhalb der Reglergrenzen, Standard-Preset gleich Standardeinstellung, kein anderes Preset gleich den Voreinstellungen, und dass jedes Preset zeigt, was sein Name sagt.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler am Minimum und Maximum), jedes Preset, jede Strafform, jede Schwelle, jedes Gewicht, Größen um die Aufzählungsgrenze, alle Schalter, Footer-Test, Layout.
- **Claims:** jede Zahl dieser README in `tests/test_claims.py`.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `hgw_algorithm.py` | Kern: Aufzählung, Schwelle, Zielplan, Strafformen, λ\*, Fenster, HiGHS |
| `hgw_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `hgw_evaluation.py` | Experimente, Sweeps, Verdikte |
| `hgw_layout.py` | Fester Kopf (Stylesheet) |
| `hgw_pdf_export.py` | PDF-Export des Ergebnisses |
| `hgw_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste), Zufalls-Seed-Button |
| `hgw_scenario.py` | Instanzen (zwei Matrizen, drei Familien) |
| `hgw_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `tests/` | Handbeispiel, Orakel, Eigenschaften, Randfälle, Verdikte, Presets, Claims, Layout, AppTest-Smoke-Test, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Kein stetiger Fall:** Die Folge unrestringierter Probleme gegen einen KKT-Punkt zeigt `straf-barriere-demo` (Nichtlineare-Optimierung-Reihe); hier eine endliche Planmenge.
- **Keine anderen Löser und Strafformen:** Gurobi, CP-SAT, Huber- und Barrierestrafe sind nicht gemessen.
- **Keine Front und kein Vorrang:** Pareto-Front, Hülle und ε-Constraint sind Stück 3, Vorrang und Gewicht `M` Stück 4; hier nur die Schwelle im Strafterm.
- **Keine Ursache für die Fehlwahl bei λ ≈ 10¹²:** Skalierung, Gap und Presolve sind nicht untersucht, die Dokumentation sagt dazu nichts.
- **Keine Inverse-Probleme-Parameterwahl:** Tikhonov-Strafterme gegen Rauschen (Parameter nach Daten, nicht nach Schwelle) sind eine eigene Linie.

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

Gebaut mit Streamlit, Plotly, NumPy und SciPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
