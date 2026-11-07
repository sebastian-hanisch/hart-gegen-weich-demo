"""Hart gegen weich - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 6 der Reihe „Zielfunktionsdesign“: dieselbe Zuordnung wie in Stück 1 bis 5, jetzt mit einer Schwelle auf der Zeitsumme. Hart heißt: kleinste Kosten unter B <= T.
Weich heißt: die Schwelle wird zum Strafterm im Ziel. Die Demo zeigt, wann die Strafe die Schwelle bricht, ab welchem Gewicht sie exakt ist, was eine Gewichtssumme ohne
Schwelle nicht erreicht, was bei unzulässiger Schwelle geschieht und wo ein Löser „hart“ und „steif“ auslegt.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import hgw_algorithm as ALG
import hgw_constants as C
from hgw_evaluation import (
    Settings,
    analyse,
    big_lam_sweep,
    delta_text,
    highs_rows,
    highs_sweep,
    lam_curve,
    lam_summary,
    lam_text,
    pct,
    penalty_result,
    plan_text,
    point_window_text,
    shift_sweep,
    small_lam_sweep,
    threshold_summary,
    tolerance_rows,
    tolerance_sweep,
    value_text,
    verdict_big_lam_sweep,
    verdict_curve,
    verdict_highs,
    verdict_highs_sweep,
    verdict_lambda,
    verdict_penalty,
    verdict_shift_sweep,
    verdict_small_lam_sweep,
    verdict_threshold,
    verdict_tolerance,
    verdict_tolerance_sweep,
)
from hgw_layout import dunkles_design, kopf_css
from hgw_pdf_export import build_pdf
from hgw_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from hgw_visualization import build_big_lam, build_cloud, build_curve, build_reach_bars, build_small_lam

st.set_page_config(page_title="Hart gegen weich – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner="HiGHS rechnet ...")
def _highs(settings):
    a = _analysis(settings)
    return highs_rows(a, settings.lam), tolerance_rows(a)


@st.cache_data(show_spinner="20 feste Instanzen werden gerechnet ...")
def _shift_sweep():
    return shift_sweep()


@st.cache_data(show_spinner="20 feste Instanzen werden gerechnet ...")
def _small_sweep():
    return small_lam_sweep()


@st.cache_data(show_spinner="20 feste Instanzen werden gerechnet ...")
def _big_sweep():
    return big_lam_sweep()


@st.cache_data(show_spinner="HiGHS rechnet auf 20 festen Instanzen ...")
def _highs_sweep():
    return highs_sweep()


@st.cache_data(show_spinner="HiGHS rechnet auf 20 festen Instanzen ...")
def _tol_sweep():
    return tolerance_sweep()


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in hgw_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("⚖️ Hart gegen weich")
st.markdown(
    """
**Was kostet ein Strafterm, und wann wird er zum Schlupf?** Dieselbe Aufgabe wie in Stück 1 bis 5: **n Personen** übernehmen **n Aufgaben**, jede Paarung hat Kosten **A** und eine Zeit **B**.
Die Zeitsumme darf eine **Schwelle T** nicht überschreiten. **Hart** heißt: kleinste Kosten unter B ≤ T. **Weich** heißt: die Schwelle wird zum Strafterm im Ziel, `A + λ·v` mit der Verletzung `v = max(0, B − T)`.
Ein Strafterm ist ein **Gewicht** und hat einen Bereich: zu klein bricht er die Schwelle, ab einem endlichen Wert ist er exakt, zu groß kann er Zahlen verschlucken, und „hart“ ist im Löser auch nur eine Toleranz.

Weiter unten: **🔬 Die Schwelle**, **🔬 Das Strafgewicht**, **🔬 Die Gewichtssumme**, **🔬 Unzulässige Schwelle und zu großes Gewicht**, **🔬 Der Löser**, **🔬 Wie oft und wie viel?**, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert der Vergleich", expanded=True):
    st.markdown(
        """
1. **Die Schwelle** gilt für die Zeitsumme B eines Plans. Das **harte Optimum** hat die kleinsten Kosten A unter allen Plänen mit B ≤ T. Sie liegt zwischen der kleinsten möglichen Zeitsumme und der Zeitsumme des Kostenoptimums (Regler „Schwelle“); unter 0 % ist sie in der Regel unzulässig.
2. **Drei Strafformen:** linear `A + λ·v`, quadratisch `A + λ·v²` und zum Vergleich die **Gewichtssumme** `A + λ·B` ohne Schwelle (Stück 3). Das Gewicht λ ist eine Potenz von 2 (exakt in float64).
3. **λ\\*** ist das kleinste Gewicht, ab dem die Strafe das harte Optimum wählt; es wird exakt aus der endlichen Planmenge gerechnet. Bei Gleichstand der Strafe gewinnt der Plan mit der kleineren Verletzung, dann der erste der Aufzählung.
4. **Messungen, kein Satz:** Alle Zahlen sind Zählungen und Spannen über feste Instanzen dieser Familien; die Löser-Zahlen gelten für HiGHS in der benutzten Version.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Instanz**")
    n = st.slider("Personen und Aufgaben n", *bounds("n_slider"), key="n_slider",
                  help=f"Bis n = {C.ENUM_MAX} werden alle Pläne aufgezählt (exakt in ganzen Zahlen); bis n = {C.N_MAX} rechnet nur HiGHS.")
    family = st.selectbox("Wertegröße der Belastungen", list(range(len(C.FAMILIES))), key="family_select", format_func=lambda i: C.FAMILIES[i],
                          help="Kosten A und Zeit B sind zwei unabhängige Matrizen.")
    seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed)
    st.markdown("**Schwelle**")
    tau = st.slider("Schwelle T in Prozent des Wegs vom Zeitminimum zum Kostenoptimum", *bounds("tau_slider"), key="tau_slider", step=C.TAU_STEP,
                    help="100 %: die Schwelle bindet nicht (das Kostenoptimum erfüllt sie); 0 %: nur Pläne mit der kleinsten Zeitsumme sind zulässig; unter 0 %: in der Regel kein Plan zulässig (bei kleinem n kann das Kostenoptimum zugleich das schnellste sein).")
    st.markdown("**Strafterm**")
    penalty = st.selectbox("Strafform", list(range(len(C.PENALTIES))), key="penalty_select", format_func=lambda i: C.PENALTY_NAMES[C.PENALTIES[i]],
                           help="Linear und quadratisch sehen die Schwelle; die Gewichtssumme nicht.")
    lam_exp = st.slider("Gewicht λ = 2^e", *bounds("lam_exp_slider"), key="lam_exp_slider",
                        help="Das Gewicht ist eine Potenz von 2, damit float64 es exakt darstellt; e = 0 ist λ = 1, e = 56 ist λ = 2^56 (rund 7·10^16).")

sync_query_params({"n_slider": int(n), "family_select": int(family), "seed_input": int(seed), "tau_slider": int(tau), "penalty_select": int(penalty), "lam_exp_slider": int(lam_exp)})

settings = Settings(int(n), int(family), int(seed), int(tau), int(penalty), int(lam_exp))
a = _analysis(settings)
lam = settings.lam
form = settings.form

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis: was wählt die Strafe, und hält sie die Schwelle?")
if a.mode == "enum":
    ts = threshold_summary(a)
    ls = lam_summary(a)
    r = penalty_result(a, form, lam)
    m1, m2, m3 = st.columns(3)
    m1.metric("Schwelle T", value_text(ts["T"]), delta=None if ts["feasible"] else "unzulässig", delta_color="off")
    m2.metric("Hartes Optimum (Kosten A)" if ts["feasible"] else "Bester Plan (Kosten A)", value_text(ts["a_t"]),
              delta=f"{pct(ts['rise'])} mehr als ohne Schwelle" if ts["feasible"] and ts["binds"] else None, delta_color="off")
    m3.metric(f"Gewählt: {C.PENALTY_SHORT[form]}, λ = {lam_text(lam)}", f"A = {value_text(r['a'])}", delta="Treffer" if r["hit"] else f"Verletzung {value_text(r['v'])}", delta_color="off")
    st.plotly_chart(build_cloud(a, ts, r), width="stretch", key="cloud_chart")
    st.caption("Jeder Punkt ist ein Plan (Zeitsumme gegen Kosten). Die senkrechte Linie ist die Schwelle; links davon liegen die zulässigen Pläne (blau). Der Stern ist das Kostenoptimum, die Raute der Zielplan (das harte Optimum), das Kreuz der Plan, den die gewählte Strafe bei diesem Gewicht wählt.")
    st.download_button("📄 Ergebnis als PDF", data=build_pdf(settings, a), file_name="hart-gegen-weich.pdf", mime="application/pdf", key="pdf_download")
else:
    st.info(f"Bei n = {a.inst.n} werden nicht alle Pläne aufgezählt (nur bis n = {C.ENUM_MAX}); Schwelle, λ* und Gewichtssumme gibt es hier nicht. Der Löser unten (HiGHS) läuft bis n = {C.N_MAX}.")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Hält die weiche Formulierung die Schwelle?")
if a.mode == "enum":
    kind, text = verdict_penalty(r, ts, ls)
    getattr(st, kind)(text)

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Die Schwelle")
st.caption("Wo die Schwelle liegt, ob sie bindet und was sie an Kosten verlangt: das harte Optimum gegen das Kostenoptimum.")
if a.mode == "enum":
    kind, text = verdict_threshold(ts)
    getattr(st, kind)(text)
    p = a.prob
    rows_md = ["| Plan | Kosten A | Zeit B | Verletzung | Person → Aufgabe |", "|---|---|---|---|---|"]
    rows_md.append(f"| **Kostenoptimum** | {value_text(a.a_opt)} | {value_text(a.bA)} | {value_text(max(0.0, a.bA - a.T))} | {plan_text(ALG.all_plans(a.inst.n)[p.iA], a.inst.n)} |")
    rows_md.append(f"| **{'Hartes Optimum' if ts['feasible'] else 'Kleinste Verletzung'}** | {value_text(ts['a_t'])} | {value_text(ts['b_t'])} | {value_text(ts['v_t'])} | {plan_text(ALG.all_plans(a.inst.n)[ts['target']], a.inst.n)} |")
    rows_md.append(f"| **Gewählt ({C.PENALTY_SHORT[form]}, λ = {lam_text(lam)})** | {value_text(r['a'])} | {value_text(r['b'])} | {value_text(r['v'])} | {plan_text(r['plan'], a.inst.n)} |")
    st.markdown("\n".join(rows_md))
else:
    st.info("Nur bis n = 8 (Aufzählung).")

st.markdown("---")

st.subheader("🔬 Das Strafgewicht")
st.caption("Der gewählte Plan bei jedem Gewicht 2^e: zu klein, und die Strafe bricht die Schwelle; ab λ* ist sie exakt. Grün: der Zielplan.")
if a.mode == "enum":
    kind, text = verdict_lambda(ls, ts)
    getattr(st, kind)(text)
    rows = lam_curve(a, form)
    st.plotly_chart(build_curve(rows, ts["a_t"], form), width="stretch", key="curve_chart")
    kind, text = verdict_curve(rows, form)
    getattr(st, kind)(text)
else:
    st.info("Nur bis n = 8 (Aufzählung).")

st.markdown("---")

st.subheader("🔬 Die Gewichtssumme")
st.caption("Die Gewichtssumme `A + λ·B` sieht die Schwelle nicht. Sie findet nur Pläne auf der unteren Hülle der Punkte (A, B) (Stück 3); das harte Optimum liegt oft nicht darauf. Wählen Sie links „Gewichtssumme“.")
if a.mode == "enum":
    if ls["point"]:
        st.info(f"Der Zielplan ist {point_window_text(ls)}.")
    elif ls["reachable"]:
        st.info(f"Der Zielplan ist für λ ≥ {lam_text(ls['lo'])}" + ("" if ls["hi"] == float("inf") else f" bis {lam_text(ls['hi'])}") + " ein Minimierer der Gewichtssumme.")
    else:
        st.warning(f"Für **kein** λ ist der Zielplan ein Minimierer der Gewichtssumme (untere Grenze {lam_text(ls['lo'])}, obere {lam_text(ls['hi'])}): die Schwelle im Strafterm erreicht, was ein Gewicht nicht erreicht.")
else:
    st.info("Nur bis n = 8 (Aufzählung).")
if st.toggle("Gewichtssumme und λ* nachrechnen (je Familie und Schwelle)", value=False, key="sweep_shift"):
    shift_res = _shift_sweep()
    kind, text = verdict_shift_sweep(shift_res)
    getattr(st, kind)(text)
    st.plotly_chart(build_reach_bars(shift_res), width="stretch", key="sweep_reach_chart")

st.markdown("---")

st.subheader("🔬 Unzulässige Schwelle und zu großes Gewicht")
st.caption("Liegt die Schwelle unter der kleinsten Zeitsumme (Regler unter 0 %), ist das harte Problem unlösbar; die Strafe liefert trotzdem einen Plan. Ein zu großes Gewicht verschluckt dann in float64 die Kosten (λ = 2^53 und mehr).")
if a.mode == "enum":
    if ts["feasible"]:
        st.info("Die Schwelle ist hier zulässig. Stellen Sie sie unter 0 % (zum Beispiel −25 %), um den Schlupf-Fall zu sehen; das Preset „Unzulässige Schwelle“ lädt ein Beispiel.")
    else:
        kind, text = verdict_penalty(r, ts, ls)
        getattr(st, kind)(text)
else:
    st.info("Nur bis n = 8 (Aufzählung).")
if st.toggle("Zu kleine und zu große Gewichte nachrechnen (je Familie)", value=False, key="sweep_small"):
    small_res = _small_sweep()
    kind, text = verdict_small_lam_sweep(small_res)
    getattr(st, kind)(text)
    st.plotly_chart(build_small_lam(small_res), width="stretch", key="sweep_small_chart")
    big_res = _big_sweep()
    kind, text = verdict_big_lam_sweep(big_res)
    getattr(st, kind)(text)
    st.plotly_chart(build_big_lam(big_res), width="stretch", key="sweep_big_chart")

st.markdown("---")

st.subheader("🔬 Der Löser")
st.caption("HiGHS (`scipy.optimize.milp`, Voreinstellungen) mit Schlupfvariable `s ≥ B − T`, `s ≥ 0` und Strafe `A + λ·s`, und mit harter Nebenbedingung `B ≤ T`. Gemessen ist die benutzte Version; die Ungarische Methode trägt die Strafe nicht (die Summe `Σ b` ist nicht in Paarkosten zerlegbar).")
hrows, trows = _highs(settings)
rows_md = ["| Weg | Kosten A | Zeit B | Verletzung | Treffer |", "|---|---|---|---|---|"]
for hr in hrows:
    if hr["plan"] is None:
        rows_md.append(f"| **{hr['label']}** | – | – | – | unlösbar |")
    else:
        rows_md.append(f"| **{hr['label']}** | {value_text(hr['a'])} | {value_text(hr['b'])} | {value_text(hr['v'])} | " + ("–" if hr["label"].startswith("hart") else "ja" if hr["hit"] else "nein" + (f" (+{pct(hr['loss'])})" if hr["loss"] and hr["v"] <= 1e-9 else "")) + " |")
st.markdown("\n".join(rows_md))
kind, text = verdict_highs(hrows)
getattr(st, kind)(text)
st.markdown("**Ist „hart“ im Löser hart?** Die Schwelle liegt δ unter der Zeitsumme des Kostenoptimums; das Kostenoptimum verletzt sie also um δ.")
rows_md = ["| δ | harte Nebenbedingung liefert das nur um δ verletzende Kostenoptimum | Strafe mit λ = 10^6 liefert es |", "|---|---|---|"]
for tr in trows:
    rows_md.append(f"| {delta_text(tr['delta'])} | {'ja' if tr['hard_violates'] else 'nein'} | {'ja' if tr['soft_violates'] else 'nein'} |")
st.markdown("\n".join(rows_md))
kind, text = verdict_tolerance(trows)
getattr(st, kind)(text)
st.caption("Laut Dokumentation (HiGHS, Optionen) ist `primal_feasibility_tolerance` voreingestellt auf 10⁻⁷ („Primal feasibility tolerance“); die Seite sagt nicht, ob sie auf skalierte oder unskalierte Zeilen wirkt. Die Zahlen oben sind Messungen der benutzten Version.")

st.markdown("---")

st.subheader("🔬 Wie oft und wie viel?")
st.caption(f"Je {len(C.SWEEP_SEEDS)} feste Instanzen (SWEEP_SEEDS), n = {C.SWEEP_N}, alle aufgezählt; die Zahlen sind Zählungen und Spannen über Instanzen, kein Satz.")
if st.toggle("HiGHS mit Schlupfvariable über die festen Instanzen vergleichen", value=False, key="sweep_highs"):
    kind, text = verdict_highs_sweep(_highs_sweep())
    getattr(st, kind)(text)
if st.toggle("Die harte Nebenbedingung in HiGHS über die festen Instanzen prüfen (reelle Werte)", value=False, key="sweep_tol"):
    kind, text = verdict_tolerance_sweep(_tol_sweep())
    getattr(st, kind)(text)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Eine Schwelle auf der Summe von B** | Mehrere Schwellen, eine Schwelle auf dem Maximum (Engpass) oder Nebenbedingungen auf den Plänen selbst ändern λ* und die Fenster; nicht gemessen | Stück 3 (Zwei Kriterien) |
| **Endliche Planmenge** | Hier ist **jede** Strafe in exakter Rechnung ab einem endlichen Gewicht exakt, weil es nur endlich viele Verletzungen gibt; im stetigen Fall braucht die quadratische Strafe ein Gewicht gegen unendlich | Straf-/Barriere-Verfahren (Nichtlineare Optimierung) |
| **Die Einheit von B** | λ* der linearen Strafe skaliert mit 1/c, das der quadratischen mit 1/c², wenn man B mit c multipliziert: das Gewicht hat eine Einheit | – |
| **HiGHS in einer Version** | Gemessen ist `scipy.optimize.milp` (HiGHS) mit Voreinstellungen in der benutzten Version; andere Löser und Versionen können andere Toleranzen und andere Grenzen der Steifheit haben. Die Ursache der Fehlwahl bei λ ≈ 10¹² ist nicht untersucht | Dokumentation des Lösers |
| **n = 8 aufgezählt, HiGHS bis n = 12** | Schwelle, λ*, Fenster und Zielplan-Vergleich gibt es nur bis n = 8; darüber nur HiGHS | – |
| **Gleichstand der Strafe** | Bei Gleichstand gewinnt die kleinere Verletzung, dann der erste Plan der Aufzählung: eine andere Regel wählt anders (Stück 5) | Stück 5 (Gleichstände) |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Problem.** Gegeben $n$ Personen, $n$ Aufgaben und Matrizen $a_{ij}$ (Kosten) und $b_{ij}$ (Zeit). Ein Plan ist eine Permutation $\pi$; $A(\pi) = \sum_i a_{i\pi(i)}$, $B(\pi) = \sum_i b_{i\pi(i)}$. Die Schwelle $T$ liegt bei $T = B_{\min} + \tau\,(B(\pi_A) - B_{\min})$, wobei $\pi_A$ das Kostenoptimum ist (bei ganzen Zahlen abgerundet).

**Hart.** $\min A(\pi)$ unter $B(\pi) \le T$. **Weich.** Mit der Verletzung $v(\pi) = \max(0,\, B(\pi) - T)$: $f_{\text{lin}} = A + \lambda v$, $f_{\text{quad}} = A + \lambda v^2$, und ohne Schwelle $f_{\text{sum}} = A + \lambda B$. Als MIP mit Schlupfvariable: $\min A(x) + \lambda s$ unter $B(x) - s \le T$, $s \ge 0$.

**Zielplan.** $\pi^*$ minimiert $(v, A)$ lexikographisch, bei Gleichstand dann $B$ und der erste Plan. Bei zulässiger Schwelle ist es das harte Optimum ($v = 0$).

**$\lambda^*$.** Die Strafe $A + \lambda v^p$ wählt den Zielplan genau dann, wenn für jeden Plan $\pi$ mit $A(\pi) < A(\pi^*)$ und $v(\pi) > v(\pi^*)$ gilt $\lambda \ge \dfrac{A(\pi^*) - A(\pi)}{v(\pi)^p - v(\pi^*)^p}$. Also $\lambda^*_p = \max$ dieser Quotienten (0, wenn es keinen solchen Plan gibt). Bei ganzen Zahlen ist $v \ge 1$ und daher $\lambda^*_1 \le A(\pi^*) - A_{\min}$. Skaliert man $b$ mit $c$, so $\lambda^*_1 \to \lambda^*_1 / c$ und $\lambda^*_2 \to \lambda^*_2 / c^2$.

**Gewichtssumme.** $A(\pi^*) + \lambda B(\pi^*) \le A(\pi) + \lambda B(\pi)$ für alle $\pi$ ergibt: Pläne mit $B > B^*$ geben die untere Schranke $\lambda \ge \max \frac{A^* - A}{B - B^*}$, Pläne mit $B < B^*$ die obere $\lambda \le \min \frac{A - A^*}{B^* - B}$. Das Fenster ist leer (untere Schranke über der oberen), wenn der Zielplan nicht auf der unteren Hülle liegt; sind beide Schranken gleich, besteht es aus einem einzigen $\lambda$, und der Zielplan ist dort nur im Gleichstand mit anderen Plänen ein Minimierer.

**Gleitkomma.** float64 hat 53 Bit Mantisse; ist $\lambda v \ge 2^{53}$, ändert das Addieren der Kosten $A$ die Summe nicht mehr, und Pläne mit gleicher Verletzung sind in der Strafe nicht mehr zu unterscheiden.

**Literatur.** Für die ℓ1-Strafe genügt ein endliches Gewicht oberhalb des größten Lagrange-Multiplikators; die klassische quadratische Außenstrafe braucht im stetigen Fall ein Gewicht gegen unendlich (Suchtreffer, Primärtext nicht gelesen: Berkeley ERL-M79/31, 1979; arXiv 1709.07073). HiGHS-Dokumentation (Optionen, gelesen): `primal_feasibility_tolerance` 10⁻⁷, `mip_feasibility_tolerance` 10⁻⁶ (Ganzzahligkeit), `mip_rel_gap` 10⁻⁴. Diese Demo übernimmt keine Zahl daraus; alle Zahlen sind Messungen an den Instanzfamilien der Demo.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
