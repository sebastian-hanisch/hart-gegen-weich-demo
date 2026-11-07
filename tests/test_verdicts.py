"""Verdikte: Art und Wortlaut in den Fällen, die sie unterscheiden (DEMO-PLAYBOOK §6): jede Zahl im Text kommt aus der Rechnung."""

import hgw_constants as C
import hgw_evaluation as ev


def _a(name_or_settings):
    s = name_or_settings if isinstance(name_or_settings, ev.Settings) else ev.Settings(**{k: v for k, v in C.PRESETS[name_or_settings].items()})
    return s, ev.analyse(s)


def _all(name):
    s, a = _a(name)
    ts, ls = ev.threshold_summary(a), ev.lam_summary(a)
    return s, a, ts, ls, ev.penalty_result(a, s.form, s.lam)


(P_SMALL, P_EXACT, P_QUAD, P_SUM, P_N3, P_INF, P_BIG, P_HIGHS) = list(C.PRESETS)


def test_threshold_verdict_cases():
    s, a, ts, ls, r = _all(P_SMALL)
    kind, text = ev.verdict_threshold(ts)
    assert kind == "info" and "T = 26 bindet" in text and "**A = 27** statt 12 (125,0 % mehr)" in text and "Zulässig sind 394 von 40320" in text
    s, a, ts, ls, r = _all(P_INF)
    kind, text = ev.verdict_threshold(ts)
    assert kind == "warning" and "unter der kleinsten Zeitsumme** (16)" in text and "kleinsten Verletzung (10)" in text
    a100 = ev.analyse(ev.Settings(8, 0, 53, 100))
    kind, text = ev.verdict_threshold(ev.threshold_summary(a100))
    assert kind == "info" and "bindet hier nicht" in text and "A = 12" in text


def test_penalty_verdict_distinguishes_violation_hit_feasible_but_costly_and_infeasible():
    s, a, ts, ls, r = _all(P_SMALL)
    kind, text = ev.verdict_penalty(r, ts, ls)
    assert kind == "warning" and "um **9** überschreitet (B = 35 statt höchstens 26)" in text and "A = 12 statt 27, also **55,6 % weniger" in text and "Ab λ = 2,25" in text
    s, a, ts, ls, r = _all(P_EXACT)
    kind, text = ev.verdict_penalty(r, ts, ls)
    assert kind == "success" and text.startswith("Bei λ = 4 wählt die lineare Strafe das harte Optimum (A = 27, B = 26)")
    s, a, ts, ls, r = _all(P_QUAD)
    kind, text = ev.verdict_penalty(r, ts, ls)
    assert kind == "success" and "quadratische Strafe" in text and text.startswith("Bei λ = 1")
    s, a, ts, ls, r = _all(P_SUM)
    kind, text = ev.verdict_penalty(r, ts, ls)
    assert kind == "info" and "Bei λ = 2 wählt die Gewichtssumme einen zulässigen Plan, aber nicht das harte Optimum: A = 31 statt 27 (14,8 % mehr)" in text and "ein zulässiger Plan ist Zufall und kein λ trifft das harte Optimum." in text
    r1 = ev.penalty_result(a, "weighted", 1.0)
    kind, text = ev.verdict_penalty(r1, ts, ls)
    assert kind == "warning" and "Die Gewichtssumme sieht die Schwelle nicht: kein λ trifft das harte Optimum." in text and "um **9** überschreitet" in text
    s, a, ts, ls, r = _all(P_INF)
    kind, text = ev.verdict_penalty(r, ts, ls)
    assert kind == "success" and "bei λ = 64 den Plan mit der kleinsten Verletzung (10)" in text and "Hart gäbe es nichts" in text
    s, a, ts, ls, r = _all(P_BIG)
    kind, text = ev.verdict_penalty(r, ts, ls)
    assert kind == "warning" and "bei λ = 2^56 **nicht**" in text and "gleiche Verletzung (10), aber A = 48 statt 40" in text and "A geht in float64 verloren" in text


def test_penalty_verdict_too_small_weight_for_an_infeasible_threshold_and_feasible_but_costly_weighted_choice():
    s, a = _a(ev.Settings(8, 0, 4, -25, 0, 0))                          # λ = 1 bei unzulässiger Schwelle: zu klein
    ts, ls = ev.threshold_summary(a), ev.lam_summary(a)
    kind, text = ev.verdict_penalty(ev.penalty_result(a, "linear", 1.0), ts, ls)
    assert kind == "warning" and "das Gewicht ist zu klein" in text and "A geht in float64" not in text
    s, a = _a(ev.Settings(8, 0, 53, 50, 2, 6))                           # Gewichtssumme mit großem λ: zulässig, aber nicht das harte Optimum (hier kein Hinweis auf kein λ nötig, aber er steht da)
    ts, ls = ev.threshold_summary(a), ev.lam_summary(a)
    r = ev.penalty_result(a, "weighted", 2.0 ** 6)
    kind, text = ev.verdict_penalty(r, ts, ls)
    assert r["v"] == 0 and not r["hit"] and kind == "info" and "zulässigen Plan, aber nicht das harte Optimum" in text


def test_lambda_verdict_cases():
    s, a, ts, ls, r = _all(P_SMALL)
    kind, text = ev.verdict_lambda(ls, ts)
    assert kind == "info" and "ab **λ = 2,25**, die quadratische ab **λ = 0,75**" in text and "Schranke λ ≥ 15" in text and "für **kein λ**" in text
    s, a, ts, ls, r = _all(P_INF)
    kind, text = ev.verdict_lambda(ls, ts)
    assert "den Plan mit der kleinsten Verletzung ab **λ = 3**" in text and "nur für λ ≥ 3" in text and "inf" not in text
    a100 = ev.analyse(ev.Settings(8, 0, 53, 100))
    ts100, ls100 = ev.threshold_summary(a100), ev.lam_summary(a100)
    assert ev.verdict_lambda(ls100, ts100)[1].startswith("Schon bei λ = 0 gilt")


def test_curve_verdict_cases():
    a = _a(P_SMALL)[1]
    kind, text = ev.verdict_curve(ev.lam_curve(a, "linear"), "linear")
    assert kind == "info" and "ab λ = 2^2 und bei jedem größeren Gewicht" in text
    kind, text = ev.verdict_curve(ev.lam_curve(a, "weighted"), "weighted")
    assert kind == "warning" and "Für kein Gewicht der Kurve" in text
    a = _a(P_INF)[1]
    kind, text = ev.verdict_curve(ev.lam_curve(a, "linear"), "linear")
    assert kind == "warning" and "ab λ = 2^2, aber nicht bei jedem größeren Gewicht: zuletzt verfehlt bei λ = 2^60" in text


def test_highs_verdicts():
    s, a = _a(P_HIGHS)
    rows = ev.highs_rows(a, s.lam)
    kind, text = ev.verdict_highs(rows)
    assert kind == "warning" and "bei 3 Gewichten einen zulässigen Plan" in text and "86,9 %" in text and "keine Warnung" in text
    s, a = _a(P_SMALL)
    kind, text = ev.verdict_highs(ev.highs_rows(a, s.lam))
    assert kind == "info" and "Bei 2 der 10 Gewichte ist die Strafe zu klein" in text and "zuletzt bei λ = 1)" in text
    s, a = _a(ev.Settings(8, 0, 4, -25, 0, 0))
    kind, text = ev.verdict_highs(ev.highs_rows(a, 1.0))
    assert kind == "info" and "findet HiGHS **keinen** Plan" in text


def test_tolerance_verdict_cases():
    rows = [{"delta": 1e-3, "hard_violates": False, "soft_violates": False}, {"delta": 1e-7, "hard_violates": True, "soft_violates": True}]
    kind, text = ev.verdict_tolerance(rows)
    assert kind == "warning" and "δ = 10^-7 unter der Zeitsumme" in text
    assert ev.verdict_tolerance([{"delta": 1e-3, "hard_violates": False, "soft_violates": False}])[0] == "info"


def test_a_window_of_a_single_point_is_reachable_only_as_a_tie_in_all_verdicts():
    """Familie 1 bis 9, n = 8, Seed 100008, Schwelle 25 %: lo == hi == 1,5. Nicht „für kein λ“, sondern: einzelnes λ, nur im Gleichstand, Gleichstandsregel entscheidet."""
    s = ev.Settings(8, 0, 100008, 25, 2, 0)
    a = ev.analyse(s)
    ts, ls = ev.threshold_summary(a), ev.lam_summary(a)
    assert ls["reachable"] and ls["point"] and ls["lo"] == ls["hi"] == 1.5
    kind, text = ev.verdict_lambda(ls, ts)
    assert kind == "info" and "kein λ" not in text and "trifft ihn nur für das einzelne λ = 1,50, und dort nur im Gleichstand mit anderen Plänen" in text and "Gleichstandsregel" in text
    kind, text = ev.verdict_penalty(ev.penalty_result(a, "weighted", 1.0), ts, ls)
    assert "kein λ" not in text and "das harte Optimum ist nur für das einzelne λ = 1,50 ein Minimierer der Gewichtssumme, und dort nur im Gleichstand" in text
    ok = ev.lam_summary(ev.analyse(ev.Settings(8, 0, 53, 50, 2, 6)))
    assert not ok["reachable"] and not ok["point"] and ok["lo"] > ok["hi"]                  # das Preset „Die Gewichtssumme erreicht es nicht“ bleibt ein leeres Fenster


def test_sweep_verdicts_carry_the_phase_zero_numbers():
    shift = ev.shift_sweep()
    kind, text = ev.verdict_shift_sweep(shift)
    assert kind == "info" and "K = 9, Schwelle 25 %: A steigt um Median 58,4 % (bis 145,5 %)" in text and "Gewichtssumme erreicht in 15 von 20 (davon nur für ein einziges λ, im Gleichstand: 2; dort wählt die Gleichstandsregel den Zielplan in 0)" in text and "Gewichtssumme erreicht in 10 von 20; K = 99, Schwelle 50 %" in text
    small = ev.small_lam_sweep()
    assert [small[("linear", 0, lam)]["violated"] for lam in (0.25, 1.0, 4.0, 8.0)] == [18, 6, 2, 0] and [small[("linear", 2, lam)]["violated"] for lam in (1.0, 8.0)] == [12, 4]
    assert "K = 9, λ = 1: 6 von 20" in ev.verdict_small_lam_sweep(small)[1]
    big = ev.big_lam_sweep()
    assert [big[0]["infeasible"][e] for e in (4, 52, 53, 56, 58, 60)] == [20, 20, 19, 16, 15, 14] and big[0]["ties"] == 12 and {big[0]["feasible"][e] for e in big[0]["feasible"] if e >= 4} == {20} and [big[0]["feasible"][e] for e in (0, 2)] == [14, 18]
    assert "2^53: 19, 2^56: 16, 2^58: 15, 2^60: 14 (Instanzen mit mehreren Plänen bei kleinster Verletzung: 12)" in ev.verdict_big_lam_sweep(big)[1]
