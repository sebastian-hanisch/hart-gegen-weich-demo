"""Jede Zahl aus dem README wird hier nachgerechnet (DEMO-PLAYBOOK §4). Aufzählung und feste Instanzerzeugung sind deterministisch;
die HiGHS-Zeilen sind Messwerte der benutzten Version (Voreinstellungen)."""

from pathlib import Path

import pytest

import hgw_algorithm as A
import hgw_constants as C
import hgw_evaluation as ev

README = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
(P_SMALL, P_EXACT, P_QUAD, P_SUM, P_N3, P_INF, P_BIG, P_HIGHS) = list(C.PRESETS)


def _claim(text):
    assert text in README, text


def _slash(values, digits=0):
    return " / ".join(ev.de(v, digits) for v in values)


def _preset(name):
    s = ev.Settings(**C.PRESETS[name])
    return s, ev.analyse(s)


@pytest.fixture(scope="module")
def shift():
    return ev.shift_sweep()


@pytest.fixture(scope="module")
def small():
    return ev.small_lam_sweep()


@pytest.fixture(scope="module")
def big():
    return ev.big_lam_sweep()


@pytest.fixture(scope="module")
def infeasible():
    return ev.infeasible_sweep(exps=(0, 2, 4, 6, 20))


@pytest.fixture(scope="module")
def highs():
    return ev.highs_sweep()


@pytest.fixture(scope="module")
def tolerance():
    return ev.tolerance_sweep()


def test_sweep_instances_are_the_documented_ones():
    assert C.SWEEP_SEEDS == tuple(range(100000, 100020)) and C.SWEEP_N == 8 and C.SWEEP_TAUS == (25, 50, 75) and C.GRANULARITIES == (9, 99, 0)
    _claim("Seeds 100000 bis 100019")


def test_threshold_claims(shift):
    rise = lambda fam: [shift[(fam, t)]["rise"] for t in (25, 50, 75)]
    cells = lambda fam: " / ".join(f"**{ev.pct(r[1])}** ({ev.pct(r[2])})" for r in rise(fam))
    assert [cells(f) for f in (0, 1, 2)] == ["**58,4 %** (145,5 %) / **29,9 %** (85,7 %) / **12,1 %** (45,5 %)", "**101,2 %** (240,3 %) / **28,0 %** (151,4 %) / **13,6 %** (64,6 %)",
                                              "**94,9 %** (225,8 %) / **51,1 %** (139,4 %) / **18,2 %** (75,0 %)"]
    assert [shift[(f, t)]["none"] for f in (0, 1, 2) for t in (25, 50, 75)] == [0, 0, 3, 0, 0, 0, 0, 0, 0]
    for fam in (0, 1, 2):
        _claim({0: "1 bis 9: ", 1: "1 bis 99: ", 2: "reell: "}[fam] + cells(fam))
    _claim("in **3** von 20")


def test_lambda_star_claims(shift):
    for power, key in ((1, "lin"), (2, "quad")):
        for fam, label in ((0, "1 bis 9: "), (1, "1 bis 99: "), (2, "reell: ")):
            cells = " / ".join(f"**{ev.de(shift[(fam, t)][key][1], 2)}** ({ev.de(shift[(fam, t)][key][2], 2)})" for t in (25, 50, 75))
            _claim(label + cells)
    assert (shift[(0, 50)]["lin"][1], shift[(0, 50)]["quad"][2]) == (pytest.approx(0.9, abs=0.01), 6.0)
    assert shift[(2, 75)]["quad"][2] == pytest.approx(500.55, abs=0.01) and shift[(2, 75)]["lin"][2] == pytest.approx(109.84, abs=0.01)


def test_weighted_claims(shift):
    assert [[shift[(f, t)]["reach"] for t in (25, 50, 75)] for f in (0, 1, 2)] == [[15, 13, 16], [10, 12, 14], [9, 9, 11]]
    assert [[shift[(f, t)]["point"] for t in (25, 50, 75)] for f in (0, 1, 2)] == [[2, 2, 2], [0, 0, 0], [0, 0, 0]]
    assert [shift[(0, t)]["point_hit"] for t in (25, 50, 75)] == [0, 2, 0]
    for text in ("1 bis 9: **15 / 13 / 16**; 1 bis 99: **10 / 12 / 14**; reell: **9 / 9 / 11**", "davon nur für ein einziges λ und nur im Gleichstand: 1 bis 9: **2 / 2 / 2**, sonst 0",
                 "die Gleichstandsregel wählt dort in **2** von 6 den Zielplan"):
        _claim(text)


def test_small_weight_claims(small):
    lams = (0.25, 1.0, 4.0, 8.0, 16.0)
    assert [[small[("linear", f, l)]["violated"] for l in lams] for f in (0, 1, 2)] == [[18, 6, 2, 0, 0], [19, 9, 4, 2, 0], [17, 12, 5, 4, 2]]
    assert [small[("linear", f, 32.0)]["violated"] for f in (0, 1, 2)] == [0, 0, 0]
    assert [small[("linear", f, 1.0)]["v_med"] for f in (0, 1, 2)] == [1.0, 6.0, pytest.approx(10.8, abs=0.05)]
    assert [ev.pct(small[("linear", f, 1.0)]["gain"]) for f in (0, 1, 2)] == ["18,8 %", "15,5 %", "17,3 %"]
    assert [[small[("quadratic", f, l)]["violated"] for l in (1.0, 4.0, 16.0)] for f in (0, 1, 2)] == [[6, 2, 0], [4, 1, 0], [4, 4, 0]]
    for text in ("1 bis 9: **18 / 6 / 2 / 0 / 0**; 1 bis 99: **19 / 9 / 4 / 2 / 0**; reell: **17 / 12 / 5 / 4 / 2**", "Verletzung **1 / 6 / 10,8** Zeiteinheiten", "**18,8 % / 15,5 % / 17,3 %**",
                 "1 bis 9: **6 / 2 / 0**; 1 bis 99: **4 / 1 / 0**; reell: **4 / 4 / 0**"):
        _claim(text)


def test_infeasible_and_float64_claims(big, infeasible):
    assert [[infeasible[f]["hits"][e] for e in (0, 2, 4, 6, 20)] for f in (0, 1, 2)] == [[2, 12, 20, 20, 20], [0, 7, 18, 19, 20], [0, 7, 15, 20, 20]]
    assert [big[0]["infeasible"][e] for e in (52, 53, 54, 56, 58, 60)] == [20, 19, 19, 16, 15, 14]
    assert all(big[f]["infeasible"][e] == 20 for f in (1, 2) for e in (52, 53, 54, 56, 58, 60)) and (big[0]["ties"], big[1]["ties"]) == (12, 0)
    for f, cells in ((0, [14, 18, 20]), (1, [11, 16, 20]), (2, [8, 15, 18])):
        assert [big[f]["feasible"][e] for e in (0, 2, 4)] == cells and all(big[f]["feasible"][e] == 20 for e in (6, 20, 50, 52, 53, 54, 56, 58, 60))
    for text in ("1 bis 9: **2 / 12 / 20 / 20 / 20**; 1 bis 99: **0 / 7 / 18 / 19 / 20**; reell: **0 / 7 / 15 / 20 / 20**", "1 bis 9: **20 / 19 / 19 / 16 / 15 / 14**; 1 bis 99 und reell: **20** bei jeder Stufe",
                 "**12** (1 bis 9) und **0** (1 bis 99)", "1 bis 9: **14 / 18 / 20** und **20**; 1 bis 99: **11 / 16 / 20** und **20**; reell: **8 / 15 / 18** und **20**"):
        _claim(text)


def test_highs_claims(highs):
    hit = lambda f, l: highs[(f, l)]["hits"]
    vio = lambda f, l: highs[(f, l)]["violated"]
    small = (0.25, 1.0, 4.0, 16.0)
    assert [[(hit(f, l), vio(f, l)) for l in small] for f in (0, 1, 2)] == [[(2, 18), (12, 8), (18, 2), (20, 0)], [(1, 19), (10, 10), (16, 4), (20, 0)], [(3, 17), (8, 12), (15, 5), (18, 2)]]
    assert all(hit(f, l) == 20 for f in (0, 1, 2) for l in (1e3, 1e6, 1e9, 1e11)) and all(hit(f, l) == 20 for f in (0, 1) for l in (1e12, 1e14))
    assert (hit(2, 1e12), hit(2, 1e14), vio(2, 1e12)) == (7, 7, 0) and ev.pct(highs[(2, 1e12)]["loss"]) == "168,2 %"
    for text in ("1 bis 9: **2** (18) / **12** (8) / **18** (2) / **20** (0); 1 bis 99: **1** (19) / **10** (10) / **16** (4) / **20** (0); reell: **3** (17) / **8** (12) / **15** (5) / **18** (2)",
                 "nur **7** von 20, alle zulässig, aber mit bis zu **168,2 %** höheren Kosten"):
        _claim(text)


def test_tolerance_claims(tolerance):
    deltas = (1e-9, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-1)
    assert [tolerance[d]["hard"] for d in deltas] == [20, 20, 19, 20, 4, 0, 0] and [tolerance[d]["soft"] for d in deltas] == [20, 20, 20, 20, 4, 0, 0]
    _claim("harte Nebenbedingung: **20 / 20 / 19 / 20 / 4 / 0 / 0**; Strafe mit λ = 10⁶: **20 / 20 / 20 / 20 / 4 / 0 / 0**")


def test_preset_claims():
    s, a = _preset(P_SMALL)
    ts, ls, r = ev.threshold_summary(a), ev.lam_summary(a), ev.penalty_result(a, s.form, s.lam)
    assert (ts["T"], ts["a_t"], ts["a_opt"], ev.pct(ts["rise"]), ts["n_feasible"], ts["total"]) == (26.0, 27, 12, "125,0 %", 394, 40320)
    assert (r["v"], r["b"], r["a"], ev.pct(r["gain"]), ls["lin"], ls["quad"], ls["reachable"]) == (9.0, 35, 12, "55,6 %", 2.25, 0.75, False)
    _claim("`T` = **26**, hartes Optimum A = **27** statt **12** (**125,0 %** mehr), **394** von 40 320 Plänen zulässig; bei λ = 1 überschreitet die Strafe die Schwelle um **9** (B = 35) und kostet A = 12, also **55,6 %** weniger; "
           "λ\\* = **2,25** (quadratisch **0,75**); die Gewichtssumme trifft es für kein λ")
    s, a = _preset(P_N3)
    ts, ls = ev.threshold_summary(a), ev.lam_summary(a)
    assert (ts["T"], ts["n_feasible"], ts["total"], ts["a_t"], ts["a_opt"], ls["lin"], ev.de(ls["quad"], 2), ls["reachable"]) == (14.0, 2, 6, 26, 14, 4.0, "1,33", False)
    _claim("`T` = **14**, **2** von 6 Plänen zulässig, hartes Optimum A = **26** statt **14**; λ\\* = **4** (quadratisch **1,33**); die Gewichtssumme trifft es nie")
    s, a = _preset(P_INF)
    ts, ls = ev.threshold_summary(a), ev.lam_summary(a)
    big_s, big_a = _preset(P_BIG)
    assert (ts["T"], ts["bmin"], ts["v_t"], ts["a_t"], ls["lin"], ev.penalty_result(big_a, big_s.form, big_s.lam)["a"]) == (6.0, 16, 10.0, 40, 3.0, 48)
    _claim("`T` = **6** unter der kleinsten Zeitsumme **16**: kleinste Verletzung **10**, A = **40**; die Strafe trifft das ab λ\\* = **3**; bei λ = 2⁵⁶ wählt sie A = **48** (A geht in float64 verloren)")
    s, a = _preset(P_HIGHS)
    ts = ev.threshold_summary(a)
    rows = {r["label"]: r for r in ev.highs_rows(a, s.lam)}
    assert (ev.de(ts["a_t"]), ev.de(ts["a_opt"]), ev.pct(ts["rise"]), ev.pct(rows["Strafe, λ = 2^40"]["loss"])) == ("191,6", "105,1", "82,3 %", "86,9 %")
    _claim("hartes Optimum A = **191,6** statt **105,1** (**82,3 %** mehr); die Aufzählung trifft es bei λ = 2⁴⁰, HiGHS liefert dort **86,9 %** höhere Kosten")


def test_threshold_at_one_hundred_percent_is_the_cost_optimums_time_sum():
    _claim("`τ = 100 %` ist genau `B(π_A)`")
    from hgw_scenario import make_instance
    for fam in (0, 1, 2):
        i = make_instance(8, fam, 100003)
        p = A.make_problem(i.arrays(), 8, i.exact)
        assert A.threshold(p, 100) == p.bA
