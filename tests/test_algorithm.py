"""Handwerte: jede Einheit des Kerns an einem Beispiel, das von Hand nachgerechnet ist (DEMO-PLAYBOOK §1b)."""

import itertools

import numpy as np
import pytest

import hgw_algorithm as A
import hgw_constants as C
import hgw_evaluation as ev

PLANS = list(itertools.permutations(range(3)))
HAND = {(0, 1, 2): (11, 6), (0, 2, 1): (12, 5), (1, 0, 2): (6, 10), (1, 2, 0): (5, 8), (2, 0, 1): (12, 9), (2, 1, 0): (10, 8)}


def _plan(idx):
    return tuple(int(x) for x in A.all_plans(3)[idx])


def test_hand_example_sums_of_all_six_plans(hand_problem):
    for k, plan in enumerate(PLANS):
        assert (int(hand_problem.sA[k]), int(hand_problem.sB[k])) == HAND[plan]
    assert (hand_problem.bmin, hand_problem.bA, _plan(hand_problem.iA)) == (5, 8, (1, 2, 0))


def test_hand_example_threshold_target_and_violations(hand_problem):
    p = hand_problem
    assert A.threshold(p, 50) == 6.0 and A.threshold(p, 0) == 5.0 and A.threshold(p, 100) == 8.0 and A.threshold(p, -25) == 4.0
    T = A.threshold(p, 50)
    assert A.feasible(p, T) and not A.feasible(p, 4.0)
    assert _plan(A.target(p, T)) == (0, 1, 2) and p.sA[A.target(p, T)] == 11
    assert {PLANS[k]: float(v) for k, v in enumerate(A.viol(p, T))} == {(0, 1, 2): 0, (0, 2, 1): 0, (1, 0, 2): 4, (1, 2, 0): 2, (2, 0, 1): 3, (2, 1, 0): 2}


def test_hand_example_lambda_star_and_window(hand_problem):
    p, T = hand_problem, 6.0
    assert A.lam_star(p, T, 1) == 3.0 and A.lam_star(p, T, 2) == 1.5
    assert A.weighted_window(p, T) == (3.0, 1.0)                      # leer: 3 > 1
    assert A.int_bound(p, T) == 6.0                                    # A_Ziel - A_min = 11 - 5


def test_hand_example_penalty_choices(hand_problem):
    p, T = hand_problem, 6.0
    # lam = 1: f = A + v: (1,2,0) hat 5 + 2 = 7 und gewinnt, die Schwelle wird um 2 gebrochen
    j = A.chosen(p, T, "linear", 1.0)
    assert _plan(j) == (1, 2, 0) and A.viol(p, T)[j] == 2 and not A.hit(p, T, j)
    # lam = 3: Gleichstand (1,2,0): 5 + 3*2 = 11 mit (0,1,2): 11; kleinere Verletzung gewinnt
    assert _plan(A.chosen(p, T, "linear", 3.0)) == (0, 1, 2) and A.hit(p, T, A.chosen(p, T, "linear", 3.0))
    # quadratisch: lam = 1,5 ist die Grenze; 1 bricht die Schwelle, 2 nicht
    assert not A.hit(p, T, A.chosen(p, T, "quadratic", 1.0)) and A.hit(p, T, A.chosen(p, T, "quadratic", 1.5)) and A.hit(p, T, A.chosen(p, T, "quadratic", 2.0))
    # Gewichtssumme: kein lam trifft das harte Optimum
    assert all(not A.hit(p, T, A.chosen(p, T, "weighted", lam)) for lam in (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 8.0, 100.0))


def test_hand_example_unfeasible_threshold_has_lambda_star_seven_thirds(hand_problem):
    p, T = hand_problem, 4.0
    assert _plan(A.target(p, T)) == (0, 2, 1) and A.viol(p, T)[A.target(p, T)] == 1 and p.sA[A.target(p, T)] == 12
    assert A.lam_star(p, T, 1) == pytest.approx(7 / 3)
    assert not A.hit(p, T, A.chosen(p, T, "linear", 2.0)) and A.hit(p, T, A.chosen(p, T, "linear", 2.5))
    assert A.int_bound(p, T) is None                                   # nur bei zulässiger Schwelle


def test_penalty_values_forms_and_unknown_form(hand_problem):
    p = hand_problem
    assert list(A.penalty_values(p, 6.0, "linear", 2.0)) == [11, 12, 6 + 8, 5 + 4, 12 + 6, 10 + 4]
    assert list(A.penalty_values(p, 6.0, "quadratic", 2.0)) == [11, 12, 6 + 32, 5 + 8, 12 + 18, 10 + 8]
    assert list(A.penalty_values(p, 6.0, "weighted", 2.0)) == [11 + 12, 12 + 10, 6 + 20, 5 + 16, 12 + 18, 10 + 16]
    with pytest.raises(ValueError):
        A.penalty_values(p, 6.0, "huber", 1.0)


def test_highs_matches_the_enumeration_on_the_hand_example(hand_arrays):
    A_, B_ = hand_arrays
    hard = A.highs_plan(hand_arrays, 6.0, hard=True)
    assert A.plan_value(A_, hard) == 11 and A.plan_value(B_, hard) == 6
    assert A.highs_plan(hand_arrays, 4.0, hard=True) is None           # unlösbar
    soft4 = A.highs_plan(hand_arrays, 6.0, 4.0)
    soft1 = A.highs_plan(hand_arrays, 6.0, 1.0)
    assert (A.plan_value(A_, soft4), A.plan_value(B_, soft4)) == (11, 6) and (A.plan_value(A_, soft1), A.plan_value(B_, soft1)) == (5, 8)
    assert A.check_feasible(soft1, 3)


def test_analysis_and_summaries_for_the_hand_example(hand):
    a = ev.Analysis(hand, "enum", hand.arrays(), A.make_problem(hand.arrays(), 3, True), 6.0, 5, 8, 5)
    ts = ev.threshold_summary(a)
    assert (ts["T"], ts["a_t"], ts["b_t"], ts["n_feasible"], ts["total"], ts["feasible"], ts["binds"]) == (6.0, 11, 6, 2, 6, True, True)
    assert ts["rise"] == pytest.approx(6 / 5)
    ls = ev.lam_summary(a)
    assert (ls["lin"], ls["quad"], ls["reachable"], ls["bound"]) == (3.0, 1.5, False, 6.0)
    r = ev.penalty_result(a, "linear", 1.0)
    assert (r["a"], r["b"], r["v"], r["hit"], r["a_t"]) == (5, 8, 2.0, False, 11) and r["gain"] == pytest.approx(6 / 11)


def test_curve_has_a_hit_from_lambda_star_on(hand):
    a = ev.Analysis(hand, "enum", hand.arrays(), A.make_problem(hand.arrays(), 3, True), 6.0, 5, 8, 5)
    rows = ev.lam_curve(a, "linear")
    assert [r["hit"] for r in rows if r["e"] <= 1] == [False] * len([r for r in rows if r["e"] <= 1])     # lam = 2^-3 bis 2: unter lam* = 3
    assert all(r["hit"] for r in rows if r["e"] >= 2)                                                      # ab 4: exakt, auch bei 2^60


def test_check_feasible_and_enumeration_limits():
    assert A.check_feasible((2, 0, 1), 3) and not A.check_feasible((0, 0, 1), 3) and not A.check_feasible((0, 1), 3)
    with pytest.raises(ValueError):
        A.all_plans(10)
    assert len(A.all_plans(5)) == 120 and C.GRANULARITIES == (9, 99, 0)


def test_texts_for_lambda_and_delta():
    assert ev.lam_text(1.0) == "1" and ev.lam_text(0.25) == "0,25" and ev.lam_text(2.0 ** 56) == "2^56" and ev.lam_text(1e12) == "10^12" and ev.lam_text(1e6) == "10^6" and ev.lam_text(4.0) == "4"
    assert ev.delta_text(1e-5) == "10^-5" and ev.delta_text(1e-1) == "10^-1"
    assert ev.de(3.14159, 2) == "3,14" and ev.pct(0.5556) == "55,6 %" and ev.value_text(5.0) == "5" and ev.value_text(5.3) == "5,3"
    assert ev.plan_text((2, 0, 1), 3) == "P1→A3, P2→A1, P3→A2"


def test_hit_ignores_a_different_b_when_the_cost_is_equal():
    """Zielplan und Minimierer haben gleiches A und gleiche Verletzung, aber verschiedenes B: das gilt als Treffer (Gleichstand in A, Stück 5)."""
    p = A.Problem(3, True, np.array([5, 5, 9]), np.array([4, 3, 1]))
    T = 4.0
    assert A.target(p, T) == 1                                          # kleinstes A, dann kleinstes B
    assert A.hit(p, T, 0) and A.hit(p, T, 1) and not A.hit(p, T, 2)
