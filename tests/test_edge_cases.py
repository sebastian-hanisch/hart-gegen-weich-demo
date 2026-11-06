"""Randfälle: kleinste und größte Instanz, Schwelle an den Rändern, konstante Matrix, nicht aufgezählte Größen, Wertebereiche."""

import numpy as np
import pytest

import hgw_algorithm as A
import hgw_constants as C
import hgw_evaluation as ev
from hgw_scenario import Instance, make_instance


def _flat(n: int, value: int = 2) -> Instance:
    m = tuple(tuple(value for _ in range(n)) for _ in range(n))
    return Instance((m, m), True)


def _analysis(inst, tau=50):
    p = A.make_problem(inst.arrays(), inst.n, inst.exact)
    return ev.Analysis(inst, "enum", inst.arrays(), p, A.threshold(p, tau), p.bmin, p.bA, p.sA.min().item())


def test_constant_matrices_make_every_plan_equal_and_the_threshold_irrelevant():
    a = _analysis(_flat(4))
    ts = ev.threshold_summary(a)
    assert ts["feasible"] and not ts["binds"] and ts["rise"] == 0 and ts["n_feasible"] == 24
    ls = ev.lam_summary(a)
    assert (ls["lin"], ls["quad"]) == (0.0, 0.0) and ls["reachable"] is True
    kind, text = ev.verdict_threshold(ts)
    assert kind == "info" and "bindet hier nicht" in text
    r = ev.penalty_result(a, "linear", 1.0)
    assert r["hit"] and r["v"] == 0


def test_a_threshold_equal_to_the_smallest_time_sum_leaves_the_fastest_plans_only():
    inst = make_instance(6, 0, 3)
    p = A.make_problem(inst.arrays(), 6, True)
    T = A.threshold(p, 0)
    assert T == p.bmin and all(p.sB[i] == p.bmin for i in np.flatnonzero(p.sB <= T))
    t = A.target(p, T)
    assert p.sB[t] == p.bmin and p.sA[t] == p.sA[p.sB == p.bmin].min()


def test_the_degenerate_case_where_the_cost_optimum_is_also_the_fastest_plan():
    a = np.array([[1, 5], [5, 1]])
    b = np.array([[1, 5], [5, 1]])
    p = A.make_problem([a, b], 2, True)
    assert p.bmin == p.bA == 2 and A.threshold(p, 50) == 2.0 and A.threshold(p, -50) == 2.0       # Spanne null: die Schwelle bleibt gleich der kleinsten Zeitsumme
    assert A.feasible(p, 2.0) and A.lam_star(p, 2.0) == 0.0


def test_negative_tau_makes_the_threshold_infeasible_and_the_target_the_smallest_violation():
    inst = make_instance(6, 0, 4)
    p = A.make_problem(inst.arrays(), 6, True)
    T = A.threshold(p, -25)
    assert T < p.bmin and not A.feasible(p, T)
    t = A.target(p, T)
    v = A.viol(p, T)
    assert v[t] == v.min() > 0 and p.sB[t] == p.bmin
    a = _analysis(inst, -25)
    ts = ev.threshold_summary(a)
    assert not ts["feasible"] and not ts["binds"] and ev.verdict_threshold(ts)[0] == "warning"
    assert ev.lam_summary(a)["bound"] is None


def test_smallest_instance_runs_every_experiment_for_every_family_and_form():
    for family in range(3):
        for tau in (-25, 0, 50, 100):
            a = ev.analyse(ev.Settings(3, family, 0, tau))
            assert a.mode == "enum" and len(a.prob.sA) == 6
            ts, ls = ev.threshold_summary(a), ev.lam_summary(a)
            ev.verdict_threshold(ts), ev.verdict_lambda(ls, ts)
            for form in C.PENALTIES:
                for e in (-3, 0, 5, 56, 60):
                    ev.verdict_penalty(ev.penalty_result(a, form, 2.0 ** e), ts, ls)
                ev.verdict_curve(ev.lam_curve(a, form), form)


def test_n_above_the_enumeration_limit_is_not_enumerated_but_highs_still_answers():
    for n in (C.ENUM_MAX + 1, C.N_MAX):
        a = ev.analyse(ev.Settings(n, 0, 2, 50))
        assert a.mode == "large" and a.prob is None and a.bmin <= a.T <= a.bA
        rows = ev.highs_rows(a, 1.0)
        assert rows[0]["label"].startswith("hart") and rows[0]["plan"] is not None and A.check_feasible(rows[0]["plan"], n)
        assert rows[0]["b"] <= a.T + 1e-9 and ev.verdict_highs(rows)[0] in ("info", "warning")
        assert len(ev.tolerance_rows(a)) == len(C.SOLVER_DELTAS)


def test_large_instance_with_an_infeasible_threshold_has_no_hard_plan():
    a = ev.analyse(ev.Settings(10, 0, 2, -25))
    rows = ev.highs_rows(a, 1.0)
    assert rows[0]["plan"] is None and ev.verdict_highs(rows)[1].startswith("Mit harter Nebenbedingung findet HiGHS **keinen** Plan")


def test_enumeration_refuses_ten_persons():
    with pytest.raises(ValueError):
        A.all_plans(10)
    with pytest.raises(ValueError):
        A.all_plans(0)


def test_value_ranges_follow_the_family_and_the_instances_are_deterministic():
    for family, k in enumerate(C.GRANULARITIES):
        for m in make_instance(8, family, 1).arrays():
            assert (m.min() >= 1 and m.max() <= k) if k else (m.min() >= 0 and m.max() <= 100)
        assert make_instance(6, family, 4) == make_instance(6, family, 4) and make_instance(6, family, 4) != make_instance(6, family, 5)
    assert make_instance(5, 2, 3).arrays()[0].dtype == float and make_instance(5, 0, 3).arrays()[0].dtype == np.int64


def test_unknown_form_raises_and_zero_weight_chooses_the_cost_optimum():
    inst = make_instance(5, 1, 1)
    p = A.make_problem(inst.arrays(), 5, True)
    T = A.threshold(p, 50)
    with pytest.raises(ValueError):
        A.chosen(p, T, "huber", 1.0)
    for form in ("linear", "quadratic", "weighted"):
        j = A.chosen(p, T, form, 0.0)
        assert p.sA[j] == p.sA.min()


def test_float64_loses_the_cost_when_the_penalty_is_far_larger_than_two_to_the_fifty_three():
    """Der Mechanismus hinter „zu großes Gewicht“: bei v = 1 und lam = 2^60 ändert das Addieren von A = 40 oder 48 die float64-Summe nicht."""
    assert 2.0 ** 60 + 40 == 2.0 ** 60 + 48 and 2.0 ** 52 + 40 != 2.0 ** 52 + 48
    p = A.Problem(3, True, np.array([40, 48, 90]), np.array([6, 6, 5]))
    T = 4.0                                                           # unzulässig: v = 2, 2, 1
    assert A.target(p, T) == 2
    assert A.chosen(p, T, "linear", 2.0 ** 6) == 2
    p2 = A.Problem(3, True, np.array([40, 48, 90]), np.array([5, 5, 5]))
    assert A.target(p2, 4.0) == 0                                      # gleiche Verletzung 1: kleinstes A
    assert A.chosen(p2, 4.0, "linear", 2.0 ** 60) == 0                 # alle Strafen gleich: bei Gleichstand der erste Plan (hier zufällig der richtige)
    p3 = A.Problem(3, True, np.array([48, 40, 90]), np.array([5, 5, 5]))
    assert A.target(p3, 4.0) == 1 and A.chosen(p3, 4.0, "linear", 2.0 ** 60) == 0      # der erste Plan, nicht der billigste
    assert A.chosen(p3, 4.0, "linear", 2.0 ** 6) == 1


def test_text_for_a_zero_weight_and_for_large_weights_is_stable():
    assert ev.lam_text(0.0625) == "0,062"
    assert ev.lam_text(2.0 ** 20) == "2^20" and ev.lam_text(2.0 ** 19) == "524288"
