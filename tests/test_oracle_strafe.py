"""Orakel-Test hart-gegen-weich-demo: der Kern gegen einen ANDEREN Rechenweg (ORACLE-PLAYBOOK §2, DEMO-PLAYBOOK §12).

Reines Python (itertools, Tupelvergleich, Fraction, kein numpy im Referenzweg) gegen die Vektorrechnung, und HiGHS gegen die Aufzählung.
Mini-Instanzen n = 3 bis 6, feste Seeds, alle drei Familien."""

import itertools
from fractions import Fraction

import pytest

import hgw_algorithm as A
import hgw_evaluation as ev
from hgw_scenario import make_instance

CASES = [(n, family, seed) for n in (3, 4, 5, 6) for family in (0, 1, 2) for seed in range(4)]


def _plans(inst):
    """Alle Pläne mit (A, B) in reinem Python (itertools-Reihenfolge)."""
    n = inst.n
    a, b = inst.mats
    return [(plan, sum(a[i][plan[i]] for i in range(n)), sum(b[i][plan[i]] for i in range(n))) for plan in itertools.permutations(range(n))]


def _setup(n, family, seed, tau=50):
    inst = make_instance(n, family, seed)
    plans = _plans(inst)
    prob = A.make_problem(inst.arrays(), n, inst.exact)
    return inst, plans, prob, A.threshold(prob, tau)


def _ref_target(plans, T):
    """Zielplan nach Definition: lexikographisch kleinste (Verletzung, A, B), bei Gleichstand der erste."""
    best = min(plans, key=lambda x: (max(0, x[2] - T), x[1], x[2]))
    return best


@pytest.mark.parametrize("n,family,seed", CASES)
def test_the_threshold_matches_its_definition(n, family, seed):
    inst, plans, prob, _ = _setup(n, family, seed)
    bmin = min(b for _, _, b in plans)
    best_a = min(plans, key=lambda x: x[1])
    d = best_a[2] - bmin
    for tau in (0, 25, 50, 75, 100, -25):
        expected = best_a[2] if tau == 100 else bmin + tau / 100 * d
        got = A.threshold(prob, tau)
        assert got == pytest.approx(float(int(expected // 1)) if inst.exact else expected, abs=1e-9)


@pytest.mark.parametrize("n,family,seed", CASES)
def test_the_target_matches_the_definition_for_feasible_and_infeasible_thresholds(n, family, seed):
    inst, plans, prob, _ = _setup(n, family, seed)
    for tau in (0, 25, 50, 75, 100, -25, -50):
        if tau == 100 and not inst.exact:
            continue            # reell: die Summationsreihenfolge (numpy gegen Python) ändert die letzten Bits, und T liegt dort genau auf B des Kostenoptimums
        T = A.threshold(prob, tau)
        ref = _ref_target(plans, T)
        j = A.target(prob, T)
        got = A.all_plans(n)[j]
        assert tuple(int(x) for x in got) == ref[0], (tau, ref)


@pytest.mark.parametrize("n,family,seed", CASES)
def test_chosen_plan_matches_the_definition_for_every_form(n, family, seed):
    inst, plans, prob, T = _setup(n, family, seed)
    for form in ("linear", "quadratic", "weighted"):
        for lam in (0.0, 0.5, 1.0, 4.0, 64.0):
            def f(x):
                v = max(0, x[2] - T)
                return {"linear": x[1] + lam * v, "quadratic": x[1] + lam * v * v, "weighted": x[1] + lam * x[2]}[form]
            fmin = min(f(x) for x in plans)
            ties = [x for x in plans if f(x) == fmin]
            expected = min(ties, key=lambda x: max(0, x[2] - T))          # kleinste Verletzung, dann der erste (min nimmt den ersten)
            assert tuple(int(c) for c in A.all_plans(n)[A.chosen(prob, T, form, lam)]) == expected[0], (form, lam)


@pytest.mark.parametrize("n,family,seed", [c for c in CASES if c[1] < 2])
def test_lambda_star_matches_an_exact_fraction_search(n, family, seed):
    """Ganze Zahlen: lam* ist ein Bruch (A-Differenz durch Verletzung hoch p); mit Fraction nachgerechnet, ohne Gleitkomma."""
    inst, plans, prob, T = _setup(n, family, seed)
    t = _ref_target(plans, T)
    vt = max(0, t[2] - T)
    for power in (1, 2):
        best = Fraction(0)
        for _, a_, b_ in plans:
            v = max(0, b_ - T)
            if a_ < t[1] and v > vt:
                best = max(best, Fraction(t[1] - a_) / (Fraction(v) ** power - Fraction(vt) ** power))
        assert A.lam_star(prob, T, power) == pytest.approx(float(best), rel=1e-12)


@pytest.mark.parametrize("n,family,seed", CASES)
def test_lambda_star_is_the_exact_threshold_against_a_grid_above_and_below(n, family, seed):
    inst, plans, prob, T = _setup(n, family, seed)
    t = A.target(prob, T)
    for power, form in ((1, "linear"), (2, "quadratic")):
        ls = A.lam_star(prob, T, power)
        assert A.hit(prob, T, A.chosen(prob, T, form, ls * 1.001 + 1e-12))
        assert A.hit(prob, T, A.chosen(prob, T, form, ls * 100 + 1))
        if ls > 1e-6:
            assert not A.hit(prob, T, A.chosen(prob, T, form, ls * 0.999)), (form, ls)


@pytest.mark.parametrize("n,family,seed", CASES)
def test_the_weighted_window_matches_a_grid_of_weights(n, family, seed):
    inst, plans, prob, T = _setup(n, family, seed)
    lo, hi = A.weighted_window(prob, T)
    t = A.target(prob, T)
    a_t, b_t = prob.sA[t], prob.sB[t]
    grid = [0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 50.0, 1000.0] + [lo, hi if hi < 1e9 else 1e4, (lo + hi) / 2 if hi < 1e9 else lo + 1]
    for lam in grid:
        values = [a_ + lam * b_ for _, a_, b_ in plans]
        is_min = a_t + lam * b_t <= min(values) + 1e-9
        if lo + 1e-9 < lam < hi - 1e-9:
            assert is_min, lam
        elif lam < lo - 1e-9 or lam > hi + 1e-9:
            assert not is_min, lam


@pytest.mark.parametrize("n,family,seed", [(5, 0, 1), (6, 0, 2), (6, 1, 3), (5, 2, 1), (6, 2, 4)])
def test_highs_hard_matches_the_enumeration_and_the_soft_penalty_at_safe_weights(n, family, seed):
    inst, plans, prob, T = _setup(n, family, seed)
    mats = inst.arrays()
    t = A.target(prob, T)
    a_t = prob.sA[t].item()
    hard = A.highs_plan(mats, T, hard=True)
    assert abs(A.plan_value(mats[0], hard) - a_t) < 1e-7 and A.plan_value(mats[1], hard) <= T + 1e-9
    ls = A.lam_star(prob, T, 1)
    soft = A.highs_plan(mats, T, ls * 4 + 10)
    assert abs(A.plan_value(mats[0], soft) - a_t) < 1e-7 and A.plan_value(mats[1], soft) <= T + 1e-9
    assert A.check_feasible(hard, n) and A.check_feasible(soft, n)


@pytest.mark.parametrize("n,family,seed", [(5, 0, 1), (6, 1, 3)])
def test_highs_is_unsolvable_with_a_hard_infeasible_threshold_but_the_penalty_still_answers(n, family, seed):
    inst, plans, prob, _ = _setup(n, family, seed)
    mats = inst.arrays()
    T = A.infeasible_threshold(prob)
    assert A.highs_plan(mats, T, hard=True) is None
    plan = A.highs_plan(mats, T, 1000.0)
    t = _ref_target(plans, T)
    assert A.plan_value(mats[1], plan) == pytest.approx(t[2]) and A.plan_value(mats[0], plan) == pytest.approx(t[1])


def test_the_sweep_hit_counts_match_a_plain_python_recount_on_one_family():
    """lam* über feste Instanzen: zwei Wege (Kern und reines Python) nennen dieselbe Zahl streng erreichbarer Gewichtssummen."""
    count = 0
    ref = 0
    for seed in range(100000, 100006):
        inst = make_instance(6, 0, seed)
        plans = _plans(inst)
        prob = A.make_problem(inst.arrays(), 6, True)
        T = A.threshold(prob, 50)
        lo, hi = A.weighted_window(prob, T)
        count += lo < hi
        t = _ref_target(plans, T)
        lo2 = max([Fraction(t[1] - a_) / (b_ - t[2]) for _, a_, b_ in plans if b_ > t[2]] + [Fraction(0)])
        hi2 = min([Fraction(a_ - t[1]) / (t[2] - b_) for _, a_, b_ in plans if b_ < t[2]] + [Fraction(10 ** 9)])
        ref += lo2 < hi2
    assert count == ref
    assert ev.shift_sweep(fams=(0,), taus=(50,))[(0, 50)]["reach"] == 11
