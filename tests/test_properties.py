"""Eigenschaftstests (ORACLE-PLAYBOOK §5): Aussagen, die für jede Instanz gelten müssen, an festen Seeds und allen Familien."""

import numpy as np
import pytest

import hgw_algorithm as A
from hgw_scenario import Instance, make_instance

CASES = [(n, family, seed) for n in (3, 5, 7, 8) for family in range(3) for seed in range(3)]
TAUS = (0, 25, 50, 75, 100, -25)


def _prob(n, family, seed):
    inst = make_instance(n, family, seed)
    return inst, A.make_problem(inst.arrays(), n, inst.exact)


@pytest.mark.parametrize("n,family,seed", CASES)
def test_the_hard_optimum_is_feasible_cost_minimal_and_never_cheaper_than_the_cost_optimum(n, family, seed):
    inst, p = _prob(n, family, seed)
    for tau in TAUS:
        T = A.threshold(p, tau)
        t = A.target(p, T)
        if A.feasible(p, T):
            assert p.sB[t] <= T and p.sA[t] == p.sA[p.sB <= T].min() and p.sA[t] >= p.sA.min()
        else:
            v = A.viol(p, T)
            assert v[t] == v.min() and v[t] > 0


@pytest.mark.parametrize("n,family,seed", CASES)
def test_the_threshold_is_monotone_and_the_hard_cost_does_not_fall_when_it_tightens(n, family, seed):
    inst, p = _prob(n, family, seed)
    costs = []
    for tau in (100, 75, 50, 25, 0):
        T = A.threshold(p, tau)
        costs.append(p.sA[A.target(p, T)].item())
    assert costs == sorted(costs)                       # engere Schwelle, nie billiger
    assert A.threshold(p, 100) >= A.threshold(p, 50) >= A.threshold(p, 0) >= A.threshold(p, -25)


@pytest.mark.parametrize("n,family,seed", CASES)
def test_a_threshold_at_one_hundred_percent_does_not_bind(n, family, seed):
    inst, p = _prob(n, family, seed)
    T = A.threshold(p, 100)
    assert A.feasible(p, T) and p.sA[A.target(p, T)] == p.sA.min() and A.lam_star(p, T, 1) == 0.0


@pytest.mark.parametrize("n,family,seed", CASES)
def test_the_penalty_optimum_is_constant_from_lambda_star_on_and_never_worse_in_a_before_that(n, family, seed):
    inst, p = _prob(n, family, seed)
    T = A.threshold(p, 50)
    t = A.target(p, T)
    for power, form in ((1, "linear"), (2, "quadratic")):
        ls = A.lam_star(p, T, power)
        for lam in (ls, ls * 1.5 + 0.5, ls * 8 + 8, 1e6):
            assert A.hit(p, T, A.chosen(p, T, form, lam)), (form, lam)
        for lam in (0.0, ls * 0.5, ls * 0.99):
            j = A.chosen(p, T, form, lam)
            assert p.sA[j] <= p.sA[t] + 1e-9                    # vor λ* nie teurer als das harte Optimum: es bricht die Schwelle oder trifft sie


@pytest.mark.parametrize("n,family,seed", CASES)
def test_the_violation_of_the_chosen_plan_never_grows_with_the_weight(n, family, seed):
    """Bei kleinster Verletzung als Gleichstandsregel: mehr Gewicht, nie mehr Verletzung (lineare und quadratische Strafe)."""
    inst, p = _prob(n, family, seed)
    T = A.threshold(p, 50)
    v = A.viol(p, T)
    for form in ("linear", "quadratic"):
        seen = [v[A.chosen(p, T, form, 2.0 ** e)] for e in range(-3, 16)]
        assert seen == sorted(seen, reverse=True), form


@pytest.mark.parametrize("n,family,seed", [c for c in CASES if c[1] < 2])
def test_integer_instances_obey_the_bound_and_a_violation_of_at_least_one(n, family, seed):
    inst, p = _prob(n, family, seed)
    for tau in (0, 25, 50, 75):
        T = A.threshold(p, tau)
        v = A.viol(p, T)
        assert v[v > 0].min() >= 1 if (v > 0).any() else True
        assert A.lam_star(p, T, 1) <= A.int_bound(p, T) + 1e-12


@pytest.mark.parametrize("n,family,seed", [(5, 0, 1), (6, 1, 2), (7, 2, 3), (8, 0, 4)])
def test_scaling_b_scales_the_lambda_stars_with_the_unit(n, family, seed):
    inst = make_instance(n, family, seed)
    a_, b_ = inst.arrays()
    p1 = A.make_problem([a_, b_], n, inst.exact)
    for c in (2, 4):
        p2 = A.make_problem([a_.astype(float), b_.astype(float) * c], n, False)
        T1, T2 = A.threshold(p1, 50), A.threshold(p2, 50)
        if p1.exact:
            T2 = c * float(p1.bmin) + 0.5 * (c * float(p1.bA) - c * float(p1.bmin))      # Schwelle an derselben Stelle (ohne Abrunden), damit die Pläne dieselben sind
            T1 = float(p1.bmin) + 0.5 * (float(p1.bA) - float(p1.bmin))
            p1f = A.make_problem([a_.astype(float), b_.astype(float)], n, False)
        else:
            p1f = p1
        assert A.lam_star(p2, T2, 1) == pytest.approx(A.lam_star(p1f, T1, 1) / c, rel=1e-9)
        assert A.lam_star(p2, T2, 2) == pytest.approx(A.lam_star(p1f, T1, 2) / c ** 2, rel=1e-9)


@pytest.mark.parametrize("n,family,seed", [(5, 0, 1), (6, 1, 2), (7, 2, 3)])
def test_the_hard_optimum_value_does_not_depend_on_the_numbering(n, family, seed):
    inst = make_instance(n, family, seed)
    a_, b_ = inst.arrays()
    p = A.make_problem([a_, b_], n, inst.exact)
    T = A.threshold(p, 50)
    base = p.sA[A.target(p, T)]
    rng = np.random.default_rng(seed)
    for _ in range(4):
        r, c = rng.permutation(n), rng.permutation(n)
        p2 = A.make_problem([a_[r][:, c], b_[r][:, c]], n, inst.exact)
        T2 = A.threshold(p2, 50)
        assert T2 == pytest.approx(T) and p2.sA[A.target(p2, T2)] == pytest.approx(base)
        assert A.lam_star(p2, T2, 1) == pytest.approx(A.lam_star(p, T, 1))
        lo, hi = A.weighted_window(p, T)
        lo2, hi2 = A.weighted_window(p2, T2)
        assert (lo < hi) == (lo2 < hi2)


def test_everything_is_deterministic():
    assert make_instance(5, 1, 3) == make_instance(5, 1, 3) and make_instance(5, 1, 3) != make_instance(5, 1, 4)
    i = make_instance(6, 2, 9)
    p1, p2 = A.make_problem(i.arrays(), 6, i.exact), A.make_problem(i.arrays(), 6, i.exact)
    assert (p1.sA == p2.sA).all() and A.lam_star(p1, A.threshold(p1, 50)) == A.lam_star(p2, A.threshold(p2, 50))
    assert isinstance(i, Instance) and not i.exact
