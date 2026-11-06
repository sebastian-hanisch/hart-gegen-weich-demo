"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, und was jedes Preset zeigen soll."""

import hgw_algorithm as A
import hgw_constants as C
import hgw_evaluation as ev
import hgw_presets as P

(SMALL, EXACT, QUAD, SUM, N3, INF, BIG, HIGHS) = list(C.PRESETS)


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds():
    for preset in C.PRESETS.values():
        assert C.N_MIN <= preset["n"] <= C.N_MAX and 0 <= preset["family"] < len(C.FAMILIES) and 0 <= preset["seed"] <= C.SEED_MAX
        assert C.TAU_MIN <= preset["tau"] <= C.TAU_MAX and preset["tau"] % C.TAU_STEP == 0
        assert 0 <= preset["penalty"] < len(C.PENALTIES) and C.LAM_EXP_MIN <= preset["lam_exp"] <= C.LAM_EXP_MAX


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS[SMALL]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_no_non_default_preset_equals_the_default_settings():
    """Lehre aus Stück 4: ein Preset, das den Voreinstellungen gleicht, wäre nicht von „nichts geklickt“ zu unterscheiden."""
    for name, preset in C.PRESETS.items():
        if name != SMALL:
            assert ev.Settings(**preset) != ev.Settings(), name


def _all(name):
    s = ev.Settings(**C.PRESETS[name])
    a = ev.analyse(s)
    return s, a


def test_small_weight_preset_breaks_the_threshold_by_nine_and_looks_cheaper():
    s, a = _all(SMALL)
    ts, r = ev.threshold_summary(a), ev.penalty_result(a, s.form, s.lam)
    assert (ts["T"], ts["a_t"], ts["a_opt"], r["v"], r["a"], r["hit"]) == (26.0, 27, 12, 9.0, 12, False)


def test_exact_preset_hits_from_lambda_star_on_and_the_help_numbers_are_true():
    s, a = _all(EXACT)
    ls, r = ev.lam_summary(a), ev.penalty_result(a, s.form, s.lam)
    assert r["hit"] and ls["lin"] == 2.25 and s.lam >= ls["lin"] and "2,25" in C.PRESET_HELP[EXACT]


def test_quadratic_preset_is_exact_earlier_than_the_linear_penalty():
    s, a = _all(QUAD)
    ls = ev.lam_summary(a)
    assert s.form == "quadratic" and ls["quad"] == 0.75 and ls["quad"] < s.lam < ls["lin"] and ev.penalty_result(a, "quadratic", s.lam)["hit"] and not ev.penalty_result(a, "linear", s.lam)["hit"]


def test_weighted_preset_never_reaches_the_target():
    s, a = _all(SUM)
    ls = ev.lam_summary(a)
    assert s.form == "weighted" and not ls["reachable"] and all(not r["hit"] for r in ev.lam_curve(a, "weighted"))


def test_smallest_example_has_six_plans_and_the_documented_numbers():
    s, a = _all(N3)
    ls = ev.lam_summary(a)
    assert a.inst.n == 3 and len(a.prob.sA) == 6 and (ls["lin"], ls["quad"], ls["reachable"]) == (4.0, 4 / 3, False)


def test_infeasible_preset_has_no_hard_plan_but_a_soft_answer():
    s, a = _all(INF)
    ts, r = ev.threshold_summary(a), ev.penalty_result(a, s.form, s.lam)
    assert not ts["feasible"] and r["hit"] and r["v"] == 10 and r["a"] == 40
    assert A.highs_plan(a.mats, a.T, hard=True) is None


def test_float64_preset_loses_the_cost_at_the_large_weight():
    s, a = _all(BIG)
    r = ev.penalty_result(a, s.form, s.lam)
    assert s.lam == 2.0 ** 56 and not r["hit"] and r["v"] == 10 and r["a"] == 48 and ev.penalty_result(a, s.form, 2.0 ** 6)["hit"]


def test_highs_preset_is_real_valued_and_highs_misses_at_the_set_weight():
    s, a = _all(HIGHS)
    assert not a.inst.exact and ev.penalty_result(a, s.form, s.lam)["hit"]              # die Aufzählung trifft
    rows = {r["label"]: r for r in ev.highs_rows(a, s.lam)}
    key = "Strafe, λ = 2^40"
    assert key in rows and not rows[key]["hit"] and rows[key]["v"] <= 1e-9 and 0.8 < rows[key]["loss"] < 0.9
    assert rows["Strafe, λ = 10^9"]["hit"]
