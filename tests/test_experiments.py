"""Experimente der App: jedes Preset, jede Strafform, jede Schwelle, jedes Gewicht, jede Größe und alle Schalter laufen ohne Fehler."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import hgw_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texts(at) -> str:
    return " ".join(m.value for kind in ("success", "info", "warning") for m in getattr(at, kind))


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_runs_and_shows_a_verdict(name):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert not at.exception, (name, [str(e) for e in at.exception])
    text = _texts(at)
    assert "Strafe" in text or "Gewichtssumme" in text


def test_default_preset_breaks_the_threshold_by_nine():
    assert "um **9** überschreitet" in _texts(_app())


@pytest.mark.parametrize("penalty", range(len(C.PENALTIES)))
def test_every_penalty_form_runs_and_is_named(penalty):
    at = _app()
    at.sidebar.selectbox(key="penalty_select").set_value(penalty).run()
    assert not at.exception, [str(e) for e in at.exception]
    assert C.PENALTY_SHORT[C.PENALTIES[penalty]] in _texts(at)


@pytest.mark.parametrize("tau", [C.TAU_MIN, -25, 0, 25, 100])
def test_every_threshold_runs(tau):
    at = _app()
    at.sidebar.slider(key="tau_slider").set_value(tau).run()
    assert not at.exception, [str(e) for e in at.exception]
    if tau < 0:
        assert "kein Plan ist zulässig" in _texts(at)


@pytest.mark.parametrize("e", [C.LAM_EXP_MIN, 0, 2, 20, 53, 56, C.LAM_EXP_MAX])
def test_every_weight_runs(e):
    at = _app()
    at.sidebar.slider(key="lam_exp_slider").set_value(e).run()
    assert not at.exception, [str(e_) for e_ in at.exception]


@pytest.mark.parametrize("family", range(len(C.FAMILIES)))
def test_every_family_runs(family):
    at = _app()
    at.sidebar.selectbox(key="family_select").set_value(family).run()
    assert not at.exception, [str(e) for e in at.exception]


@pytest.mark.parametrize("n", [3, 8, 9, 12])
def test_sizes_around_the_enumeration_limit_switch_the_experiments(n):
    at = _app()
    at.sidebar.slider(key="n_slider").set_value(n).run()
    assert not at.exception, [str(e) for e in at.exception]
    if n > C.ENUM_MAX:
        assert "nicht alle Pläne aufgezählt" in " ".join(m.value for m in at.info) and "HiGHS" in " ".join(m.value for m in at.markdown)
    else:
        assert "bindet" in _texts(at)


def test_the_sweeps_run_when_switched_on():
    at = _app()
    for key in ("sweep_shift", "sweep_small", "sweep_highs", "sweep_tol"):
        at.toggle(key=key).set_value(True).run()
        assert not at.exception, (key, [str(e) for e in at.exception])
    text = _texts(at)
    assert "Schwelle, λ* und Gewichtssumme" in text and "Instanzen mit verletzter Schwelle" in text and "Treffer der linearen Strafe in float64" in text
    assert "HiGHS mit Schlupfvariable, Treffer" in text and "Schwelle δ unter der Zeitsumme" in text
