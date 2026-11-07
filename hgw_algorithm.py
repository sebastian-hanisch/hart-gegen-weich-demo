"""Kern der Demo (numpy, scipy): eine harte Schwelle gegen einen Strafterm.

Zwei Matrizen A (Kosten, minimieren) und B (Zeit); ein Plan hat die Summen A(π) und B(π). Die Schwelle T gilt für B.
  * Hart: kleinstes A unter B <= T. Gibt es keinen zulässigen Plan (T unter der kleinsten Zeitsumme), ist das harte Problem unlösbar.
  * Weich: Strafe auf die Verletzung v = max(0, B - T): linear A + lam*v, quadratisch A + lam*v^2; zum Vergleich die Gewichtssumme A + lam*B ohne Schwelle.
  * `target`: der Plan, den „hart“ meint: kleinste Verletzung, dann kleinstes A, dann kleinstes B, dann der erste Plan (bei zulässiger Schwelle das harte Optimum).
  * `lam_star`: kleinstes lam, ab dem das Strafoptimum den Zielplan enthält; exakt aus der endlichen Planmenge gerechnet (kein Gitter).
  * `weighted_window`: Fenster der lam, in denen die Gewichtssumme den Zielplan minimiert (leer: er liegt nicht auf der unteren Hülle; ein einzelner Punkt: er ist nur im Gleichstand ein Minimierer).
  * `highs_plan`: dasselbe als ganzzahliges Programm mit Schlupfvariable (HiGHS über scipy.optimize.milp, Voreinstellungen) oder mit harter Nebenbedingung.
Der Minimierer bei Gleichstand der Strafe ist der Plan mit der kleinsten Verletzung, dann der erste der Aufzählung (siehe README).
Quelle: Aufzählung und HiGHS-Aufruf stammen aus Stück 1 bis 5 der Linie (zielvergleich-demo bis gleichstand-demo); neu sind Schwelle, Zielplan, lam*, Fenster, Schlupf-Fall.
"""

import itertools
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


@lru_cache(maxsize=None)
def all_plans(n: int) -> np.ndarray:
    """Alle n! Pläne als Zeilen (Reihenfolge von itertools.permutations); nur bis n = 9."""
    if not 1 <= n <= 9:
        raise ValueError(f"n = {n}: Aufzählung nur für 1 bis 9 Personen")
    return np.array(list(itertools.permutations(range(n))), dtype=np.int16)


def check_feasible(plan, n: int) -> bool:
    """Unabhängiger Prüfer: der Plan ist eine Zuordnung, jede Aufgabe genau einmal."""
    plan = [int(j) for j in plan]
    return len(plan) == n and sorted(plan) == list(range(n))


def plan_sums(m: np.ndarray, n: int) -> np.ndarray:
    """Summe einer Matrix über alle Pläne."""
    return m[np.arange(n)[None, :], all_plans(n)].sum(axis=1)


def plan_value(m: np.ndarray, plan) -> float:
    return m[np.arange(m.shape[0]), np.asarray(plan, dtype=int)].sum().item()


@dataclass
class Problem:
    n: int
    exact: bool
    sA: np.ndarray            # Kosten-Summe je Plan
    sB: np.ndarray            # Zeit-Summe je Plan

    @property
    def bmin(self) -> float:
        return self.sB.min().item()

    @property
    def iA(self) -> int:
        """Kostenoptimum (erster Plan mit kleinstem A)."""
        return int(np.argmin(self.sA))

    @property
    def bA(self) -> float:
        return self.sB[self.iA].item()


def make_problem(mats: list[np.ndarray], n: int, exact: bool) -> Problem:
    return Problem(n, exact, plan_sums(mats[0], n), plan_sums(mats[1], n))


def threshold(p: Problem, tau_pct: int) -> float:
    """Schwelle: T = Bmin + tau * (B(Kostenoptimum) - Bmin), tau in Prozent; bei ganzen Zahlen abgerundet. Unter 0 liegt T unter der kleinsten Zeitsumme."""
    t = p.bA if tau_pct == 100 else p.bmin + (tau_pct / 100.0) * (p.bA - p.bmin)      # 100 %: genau B des Kostenoptimums (bmin + (bA - bmin) rundet in float64 sonst knapp darunter)
    return float(np.floor(t)) if p.exact else float(t)


def infeasible_threshold(p: Problem) -> float:
    """Eine unzulässige Schwelle für die Sweeps: eins unter der kleinsten Zeitsumme (ganze Zahlen) beziehungsweise 95 % davon (reell)."""
    return p.bmin - 1 if p.exact else 0.95 * p.bmin


def feasible(p: Problem, T: float) -> bool:
    return p.bmin <= T


def viol(p: Problem, T: float) -> np.ndarray:
    return np.maximum(0.0, p.sB - T)


def target(p: Problem, T: float) -> int:
    """Index des Zielplans: kleinste Verletzung, dann kleinstes A, dann kleinstes B, dann der erste."""
    v = viol(p, T)
    cand = np.flatnonzero(v == v.min())
    cand = cand[p.sA[cand] == p.sA[cand].min()]
    return int(cand[np.argmin(p.sB[cand])])


def penalty_values(p: Problem, T: float, form: str, lam: float) -> np.ndarray:
    """Zielwert je Plan: linear A + lam*v, quadratisch A + lam*v^2, Gewichtssumme A + lam*B (float64)."""
    if form == "linear":
        return p.sA + lam * viol(p, T)
    if form == "quadratic":
        return p.sA + lam * viol(p, T) ** 2
    if form == "weighted":
        return p.sA + lam * p.sB
    raise ValueError(f"unbekannte Strafform: {form}")


def chosen(p: Problem, T: float, form: str, lam: float) -> int:
    """Minimierer der Strafe; bei Gleichstand der Wert die kleinste Verletzung, dann der erste Plan."""
    f = penalty_values(p, T, form, lam)
    m = np.flatnonzero(f == f.min())
    if len(m) == 1:
        return int(m[0])
    v = viol(p, T)[m]
    return int(m[np.argmin(v)])


def hit(p: Problem, T: float, idx: int) -> bool:
    """Trifft der gewählte Plan den Zielplan (gleiche Verletzung und gleiches A; bei Gleichstand darf B anders sein)?"""
    t = target(p, T)
    v = viol(p, T)
    return bool(v[idx] == v[t] and p.sA[idx] == p.sA[t])


def lam_star(p: Problem, T: float, power: int = 1) -> float:
    """Kleinstes lam, ab dem A + lam*v^power den Zielplan unter den Minimierern hat: max über Pläne mit kleinerem A und größerer Verletzung von
    (A_Ziel - A) / (v^power - v_Ziel^power); 0, wenn es keinen solchen Plan gibt."""
    t = target(p, T)
    v = viol(p, T)
    m = (p.sA < p.sA[t]) & (v > v[t])
    if not m.any():
        return 0.0
    return float(np.max((p.sA[t] - p.sA[m]) / (v[m] ** power - v[t] ** power)))


def weighted_window(p: Problem, T: float) -> tuple[float, float]:
    """Fenster [lo, hi] der lam >= 0, in denen A + lam*B den Zielplan minimiert.
    Aus A_t + lam*B_t <= A + lam*B: Pläne mit B > B_t geben die untere, Pläne mit B < B_t die obere Schranke. Leer, wenn lo > hi;
    bei lo == hi besteht das Fenster aus einem einzigen lam (der Zielplan liegt auf einer Kante der unteren Hülle und ist dort ein Minimierer im Gleichstand)."""
    t = target(p, T)
    a_t, b_t = p.sA[t], p.sB[t]
    more, less = p.sB > b_t, p.sB < b_t
    lo = max(0.0, float(np.max((a_t - p.sA[more]) / (p.sB[more] - b_t)))) if more.any() else 0.0
    hi = float(np.min((p.sA[less] - a_t) / (b_t - p.sB[less]))) if less.any() else float("inf")
    return lo, hi


def window_reachable(lo: float, hi: float) -> bool:
    """Der Zielplan minimiert A + lam*B für mindestens ein lam: das Fenster ist nicht leer (lo <= hi; lo == hi ist ein einzelner Punkt und zählt mit)."""
    return bool(lo <= hi)


def int_bound(p: Problem, T: float) -> float | None:
    """Bei ganzen Zahlen gilt v >= 1; daher genügt lam >= A_Ziel - A_min für die lineare Strafe (nur bei zulässiger Schwelle)."""
    if not p.exact or not feasible(p, T):
        return None
    return float(p.sA[target(p, T)] - p.sA.min())


# --- HiGHS -----------------------------------------------------------------------------------------------------------------------------------

def highs_plan(mats: list[np.ndarray], T: float, lam: float | None = None, hard: bool = False):
    """Zuordnung als ganzzahliges Programm (HiGHS über scipy.optimize.milp, Voreinstellungen).
    Weich: min A + lam*s mit s >= B - T und s >= 0 (Schlupfvariable). Hart: B <= T als Nebenbedingung. Rückgabe: Plan oder None (unlösbar)."""
    mA, mB = mats[0], mats[1]
    n = mA.shape[0]
    nv = n * n + (0 if hard else 1)
    rows = np.zeros((2 * n, nv))
    for i in range(n):
        rows[i, i * n:(i + 1) * n] = 1
    for j in range(n):
        rows[n + j, j:n * n:n] = 1
    b = np.zeros(nv)
    b[:n * n] = mB.ravel()
    if not hard:
        b[n * n] = -1.0
    cons = [LinearConstraint(rows, 1, 1), LinearConstraint(b.reshape(1, -1), -np.inf, T)]
    c = np.zeros(nv)
    c[:n * n] = mA.ravel()
    if not hard:
        c[n * n] = lam
    integ = np.zeros(nv)
    integ[:n * n] = 1
    ub = np.ones(nv)
    if not hard:
        ub[n * n] = np.inf
    res = milp(c, constraints=cons, integrality=integ, bounds=Bounds(0, ub))
    if res.x is None:
        return None
    return np.round(res.x[:n * n]).reshape(n, n).argmax(axis=1)
