"""Kennzahlen, Verdikt-Logik und Sweeps. Verdikte sind reine Funktionen (Art, Text); jede Zahl im Text kommt aus der Rechnung.

Hart heißt: kleinstes A unter B <= T. Weich heißt: Strafe auf die Verletzung v = max(0, B - T). Gefragt wird, was die Schwelle kostet, ab welchem Gewicht die Strafe
das harte Optimum wählt (lam*), ob eine Gewichtssumme ohne Schwelle es überhaupt erreicht, was bei unzulässiger Schwelle geschieht und wo ein Löser „hart“ und „steif“ auslegt."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

import hgw_algorithm as A
import hgw_constants as C
from hgw_scenario import Instance, make_instance


def de(x: float, digits: int = 1) -> str:
    """Zahl mit Dezimalkomma; kein „-0,0“."""
    text = f"{x:.{digits}f}"
    if float(text) == 0.0:
        text = f"{0.0:.{digits}f}"
    return text.replace(".", ",")


def pct(x: float, digits: int = 1) -> str:
    """Anteil als Prozent: 0,05 -> „5,0 %“."""
    return de(x * 100.0, digits) + " %"


def value_text(v) -> str:
    """Kennzahl: ganze Zahlen ohne Nachkommastelle, sonst mit einer."""
    return str(int(v)) if float(v).is_integer() else de(float(v), 1)


def lam_text(lam: float) -> str:
    """Gewicht lesbar: kleine Werte als Dezimalzahl, große als Potenz von 2 (Exponent), nicht als Zahl mit 17 Stellen."""
    e = int(np.log2(lam)) if lam > 0 and float(np.log2(lam)).is_integer() else None
    if e is not None and e >= 20:
        return f"2^{e}"
    if lam >= 1e6:
        k = int(round(np.log10(lam)))
        return f"10^{k}" if abs(lam / 10.0 ** k - 1.0) < 1e-9 else f"{lam:.3g}"
    return value_text(lam) if float(lam).is_integer() else de(lam, 3 if lam < 0.1 else 2)


def delta_text(d: float) -> str:
    """Abstand als Potenz von 10: 1e-05 -> „10^-5“."""
    k = int(round(np.log10(d)))
    return f"10^{k}" if abs(d / 10.0 ** k - 1.0) < 1e-9 else f"{d:g}"


def plan_text(plan, n: int) -> str:
    return ", ".join(f"P{i + 1}→A{int(j) + 1}" for i, j in enumerate(plan)) if n <= 12 else "siehe Rechnung"


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    family: int = C.DEFAULT_FAMILY
    seed: int = C.DEFAULT_SEED
    tau: int = C.DEFAULT_TAU
    penalty: int = C.DEFAULT_PENALTY
    lam_exp: int = C.DEFAULT_LAM_EXP

    @property
    def form(self) -> str:
        return C.PENALTIES[self.penalty]

    @property
    def lam(self) -> float:
        return 2.0 ** self.lam_exp


@dataclass
class Analysis:
    inst: Instance
    mode: str                  # "enum" (alle Pläne aufgezählt) oder "large" (nur Löser)
    mats: list                 # die beiden Matrizen als Arrays
    prob: A.Problem | None     # Kennzahlen aller Pläne (nur bei Aufzählung)
    T: float                   # Schwelle
    bmin: float                # kleinste Zeitsumme
    bA: float                  # Zeitsumme des Kostenoptimums
    a_opt: float               # Kosten des Kostenoptimums


def _lap_plan(m: np.ndarray) -> np.ndarray:
    rows, cols = linear_sum_assignment(m.astype(float))
    plan = np.empty(m.shape[0], dtype=int)
    plan[rows] = cols
    return plan


def analyse(settings: Settings) -> Analysis:
    inst = make_instance(settings.n, settings.family, settings.seed)
    mats = inst.arrays()
    if inst.n <= C.ENUM_MAX:
        p = A.make_problem(mats, inst.n, inst.exact)
        return Analysis(inst, "enum", mats, p, A.threshold(p, settings.tau), p.bmin, p.bA, p.sA.min().item())
    # große Instanzen: Zeitminimum und Kostenoptimum über die Ungarische Methode, die Schwelle wie bei der Aufzählung
    bmin = A.plan_value(mats[1], _lap_plan(mats[1]))
    plan_a = _lap_plan(mats[0])
    a_opt, b_a = A.plan_value(mats[0], plan_a), A.plan_value(mats[1], plan_a)
    t = b_a if settings.tau == 100 else bmin + (settings.tau / 100.0) * (b_a - bmin)
    return Analysis(inst, "large", mats, None, float(np.floor(t)) if inst.exact else float(t), bmin, b_a, a_opt)


# --- Experimente auf einer Instanz ---------------------------------------------------------------------------------------------------------

def threshold_summary(a: Analysis) -> dict:
    """Die Schwelle: zulässig? bindet sie? hartes Optimum (Zielplan) und sein Anstieg gegen das Kostenoptimum."""
    p = a.prob
    feasible = A.feasible(p, a.T)
    t = A.target(p, a.T)
    v = A.viol(p, a.T)
    a_opt = a.a_opt
    return {"T": a.T, "bmin": a.bmin, "bA": a.bA, "a_opt": a_opt, "feasible": feasible, "target": t, "a_t": p.sA[t].item(), "b_t": p.sB[t].item(), "v_t": v[t].item(),
            "binds": feasible and p.sA[t] > a_opt, "rise": (p.sA[t].item() - a_opt) / a_opt if a_opt else 0.0, "total": len(p.sA), "n_feasible": int((p.sB <= a.T).sum())}


def penalty_result(a: Analysis, form: str, lam: float) -> dict:
    """Der Plan, den die Strafe bei diesem Gewicht wählt, und sein Vergleich mit dem Zielplan."""
    p = a.prob
    j = A.chosen(p, a.T, form, lam)
    t = A.target(p, a.T)
    v = A.viol(p, a.T)
    a_t = p.sA[t].item()
    return {"form": form, "lam": lam, "idx": j, "a": p.sA[j].item(), "b": p.sB[j].item(), "v": v[j].item(), "hit": A.hit(p, a.T, j), "target": t, "a_t": a_t, "b_t": p.sB[t].item(),
            "v_t": v[t].item(), "gain": (a_t - p.sA[j].item()) / a_t if a_t else 0.0, "plan": A.all_plans(a.inst.n)[j].astype(int)}


def lam_summary(a: Analysis) -> dict:
    """lam* der linearen und der quadratischen Strafe, Fenster der Gewichtssumme, Schranke bei ganzen Zahlen."""
    p = a.prob
    lo, hi = A.weighted_window(p, a.T)
    return {"lin": A.lam_star(p, a.T, 1), "quad": A.lam_star(p, a.T, 2), "lo": lo, "hi": hi, "reachable": lo < hi, "bound": A.int_bound(p, a.T), "feasible": A.feasible(p, a.T)}


LAM_CURVE_EXPS = tuple(range(-3, 13)) + (16, 20, 30, 40, 50, 53, 56, 60)


def lam_curve(a: Analysis, form: str, exps=LAM_CURVE_EXPS) -> list[dict]:
    """Der gewählte Plan je Gewicht 2^e: Kosten, Verletzung, Treffer."""
    rows = []
    for e in exps:
        r = penalty_result(a, form, 2.0 ** e)
        rows.append({"e": e, "lam": r["lam"], "a": r["a"], "v": r["v"], "hit": r["hit"]})
    return rows


def _highs_row(a: Analysis, label: str, lam: float | None, hard: bool, T: float | None = None, ref_a: float | None = None) -> dict:
    T = a.T if T is None else T
    plan = A.highs_plan(a.mats, T, lam, hard=hard)
    if plan is None:
        return {"label": label, "lam": lam, "plan": None, "a": None, "b": None, "v": None, "hit": False, "loss": None}
    va, vb = A.plan_value(a.mats[0], plan), A.plan_value(a.mats[1], plan)
    v = max(0.0, vb - T)
    ok = ref_a is not None and abs(va - ref_a) < 1e-7 and v <= 1e-9
    return {"label": label, "lam": lam, "plan": plan, "a": va, "b": vb, "v": v, "hit": ok, "loss": None if ref_a is None or not ref_a else (va - ref_a) / ref_a}


def highs_rows(a: Analysis, lam_now: float) -> list[dict]:
    """HiGHS auf der Instanz: harte Nebenbedingung und Strafe mit Schlupfvariable bei einem Gewichtsgitter (und beim eingestellten Gewicht)."""
    hard = _highs_row(a, "hart (B ≤ T)", None, True)
    ref = hard["a"]
    rows = [hard]
    lams = sorted(set(C.SOLVER_LAMS) | {lam_now})
    for lam in lams:
        rows.append(_highs_row(a, f"Strafe, λ = {lam_text(lam)}", lam, False, ref_a=ref))
    return rows


def tolerance_rows(a: Analysis) -> list[dict]:
    """Die Schwelle liegt delta unter der Zeitsumme des Kostenoptimums: liefert HiGHS (hart und mit Strafe, lam = 10^6) das nur um delta verletzende Kostenoptimum?"""
    rows = []
    for d in C.SOLVER_DELTAS:
        T = a.bA - d
        hard = _highs_row(a, "hart", None, True, T=T, ref_a=a.a_opt)
        soft = _highs_row(a, "weich", 1e6, False, T=T, ref_a=a.a_opt)
        rows.append({"delta": d, "hard_violates": hard["plan"] is not None and hard["b"] > T and abs(hard["a"] - a.a_opt) < 1e-9,
                     "soft_violates": soft["plan"] is not None and soft["b"] > T and abs(soft["a"] - a.a_opt) < 1e-9})
    return rows


# --- Sweeps über feste Instanzen -----------------------------------------------------------------------------------------------------------

def _problems(fam: int, n: int = C.SWEEP_N, seeds=C.SWEEP_SEEDS):
    out = []
    for s in seeds:
        inst = make_instance(n, fam, s)
        out.append((inst, inst.arrays(), A.make_problem(inst.arrays(), n, inst.exact)))
    return out


def _q3(vals) -> tuple[float, float, float]:
    v = np.asarray(vals, dtype=float)
    return float(v.min()), float(np.median(v)), float(v.max())


def shift_sweep(fams=(0, 1, 2), taus=C.SWEEP_TAUS) -> dict:
    """Je Familie und Schwelle: Anstieg von A des harten Optimums (Min/Median/Max), Instanzen ohne Wirkung, lam* linear und quadratisch, Gewichtssumme erreicht."""
    out = {}
    for fam in fams:
        probs = _problems(fam)
        for tau in taus:
            rises, lin, quad, none, reach, weak = [], [], [], 0, 0, 0
            for _, _, p in probs:
                T = A.threshold(p, tau)
                t = A.target(p, T)
                a_opt = p.sA.min().item()
                rises.append((p.sA[t].item() - a_opt) / a_opt)
                none += p.sA[t] == a_opt
                lin.append(A.lam_star(p, T, 1))
                quad.append(A.lam_star(p, T, 2))
                lo, hi = A.weighted_window(p, T)
                reach += lo < hi
                weak += lo <= hi
            out[(fam, tau)] = {"rise": _q3(rises), "none": int(none), "lin": _q3(lin), "quad": _q3(quad), "reach": int(reach), "weak": int(weak), "seeds": len(probs)}
    return out


def small_lam_sweep(fams=(0, 1, 2), tau: int = 50, lams=C.SWEEP_LAMS) -> dict:
    """Zu kleines Gewicht (lineare und quadratische Strafe): Instanzen mit verletzter Schwelle, Verletzung (Median über die verletzten), A-Gewinn gegen das harte Optimum (Median)."""
    out = {}
    for fam in fams:
        probs = _problems(fam)
        for form in ("linear", "quadratic"):
            for lam in lams:
                nv, vs, gains = 0, [], []
                for _, _, p in probs:
                    T = A.threshold(p, tau)
                    j = A.chosen(p, T, form, lam)
                    v = A.viol(p, T)
                    if v[j] > 0:
                        nv += 1
                        vs.append(v[j])
                        a_t = p.sA[A.target(p, T)]
                        gains.append((a_t - p.sA[j]) / a_t)
                out[(form, fam, lam)] = {"violated": nv, "v_med": float(np.median(vs)) if vs else None, "v_max": float(max(vs)) if vs else None,
                                         "gain": float(np.median(gains)) if gains else None, "seeds": len(probs)}
    return out


def big_lam_sweep(fams=(0, 1, 2), exps=C.SWEEP_BIG_EXPS, tau: int = 50) -> dict:
    """Zu großes Gewicht in float64 (lineare Strafe, lam = 2^e): Treffer bei zulässiger Schwelle und bei unzulässiger Schwelle; Instanzen mit mehreren Plänen bei kleinster Verletzung."""
    out = {}
    for fam in fams:
        probs = _problems(fam)
        feas, infeas = {e: 0 for e in exps}, {e: 0 for e in exps}
        ties = 0
        for _, _, p in probs:
            T1, T2 = A.threshold(p, tau), A.infeasible_threshold(p)
            if p.exact:
                ties += int(np.sum(A.viol(p, T2) == A.viol(p, T2).min()) > 1)
            for e in exps:
                lam = 2.0 ** e
                feas[e] += A.hit(p, T1, A.chosen(p, T1, "linear", lam))
                infeas[e] += A.hit(p, T2, A.chosen(p, T2, "linear", lam))
        out[fam] = {"feasible": {e: int(v) for e, v in feas.items()}, "infeasible": {e: int(v) for e, v in infeas.items()}, "ties": ties, "seeds": len(probs)}
    return out


def infeasible_sweep(fams=(0, 1, 2), exps=(0, 2, 4, 6, 20)) -> dict:
    """Unzulässige Schwelle (eins unter der kleinsten Zeitsumme, reell 95 %): Treffer „kleinste Verletzung, dann kleinste Kosten“ je Gewicht 2^e."""
    out = {}
    for fam in fams:
        hits = {e: 0 for e in exps}
        probs = _problems(fam)
        for _, _, p in probs:
            T = A.infeasible_threshold(p)
            for e in exps:
                hits[e] += A.hit(p, T, A.chosen(p, T, "linear", 2.0 ** e))
        out[fam] = {"hits": {e: int(v) for e, v in hits.items()}, "seeds": len(probs)}
    return out


def highs_sweep(fams=(0, 1, 2), lams=C.SOLVER_LAMS, tau: int = C.SOLVER_TAU) -> dict:
    """HiGHS mit Schlupfvariable: je Gewicht Treffer (zulässig und A gleich dem harten Optimum), verletzte Instanzen, größter Mehrpreis in A (nur zulässige Ergebnisse)."""
    out = {}
    for fam in fams:
        probs = _problems(fam)
        for lam in lams:
            hit, nv, loss = 0, 0, 0.0
            for _, mats, p in probs:
                T = A.threshold(p, tau)
                a_t = p.sA[A.target(p, T)].item()
                plan = A.highs_plan(mats, T, lam)
                va, vb = A.plan_value(mats[0], plan), A.plan_value(mats[1], plan)
                ok = vb - T <= 1e-9
                hit += ok and abs(va - a_t) < 1e-7
                nv += not ok
                if ok:
                    loss = max(loss, (va - a_t) / a_t)
            out[(fam, lam)] = {"hits": int(hit), "violated": int(nv), "loss": loss, "seeds": len(probs)}
    return out


def tolerance_sweep(fam: int = 2, deltas=C.SOLVER_DELTAS) -> dict:
    """Reelle Werte: die Schwelle liegt delta unter der Zeitsumme des Kostenoptimums; in wie vielen Instanzen liefert HiGHS (hart beziehungsweise Strafe lam = 10^6) das nur um delta verletzende Kostenoptimum?"""
    out = {d: {"hard": 0, "soft": 0} for d in deltas}
    probs = _problems(fam)
    for _, mats, p in probs:
        a_opt, b_a = p.sA.min().item(), p.bA
        for d in deltas:
            T = b_a - d
            ph, ps = A.highs_plan(mats, T, hard=True), A.highs_plan(mats, T, 1e6)
            out[d]["hard"] += ph is not None and A.plan_value(mats[1], ph) > T and abs(A.plan_value(mats[0], ph) - a_opt) < 1e-9
            out[d]["soft"] += ps is not None and A.plan_value(mats[1], ps) > T and abs(A.plan_value(mats[0], ps) - a_opt) < 1e-9
    return {d: {"hard": int(v["hard"]), "soft": int(v["soft"]), "seeds": len(probs)} for d, v in out.items()}


# --- Verdikte ------------------------------------------------------------------------------------------------------------------------------

def verdict_threshold(ts: dict) -> tuple[str, str]:
    """Experiment Schwelle: bindet sie, und was kostet sie?"""
    if not ts["feasible"]:
        return "warning", (f"Die Schwelle T = {value_text(ts['T'])} liegt **unter der kleinsten Zeitsumme** ({value_text(ts['bmin'])}): kein Plan ist zulässig, das harte Problem ist unlösbar. "
                           f"Eine weiche Formulierung liefert trotzdem einen Plan: den mit der kleinsten Verletzung ({value_text(ts['v_t'])}).")
    if not ts["binds"]:
        return "info", (f"Die Schwelle bindet hier nicht: das Kostenoptimum (A = {value_text(ts['a_opt'])}) hat B = {value_text(ts['bA'])} und erfüllt sie schon (T = {value_text(ts['T'])}).")
    return "info", (f"Die Schwelle T = {value_text(ts['T'])} bindet: das harte Optimum kostet **A = {value_text(ts['a_t'])}** statt {value_text(ts['a_opt'])} ({pct(ts['rise'])} mehr), "
                    f"B = {value_text(ts['b_t'])}. Zulässig sind {ts['n_feasible']} von {ts['total']} Plänen.")


def verdict_penalty(r: dict, ts: dict, ls: dict) -> tuple[str, str]:
    """Experiment Strafgewicht: was wählt die Strafe bei diesem λ?"""
    name = C.PENALTY_SHORT[r["form"]]
    lam = lam_text(r["lam"])
    if not ts["feasible"]:
        if r["hit"]:
            return "success", (f"Die Schwelle ist unzulässig; die {name} wählt bei λ = {lam} den Plan mit der kleinsten Verletzung ({value_text(r['v'])}) und den kleinsten Kosten dabei (A = {value_text(r['a'])}). "
                               f"Hart gäbe es nichts.")
        why = ("A geht in float64 verloren: bei diesem Gewicht ist λ · v so groß, dass die Kosten A die Summe nicht mehr verändern, und unter den Plänen mit gleicher Strafe entscheidet die Reihenfolge"
               if r["form"] == "linear" and r["lam"] >= ls["lin"] else "das Gewicht ist zu klein: ein Plan mit größerer Verletzung, aber kleineren Kosten gewinnt")
        vtxt = f"Verletzung {value_text(r['v'])} statt {value_text(ts['v_t'])}, " if r["v"] != ts["v_t"] else f"gleiche Verletzung ({value_text(r['v'])}), aber "
        return "warning", (f"Die Schwelle ist unzulässig; die {name} wählt bei λ = {lam} **nicht** den besten Plan (kleinste Verletzung, dann kleinste Kosten): {vtxt}"
                           f"A = {value_text(r['a'])} statt {value_text(ts['a_t'])}. Grund: {why}.")
    if r["hit"]:
        return "success", (f"Bei λ = {lam} wählt die {name} das harte Optimum (A = {value_text(r['a_t'])}, B = {value_text(r['b_t'])}): die Schwelle ist eingehalten.")
    if r["v"] > 0:
        extra = ""
        if r["form"] == "weighted":
            extra = (" Die Gewichtssumme sieht die Schwelle nicht: " + ("kein λ trifft das harte Optimum." if not ls["reachable"] else ("das harte Optimum ist nur für λ ≥ " + lam_text(ls["lo"]) + ("" if not np.isfinite(ls["hi"]) else " bis " + lam_text(ls["hi"])) + " ein Minimierer.")))
        else:
            need = ls["lin"] if r["form"] == "linear" else ls["quad"]
            extra = f" Ab λ = {lam_text(need)} würde sie das harte Optimum wählen."
        return "warning", (f"Bei λ = {lam} wählt die {name} einen Plan, der die Schwelle um **{value_text(r['v'])}** überschreitet (B = {value_text(r['b'])} statt höchstens {value_text(ts['T'])}); "
                           f"er kostet A = {value_text(r['a'])} statt {value_text(r['a_t'])}, also **{pct(r['gain'])} weniger, weil er die Schwelle bricht**.{extra}")
    return "info", (f"Bei λ = {lam} wählt die {name} einen zulässigen Plan, aber nicht das harte Optimum: A = {value_text(r['a'])} statt {value_text(r['a_t'])} "
                    f"({pct((r['a'] - r['a_t']) / r['a_t'])} mehr)." + (" Die Gewichtssumme sieht die Schwelle nicht: ein zulässiger Plan ist Zufall" + (" und kein λ trifft das harte Optimum." if not ls["reachable"] else ".")))


def verdict_lambda(ls: dict, ts: dict) -> tuple[str, str]:
    """Experiment λ*: ab welchem Gewicht ist die Strafe exakt, und erreicht die Gewichtssumme es?"""
    tail = ("Die Gewichtssumme A + λ·B trifft ihn für **kein λ**: er liegt nicht auf der unteren Hülle der Punkte (A, B)." if not ls["reachable"]
            else (f"Die Gewichtssumme A + λ·B trifft ihn nur für λ ≥ {lam_text(ls['lo'])}." if not np.isfinite(ls["hi"])
                  else f"Die Gewichtssumme A + λ·B trifft ihn nur für λ zwischen {lam_text(ls['lo'])} und {lam_text(ls['hi'])}."))
    what = "den Plan mit der kleinsten Verletzung" if not ts["feasible"] else "das harte Optimum"
    if ls["lin"] == 0 and ls["quad"] == 0:
        return "info", f"Schon bei λ = 0 gilt: der Kostenoptimum-Plan ist {what}. {tail}"
    bound = f" Bei ganzen Zahlen gilt v ≥ 1 und damit die Schranke λ ≥ {value_text(ls['bound'])}." if ls["bound"] is not None else ""
    return "info", (f"Die lineare Strafe wählt {what} ab **λ = {lam_text(ls['lin'])}**, die quadratische ab **λ = {lam_text(ls['quad'])}** (bei Gleichstand gewinnt der Plan mit der kleineren Verletzung).{bound} {tail}")


def verdict_curve(rows: list[dict], form: str) -> tuple[str, str]:
    """Kurve des gewählten Plans über λ: ab wann bleibt der Treffer?"""
    first = next((r["e"] for r in rows if r["hit"]), None)
    last_bad = max((r["e"] for r in rows if not r["hit"]), default=None)
    name = C.PENALTY_SHORT[form]
    if first is None:
        return "warning", f"Für kein Gewicht der Kurve (2^-3 bis 2^60) trifft die {name} den Zielplan."
    if last_bad is None:
        return "info", f"Die {name} trifft den Zielplan bei jedem Gewicht der Kurve (2^-3 bis 2^60)."
    if last_bad > first:
        return "warning", (f"Die {name} trifft den Zielplan ab λ = 2^{first}, aber nicht bei jedem größeren Gewicht: zuletzt verfehlt bei λ = 2^{last_bad}. Die Treffer bilden ein Fenster, kein „ab hier immer“.")
    return "info", f"Die {name} trifft den Zielplan ab λ = 2^{first} und bei jedem größeren Gewicht der Kurve (bis 2^60)."


def verdict_highs(rows: list[dict]) -> tuple[str, str]:
    """Experiment Löser: wo wird die Strafe in HiGHS steif oder die harte Grenze weich?"""
    hard = rows[0]
    if hard["plan"] is None:
        return "info", "Mit harter Nebenbedingung findet HiGHS **keinen** Plan (unzulässig); die Strafe liefert immer einen."
    bad = [r for r in rows[1:] if r["plan"] is not None and r["v"] <= 1e-9 and not r["hit"]]
    viol = [r for r in rows[1:] if r["plan"] is not None and r["v"] > 1e-9]
    if bad:
        worst = max(bad, key=lambda r: r["loss"])
        return "warning", (f"HiGHS liefert bei {len(bad)} Gewichten einen zulässigen Plan, der **nicht** das harte Optimum ist: am schlechtesten bei {worst['label'].split(', ')[1]} mit {pct(worst['loss'])} höheren Kosten. "
                           f"Die Strafe wird numerisch steif, ohne dass jemand eine Warnung sieht.")
    if viol:
        return "info", f"Bei {len(viol)} der {len(rows) - 1} Gewichte ist die Strafe zu klein: HiGHS wählt einen Plan, der die Schwelle verletzt (zuletzt bei {viol[-1]['label'].split(', ')[1]}); darüber trifft es das harte Optimum."
    return "info", f"HiGHS trifft das harte Optimum bei allen {len(rows) - 1} Gewichten des Gitters."


def verdict_tolerance(rows: list[dict]) -> tuple[str, str]:
    """Experiment Toleranz: hält die harte Nebenbedingung?"""
    viol = [r for r in rows if r["hard_violates"]]
    if not viol:
        return "info", "HiGHS hält die harte Nebenbedingung für jede der geprüften Schwellen ein (kein Plan, der sie nur um δ verletzt)."
    d = max(r["delta"] for r in viol)
    return "warning", (f"„Hart“ ist im Löser eine Toleranz: Liegt die Schwelle nur δ = {delta_text(d)} unter der Zeitsumme des Kostenoptimums, liefert HiGHS mit harter Nebenbedingung trotzdem das Kostenoptimum, "
                       f"das sie um δ verletzt. Bei den größeren δ der Tabelle nicht.")


def _fam(fam: int) -> str:
    return "K = " + str(C.GRANULARITIES[fam]) if C.GRANULARITIES[fam] else "reell"


def verdict_shift_sweep(res: dict) -> tuple[str, str]:
    parts = []
    for (fam, tau), r in res.items():
        lo, med, hi = r["rise"]
        parts.append(f"{_fam(fam)}, Schwelle {tau} %: A steigt um Median {pct(med)} (bis {pct(hi)}), λ* linear Median {de(r['lin'][1], 2)} (Maximum {de(r['lin'][2], 2)}), quadratisch Median {de(r['quad'][1], 2)} "
                     f"(Maximum {de(r['quad'][2], 2)}), Gewichtssumme erreicht in {r['reach']} von {r['seeds']}")
    return "info", f"Schwelle, λ* und Gewichtssumme, n = {C.SWEEP_N}, je {len(C.SWEEP_SEEDS)} feste Instanzen: " + "; ".join(parts) + "."


def verdict_small_lam_sweep(res: dict) -> tuple[str, str]:
    parts = []
    for (form, fam, lam), r in res.items():
        if form != "linear":
            continue
        parts.append(f"{_fam(fam)}, λ = {lam_text(lam)}: {r['violated']} von {r['seeds']}")
    q = [f"{_fam(fam)}, λ = {lam_text(lam)}: {r['violated']}" for (form, fam, lam), r in res.items() if form == "quadratic" and lam in (0.25, 1.0, 4.0, 16.0)]
    return "info", ("Instanzen mit verletzter Schwelle, lineare Strafe, Schwelle 50 %: " + "; ".join(parts) + ". Quadratische Strafe: " + "; ".join(q) + ".")


def verdict_big_lam_sweep(res: dict) -> tuple[str, str]:
    parts = []
    for fam, r in res.items():
        inf = ", ".join(f"2^{e}: {v}" for e, v in r["infeasible"].items() if e in (4, 52, 53, 56, 58, 60))
        fe = ", ".join(f"2^{e}: {v}" for e, v in r["feasible"].items() if e in (4, 53, 60))
        parts.append(f"{_fam(fam)}: unzulässige Schwelle {inf} (Instanzen mit mehreren Plänen bei kleinster Verletzung: {r['ties']}); zulässige Schwelle {fe}")
    return "info", f"Treffer der linearen Strafe in float64 je Gewicht, von {len(C.SWEEP_SEEDS)}: " + "; ".join(parts) + "."


def verdict_highs_sweep(res: dict) -> tuple[str, str]:
    parts = []
    for fam in (0, 1, 2):
        cells = ", ".join(f"λ = {lam_text(lam)}: {r['hits']} ({r['violated']})" for (f, lam), r in res.items() if f == fam)
        loss = max(r["loss"] for (f, _), r in res.items() if f == fam)
        parts.append(f"{_fam(fam)}: {cells}; größter Mehrpreis der zulässigen Ergebnisse {pct(loss)}")
    return "info", f"HiGHS mit Schlupfvariable, Treffer (verletzt) von {len(C.SWEEP_SEEDS)}: " + "; ".join(parts) + "."


def verdict_tolerance_sweep(res: dict) -> tuple[str, str]:
    cells = ", ".join(f"δ = {delta_text(d)}: {r['hard']} (Strafe {r['soft']})" for d, r in res.items())
    return "info", f"Reelle Werte, Schwelle δ unter der Zeitsumme des Kostenoptimums: Instanzen von {next(iter(res.values()))['seeds']}, in denen HiGHS das nur um δ verletzende Kostenoptimum liefert (hart; Strafe mit λ = 10^6): {cells}."
