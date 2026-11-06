"""PDF-Export des aktuellen Ergebnisses (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Sonderzeichen werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

import hgw_constants as C
from hgw_evaluation import Analysis, Settings, lam_summary, lam_text, penalty_result, threshold_summary, value_text

_REPLACE = {"→": "->", "–": "-", "—": "-", "−": "-", "≤": "<=", "≥": ">=", "·": "*", "×": "x", "Δ": "Delta", "δ": "delta", "λ": "lambda", "↔": "<->", "„": '"', "“": '"', "²": "2", "∞": "unendlich", "⁵": "5", "⁶": "6"}


def _s(text) -> str:
    out = str(text).replace("**", "").replace("`", "")
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(s: Settings, a: Analysis) -> bytes:
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Hart gegen weich"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(f"{s.n} Personen, {s.n} Aufgaben, Familie: {C.FAMILIES[s.family]}, Seed {s.seed}, Schwelle T = {value_text(a.T)} (Anteil {s.tau} %). Alles simuliert."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    if a.mode != "enum":
        pdf.multi_cell(0, 5, _s(f"Bei n = {s.n} werden nicht alle Pläne aufgezählt (nur bis n = {C.ENUM_MAX}); der Löservergleich steht in der App."), new_x="LMARGIN", new_y="NEXT")
        return bytes(pdf.output())
    ts, ls = threshold_summary(a), lam_summary(a)
    r = penalty_result(a, s.form, s.lam)
    if ts["feasible"]:
        pdf.multi_cell(0, 5, _s(f"Hartes Optimum: A = {value_text(ts['a_t'])}, B = {value_text(ts['b_t'])}; Kostenoptimum A = {value_text(ts['a_opt'])}."), new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.multi_cell(0, 5, _s(f"Die Schwelle ist unzulässig (kleinste Zeitsumme {value_text(ts['bmin'])}); kleinste Verletzung {value_text(ts['v_t'])}."), new_x="LMARGIN", new_y="NEXT")
    pdf.multi_cell(0, 5, _s(f"lambda* der linearen Strafe: {lam_text(ls['lin'])}; der quadratischen: {lam_text(ls['quad'])}; Gewichtssumme erreicht den Zielplan: {'ja' if ls['reachable'] else 'nein'}."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    widths = (60, 30, 30, 30, 30)
    pdf.set_font("Helvetica", "B", 9)
    for w, h in zip(widths, ("Strafe", "lambda", "A", "B", "Verletzung")):
        pdf.cell(w, 6, _s(h), border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for w, v in zip(widths, (C.PENALTY_SHORT[r["form"]], lam_text(r["lam"]), value_text(r["a"]), value_text(r["b"]), value_text(r["v"]))):
        pdf.cell(w, 5.5, _s(v), border=1)
    pdf.ln(9)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Minimierer bei Gleichstand: kleinste Verletzung, dann der erste Plan der Aufzählung. Quelle: hart-gegen-weich-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
