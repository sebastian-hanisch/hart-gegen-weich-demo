"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

N_MIN, N_MAX, DEFAULT_N = 3, 12, 8
ENUM_MAX = 8                                  # bis hierher werden alle n! Pläne aufgezählt (exakt in ganzen Zahlen)
SEED_MAX, DEFAULT_SEED = 999999, 53
GRANULARITIES = (9, 99, 0)                    # ganze Zahlen 1 bis K; 0: reell gleichverteilt 0 bis 100
FAMILIES = ("Ganze Zahlen 1 bis 9", "Ganze Zahlen 1 bis 99", "Reell gleichverteilt 0 bis 100")
DEFAULT_FAMILY = 0

# Schwelle T = Bmin + tau * (B(Kostenoptimum) - Bmin); tau in Prozent. Unter 0 liegt T unter der kleinsten Zeitsumme (in der Regel unzulässig).
TAU_MIN, TAU_MAX, TAU_STEP, DEFAULT_TAU = -50, 100, 5, 50

PENALTIES = ("linear", "quadratic", "weighted")
PENALTY_NAMES = {"linear": "linear: A + λ · v", "quadratic": "quadratisch: A + λ · v²", "weighted": "Gewichtssumme ohne Schwelle: A + λ · B"}
PENALTY_SHORT = {"linear": "lineare Strafe", "quadratic": "quadratische Strafe", "weighted": "Gewichtssumme"}
DEFAULT_PENALTY = 0
LAM_EXP_MIN, LAM_EXP_MAX, DEFAULT_LAM_EXP = -3, 60, 0      # lambda = 2 hoch Exponent (exakt in float64)

SWEEP_SEEDS = tuple(range(100000, 100020))    # feste Instanzen für alle Sweeps: nie ein einzelner Seed als Befund
SWEEP_N = 8                                   # Instanzgröße der Sweeps (alle aufgezählt)
SWEEP_TAUS = (25, 50, 75)                     # Schwelle in Prozent des Wegs vom Zeitminimum zum Kostenoptimum
SWEEP_LAMS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)    # kleine Gewichte (Potenzen von 2)
SWEEP_BIG_EXPS = (0, 2, 4, 6, 20, 50, 52, 53, 54, 56, 58, 60)
SOLVER_LAMS = (0.25, 1.0, 4.0, 16.0, 1e3, 1e6, 1e9, 1e11, 1e12, 1e14)
SOLVER_DELTAS = (1e-9, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-1)
SOLVER_TAU = 50

COLOR_ALL = "rgba(150,150,150,0.35)"
COLOR_FEAS = "#4c78a8"
COLOR_OK = "#2F6B65"
COLOR_BAD = "#c0504d"
COLOR_MARK = "#d8a13a"

PRESETS = {
    "Zu kleines Gewicht bricht die Schwelle": {"n": 8, "family": 0, "seed": 53, "tau": 50, "penalty": 0, "lam_exp": 0},
    "Ab λ* ist die Strafe exakt": {"n": 8, "family": 0, "seed": 53, "tau": 50, "penalty": 0, "lam_exp": 2},
    "Quadratisch ist früher exakt": {"n": 8, "family": 0, "seed": 53, "tau": 50, "penalty": 1, "lam_exp": 0},
    "Die Gewichtssumme erreicht es nicht": {"n": 8, "family": 0, "seed": 53, "tau": 50, "penalty": 2, "lam_exp": 1},
    "Kleinstes Beispiel (n = 3)": {"n": 3, "family": 0, "seed": 4, "tau": 50, "penalty": 0, "lam_exp": 0},
    "Unzulässige Schwelle: weich liefert einen Plan": {"n": 8, "family": 0, "seed": 4, "tau": -25, "penalty": 0, "lam_exp": 6},
    "Zu großes Gewicht in float64": {"n": 8, "family": 0, "seed": 4, "tau": -25, "penalty": 0, "lam_exp": 56},
    "HiGHS wird bei λ = 10¹² steif (reell)": {"n": 8, "family": 2, "seed": 0, "tau": 50, "penalty": 0, "lam_exp": 40},
}
PRESET_HELP = {
    "Zu kleines Gewicht bricht die Schwelle": "Bei λ = 1 wählt die Strafe einen Plan, der die Schwelle um 9 überschreitet und so die Kosten von 27 auf 12 drückt.",
    "Ab λ* ist die Strafe exakt": "Dieselbe Instanz mit λ = 4: ab λ* = 2,25 wählt die lineare Strafe das harte Optimum.",
    "Quadratisch ist früher exakt": "Die quadratische Strafe ist hier schon bei λ = 1 exakt (λ* = 0,75), die lineare erst ab 2,25: das hängt an der Einheit von B.",
    "Die Gewichtssumme erreicht es nicht": "A + λ·B ohne Schwelle liefert das harte Optimum für kein λ: bei λ = 2 ist der Plan zulässig, aber um 15 % teurer; bei λ = 1 bricht er die Schwelle.",
    "Kleinstes Beispiel (n = 3)": "Sechs Pläne: die lineare Strafe braucht λ ≥ 4, die quadratische λ ≥ 1,33, die Gewichtssumme erreicht das harte Optimum nie.",
    "Unzulässige Schwelle: weich liefert einen Plan": "Die Schwelle liegt unter der kleinsten Zeitsumme: hart ist unlösbar, die Strafe liefert den Plan mit der kleinsten Verletzung.",
    "Zu großes Gewicht in float64": "Dieselbe unzulässige Schwelle mit λ = 2⁵⁶: A geht in float64 verloren, die Strafe wählt einen Plan mit zu hohen Kosten.",
    "HiGHS wird bei λ = 10¹² steif (reell)": "Bei reellen Werten und λ ≈ 10¹² liefert HiGHS einen zulässigen Plan, der rund 87 % mehr kostet als das harte Optimum.",
}
