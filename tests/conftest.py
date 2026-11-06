"""Gemeinsame Fixtures: das Handbeispiel (n = 3), an dem jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hgw_algorithm as A  # noqa: E402
from hgw_scenario import Instance  # noqa: E402

HAND_A = ((4, 1, 5), (2, 4, 3), (1, 5, 3))
HAND_B = ((1, 5, 5), (1, 1, 1), (2, 3, 4))


def make_hand() -> Instance:
    return Instance((HAND_A, HAND_B), True)


@pytest.fixture
def hand() -> Instance:
    """Handbeispiel n = 3 (A Kosten, B Zeit; Zeile Person, Spalte Aufgabe). Sechs Pläne mit (A, B), in der Reihenfolge von itertools.permutations:
    (0,1,2): (11, 6); (0,2,1): (12, 5); (1,0,2): (6, 10); (1,2,0): (5, 8); (2,0,1): (12, 9); (2,1,0): (10, 8).
    Kostenoptimum (1,2,0); kleinste Zeitsumme 5; Schwelle bei 50 %: T = 5 + floor(0,5 * 3) = 6; hartes Optimum (0,1,2) mit A = 11."""
    return make_hand()


@pytest.fixture
def hand_arrays(hand) -> list[np.ndarray]:
    return hand.arrays()


@pytest.fixture
def hand_problem(hand_arrays) -> A.Problem:
    return A.make_problem(hand_arrays, 3, True)
