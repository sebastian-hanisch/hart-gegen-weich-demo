"""Szenario: n Personen, n Aufgaben, zwei Belastungen je Paar: A (Kosten, minimieren) und B (Zeit, die Summe darf die Schwelle nicht überschreiten). Alles simuliert.

Ganze Zahlen 1 bis K (K = 9, 99) oder reell gleichverteilt 0 bis 100. Die Erzeugung entspricht den Messreihen der Planung (gleiche Seeds, gleiche Reihenfolge der Ziehungen)."""

from dataclasses import dataclass

import numpy as np

import hgw_constants as C


@dataclass(frozen=True)
class Instance:
    mats: tuple               # zwei Matrizen (A, B) als Tupel von Tupeln (ganze Zahlen oder Gleitkommazahlen)
    exact: bool               # ganze Zahlen (exakter Vergleich)?

    @property
    def n(self) -> int:
        return len(self.mats[0])

    def arrays(self) -> list[np.ndarray]:
        return [np.array(m, dtype=np.int64 if self.exact else float) for m in self.mats]


def _tup(m) -> tuple:
    return tuple(tuple(x.item() for x in row) for row in m)


def make_instance(n: int, family: int, seed: int) -> Instance:
    """Deterministisch aus (n, Familie, Seed): gleiche Eingabe, gleiche Instanz."""
    rng = np.random.default_rng(seed)
    k = C.GRANULARITIES[family]
    if k == 0:
        return Instance(tuple(_tup(rng.uniform(0.0, 100.0, (n, n)).astype(float)) for _ in range(2)), False)
    return Instance(tuple(_tup(rng.integers(1, k + 1, (n, n)).astype(np.int64)) for _ in range(2)), True)
