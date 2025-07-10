"""Global constants for the quantum engine."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Constants:
    lP: float = 1.0
    alpha0: float = 1e-2
    beta0: float = 1e-2
    xi: float = 0.0


CONSTS = Constants()
