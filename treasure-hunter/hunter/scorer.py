from typing import List
import numpy as np

from .config import load_weights


class RuleEngine:
    def __init__(self, weights: List[float] | None = None):
        self.weights = np.array(weights or load_weights())

    def score(self, text: str) -> float:
        """Return a dummy score using text length heuristics."""
        norm = np.linalg.norm(self.weights)
        base = min(len(text) / 1000, 1.0)
        return float(base * norm / 3)
