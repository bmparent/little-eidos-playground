import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import QuantumToy


def test_bell_state():
    q = QuantumToy(n=2)
    q.apply("H", 0)
    q.apply("CNOT", 0, 1)
    probs = np.abs(q.state) ** 2
    assert np.isclose(probs[0], 0.5, atol=1e-2)
    assert np.isclose(probs[3], 0.5, atol=1e-2)
