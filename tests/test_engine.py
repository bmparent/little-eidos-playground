import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine import QuantumEngine


def test_x_gate():
    q = QuantumEngine(1)
    q.apply("X", 0)
    assert np.allclose(q.state, np.array([0, 1]))


def test_cnot_entangle():
    q = QuantumEngine(2)
    q.apply("H", 0)
    q.apply("CNOT", 0, 1)
    probs = np.abs(q.state) ** 2
    assert np.isclose(probs[0], 0.5)
    assert np.isclose(probs[3], 0.5)
