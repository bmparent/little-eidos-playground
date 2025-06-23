import numpy as np
from types import SimpleNamespace

from sensors import gcp_rng
from parser import Parser
from repl import REPL
from engine import QuantumToy


def test_gcp_rotation(monkeypatch):
    def fake_get(url, timeout=60):
        return SimpleNamespace(text='2024 01 01 00 00 0 2.5')

    monkeypatch.setattr(gcp_rng.requests, 'get', fake_get)
    z = gcp_rng.latest_z()
    assert z == 2.5

    q = QuantumToy()
    repl = REPL(q)
    stmt = Parser('⚡').parse()
    repl.execute(stmt)
    expected = np.exp(-1j * np.pi / 8)
    assert np.allclose(q.state[0], expected)
