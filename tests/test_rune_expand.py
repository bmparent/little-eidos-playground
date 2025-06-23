import sys
from pathlib import Path as P

sys.path.insert(0, str(P(__file__).resolve().parents[1]))

from repl import expand_sequence


def test_expand_rune():
    ctx = {"sentiment": 0.7, "rng_z": 0.2}
    seq = expand_sequence("ᚦ", ctx)
    assert [c.gate for c in seq] == ["RX", "H", "RZ"]
    assert seq[0].theta == 0.7
    assert seq[1].theta is None
    assert seq[2].theta == 0.2
