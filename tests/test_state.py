import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import state


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    monkeypatch.setattr(state, "STATE_PATH", path)

    ck = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
    data = state.DEFAULT_STATE.copy()
    data.update({"ck": ck, "sigma2": 0.5, "qubit_state": ck.copy()})

    state.save_state(data)
    loaded = state.load_state()

    assert np.allclose(loaded["ck"], ck)
    assert np.isclose(loaded["sigma2"], 0.5)
    assert np.allclose(loaded["qubit_state"], ck)
