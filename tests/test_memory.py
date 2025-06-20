import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import memory


@pytest.mark.fast
def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    path = tmp_path / ".eidos_memory.json"
    monkeypatch.setattr(memory, "MEM_PATH", path)

    ck = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
    mem = {
        "timestamp": "2025-01-01T00:00:00",
        "ck": ck,
        "sigma2": 0.5,
        "state_amplitudes": ck.copy(),
    }

    memory.save_memory(mem)
    loaded = memory.load_memory()

    assert np.allclose(loaded["ck"], ck)
    assert np.isclose(loaded["sigma2"], 0.5)
    assert np.allclose(loaded["state_amplitudes"], ck)
