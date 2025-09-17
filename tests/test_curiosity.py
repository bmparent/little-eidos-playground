import numpy as np
import pytest
import curiosity


def test_calc_surprise_updates(tmp_path, monkeypatch):
    store = tmp_path / 'store.npy'
    monkeypatch.setattr(curiosity, 'STORE', store)
    s1 = curiosity.calc_surprise(np.array([1.0, 0.0]))
    assert store.exists()
    s2 = curiosity.calc_surprise(np.array([0.0, 1.0]))
    assert s1 >= 0 and s2 >= 0


def test_bytes_error():
    with pytest.raises(ValueError):
        curiosity.calc_surprise(b"\x00\x01")


def test_basic_cosine():
    v = np.ones(2, dtype=np.float32)
    assert 0 <= curiosity.calc_surprise(v) <= 2.0
