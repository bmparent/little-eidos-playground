import json
from types import SimpleNamespace

import numpy as np
import pytest

from agents import get


class DummyStudy:
    def __init__(self):
        self.best_params = {"lam": 0.9, "p": 1, "q": 1, "garch": False}
        self.best_value = 1.0

    def optimize(self, func, n_trials=1, timeout=None):
        func(SimpleNamespace(suggest_float=lambda *a, **k: 0.9,
                             suggest_int=lambda *a, **k: 1,
                             suggest_categorical=lambda *a, **k: False))


def test_tuner_creates_json(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mem = {"prices": list(np.arange(40, dtype=float))}
    with open('memory.json', 'w') as f:
        json.dump(mem, f)

    tuner = get('tuner')
    monkeypatch.setattr(tuner.optuna, 'create_study', lambda direction: DummyStudy())
    tuner.main()
    assert (tmp_path / 'tuning.json').exists()
