import yaml
from types import SimpleNamespace
import numpy as np
import executor


def test_execute_card(tmp_path, monkeypatch):
    todo = tmp_path / 'todo'
    exp = tmp_path / 'exp'
    todo.mkdir()
    card = {'kind': 'creator', 'name': 'dummy'}
    with (todo / 'task.yaml').open('w') as f:
        yaml.safe_dump(card, f)
    monkeypatch.setattr(executor, 'TODO_DIR', todo)
    monkeypatch.setattr(executor, 'EXPERIENCES_DIR', exp)
    monkeypatch.setattr(executor, 'global_pause', lambda: False)
    monkeypatch.setattr(executor, 'within_budget', lambda *a, **k: True)
    monkeypatch.setattr(executor, 'calc_surprise', lambda v: 0.0)
    monkeypatch.setattr(executor, 'embed', lambda text: np.zeros(1, dtype=np.float32))
    monkeypatch.setattr(executor, 'evaluate', lambda *a, **k: None)
    monkeypatch.setattr(executor, 'import_module', lambda name: SimpleNamespace(main=lambda: 'ok'))
    executor.main()
    assert not (todo / 'task.yaml').exists()
    assert any(exp.iterdir())
