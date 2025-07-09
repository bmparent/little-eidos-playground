import json
from pathlib import Path
import observer


def test_observe_and_plan(tmp_path, monkeypatch):
    todo = tmp_path / 'todo'
    exp_dir = tmp_path / 'experiences/2024-01-01'
    metrics_dir = tmp_path / 'metrics'
    todo.mkdir()
    exp_dir.mkdir(parents=True)
    metrics_dir.mkdir()

    card = todo / 'card.yaml'
    card.write_text('kind: note\nname: test')
    (metrics_dir / 'actions.jsonl').write_text('{}\n')

    monkeypatch.setattr(observer, 'TODO_DIR', todo)
    monkeypatch.setattr(observer, 'EXPERIENCES_DIR', tmp_path / 'experiences')
    monkeypatch.setattr(observer, 'METRICS_FILE', metrics_dir / 'actions.jsonl')
    monkeypatch.setattr(observer, 'OBSERVER_LOG', metrics_dir / 'observer.jsonl')
    monkeypatch.setattr(observer, 'shadow', type('s', (), {'chat_with_shadow': lambda m: {'type': 'card', 'card': 'kind: note\nname: new'}}))

    result = observer.observe_and_plan()
    assert (metrics_dir / 'observer.jsonl').exists()
    assert result['type'] == 'card'
