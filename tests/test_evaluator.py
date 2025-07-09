from pathlib import Path
import json
import evaluator


def test_log_metrics(tmp_path, monkeypatch):
    path = tmp_path / 'metrics.jsonl'
    monkeypatch.setattr(evaluator, 'METRIC_PATH', path)
    evaluator.log_metrics({'a': 1})
    assert path.exists()
    data = json.loads(path.read_text().splitlines()[0])
    assert data['a'] == 1
