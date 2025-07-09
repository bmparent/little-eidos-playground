import yaml
from agents import planner


def test_plan_creates_file(tmp_path, monkeypatch):
    monkeypatch.setattr(planner, 'TODO_DIR', tmp_path)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    path = planner.plan()
    assert path.exists()
    with path.open() as f:
        data = yaml.safe_load(f)
    assert 'kind' in data
