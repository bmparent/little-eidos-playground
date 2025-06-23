import json
from types import SimpleNamespace

from agents import get


def test_visionary_skip(tmp_path, monkeypatch):
    vis = get('visionary')
    monkeypatch.chdir(tmp_path)
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / 'docs').mkdir()
    with open('memory.json', 'w') as f:
        json.dump({'last_alignment': 0.8}, f)
    monkeypatch.setenv('GH_BOT_TOKEN', 'x')

    monkeypatch.setattr(vis.ai_buddy, 'alignment_score', lambda weights=None: 0.9)
    monkeypatch.setattr(vis.ai_buddy, 'fetch_metrics', lambda: (1.0, 0, 0))

    called = []

    def fake_api_request(*a, **k):
        called.append(a[1])
        return SimpleNamespace(status_code=201, json=lambda: {})

    monkeypatch.setattr(vis, 'api_request', fake_api_request)
    rc = vis.main()
    assert rc == 0
    assert called == []
