import json
from types import SimpleNamespace
import re
import responses

from agents import get

API = 'https://api.github.com'


def test_visionary_pr(tmp_path, monkeypatch):
    vis = get('visionary')
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'docs').mkdir()
    with open('memory.json', 'w') as f:
        json.dump({'last_alignment': 0.0}, f)
    monkeypatch.setenv('GH_BOT_TOKEN', 'x')

    monkeypatch.setattr(vis.ai_buddy, 'alignment_score', lambda weights=None: 1.0)
    monkeypatch.setattr(vis.ai_buddy, 'fetch_metrics', lambda: (1.0, 2, 3))

    repo = f"{vis.OWNER}/{vis.REPO}"
    with responses.RequestsMock() as rs:
        rs.add(responses.GET, f"{API}/repos/{repo}", json={'default_branch':'main'})
        rs.add(responses.GET, re.compile(f"{API}/repos/{repo}/git/ref/heads/main"), json={'object':{'sha':'abc'}})
        rs.add(responses.POST, f"{API}/repos/{repo}/git/refs", json={}, status=201)
        rs.add(responses.PUT, re.compile(f"{API}/repos/{repo}/contents/.*"), json={}, status=201)
        rs.add(responses.POST, f"{API}/repos/{repo}/pulls", json={}, status=201)
        rc = vis.main()
        called_urls = [c.request.url for c in rs.calls]

    assert rc == 0
    assert any('/git/refs' in u for u in called_urls)
    assert any('/pulls' in u for u in called_urls)
