import types
from datetime import datetime, timezone

from sensors import ai_buddy


def test_alignment_score(monkeypatch):
    now = datetime(2024, 1, 2, tzinfo=timezone.utc)

    def fake_api_request(method, url, **kwargs):
        if 'commits' in url:
            return types.SimpleNamespace(json=lambda: [{
                'commit': {'committer': {'date': '2024-01-01T00:00:00Z'}}
            }])
        return types.SimpleNamespace(json=lambda: {
            'open_issues_count': 5,
            'stargazers_count': 20,
        })

    class FakeDT:
        @staticmethod
        def now(tz=None):
            return now

        @staticmethod
        def fromisoformat(s: str):
            return datetime.fromisoformat(s)

    monkeypatch.setattr(ai_buddy, 'api_request', fake_api_request)
    monkeypatch.setattr(ai_buddy, 'datetime', FakeDT)
    score = ai_buddy.alignment_score({'commits': 1.0, 'issues': 0.5, 'stars': 0.2})
    expected = 1.0 * ((72 - 24) / 72) + 0.5 * (-5) + 0.2 * (20 / 100)
    assert round(expected, 3) == score
