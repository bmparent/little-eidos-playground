import os
import time
from pathlib import Path

import pytest
import responses

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gh_api


@pytest.mark.fast
@responses.activate
def test_pagination_and_rate_limit(monkeypatch):
    os.environ['GH_BOT_TOKEN'] = 'x'
    sleeps = []
    monkeypatch.setattr(time, 'sleep', lambda s: sleeps.append(s))

    responses.add(
        responses.GET,
        'https://api.github.com/repos/o/r/branches',
        json=[{'name': 'a'}, {'name': 'b'}],
        headers={
            'Link': '<https://api.github.com/repos/o/r/branches?page=2>; rel="next"',
            'X-RateLimit-Remaining': '15',
        },
        status=200,
    )
    responses.add(
        responses.GET,
        'https://api.github.com/repos/o/r/branches?page=2',
        json=[{'name': 'c'}],
        headers={'X-RateLimit-Remaining': '0', 'X-RateLimit-Reset': str(int(time.time()) + 1)},
        status=200,
    )

    branches = gh_api.list_branches('o', 'r')
    assert [b['name'] for b in branches] == ['a', 'b', 'c']
    assert sleeps and sleeps[0] >= 1


def test_missing_token(monkeypatch):
    monkeypatch.delenv('GH_BOT_TOKEN', raising=False)
    with pytest.raises(RuntimeError):
        gh_api.api_request('GET', '/repos/foo/bar')
