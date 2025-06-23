from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Tuple

from gh_api import api_request, API

OWNER = 'bmparent'
REPO = 'ai-devops-buddy'


def fetch_metrics() -> Tuple[float, int, int]:
    """Return hours_since_commit, open_issues, stars for the repo."""
    repo = f"{OWNER}/{REPO}"
    commit_resp = api_request('GET', f'{API}/repos/{repo}/commits?per_page=1')
    commit_date = commit_resp.json()[0]['commit']['committer']['date']
    dt = datetime.fromisoformat(commit_date.replace('Z', '+00:00'))
    hours_since = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
    repo_resp = api_request('GET', f'{API}/repos/{repo}')
    data = repo_resp.json()
    issues = int(data.get('open_issues_count', 0))
    stars = int(data.get('stargazers_count', 0))
    return hours_since, issues, stars


def alignment_score(weights: Dict[str, float] | None = None) -> float:
    """Compute alignment score based on repo metrics."""
    if weights is None:
        weights = {'commits': 1.0, 'issues': 0.5, 'stars': 0.2}
    hours_since, issues, stars = fetch_metrics()
    delta_commits = max(0, 72 - hours_since) / 72
    w = weights
    score = w['commits'] * delta_commits + w['issues'] * (-issues) + w['stars'] * (stars / 100)
    return round(score, 3)
