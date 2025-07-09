"""Safety helpers for the action loop."""

import os
import requests

BUDGETS = {
    "openai": 10000,
}


def within_budget(service: str, tokens: int, limit: int | None = None) -> bool:
    """Return True if ``tokens`` for ``service`` are below ``limit``."""
    limit = limit or BUDGETS.get(service, 10000)
    return tokens <= limit


def moderate_text(text: str) -> bool:
    """Simple text length moderation."""
    return len(text) < 2000


def global_pause() -> bool:
    """Check open GitHub issues for a pause flag."""
    token = os.getenv("GH_BOT_TOKEN")
    repo = os.getenv("GITHUB_REPOSITORY")
    if not token or not repo:
        return False
    url = f"https://api.github.com/repos/{repo}/issues"
    try:
        resp = requests.get(url, headers={"Authorization": f"token {token}"}, timeout=10)
        for issue in resp.json():
            if issue.get("state") == "open" and "PAUSE EIDOS" in issue.get("title", ""):
                return True
    except Exception:
        return False
    return False
