import os
import time
from typing import Any, Dict, List, Optional

import requests

BASE_URL = "https://api.github.com"


def _get_token() -> str:
    token = os.getenv("GH_BOT_TOKEN")
    if not token:
        raise RuntimeError("GH_BOT_TOKEN environment variable not set")
    return token


def _raw_request(method: str, path: str, **kwargs: Any) -> requests.Response:
    url = BASE_URL + path
    headers = kwargs.pop("headers", {})
    headers.setdefault("Authorization", f"Bearer {_get_token()}")
    headers.setdefault("Accept", "application/vnd.github+json")

    while True:
        resp = requests.request(method, url, headers=headers, **kwargs)
        if resp.status_code == 403 and resp.headers.get("X-RateLimit-Remaining") == "0":
            reset = int(resp.headers.get("X-RateLimit-Reset", "0"))
            time.sleep(max(reset - int(time.time()), 1))
            continue
        if resp.status_code >= 400:
            raise RuntimeError(f"GitHub API error {resp.status_code}: {resp.text}")
        if resp.headers.get("X-RateLimit-Remaining") == "0":
            reset = int(resp.headers.get("X-RateLimit-Reset", "0"))
            time.sleep(max(reset - int(time.time()), 1))
        return resp


def api_request(method: str, path: str, **kwargs: Any) -> Dict:
    """Generic GitHub API request returning parsed JSON."""
    resp = _raw_request(method, path, **kwargs)
    if resp.status_code == 204:
        return {}
    return resp.json()


def get_repo(owner: str, repo: str) -> Dict:
    return api_request("GET", f"/repos/{owner}/{repo}")


def list_branches(owner: str, repo: str, prefix: Optional[str] = None) -> List[Dict]:
    branches: List[Dict] = []
    path = f"/repos/{owner}/{repo}/branches"
    params = {"per_page": 100}
    while True:
        resp = _raw_request("GET", path, params=params)
        data = resp.json()
        for b in data:
            if prefix is None or b["name"].startswith(prefix):
                branches.append(b)
        if "next" not in resp.links:
            break
        next_url = resp.links["next"]["url"]
        path = next_url.replace(BASE_URL, "")
        params = {}
    return branches


def delete_branch(owner: str, repo: str, branch: str) -> None:
    api_request("DELETE", f"/repos/{owner}/{repo}/git/refs/heads/{branch}")


def search_code(
    query: str,
    owner: Optional[str] = None,
    repo: Optional[str] = None,
    per_page: int = 30,
) -> List[Dict]:
    q = query
    if owner and repo:
        q += f" repo:{owner}/{repo}"
    page = 1
    items: List[Dict] = []
    while True:
        data = api_request(
            "GET",
            "/search/code",
            params={"q": q, "per_page": per_page, "page": page},
        )
        batch = data.get("items", [])
        items.extend(batch)
        if len(batch) < per_page:
            break
        page += 1
    return items
