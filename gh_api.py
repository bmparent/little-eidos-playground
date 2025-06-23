import os
import requests

API = 'https://api.github.com'


def _headers():
    token = os.getenv('GH_BOT_TOKEN')
    if not token:
        raise FileNotFoundError('GH_BOT_TOKEN not set')
    return {'Authorization': f'token {token}', 'Accept': 'application/vnd.github+json'}


def list_branches(repo):
    url = f'{API}/repos/{repo}/branches'
    resp = requests.get(url, headers=_headers())
    resp.raise_for_status()
    return resp.json()


def delete_branch(repo, name):
    url = f'{API}/repos/{repo}/git/refs/heads/{name}'
    resp = requests.delete(url, headers=_headers())
    if resp.status_code not in (204, 422):
        resp.raise_for_status()
    return resp.status_code
