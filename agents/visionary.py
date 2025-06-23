import base64
import json
from datetime import datetime, timezone
from pathlib import Path

from sensors import ai_buddy
from gh_api import api_request, API

OWNER = ai_buddy.OWNER
REPO = ai_buddy.REPO
MEM_PATH = Path('memory.json')
DOCS_DIR = Path('docs')


def load_mem() -> dict:
    if MEM_PATH.exists():
        try:
            with MEM_PATH.open() as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_mem(mem: dict) -> None:
    with MEM_PATH.open('w') as f:
        json.dump(mem, f)


def create_branch(branch: str, sha: str) -> str:
    repo = f'{OWNER}/{REPO}'
    url = f'{API}/repos/{repo}/git/refs'
    resp = api_request('POST', url, json={'ref': f'refs/heads/{branch}', 'sha': sha})
    if resp.status_code == 422:
        branch = f"{branch}{datetime.now(timezone.utc).strftime('%S')}"
        api_request('POST', url, json={'ref': f'refs/heads/{branch}', 'sha': sha})
    return branch


def commit_doc(branch: str, path: Path) -> None:
    repo = f'{OWNER}/{REPO}'
    url = f'{API}/repos/{repo}/contents/{path.name}'
    with path.open('rb') as f:
        content = base64.b64encode(f.read()).decode()
    api_request('PUT', url, json={'message': 'vision update', 'content': content, 'branch': branch})


def open_pr(branch: str, score: float) -> None:
    repo = f'{OWNER}/{REPO}'
    url = f'{API}/repos/{repo}/pulls'
    api_request('POST', url, json={'title': f'\ud83c\udf31 Emergent Vision Update – {score}', 'head': branch, 'base': 'main'})


def main() -> int:
    score = ai_buddy.alignment_score()
    hours, issues, stars = ai_buddy.fetch_metrics()
    mem = load_mem()
    last = mem.get('last_alignment', 0.0)
    mem['last_alignment'] = score
    save_mem(mem)
    if score < last + 0.3:
        return 0

    DOCS_DIR.mkdir(exist_ok=True)
    now = datetime.now(timezone.utc)
    doc_path = DOCS_DIR / f'vision_{now.strftime("%Y%m%d")}.md'
    with doc_path.open('w') as f:
        f.write(f'# Vision {now.strftime("%Y-%m-%d")}\n\n')
        f.write(f'Alignment score: {score}\n\n')
        f.write('* Hours since last commit: {0:.1f}\n'.format(hours))
        f.write(f'* Open issues: {issues}\n')
        f.write(f'* Stars: {stars}\n')

    repo = f'{OWNER}/{REPO}'
    repo_data = api_request('GET', f'{API}/repos/{repo}').json()
    default_branch = repo_data['default_branch']
    ref = api_request('GET', f'{API}/repos/{repo}/git/ref/heads/{default_branch}').json()
    sha = ref['object']['sha']
    branch = create_branch(now.strftime('vision/%Y%m%d-%H%M'), sha)
    commit_doc(branch, doc_path)
    open_pr(branch, score)
    return 0


def cli(argv=None):
    raise SystemExit(main())


if __name__ == '__main__':
    cli()
