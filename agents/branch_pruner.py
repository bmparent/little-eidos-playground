import argparse
import os
from datetime import datetime, timezone, timedelta

from gh_api import list_branches, delete_branch


DEFAULT_KEEP = {'main', 'master'}


def parse_args(args=None):
    p = argparse.ArgumentParser(description='Prune stale branches')
    p.add_argument('--repo', default=os.getenv('GITHUB_REPOSITORY'))
    p.add_argument('--days', type=int, default=30)
    p.add_argument('--keep', nargs='*', default=list(DEFAULT_KEEP))
    return p.parse_args(args)


def main(argv=None):
    opts = parse_args(argv)
    cutoff = datetime.now(timezone.utc) - timedelta(days=opts.days)
    branches = list_branches(opts.repo)
    for br in branches:
        name = br['name']
        if name in opts.keep:
            continue
        commit_date = br['commit']['commit']['committer']['date']
        dt = datetime.fromisoformat(commit_date.replace('Z', '+00:00'))
        if dt < cutoff:
            delete_branch(opts.repo, name)


def cli(argv=None):
    main(argv)


if __name__ == '__main__':
    cli()
