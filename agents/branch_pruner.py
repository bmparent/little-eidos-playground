import argparse
import datetime as dt
from typing import List, Tuple

import gh_api


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prune old Git branches via API")
    parser.add_argument("--owner", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--keep", type=int, default=5)
    parser.add_argument("--prefix", default="")
    return parser.parse_args()


def fetch_branch_date(owner: str, repo: str, branch: str) -> dt.datetime:
    commit = gh_api.api_request(
        "GET", f"/repos/{owner}/{repo}/commits/{branch}"
    )
    ts = commit["commit"]["committer"]["date"]
    return dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))


def main() -> None:
    args = parse_args()
    branches = gh_api.list_branches(args.owner, args.repo, prefix=args.prefix or None)

    info: List[Tuple[str, dt.datetime]] = []
    for b in branches:
        info.append((b["name"], fetch_branch_date(args.owner, args.repo, b["name"])))

    info.sort(key=lambda x: x[1], reverse=True)
    keep_set = {name for name, _ in info[: args.keep]}
    cutoff = dt.datetime.utcnow() - dt.timedelta(days=args.days)

    for name, date in info:
        if name in keep_set or date > cutoff:
            continue
        print(f"Deleting {name} last updated {date.isoformat()}")
        gh_api.delete_branch(args.owner, args.repo, name)


if __name__ == "__main__":
    main()
