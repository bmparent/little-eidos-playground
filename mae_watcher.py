import argparse
import subprocess
import sys

import gh_api


def main() -> None:
    parser = argparse.ArgumentParser(description="Check local repo against GitHub")
    parser.add_argument("--owner", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--branch", default=None)
    args = parser.parse_args()

    repo = gh_api.get_repo(args.owner, args.repo)
    branch = args.branch or repo["default_branch"]
    remote = gh_api.api_request(
        "GET", f"/repos/{args.owner}/{args.repo}/commits/{branch}"
    )
    remote_sha = remote["sha"]

    local_sha = (
        subprocess.check_output(["git", "rev-parse", branch]).decode().strip()
    )

    if local_sha != remote_sha:
        print(f"Local {branch} is {local_sha}, remote is {remote_sha}")
        sys.exit(1)
    print("Repository is up to date")


if __name__ == "__main__":
    main()
