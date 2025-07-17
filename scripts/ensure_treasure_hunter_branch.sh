#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'USAGE'
Usage: ensure_treasure_hunter_branch.sh
Ensures the treasure-hunter branch exists and workflow references it.

Optional environment variables:
  REPO_URL          Repository URL (default: https://github.com/bmparent/little-eidos-playground.git)
  TARGET_BRANCH     Branch name to ensure (default: treasure-hunter)
  WORKFLOW_FILE     Workflow file to patch (default: .github/workflows/treasure_hunter.yml)
  PIN_REF           When 1, pin checkout ref to TARGET_BRANCH (default: 1)
  CREATE_TAG        When 1, also create annotated tag TARGET_BRANCH if missing (default: 0)
  FROM_SHA          Commit SHA to create branch from (default: tip of default branch)
  UPDATE_ON_TARGET  Update workflow on TARGET_BRANCH instead of default branch (default: 0)
  GH_PAT_TOKEN      Personal access token for pushes (optional)
USAGE
  exit 0
fi

REPO_URL=${REPO_URL:-https://github.com/bmparent/little-eidos-playground.git}
TARGET_BRANCH=${TARGET_BRANCH:-treasure-hunter}
WORKFLOW_FILE=${WORKFLOW_FILE:-.github/workflows/treasure_hunter.yml}
PIN_REF=${PIN_REF:-1}
CREATE_TAG=${CREATE_TAG:-0}
FROM_SHA=${FROM_SHA:-}
UPDATE_ON_TARGET=${UPDATE_ON_TARGET:-0}
COMMIT_USER_NAME=${COMMIT_USER_NAME:-$(git config --global user.name || echo "CI Bot")}
COMMIT_USER_EMAIL=${COMMIT_USER_EMAIL:-$(git config --global user.email || echo "ci@example.com")}

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  current_remote=$(git remote get-url origin 2>/dev/null || echo "")
  stripped_current=${current_remote#https://}
  stripped_repo=${REPO_URL#https://}
  if [[ "$stripped_current" != "$stripped_repo" ]]; then
    workdir=$(mktemp -d)
    clone_url="$REPO_URL"
    [[ -n "${GH_PAT_TOKEN:-}" ]] && clone_url="https://${GH_PAT_TOKEN}@${REPO_URL#https://}"
    git clone "$clone_url" "$workdir"
    git -C "$workdir" remote set-url origin "$REPO_URL"
    cd "$workdir"
  fi
else
  workdir=$(mktemp -d)
  clone_url="$REPO_URL"
  [[ -n "${GH_PAT_TOKEN:-}" ]] && clone_url="https://${GH_PAT_TOKEN}@${REPO_URL#https://}"
  git clone "$clone_url" "$workdir"
  git -C "$workdir" remote set-url origin "$REPO_URL"
  cd "$workdir"
fi

DEFAULT_BRANCH=""
if command -v gh >/dev/null 2>&1; then
  DEFAULT_BRANCH=$(gh repo view --json defaultBranchRef -q .defaultBranchRef.name 2>/dev/null || true)
fi
if [[ -z "$DEFAULT_BRANCH" ]]; then
  DEFAULT_BRANCH=$(git remote show origin 2>/dev/null | awk '/HEAD branch/ {print $NF}')
fi
if [[ -z "$DEFAULT_BRANCH" ]]; then
  if git ls-remote --exit-code --heads origin main >/dev/null 2>&1; then
    DEFAULT_BRANCH=main
  elif git ls-remote --exit-code --heads origin master >/dev/null 2>&1; then
    DEFAULT_BRANCH=master
  else
    DEFAULT_BRANCH=main
  fi
fi

git fetch origin --prune
branch_exists=0
if git ls-remote --exit-code --heads origin "$TARGET_BRANCH" >/dev/null 2>&1; then
  branch_exists=1
fi

push_url="origin"
if [[ -n "${GH_PAT_TOKEN:-}" ]]; then
  push_url="https://${GH_PAT_TOKEN}@${REPO_URL#https://}"
fi

if [[ $branch_exists -eq 1 ]]; then
  echo "Branch $TARGET_BRANCH already exists" >&2
  git checkout -B "$TARGET_BRANCH" "origin/$TARGET_BRANCH"
else
  base_ref=${FROM_SHA:-"origin/$DEFAULT_BRANCH"}
  echo "Creating branch $TARGET_BRANCH from $base_ref" >&2
  git checkout -B "$TARGET_BRANCH" "$base_ref"
  git push -u "$push_url" "$TARGET_BRANCH"
fi

if [[ "$CREATE_TAG" == "1" ]]; then
  if git ls-remote --exit-code --tags origin "$TARGET_BRANCH" >/dev/null 2>&1; then
    echo "Tag $TARGET_BRANCH already exists on origin" >&2
  else
    if ! git rev-parse -q --verify "refs/tags/$TARGET_BRANCH" >/dev/null; then
      git tag -a "$TARGET_BRANCH" -m "Create $TARGET_BRANCH tag" "$TARGET_BRANCH"
    fi
    git push "$push_url" "refs/tags/$TARGET_BRANCH"
  fi
fi

if [[ "$UPDATE_ON_TARGET" == "1" ]]; then
  work_branch="$TARGET_BRANCH"
else
  git checkout "$DEFAULT_BRANCH"
  work_branch="$DEFAULT_BRANCH"
fi

if [[ "$PIN_REF" == "1" ]]; then
  replacement="      - uses: actions/checkout@v4\n        with:\n          ref: $TARGET_BRANCH\n          fetch-depth: 0\n          token: \${{ secrets.GH_PAT_TOKEN }}  # optional; needed for cross-repo or elevated perms"
else
  replacement="      - uses: actions/checkout@v4\n        with:\n          fetch-depth: 0\n          token: \${{ secrets.GH_PAT_TOKEN }}  # optional; needed for cross-repo or elevated perms"
fi
sed -i "/uses: actions\/checkout@v4/,+2c$replacement" "$WORKFLOW_FILE"

git config user.name "$COMMIT_USER_NAME"
git config user.email "$COMMIT_USER_EMAIL"

git add "$WORKFLOW_FILE"
if ! git diff --cached --quiet; then
  git commit -m "CI: ensure treasure-hunter branch + checkout fix"
  git push "$push_url" "$work_branch"
fi

echo "Branch ensured on $work_branch. Workflow updated." >&2
