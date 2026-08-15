#!/usr/bin/env bash
# Push the committed v1.0.0 to a NEW GitHub private repository.
#
# Prerequisite (one of):
#   A) A PAT with repo-creation rights exported as GITHUB_TOKEN:
#        export GITHUB_TOKEN=ghp_xxxx
#   B) You already created the empty private repo on github.com/new
#      (no README) - then just run this script without GITHUB_TOKEN.
#
# Usage: scripts/push_to_github.sh [REPO_NAME]   (default: trop2-binder-platform)
set -euo pipefail

REPO="${1:-trop2-binder-platform}"
OWNER="qiuzh37"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE="git@github.com:${OWNER}/${REPO}.git"

cd "$ROOT"

if [ -n "${GITHUB_TOKEN:-}" ]; then
    echo "[push] creating private repository ${OWNER}/${REPO} via API"
    curl -fsS -X POST https://api.github.com/user/repos \
        -H "Authorization: Bearer ${GITHUB_TOKEN}" \
        -H "Accept: application/vnd.github+json" \
        -d "{\"name\":\"${REPO}\",\"private\":true,\"description\":\"TROP2 R87-T88 cleaved-state miniprotein binder design platform (PRD v1.0)\"}" \
        | grep -E '"(full_name|private)"' | head -2 || true
fi

if git remote | grep -q '^origin$'; then
    git remote set-url origin "$REMOTE"
else
    git remote add origin "$REMOTE"
fi

echo "[push] pushing to $REMOTE"
git push -u origin main
echo "[push] done. Open: https://github.com/${OWNER}/${REPO}"
