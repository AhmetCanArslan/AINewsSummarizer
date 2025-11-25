#!/usr/bin/env bash
set -e

REMOTE=${1:-origin}
BRANCH=${2:-main}
MODEL_DIR="models/t5/t5_finetuned_2"

echo "Remote: $REMOTE | Branch: $BRANCH"
echo "Model dir: $MODEL_DIR"

# 1) Check tools
command -v git >/dev/null 2>&1 || { echo "git not found. Install git."; exit 1; }
command -v git-lfs >/dev/null 2>&1 || { echo "git-lfs not found. Install git-lfs (https://git-lfs.github.com/) and run 'git lfs install'."; exit 1; }

# 2) Determine repository root (use current dir if not in a git repo)
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || true)
if [ -z "$REPO_ROOT" ]; then
  echo "Warning: not inside a git repository. Using current directory as repo root."
  REPO_ROOT=$(pwd)
fi
echo "Repository root: $REPO_ROOT"

# 3) Resolve model dir path relative to repo root and verify it exists
MODEL_PATH="$REPO_ROOT/$MODEL_DIR"
if [ ! -d "$MODEL_PATH" ]; then
  echo "Error: model directory not found:"
  echo "  Expected: $MODEL_PATH"
  echo "Please set MODEL_DIR correctly relative to the repository root or run this script from the repo root."
  exit 1
fi

# 4) Initialize LFS in the repo root
git -C "$REPO_ROOT" lfs install

# 5) Track model files with git-lfs if not already tracked
GITATTR="$REPO_ROOT/.gitattributes"
PATTERN="$MODEL_DIR/**"

if [ -f "$GITATTR" ] && grep -Fq "$PATTERN" "$GITATTR"; then
  echo "\"$PATTERN\" already tracked in .gitattributes"
else
  git -C "$REPO_ROOT" lfs track "$PATTERN"
  git -C "$REPO_ROOT" add .gitattributes
fi

# 6) Add model files (from repo root) and commit/push
echo "Adding model files to git (this may be large)..."
git -C "$REPO_ROOT" add "$MODEL_DIR"

git -C "$REPO_ROOT" commit -m "Add latest fine-tuned T5 model" || echo "Nothing to commit."

echo "Pushing to $REMOTE/$BRANCH ..."
git -C "$REPO_ROOT" push "$REMOTE" "$BRANCH"

echo "Done"