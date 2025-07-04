#!/bin/bash


REPO_PATH=$1
BAD_COMMIT=$2

if [ -z "$REPO_PATH" ] || [ -z "$BAD_COMMIT" ]; then
    echo "Usage: $0 <repo_path> <bad_commit>"
    exit 1
fi

cd ${REPO_PATH}

if git merge-base --is-ancestor ${BAD_COMMIT} HEAD; then
    echo "The commit ${BAD_COMMIT} is an ancestor of the current HEAD."
    exit 1
else
    echo "The commit ${BAD_COMMIT} is NOT an ancestor of the current HEAD."
    exit 0
fi
