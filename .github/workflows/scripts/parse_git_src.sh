#!/bin/bash

parse_git_src() {
    PACKAGE=$1
    SRC="$2"
    echo "REPO_${PACKAGE}=$(echo "${SRC}" | cut -f1 -d#)" >> $GITHUB_OUTPUT
    echo "REF_${PACKAGE}=$(echo "${SRC}"  | cut -f2 -d#)" >> $GITHUB_OUTPUT
}