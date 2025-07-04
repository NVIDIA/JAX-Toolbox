#!/bin/bash

if [ ! -f "feature_file.txt" ]; then
    echo "Build FAILED: The feature commit was not applied (feature_file.txt is missing)."
    exit 1
fi

echo "Mock build script: Build successful (feature commit is present)."
exit 0
