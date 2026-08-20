"""Resolve the source refs used by the scale-training workflow."""

from __future__ import annotations

import datetime
import json
import os
import re
import urllib.parse
import urllib.request
from typing import Any

XLA_REPOSITORY = "https://github.com/NVIDIA/xla_staging"
XLA_OWNER = "NVIDIA"
XLA_REPO = "xla_staging"
XLA_BRANCH = "nv/staging"
TAG_PATTERN = re.compile(r"^staging-(\d{4}-\d{2}-\d{2})$")


def github_headers() -> dict[str, str]:
    """Return GitHub API headers, using the Actions token when available."""
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "JAX-Toolbox-scale-training",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token := os.environ.get("GH_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    return headers


def return_json_from_url(url: str) -> Any:
    """Fetch and decode a JSON response from the GitHub API."""
    request = urllib.request.Request(url, headers=github_headers())
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def fetch_tags() -> list[dict[str, Any]]:
    """Fetch all tags in the XLA staging repository."""
    tags: list[dict[str, Any]] = []
    page = 1
    per_page = 100

    while True:
        query = urllib.parse.urlencode({"page": page, "per_page": per_page})
        response = return_json_from_url(
            f"https://api.github.com/repos/{XLA_OWNER}/{XLA_REPO}/tags?{query}"
        )
        if not isinstance(response, list):
            raise RuntimeError("GitHub returned an unexpected response for XLA tags")

        tags.extend(response)
        if len(response) < per_page:
            return tags
        page += 1


def latest_staging_tag(tags: list[dict[str, Any]]) -> tuple[str, str]:
    """Return the newest date-formatted staging tag and its commit SHA."""
    candidates: list[tuple[datetime.date, str, str]] = []

    for tag in tags:
        name = tag.get("name")
        commit = tag.get("commit")
        if not isinstance(name, str) or not isinstance(commit, dict):
            continue

        match = TAG_PATTERN.fullmatch(name)
        sha = commit.get("sha")
        if match is None or not isinstance(sha, str):
            continue

        try:
            tag_date = datetime.date.fromisoformat(match.group(1))
        except ValueError:
            continue
        candidates.append((tag_date, name, sha))

    if not candidates:
        raise RuntimeError(
            f"No tags matching {TAG_PATTERN.pattern!r} found in "
            f"{XLA_OWNER}/{XLA_REPO}"
        )

    _, name, sha = max(candidates)
    return name, sha


def xla_ref(tags: list[dict[str, Any]]) -> dict[str, str]:
    """Build the XLA source-ref payload consumed by ci.yaml."""
    xla_tag, xla_commit = latest_staging_tag(tags)
    return {
        "repository": XLA_REPOSITORY,
        "branch": XLA_BRANCH,
        "tag": xla_tag,
        "commit": xla_commit,
        "urlref": f"{XLA_REPOSITORY}#{xla_tag}",
    }


def main() -> None:
    print(json.dumps(xla_ref(fetch_tags()), sort_keys=True))


if __name__ == "__main__":
    main()
