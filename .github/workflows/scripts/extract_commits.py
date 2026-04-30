import base64
import json
import os
import re
import urllib.parse
import urllib.request

OWNER, REPO = "openxla", "xla"
BRANCH_NAME = "nv-staging/latest"
STAGING_FILE = "STAGING.md"
DEFAULT_JAX_REPO = "https://github.com/jax-ml/jax.git"


def return_json_from_url(url: str) -> dict:
    """Function that uses urllib to return the json content

    Args:
        url (str): The url to fetch the json content from

    Returns:
        dict: The json content as a dictionary
    """
    req = urllib.request.Request(url, headers=H)
    with urllib.request.urlopen(req) as response:
        data = response.read()
    return json.loads(data)


def return_file_from_repo(path: str, ref: str) -> str:
    """Return a text file from a repo ref via the GitHub contents API."""
    encoded_ref = urllib.parse.quote(ref, safe="")
    encoded_path = urllib.parse.quote(path)
    response = return_json_from_url(
        f"https://api.github.com/repos/{OWNER}/{REPO}/contents/{encoded_path}?ref={encoded_ref}"
    )
    if response.get("encoding") != "base64":
        raise RuntimeError(
            f"Unexpected encoding for {path} at {ref}: {response.get('encoding')}"
        )
    return base64.b64decode(response["content"]).decode()


def extract_jax_source(staging_text: str) -> tuple[str, str]:
    """Extract the JAX repo URL and commit ref from STAGING.md."""
    for line in staging_text.splitlines():
        if "JAX commit" not in line:
            continue
        match = re.search(
            r"https://github\.com/([^/\s)]+/[^/\s)]+)/commit/([0-9a-f]{7,40})",
            line,
        )
        if match is not None:
            repo_path, commit = match.groups()
            return f"https://github.com/{repo_path}.git", commit

        match = re.search(r"([0-9a-f]{7,40})", line)
        if match is not None:
            return DEFAULT_JAX_REPO, match.group(1)

    raise RuntimeError(f"Failed to find the JAX commit in {STAGING_FILE}")


H = {
    "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

encoded_branch = urllib.parse.quote(BRANCH_NAME, safe="")
latest = return_json_from_url(
    f"https://api.github.com/repos/{OWNER}/{REPO}/branches/{encoded_branch}"
)
staging_text = return_file_from_repo(STAGING_FILE, BRANCH_NAME)
jax_repo, jax_commit = extract_jax_source(staging_text)
# return to the gha
print(
    json.dumps(
        {
            "name": latest["name"],
            "commit": {"sha": latest["commit"]["sha"]},
            "jax": {"repo": jax_repo, "commit": jax_commit},
        }
    )
)
