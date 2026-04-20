import json
import os
import re
import urllib.request

OWNER, REPO = "openxla", "xla"
BRANCH_PATTERN = re.compile(r"^nv-staging/(\d{4}-\d{2}-\d{2})$")


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


def branch_date(branch_name: str) -> str:
    """Extract the ISO date suffix from an nv-staging branch name."""
    match = BRANCH_PATTERN.match(branch_name)
    if match is None:
        raise ValueError(f"Unexpected branch name: {branch_name}")
    return match.group(1)


H = {
    "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}


branches = []
page = 1
# here we're analysing the pages and retrieve the branches
while True:
    batch = return_json_from_url(
        f"https://api.github.com/repos/{OWNER}/{REPO}/branches?per_page=100&page={page}"
    )
    if not batch:
        break
    branches.extend(batch)
    page += 1

matching_branches = [
    branch for branch in branches if BRANCH_PATTERN.match(branch["name"]) is not None
]

if not matching_branches:
    print("No nv-staging branches found in the repository.")
    exit(1)

# sort the branches by the date embedded in the branch name
matching_branches.sort(key=lambda branch: branch_date(branch["name"]), reverse=True)
latest = matching_branches[0]
# return to the gha
print(json.dumps({"name": latest["name"], "commit": {"sha": latest["commit"]["sha"]}}))
