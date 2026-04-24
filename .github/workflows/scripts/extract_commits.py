import json
import os
import urllib.parse
import urllib.request

OWNER, REPO = "openxla", "xla"
BRANCH_NAME = "nv-staging/latest"


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


H = {
    "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

encoded_branch = urllib.parse.quote(BRANCH_NAME, safe="")
latest = return_json_from_url(
    f"https://api.github.com/repos/{OWNER}/{REPO}/branches/{encoded_branch}"
)
# return to the gha
print(json.dumps({"name": latest["name"], "commit": {"sha": latest["commit"]["sha"]}}))
