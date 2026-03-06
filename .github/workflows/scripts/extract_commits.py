import json
import os
import urllib.request

OWNER, REPO = "sfvaroglu", "xla"


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


def commit_date(sha: str) -> str:
    """Return the commit date for a given commit sha
    Args:
        sha (str): The commit sha
    Returns:
        str: The commit date in ISO format
    """
    url = f"https://api.github.com/repos/{OWNER}/{REPO}/commits/{sha}"
    commit_data = return_json_from_url(url)
    return commit_data["commit"]["author"]["date"]


H = {
    "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}


tags = []
page = 1
# here we're analysing the pages and retrieve the tags
while True:
    batch = return_json_from_url(
        f"https://api.github.com/repos/{OWNER}/{REPO}/tags?per_page=100&page={page}"
    )
    if not batch:
        break
    tags.extend(batch)
    page += 1

if not tags:
    print("No tags found in the repository.")
    exit(1)

print(f"Found {len(tags)} tags in the repository.")
for tag in tags:
    print(f"Tag: {tag['name']}, Commit SHA: {tag['commit']['sha']}")
    current_time_info = commit_date(tag["commit"]["sha"])
    print(f"Commit date for tag {tag['name']}: {current_time_info}")
# sort the tags by commit date
tags.sort(key=lambda tag: commit_date(tag["commit"]["sha"]), reverse=True)
latest = tags[0]
# return to the gha
print(json.dumps({"name": latest["name"], "commit": {"sha": latest["commit"]["sha"]}}))
