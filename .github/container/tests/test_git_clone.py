"""CPU-only git-clone.sh regressions; no network or third-party packages needed.

Run: python3 -m unittest discover -s .github/container/tests -v
The git shim redirects only clone to a local repository. All remote/checkout
operations use real Git. The yq fixture checks the unchanged manifest merge
expression and applies its JSON payload to a JSON (valid YAML) manifest.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "git-clone.sh"


class GitCloneTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.seed = self.root / "seed"
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.git = shutil.which("git")
        self.env = {
            k: v for k, v in os.environ.items()
            if not k.startswith("GIT_") and k not in ("GITHUB_TOKEN", "GITLAB_TOKEN")
        }
        self.env.update(
            HOME=str(self.root), GIT_CONFIG_NOSYSTEM="1",
            GIT_CONFIG_GLOBAL=os.devnull, GIT_TERMINAL_PROMPT="0",
            GIT_ASKPASS="/bin/false", REAL_GIT=self.git,
            CLONE_SOURCE=str(self.seed),
        )
        self.run_git("init", "-b", "main", str(self.seed))
        self.run_git("-C", str(self.seed), "-c", "user.name=Fixture",
                     "-c", "user.email=fixture@example.invalid",
                     "commit", "--allow-empty", "-m", "Initial fixture")
        self.commit = self.run_git("-C", str(self.seed), "rev-parse", "HEAD").strip()
        self.run_git("-C", str(self.seed), "branch", "release")
        self.write_command("git", '''#!/bin/bash
set -e
if [[ "$1" == clone ]]; then
    "$REAL_GIT" clone "$CLONE_SOURCE" "$3"
    "$REAL_GIT" -C "$3" remote set-url origin "$2"
    if [[ "$2" == *gitlab* ]]; then
        echo 'gitlab-ci-token:inert-fixture' > "$3/.git/FETCH_HEAD"
    fi
else
    exec "$REAL_GIT" "$@"
fi
''')
        self.write_command("yq", '''#!/usr/bin/env python3
import json, sys
from pathlib import Path
assert sys.argv[1] == 'eval' and sys.argv[2] == '--inplace', sys.argv
assert sys.argv[3].startswith('. += '), sys.argv
update = json.loads(sys.argv[3][5:])
p = Path(sys.argv[4])
data = json.loads(p.read_text()) if p.read_text().strip() else {}
data.update(update)
p.write_text(json.dumps(data))
''')
        self.env["PATH"] = str(self.bin) + os.pathsep + os.environ["PATH"]

    def write_command(self, name, content):
        path = self.bin / name
        path.write_text(content)
        path.chmod(0o755)

    def run_git(self, *args):
        return subprocess.check_output(
            [self.git, *args], env=self.env, text=True, stderr=subprocess.PIPE,
        )

    def check_clone(self, host, credentials="", ref="main", caller_repo=False):
        caller = self.root / "caller"
        caller.mkdir()
        if caller_repo:
            self.run_git("init", str(caller))
            self.run_git("-C", str(caller), "remote", "add", "origin",
                         "https://caller.example/unchanged.git")
        destination = self.root / "checkout"
        manifest = self.root / "manifest.json"
        preserved = {"keep": {"url": "https://existing.example/repo", "mode": "fixture"}}
        manifest.write_text(json.dumps(preserved))
        clean_url = f"https://{host}/repo.git"
        url = f"https://{credentials}{host}/repo.git"
        result = subprocess.run(
            ["bash", str(SCRIPT), "-m", str(manifest), f"{url}#{ref}", str(destination)],
            cwd=caller, env=self.env, text=True, capture_output=True, timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.run_git("-C", str(destination), "rev-parse", "HEAD").strip(), self.commit)
        remotes = self.run_git("-C", str(destination), "remote").splitlines()
        if "gitlab" in host:
            self.assertNotIn("origin", remotes)
            self.assertFalse((destination / ".git/FETCH_HEAD").exists())
            branches = self.run_git("-C", str(destination), "branch", "--format=%(refname:short)").splitlines()
            self.assertEqual("main" in branches, ref == "main")
        else:
            self.assertIn("origin", remotes)
            self.assertEqual(self.run_git("-C", str(destination), "remote", "get-url", "origin").strip(), clean_url)
        if caller_repo:
            self.assertEqual(self.run_git("-C", str(caller), "remote", "get-url", "origin").strip(),
                             "https://caller.example/unchanged.git")
        self.assertEqual(json.loads(manifest.read_text()), {
            **preserved,
            "checkout": {"url": clean_url, "tracking_ref": ref,
                         "latest_verified_commit": self.commit, "mode": "git-clone"},
        })
        if credentials:
            for path in (manifest, destination / ".git/config"):
                self.assertNotIn(credentials.rstrip("@"), path.read_text())

    def test_github_from_non_repository(self):
        self.check_clone("github.example")

    def test_github_credentials_and_caller_origin_preserved(self):
        self.check_clone("github.example", "example-user:inert-password@", caller_repo=True)

    def test_github_token_only_credentials(self):
        self.check_clone("github.example", "inert-token@")

    def test_gitlab_removed_origin(self):
        self.check_clone("gitlab.example")

    def test_gitlab_credentials_and_caller_origin_preserved(self):
        self.check_clone("gitlab.example", "gitlab-ci-token:inert-password@", caller_repo=True)

    def test_gitlab_non_main_ref(self):
        self.check_clone("gitlab.example", "gitlab-ci-token:inert-password@", ref="release")


if __name__ == "__main__":
    unittest.main()
