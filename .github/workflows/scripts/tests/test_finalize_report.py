"""Offline report regression tests; requires bash, jq and PyYAML.

Run with: python -m unittest discover -s .github/workflows/scripts/tests
"""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import yaml


WORKFLOW = Path(__file__).resolve().parents[2] / "_finalize.yaml"
STEPS = yaml.safe_load(WORKFLOW.read_text())["jobs"]["report"]["steps"]
SCRIPTS = {step["name"]: step["run"] for step in STEPS if "run" in step}


class FinalizeReportTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.summary = self.root / "summary.md"
        self.env = dict(
            os.environ,
            GITHUB_STEP_SUMMARY=str(self.summary),
            REPORT_ARTIFACT_NAME="artifact-final-report",
        )

    def artifact(self, name, payload, filename="sitrep.json"):
        directory = self.root / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / filename).write_text(json.dumps(payload))

    def run_step(self, name):
        return subprocess.run(
            ["bash", "-e", "-c", SCRIPTS[name]],
            cwd=self.root,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=10,
        )

    def assert_reports(self):
        result = self.run_step("Concatenate all sitreps")
        self.assertEqual(result.returncode, 0, result.stderr)
        entries = json.loads((self.root / "sitrep.json").read_text())
        self.assertEqual(len(entries), 2)
        self.assertEqual(
            {key: value for entry in entries for key, value in entry.items()},
            {"./job one": "first\nsummary", "./job-two": "second"},
        )
        summary = self.summary.read_text()
        self.assertEqual(summary.count("first\nsummary"), 1)
        self.assertEqual(summary.count("second"), 1)
        result = self.run_step("Combine all json into final summary")
        self.assertEqual(result.returncode, 0, result.stderr)
        combined = json.loads((self.root / "combined.json").read_text())
        self.assertEqual(set(combined), {"job one", "job-two"})
        self.assertEqual(combined["job one"]["sitrep"]["summary"], "first\nsummary")
        self.assertEqual(combined["job one"]["metrics_summary"], {"rate": 1})
        self.assertEqual(combined["job-two"]["status"], {"outcome": "success"})

    def per_job_inputs(self):
        self.artifact("job one", {"summary": "first\nsummary", "outcome": "success"})
        self.artifact("job-two", {"summary": "second"})
        self.artifact("job one", {"rate": 1}, "metrics_summary.json")
        self.artifact("job-two", {"outcome": "success"}, "test-status.json")

    def test_ordinary_inputs(self):
        self.per_job_inputs()
        self.assert_reports()

    def test_retained_aggregate(self):
        self.per_job_inputs()
        self.artifact("artifact-final-report", [{"./old-job": "old summary"}])
        self.artifact("artifact-final-report", {"old-job": {}}, "combined.json")
        self.assert_reports()

    def test_custom_aggregate_name(self):
        self.per_job_inputs()
        self.env["REPORT_ARTIFACT_NAME"] = "custom report"
        self.artifact("custom report", [{"./old-job": "old summary"}])
        self.assert_reports()

    def test_root_output_is_not_reingested(self):
        self.per_job_inputs()
        (self.root / "sitrep.json").write_text('[{"old": "summary"}]')
        (self.root / "fsitrep.json").write_text("stale output")
        self.assert_reports()

    def test_missing_or_malformed_summary_is_visible(self):
        for payload in ({}, {"summary": None}, {"summary": []}, []):
            with self.subTest(payload=payload):
                self.artifact("required-job", payload)
                result = self.run_step("Concatenate all sitreps")
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((self.root / "sitrep.json").exists())
                self.assertFalse(self.summary.exists())

    def test_invalid_json_is_visible(self):
        self.artifact("required-job", {})
        (self.root / "required-job/sitrep.json").write_text("invalid JSON")
        self.assertNotEqual(self.run_step("Concatenate all sitreps").returncode, 0)

    def test_multiple_json_documents_are_rejected(self):
        self.artifact("required-job", {})
        (self.root / "required-job/sitrep.json").write_text(
            '{"summary":"one"}\n{"summary":"two"}'
        )
        self.assertNotEqual(self.run_step("Concatenate all sitreps").returncode, 0)

    def test_only_aggregate_is_not_success(self):
        self.artifact("artifact-final-report", [{"old": "summary"}])
        result = self.run_step("Concatenate all sitreps")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No per-job sitrep", result.stderr)


if __name__ == "__main__":
    unittest.main()
