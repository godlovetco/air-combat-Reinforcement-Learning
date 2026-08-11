import json
import os
import tempfile
import unittest
import zipfile

from dcs_bridge import licensing, support


class DiagnosticsTest(unittest.TestCase):
    def test_reports_core_fields(self):
        d = support.collect_diagnostics()
        self.assertEqual(d["product"]["name"], "UCAV AI Pilot")
        self.assertIn("version", d["product"])
        self.assertIn("system", d["platform"])
        self.assertIn("edition", d["license"])
        self.assertIn("numpy", d["optional_packages"])

    def test_never_leaks_secret_values(self):
        secret_key = licensing.issue_key("pro", "SecretCustomer", ["core", "llm"])
        os.environ["UCAV_LICENSE_KEY"] = secret_key
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-SUPERSECRET-VALUE"
        try:
            blob = json.dumps(support.collect_diagnostics())
        finally:
            os.environ.pop("UCAV_LICENSE_KEY", None)
            os.environ.pop("ANTHROPIC_API_KEY", None)

        # The raw secrets must not appear anywhere in the diagnostics.
        self.assertNotIn(secret_key, blob)
        self.assertNotIn("SUPERSECRET", blob)
        # ...but their presence is reported, and the license *status* is.
        d = json.loads(blob)
        self.assertTrue(d["env_present"]["UCAV_LICENSE_KEY"])
        self.assertTrue(d["env_present"]["ANTHROPIC_API_KEY"])
        self.assertEqual(d["license"]["licensee"], "SecretCustomer")

    def test_missing_checkpoint_is_flagged_not_fatal(self):
        d = support.collect_diagnostics(checkpoint="/nope/missing.npz")
        self.assertTrue(d["checkpoint"].get("missing"))

    def test_checkpoint_fingerprint_when_present(self):
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as fh:
            fh.write(b"weights")
            path = fh.name
        self.addCleanup(os.remove, path)
        d = support.collect_diagnostics(checkpoint=path)
        self.assertEqual(d["checkpoint"]["bytes"], 7)
        self.assertEqual(len(d["checkpoint"]["sha256"]), 16)


class BundleTest(unittest.TestCase):
    def test_bundle_contains_diagnostics_and_log_tail(self):
        with tempfile.TemporaryDirectory() as d:
            log = os.path.join(d, "ucav.log")
            with open(log, "w", encoding="utf-8") as fh:
                fh.write("line one\nline two\n")
            out = os.path.join(d, "bundle.zip")
            support.write_bundle(out, log_file=log)

            with zipfile.ZipFile(out) as zf:
                names = zf.namelist()
                self.assertIn("diagnostics.json", names)
                self.assertIn("ucav_pilot.log.tail", names)
                self.assertIn("line two", zf.read("ucav_pilot.log.tail").decode())
                json.loads(zf.read("diagnostics.json"))  # valid JSON

    def test_missing_log_is_noted_not_fatal(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "bundle.zip")
            support.write_bundle(out, log_file=os.path.join(d, "absent.log"))
            with zipfile.ZipFile(out) as zf:
                self.assertIn("no log available",
                              zf.read("ucav_pilot.log.tail").decode())

    def test_print_mode_writes_no_file(self):
        with tempfile.TemporaryDirectory() as d:
            cwd = os.getcwd()
            os.chdir(d)
            try:
                rc = support.main(["--print"])
            finally:
                os.chdir(cwd)
            self.assertEqual(rc, 0)
            self.assertEqual(os.listdir(d), [])


if __name__ == "__main__":
    unittest.main()
