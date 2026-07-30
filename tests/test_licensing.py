import unittest
from datetime import datetime, timezone

from dcs_bridge import licensing as lic


class KeyRoundTripTest(unittest.TestCase):
    def test_issue_and_verify_round_trip(self):
        key = lic.issue_key("pro", "ACME Order 42",
                            ["core", "cca", "wso", "llm"], expires=None)
        info = lic.verify_key(key)
        self.assertTrue(info.valid)
        self.assertEqual(info.edition, "pro")
        self.assertEqual(info.licensee, "ACME Order 42")
        self.assertTrue(info.has("llm"))

    def test_tampered_payload_fails_signature(self):
        key = lic.issue_key("standard", "x", ["core", "llm"])
        prefix, payload, sig = key.split(".")
        # Flip a character in the payload -> signature must no longer match.
        bad_payload = payload[:-1] + ("A" if payload[-1] != "A" else "B")
        with self.assertRaises(lic.LicenseError):
            lic.verify_key(f"{prefix}.{bad_payload}.{sig}")

    def test_wrong_secret_rejected(self):
        key = lic.issue_key("pro", "x", ["llm"], secret=b"vendor-secret")
        with self.assertRaises(lic.LicenseError):
            lic.verify_key(key, secret=b"different-secret")

    def test_malformed_key_raises(self):
        with self.assertRaises(lic.LicenseError):
            lic.verify_key("not-a-key")
        with self.assertRaises(lic.LicenseError):
            lic.verify_key("WRONG1.aaa.bbb")

    def test_unknown_feature_or_edition_rejected_at_issue(self):
        with self.assertRaises(ValueError):
            lic.issue_key("pro", "x", ["teleport"])
        with self.assertRaises(ValueError):
            lic.issue_key("platinum", "x", ["llm"])


class ExpiryTest(unittest.TestCase):
    def test_expired_key_is_invalid(self):
        key = lic.issue_key("standard", "x", ["core", "llm"], expires="2020-01-01")
        info = lic.verify_key(key, now=datetime(2026, 1, 1, tzinfo=timezone.utc))
        self.assertFalse(info.valid)
        self.assertIn("expired", info.reason)

    def test_not_yet_expired_key_is_valid(self):
        key = lic.issue_key("standard", "x", ["core", "llm"], expires="2999-01-01")
        info = lic.verify_key(key, now=datetime(2026, 1, 1, tzinfo=timezone.utc))
        self.assertTrue(info.valid)
        self.assertTrue(info.has("llm"))


class LoadLicenseTest(unittest.TestCase):
    def test_no_key_is_trial_without_llm(self):
        info = lic.load_license(key=None, key_file="/nonexistent/path")
        self.assertTrue(info.is_trial)
        self.assertFalse(info.has("llm"))
        self.assertTrue(info.has("cca"))  # core teaming still works in trial

    def test_valid_key_unlocks_llm(self):
        key = lic.issue_key("pro", "x", ["core", "cca", "wso", "llm"])
        info = lic.load_license(key=key, key_file="/nonexistent/path")
        self.assertFalse(info.is_trial)
        self.assertTrue(info.has("llm"))

    def test_invalid_key_falls_back_to_trial(self):
        info = lic.load_license(key="UCAV1.garbage.sig", key_file="/nonexistent/path")
        self.assertTrue(info.is_trial)
        self.assertFalse(info.has("llm"))

    def test_expired_key_falls_back_to_trial(self):
        key = lic.issue_key("standard", "x", ["llm"], expires="2000-01-01")
        info = lic.load_license(key=key, key_file="/nonexistent/path")
        self.assertTrue(info.is_trial)

    def test_key_file_is_read(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "license.key")
            key = lic.issue_key("pro", "filecustomer", ["core", "llm"])
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(key + "\n")
            info = lic.load_license(key=None, key_file=path)
            self.assertTrue(info.has("llm"))
            self.assertEqual(info.licensee, "filecustomer")


if __name__ == "__main__":
    unittest.main()
