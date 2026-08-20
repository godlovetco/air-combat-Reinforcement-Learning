import tempfile
import unittest
from pathlib import Path

from dcs_bridge import install


class InstallerTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.saved_games = Path(self._tmp.name) / "Saved Games" / "DCS"
        self.saved_games.mkdir(parents=True)
        self.export = self.saved_games / "Scripts" / "Export.lua"

    def tearDown(self):
        self._tmp.cleanup()

    def test_fresh_install_copies_addon_and_registers(self):
        install.install(self.saved_games)
        script = self.saved_games / "Scripts" / "UCAVPilot" / "UCAVPilotExport.lua"
        self.assertTrue(script.is_file())
        text = self.export.read_text(encoding="utf-8")
        self.assertIn(install.BEGIN_MARK, text)
        self.assertIn("UCAVPilotExport.lua", text)

    def test_install_is_idempotent(self):
        install.install(self.saved_games)
        install.install(self.saved_games)
        text = self.export.read_text(encoding="utf-8")
        self.assertEqual(text.count(install.BEGIN_MARK), 1)
        self.assertEqual(text.count(install.END_MARK), 1)

    def test_preserves_existing_export_content(self):
        self.export.parent.mkdir(parents=True, exist_ok=True)
        self.export.write_text("-- Tacview\ndofile('Tacview.lua')\n", encoding="utf-8")
        install.install(self.saved_games)
        text = self.export.read_text(encoding="utf-8")
        self.assertIn("Tacview", text)          # existing content kept
        self.assertIn(install.BEGIN_MARK, text)  # loader appended

    def test_uninstall_removes_block_but_keeps_other_content(self):
        self.export.parent.mkdir(parents=True, exist_ok=True)
        self.export.write_text("dofile('SRS.lua')\n", encoding="utf-8")
        install.install(self.saved_games)
        install.uninstall(self.saved_games)
        text = self.export.read_text(encoding="utf-8")
        self.assertIn("SRS.lua", text)
        self.assertNotIn(install.BEGIN_MARK, text)
        self.assertFalse((self.saved_games / "Scripts" / "UCAVPilot").exists())

    def test_uninstall_deletes_export_if_we_created_it(self):
        install.install(self.saved_games)  # created Export.lua ourselves
        install.uninstall(self.saved_games)
        self.assertFalse(self.export.exists())

    def test_dry_run_writes_nothing(self):
        actions = install.install(self.saved_games, dry_run=True)
        self.assertTrue(actions)
        self.assertFalse((self.saved_games / "Scripts" / "UCAVPilot").exists())
        self.assertFalse(self.export.exists())

    def test_register_export_reports_states(self):
        scripts = self.saved_games / "Scripts"
        self.assertEqual(install.register_export(scripts / "Export.lua"), "created")
        self.assertEqual(install.register_export(scripts / "Export.lua"), "unchanged")

    def test_bundled_addon_source_exists(self):
        src = install.default_addon_src() / "Scripts" / "UCAVPilot" / "UCAVPilotExport.lua"
        self.assertTrue(src.is_file())


if __name__ == "__main__":
    unittest.main()
