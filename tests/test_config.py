import os
import tempfile
import unittest

from dcs_bridge import config

try:
    import tomllib as _toml  # noqa: F401
    _HAVE_TOML = True
except ModuleNotFoundError:
    try:
        import tomli as _toml  # noqa: F401
        _HAVE_TOML = True
    except ModuleNotFoundError:
        _HAVE_TOML = False


def _write(text):
    fd, path = tempfile.mkstemp(suffix=".toml")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


@unittest.skipUnless(_HAVE_TOML, "no TOML parser (needs Python 3.11+ or tomli)")
class LoadConfigTest(unittest.TestCase):
    def test_flat_keys(self):
        path = _write('radio = true\nleash = "loose"\ntelemetry_port = 7000\n')
        self.addCleanup(os.remove, path)
        cfg = config.load_config(path)
        self.assertEqual(cfg, {"radio": True, "leash": "loose", "telemetry_port": 7000})

    def test_sections_are_flattened(self):
        path = _write('[radio]\nradio = true\n[cca]\nleash = "close"\n')
        self.addCleanup(os.remove, path)
        cfg = config.load_config(path)
        self.assertEqual(cfg, {"radio": True, "leash": "close"})

    def test_dashes_normalized_to_underscores(self):
        path = _write('radio-lang = "en-US"\nformation-side = "left"\n')
        self.addCleanup(os.remove, path)
        cfg = config.load_config(path)
        self.assertEqual(cfg, {"radio_lang": "en-US", "formation_side": "left"})

    def test_unknown_key_rejected_when_valid_keys_given(self):
        path = _write('radio = true\nteleport = true\n')
        self.addCleanup(os.remove, path)
        with self.assertRaises(config.ConfigError) as ctx:
            config.load_config(path, valid_keys={"radio", "leash"})
        self.assertIn("teleport", str(ctx.exception))

    def test_missing_file_raises(self):
        with self.assertRaises(config.ConfigError):
            config.load_config("/nope/does-not-exist.toml")

    def test_invalid_toml_raises(self):
        path = _write("this is = = not toml")
        self.addCleanup(os.remove, path)
        with self.assertRaises(config.ConfigError):
            config.load_config(path)


@unittest.skipUnless(_HAVE_TOML, "no TOML parser (needs Python 3.11+ or tomli)")
class ParseArgsWithConfigTest(unittest.TestCase):
    def test_config_supplies_defaults_and_cli_overrides(self):
        from dcs_bridge.run_pilot import parse_args

        path = _write('leash = "close"\nradio = true\ntarget_speed = 300.0\n')
        self.addCleanup(os.remove, path)

        # Config alone.
        args = parse_args(["--config", path])
        self.assertEqual(args.leash, "close")
        self.assertTrue(args.radio)
        self.assertEqual(args.target_speed, 300.0)

        # CLI flag wins over the config file.
        args = parse_args(["--config", path, "--leash", "loose"])
        self.assertEqual(args.leash, "loose")
        self.assertTrue(args.radio)  # still from config

    def test_unknown_config_key_errors_out(self):
        from dcs_bridge.run_pilot import parse_args

        path = _write("bogus_option = 1\n")
        self.addCleanup(os.remove, path)
        with self.assertRaises(SystemExit):  # parser.error -> SystemExit
            parse_args(["--config", path])


if __name__ == "__main__":
    unittest.main()
