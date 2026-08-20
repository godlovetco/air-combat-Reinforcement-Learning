import logging
import os
import tempfile
import unittest

from dcs_bridge import logging_setup


class LoggingSetupTest(unittest.TestCase):
    def tearDown(self):
        lg = logging.getLogger(logging_setup.LOGGER_NAME)
        for h in list(lg.handlers):
            lg.removeHandler(h)
            h.close()

    def test_console_only_when_no_file(self):
        lg = logging_setup.setup_logging(level="DEBUG", log_file="")
        self.assertEqual(lg.level, logging.DEBUG)
        self.assertEqual(len(lg.handlers), 1)  # console only
        self.assertFalse(lg.propagate)

    def test_writes_to_rotating_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "logs", "ucav.log")
            lg = logging_setup.setup_logging(level="INFO", log_file=path)
            lg.info("hello market")
            for h in lg.handlers:
                h.flush()
            self.assertTrue(os.path.exists(path))
            with open(path, encoding="utf-8") as fh:
                self.assertIn("hello market", fh.read())

    def test_idempotent_no_handler_pileup(self):
        logging_setup.setup_logging(level="INFO", log_file="")
        logging_setup.setup_logging(level="INFO", log_file="")
        lg = logging.getLogger(logging_setup.LOGGER_NAME)
        self.assertEqual(len(lg.handlers), 1)


if __name__ == "__main__":
    unittest.main()
